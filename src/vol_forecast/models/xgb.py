import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.metrics import qlike_loss_var
from .wf_util import (get_train_slice,
                      compute_start_pos,
                      compute_train_end_excl, 
                      compute_purged_val_split)
from vol_forecast.schema import require_cols
from typing import Any
from xgboost.core import XGBoostError


def make_xgb_model(
    *,
    early_stopping_rounds: int = 50,
    eval_metric: str = "rmse",
    params_overrides: dict[str, Any] | None = None,
) -> XGBRegressor:
    params: dict[str, Any] = dict(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=3,
        min_child_weight=1,   
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds = int(early_stopping_rounds),
        random_state=0,
        tree_method="hist",
        n_jobs=-1,
        eval_metric=eval_metric,
    )

    if params_overrides:
        params.update(dict(params_overrides))

    return XGBRegressor(**params)


def _xgb_predict_up_to_iter(model: XGBRegressor, X: pd.DataFrame, iter_inclusive: int) -> np.ndarray | None:
    """Predicts using trees [0, iter_inclusive] via iteration_range; returns None if unsupported."""
    it = int(iter_inclusive)
    try:
        return model.predict(X, iteration_range=(0, it + 1))
    except TypeError:
        return None


def _get_best_iter_rmse(model: XGBRegressor) -> int | None:
    """Returns early-stopping best iteration (RMSE). None if not available."""
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None:
        best_iter = getattr(model, "best_iteration_", None)
    return int(best_iter) if best_iter is not None else None


def select_best_iteration_by_qlike(
    model: XGBRegressor,
    X_va: pd.DataFrame,
    v_true_va: np.ndarray,
    *,
    max_iter: int|None = None,
    grid_points: int = 25,
    min_iter: int = 10,
) -> int | None:
    """Chooses iteration that minimizes QLIKE on v_true vs exp(mu_pred) over the validation slice."""
    v_true_va = np.asarray(v_true_va, dtype=float)
    if len(v_true_va) < 50:
        return None

    rmse_best = _get_best_iter_rmse(model)
    if max_iter is None:
        max_iter = rmse_best
    if max_iter is None:
        return None

    max_iter = int(max_iter)
    if max_iter < min_iter:
        return None

    grid_points = int(max(5, grid_points))
    candidates = np.unique(np.linspace(min_iter, max_iter, grid_points).astype(int))

    best_it = None
    best_score = float("inf")

    for it in candidates:
        mu = _xgb_predict_up_to_iter(model, X_va, int(it))
        if mu is None:
            return None
        v_pred = np.exp(mu.astype(float))
        score = qlike_loss_var(v_true_va, v_pred)
        if np.isfinite(score) and score < best_score:
            best_score = score
            best_it = int(it)

    return best_it


def _predict_mu(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predicts mu (log variance) using QLIKE-selected iteration if set; else use model.predict."""
    it = getattr(model, "_best_iteration_qlike", None)
    if it is not None:
        mu = _xgb_predict_up_to_iter(model, X, int(it))
        if mu is not None:
            return mu
    return model.predict(X)


def walk_forward_xgb_logtarget_var(
    df: pd.DataFrame,
    features: list[str],
    target_var_col: str,
    target_log_col: str,
    horizon: int,
    *,
    cfg: WalkForwardConfig | None = None,
    val_frac: float = 0.2,
    min_val_size: int = 250,
    early_stopping_rounds: int = 50,
    apply_lognormal_mean_correction: bool = True,
    var_eps_ewma_alpha: float = 0.10,
    mean_mult_cap: float | None = 2.5,
    var_eps_min_updates: int = 50,
    start_date: pd.Timestamp | None = None,
    params_overrides: dict[str, object] | None = None,
    name_prefix: str = "xgb_wf",
) -> tuple[pd.Series, pd.Series]:
    """
    Walk-forward XGB on log forward-variance.

    Refit every cfg.refit_every using data up to (pos - horizon). If a purged
    validation split exists, fit with RMSE early stopping and optionally pick a
    tree iteration by minimizing QLIKE on exp(mu) over the validation slice.

    Returns:
      - out_med: exp(mu) (median variance).
      - out_mean: exp(mu + 0.5*var_eps_eff) if enabled and var_eps is ready; else out_med.
    var_eps is EWMA variance of matured log residuals eps = y - mu (available after horizon),
    optionally capped via mean_mult_cap.
    """
    require_cols(df.columns, list(features) + [target_var_col, target_log_col], context="walk_forward_xgb_logtarget_var")

    cfg = cfg or WalkForwardConfig()

    out_med = pd.Series(index=df.index, dtype=float, name=f"{name_prefix}_median_var")
    out_mean = pd.Series(index=df.index, dtype=float, name=f"{name_prefix}_mean_var")

    needed = list(features) + [target_log_col, target_var_col]
    df2 = df.dropna(subset=needed).copy()
    n2 = len(df2)

    start_pos = compute_start_pos(
        df2.index,
        cfg=cfg,
        n_rows=n2,
        origin_start_date=start_date,
    )

    model: XGBRegressor | None = None

    # lagged, out-of-sample log-residual moments 
    mu_by_pos: dict[int, float] = {}
    eps_mean_ewma = 0.0               
    eps2_mean_ewma = 0.0               
    eps_updates = 0                    

    var_eps_cap = None
    if mean_mult_cap is not None:
        var_eps_cap = 2.0 * float(np.log(float(mean_mult_cap)))


    for pos in range(start_pos, n2):
        old_pos = pos - int(horizon)
        mu_old = mu_by_pos.pop(old_pos, None)
        if mu_old is not None and apply_lognormal_mean_correction:
            
            y_old = float(df2.iloc[old_pos][target_log_col])
            eps = y_old - float(mu_old)
            if np.isfinite(eps):
                a = float(var_eps_ewma_alpha)
                if eps_updates == 0:
                    eps_mean_ewma = float(eps)
                    eps2_mean_ewma = float(eps * eps)
                else:
                    eps_mean_ewma = (1.0 - a) * eps_mean_ewma + a * float(eps)
                    eps2_mean_ewma = (1.0 - a) * eps2_mean_ewma + a * float(eps * eps)
                eps_updates += 1

        var_eps = 0.0
        if apply_lognormal_mean_correction and eps_updates >= int(var_eps_min_updates):
            var_eps = float(max(0.0, eps2_mean_ewma - eps_mean_ewma * eps_mean_ewma))

        do_refit = (model is None) or ((pos - start_pos) % cfg.refit_every == 0)
        train_end_excl = compute_train_end_excl(pos, horizon=horizon)

        if do_refit:
            train_slice = get_train_slice(train_end_excl, cfg.window_type, cfg.rolling_window_size)
            train = df2.iloc[train_slice]

            required_min = cfg.initial_train_size if cfg.window_type == "expanding" else cfg.min_train_size
            if len(train) >= required_min:
                X_all = train[features]
                y_all = train[target_log_col]
                n_all = len(train)

                split = compute_purged_val_split(
                    n_all,
                    val_frac=val_frac,
                    min_val_size=min_val_size,
                    embargo=horizon,
                    min_train_size=100,
                    min_val_points=50,
                )

                if split is not None:
                   
                    split_end, val_size = split
                    va_start = split_end + horizon
                    va_end = va_start + val_size

                    X_tr = X_all.iloc[:split_end]
                    y_tr = y_all.iloc[:split_end]
                    X_va = X_all.iloc[va_start:va_end]
                    y_va = y_all.iloc[va_start:va_end]

                    v_true_all = train[target_var_col].values.astype(float)
                    v_va = v_true_all[va_start:va_end]

                    if len(X_tr) >= 100 and len(X_va) >= 50:
                        candidate = make_xgb_model(
                            early_stopping_rounds=early_stopping_rounds,
                            eval_metric="rmse",
                            params_overrides=params_overrides,
                        )
                        try:
                            candidate.fit(
                                X_tr, y_tr,
                                eval_set=[(X_va, y_va)],
                                verbose=False
                            )

                            best_iter_qlike = select_best_iteration_by_qlike(
                                candidate,
                                X_va,
                                v_va,
                                max_iter=_get_best_iter_rmse(candidate),
                                grid_points=25,
                                min_iter=10,
                            )
                            setattr(candidate, "_best_iteration_qlike", best_iter_qlike)
                            model = candidate
                        except (ValueError, FloatingPointError, XGBoostError):
                            pass

        if model is None:
            continue

        row = df2.iloc[[pos]]
        mu = float(_predict_mu(model, row[features])[0])

        mu_by_pos[pos] = mu

        v_med = float(np.exp(mu))
        out_med.loc[df2.index[pos]] = v_med
        var_eps_eff = var_eps
        if var_eps_cap is not None and var_eps_eff > var_eps_cap:
            var_eps_eff = var_eps_cap

        if apply_lognormal_mean_correction and var_eps_eff > 0.0:
            out_mean.loc[df2.index[pos]] = float(np.exp(mu + 0.5 * var_eps_eff))
        else:
            out_mean.loc[df2.index[pos]] = v_med

    return out_med, out_mean