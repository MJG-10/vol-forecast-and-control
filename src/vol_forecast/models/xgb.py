import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.metrics import qlike_loss_var
from .wf_util import (compute_train_end_excl, 
                                  compute_purged_val_split, 
                                  get_train_slice)

from vol_forecast.schema import require_cols

def make_xgb_model() -> XGBRegressor:
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=0,
        tree_method="hist",
        n_jobs=-1,
    )


def _xgb_fit_with_fallback(
    model: XGBRegressor,
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame,
    y_va: pd.Series,
    early_stopping_rounds: int|None,
) -> None:
    try:
        if early_stopping_rounds is not None:
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_va, y_va)],
                eval_metric="rmse",
                early_stopping_rounds=int(early_stopping_rounds),
                verbose=False,
            )
        else:
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="rmse", verbose=False)
    except TypeError:
        model.fit(X_tr, y_tr, verbose=False)


def _winsorize(x: np.ndarray, q: float = 0.01) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return x
    lo, hi = np.quantile(x, [q, 1.0 - q])
    return np.clip(x, lo, hi)


def _xgb_predict_up_to_iter(model: XGBRegressor, X: pd.DataFrame, iter_inclusive: int) -> np.ndarray|None:
    it = int(iter_inclusive)
    try:
        return model.predict(X, iteration_range=(0, it + 1))
    except TypeError:
        return None


def _get_best_iter_rmse(model: XGBRegressor) -> int|None:
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
) -> int|None:
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


def walk_forward_xgb_logtarget_var(
    df: pd.DataFrame,
    features: list[str],
    target_var_col: str,
    target_log_col: str,
    horizon: int,
    *,
    cfg: WalkForwardConfig|None = None,
    val_frac: float = 0.2,
    min_val_size: int = 250,
    early_stopping_rounds: int|None = 50,
    name_prefix: str = "xgb_wf",
    apply_lognormal_mean_correction: bool = True,
    embargo: int = -1,
    resid_winsor_q: float = 0.01,
    var_eps_ewma_alpha: float = 0.20,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns two forecasts:
      - median_var = exp(mu)
      - mean_var   = exp(mu + 0.5 * var_eps)  (var_eps estimated on validation residuals, smoothed)
    """
    def _predict_mu(model: XGBRegressor, X: pd.DataFrame) -> np.ndarray:
        it = getattr(model, "_best_iteration_qlike", None)
        if it is not None:
            mu = _xgb_predict_up_to_iter(model, X, int(it))
            if mu is not None:
                return mu
        # fallback to default predict
        return model.predict(X)

    require_cols(df.columns, list(features) + [target_var_col, target_log_col], context="walk_forward_xgb_logtarget_var")

    cfg = cfg or WalkForwardConfig()

    out_med = pd.Series(index=df.index, dtype=float, name=f"{name_prefix}_median_var")
    out_mean = pd.Series(index=df.index, dtype=float, name=f"{name_prefix}_mean_var")

    emb = horizon if embargo<0 else int(embargo)

    needed = list(features) + [target_log_col, target_var_col]
    df2 = df.dropna(subset=needed).copy()
    n2 = len(df2)
    # if n2 < 300:
    if n2 <cfg.min_rows_total:
        return out_med, out_mean

    initial_train_idx = int(cfg.initial_train_frac * n2)
    model: XGBRegressor|None = None

    var_eps_smoothed = 0.0
    have_var_eps = False

    for pos in range(initial_train_idx, n2):
        train_end_excl = compute_train_end_excl(
            pos,
            horizon=horizon,
            min_train_end_excl=51
        )
        if train_end_excl is None:
            continue

        do_refit = (model is None) or ((pos - initial_train_idx) % cfg.refit_every == 0)
        var_eps = 0.0

        if do_refit:
            train_slice = get_train_slice(train_end_excl, cfg.window_type, cfg.rolling_w)
            train = df2.iloc[train_slice]
            # if len(train) < 200:
            if len(train) < cfg.min_train_rows:
                continue

            X_all = train[features]
            y_all = train[target_log_col]
            n_all = len(train)

            split = compute_purged_val_split(
                n_all,
                val_frac=val_frac,
                min_val_size=min_val_size,
                embargo=emb,
                min_train_size=100,
                min_val_points=50,
            )
            if split is None:
                continue

            split_end, val_size = split

            X_tr = X_all.iloc[:split_end]
            y_tr = y_all.iloc[:split_end]

            va_start = split_end + emb
            va_end = va_start + val_size

            X_va = X_all.iloc[va_start:va_end]
            y_va = y_all.iloc[va_start:va_end]

            v_true_all = train[target_var_col].values.astype(float)
            v_va = v_true_all[va_start:va_end]

            if len(X_tr) < 100 or len(X_va) < 50:
                continue

            model = make_xgb_model()
            _xgb_fit_with_fallback(model, X_tr, y_tr, X_va, y_va, early_stopping_rounds)

            best_iter_qlike = select_best_iteration_by_qlike(
                model,
                X_va,
                v_va,
                max_iter=_get_best_iter_rmse(model),
                grid_points=25,
                min_iter=10,
            )
            setattr(model, "_best_iteration_qlike", best_iter_qlike)

            if apply_lognormal_mean_correction:
                yhat_va = _predict_mu(model, X_va)
                resid = (y_va.values.astype(float) - yhat_va.astype(float))
                resid = resid[np.isfinite(resid)]

                if len(resid) >= 30:
                    resid_w = _winsorize(resid, q=float(resid_winsor_q))
                    v_new = float(np.var(resid_w, ddof=1)) if len(resid_w) >= 30 else float("nan")
                    v_new = v_new if (np.isfinite(v_new) and v_new > 0.0) else 0.0

                    if not have_var_eps:
                        var_eps_smoothed = v_new
                        have_var_eps = True
                    else:
                        a = float(var_eps_ewma_alpha)
                        var_eps_smoothed = (1.0 - a) * var_eps_smoothed + a * v_new

                var_eps = var_eps_smoothed if have_var_eps else 0.0
            else:
                var_eps = 0.0

        if model is None:
            continue

        row = df2.iloc[[pos]]
        mu = float(_predict_mu(model, row[features])[0])

        v_med = float(np.exp(mu))
        out_med.loc[df2.index[pos]] = v_med
        out_mean.loc[df2.index[pos]] = float(np.exp(mu + 0.5 * var_eps)) if (apply_lognormal_mean_correction and var_eps > 0) else v_med

    return out_med, out_mean
