import numpy as np
import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.schema import require_cols
from .wf_util import (get_train_slice,
                      compute_start_pos,
                      compute_train_end_excl)
import statsmodels.api as sm


def fit_log_har_var_generic(train_df: pd.DataFrame, feature_cols: list[str], y_col: str):
    X = sm.add_constant(train_df[feature_cols], has_constant="add")
    y = train_df[y_col]
    model = sm.OLS(y, X).fit()
    sigma2 = float(model.mse_resid)
    return model, sigma2


def predict_log_har_var_generic(df: pd.DataFrame, model, sigma2: float, feature_cols: list[str]) -> pd.Series:
    X = sm.add_constant(df[feature_cols], has_constant="add")
    X = X[model.model.exog_names]
    mu = model.predict(X)
    return np.exp(mu + 0.5 * sigma2)


def walk_forward_log_har_var_generic(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    target_log_col: str,
    target_var_col: str,
    horizon: int,
    out_name: str,
    cfg: WalkForwardConfig|None = None,
    start_date: pd.Timestamp | None = None,
) -> pd.Series:

    cfg = cfg or WalkForwardConfig()
    out = pd.Series(index=df.index, dtype=float, name=out_name)
    require_cols(df.columns, list(feature_cols) + [target_log_col, target_var_col], context="walk_forward_log_har_var_generic")

    needed = list(feature_cols) + [target_log_col, target_var_col]
    df2 = df.dropna(subset=needed).copy()
    n2 = len(df2)

    start_pos = compute_start_pos(
        df2.index,
        cfg=cfg,
        n_rows=n2,
        origin_start_date=start_date,
    )
    model, sigma2 = None, 0.0

    for pos in range(start_pos, n2):
        train_end_excl = compute_train_end_excl(pos, horizon=horizon)

        do_refit = (model is None) or ((pos - start_pos) % cfg.refit_every == 0)
        if do_refit:
            train_slice = get_train_slice(train_end_excl, cfg.window_type, cfg.rolling_window_size)
            train = df2.iloc[train_slice]
            
            required_min = cfg.initial_train_size if cfg.window_type == "expanding" else cfg.min_train_size

            if len(train) >= required_min:
                try:
                    new_model, new_sigma2 = fit_log_har_var_generic(
                        train, feature_cols=feature_cols, y_col=target_log_col
                    )
                    if np.isfinite(new_sigma2) and new_sigma2 >= 0.0:
                        model, sigma2 = new_model, float(new_sigma2)
                except Exception:
                    pass

        if model is None:
            continue

        row = df2.iloc[[pos]]
        pred = predict_log_har_var_generic(row, model, sigma2, feature_cols=feature_cols).iloc[0]
        out.loc[df2.index[pos]] = float(pred)
    return out
