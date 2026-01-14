import pandas as pd
from typing import Any
from vol_forecast.wf_config import WalkForwardConfig
# from vol_forecast.data.vix import load_vix_close_series
from vol_forecast.schema import COLS
from vol_forecast.features import build_core_features
from .experiment_pipeline import (fit_forecasts,
                                  build_wf_hold_panel,
                                  compute_eval_panels)
from vol_forecast.strategy import run_strategy_holdout_cost_grid
from vol_forecast.eval.data_quality import build_data_diagnostics
from vol_forecast.data  import load_vix_close_series



def compute_experiment_report(
    base: pd.DataFrame,
    *,
    label: str,
    horizon: int,
    freq: int,
    wf_cfg: WalkForwardConfig,
    sigma_target: float,
    L_max: float,
    tcost_grid_bps: list[float],
    cash_daily_simple: pd.Series | None,
    execution_lag_days: int = 1,
    garch_dist: str = "t",
    holdout_start_date: str = "2019-01-01",
    gjr_pneg_mode: str = "implied",
    hac_lag_grid: list[int] | None = None,
) -> dict[str, Any]:
    """
    Notebook-friendly compute pipeline.
    No printing, no plotting, no file I/O.
    Returns a 'report' dict with panels + tables + strategy outputs + metadata.
    """
    cfg = wf_cfg
    ret_col = COLS.RET

    core_cols = [
        ret_col,
        COLS.DAILY_VAR, COLS.RV20_VAR, COLS.RV20_FWD_VAR,
        COLS.RW_FORECAST_VAR, COLS.EWMA_FORECAST_VAR,
        *COLS.HAR_LOG_FEATURES,
        COLS.LOG_TARGET_VAR,
    ]

    # -------------------------------------------------------------------------
    # VIX: best-effort load (do not fail the pipeline if unavailable)
    # -------------------------------------------------------------------------
    # vix_close: pd.Series | None = None
    # vix_error: str | None = None

    vix_close = load_vix_close_series(
            start_date=str(base.index.min().date()),
            end_date=None,
        ).rename(COLS.VIX_CLOSE)
        
    core_cols.extend(COLS.VIX_FEATURES)


    # except Exception as e:
    #     vix_close = None
    #     vix_error = str(e)
    
    # -------------------------------------------------------------------------
    # Core features (single source of truth)
    # -------------------------------------------------------------------------
    df = base.copy()
    df = build_core_features(
        df,
        ret_col=ret_col,
        horizon=horizon,
        freq=freq,
        vix_close=vix_close,
    )

    baseline_col = COLS.EWMA_FORECAST_VAR

    # -------------------------------------------------------------------------
    # Walk-forward forecasts
    # -------------------------------------------------------------------------
    forecasts, have_vix_cols = fit_forecasts(
        df,
        cfg=cfg,
        horizon=horizon,
        active_ret_col=ret_col,
        garch_dist=garch_dist,
        gjr_pneg_mode=gjr_pneg_mode,
    )

    # -------------------------------------------------------------------------
    # Build panel where target exists + holdout slice + strategy defaults
    # -------------------------------------------------------------------------
    wf_hold, model_cols_headline, strategy_signal_cols, n_vix_feat_holdout = build_wf_hold_panel(
        df,
        forecasts=forecasts,
        baseline_col=baseline_col,
        holdout_start_date=holdout_start_date,
    )

    data_diag = build_data_diagnostics(
        df=df,
        wf_hold=wf_hold,
        ret_col=ret_col,
        target_var_col=COLS.RV20_FWD_VAR,
        baseline_var_col=baseline_col,
        model_var_cols=model_cols_headline,
        vix_close=vix_close,
        cash_daily_simple=cash_daily_simple,
        core_cols=core_cols,
        warmup_core=200,
    )

    # -------------------------------------------------------------------------
    # Reports (pure eval functions) -> dict
    # -------------------------------------------------------------------------
    panels = compute_eval_panels(
        wf_hold,
        baseline_col=baseline_col,
        model_cols_headline=model_cols_headline,
        hac_lag_grid=hac_lag_grid,
    )

    # Effective HAC grid for meta (prefer dm.attrs, else the defaulting logic)
    effective_hac_lag_grid = panels.get("dm", pd.DataFrame()).attrs.get("hac_lag_grid", None)
    if effective_hac_lag_grid is None:
        effective_hac_lag_grid = hac_lag_grid if hac_lag_grid is not None else [20, 40, 60]

    # -------------------------------------------------------------------------
    # Strategy (pure; no printing inside strategy.py)
    # -------------------------------------------------------------------------
    strat_df = run_strategy_holdout_cost_grid(
        wf_hold,
        return_col=ret_col,
        signal_var_cols=strategy_signal_cols,
        cash_daily_simple=cash_daily_simple.reindex(wf_hold.index) if cash_daily_simple is not None else None,
        sigma_target=sigma_target,
        L_max=L_max,
        horizon=horizon,
        tcost_grid_bps=tcost_grid_bps,
        execution_lag_days=execution_lag_days,
        variants=["hold20", "tranche20"],
        freq=freq,
    )

    # -------------------------------------------------------------------------
    # Meta (inline)
    # -------------------------------------------------------------------------
    meta: dict[str, Any] = {
        "label": label,
        "horizon": horizon,
        "freq": freq,
        "holdout_start_date": holdout_start_date,
        "hac_lag_grid": list(effective_hac_lag_grid),
        "wf_cfg": {
            "initial_train_frac": cfg.initial_train_frac,
            "window_type": cfg.window_type,
            "rolling_w": cfg.rolling_w,
            "refit_every": cfg.refit_every,
        },
    }

    return {
        "wf_hold": wf_hold,
        "baseline_col": baseline_col,
        "model_cols_headline": model_cols_headline,
        "strategy_signal_cols": strategy_signal_cols,
        # eval panels
        "data_diag": data_diag,
        "availability": panels["availability"],
        "headline_full": panels["headline_full"],
        "headline_half1": panels["headline_half1"],
        "headline_half2": panels["headline_half2"],
        "split_mid": panels["split_mid"],
        "xgb_sanity": panels["xgb_sanity"],
        "calibration": panels["calibration"],
        "dm": panels["dm"],
        # strategy + meta
        "strategy": strat_df,
        "meta": meta,
    }
