import pandas as pd
from typing import Any
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.schema import COLS
from .experiment_pipeline import (fit_forecasts,
                                  build_wf_hold_panel,
                                  compute_eval_panels)
from vol_forecast.strategy import run_strategy_holdout_cost_grid
from vol_forecast.eval.data_quality import build_holdout_data_diagnostics
from vol_forecast.models_tuning.tuning import tune_xgb_params_pre_holdout
from vol_forecast.runner.experiment_pipeline import build_experiment_df


def compute_experiment_report(
    df: pd.DataFrame,
    *,
    horizon: int,
    freq: int,
    wf_cfg: WalkForwardConfig,
    execution_lag_days: int = 0,
    garch_dist: str = "t",
    holdout_start_date: str = "2019-01-01",
    hac_lag_grid: list[int] | None = None,
    run_strategy: bool = True, 
    strategy_variants: list[str] | None = None,
    sigma_target: float = 0.10,
    tcost_grid_bps: list[float] | None = None,
    xgb_params_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute evaluation panels/tables from a canonical experiment DataFrame `df`.

    `df` must already contain returns, the forward variance target, and engineered predictors
    aligned to avoid look-ahead (e.g., predictors known by close of t-1 for a label over t..t+h-1).
    `execution_lag_days` affects only strategy timing (not model fitting).
    `hac_lag_grid` affects DM standard errors/inference, not the loss differential itself.

    Returns a report dict with headline tables, DM panels, diagnostics, and optional strategy results.
    """
    cfg = wf_cfg

    if tcost_grid_bps is None:
        tcost_grid_bps = [0.0, 1.0, 2.0, 5.0, 10.0]

    if strategy_variants is None:
        strategy_variants = ["daily_reset", "band_no_trade"]

    if  hac_lag_grid is None:
        hac_lag_grid = [20, 40, 60]

    baseline_col = COLS.RW_FORECAST_VAR

 
    # Walk-forward forecasts
    forecasts = fit_forecasts(
        df,
        cfg=cfg,
        horizon=horizon,
        active_ret_col=COLS.RET,
        garch_dist=garch_dist,
        holdout_start_date=holdout_start_date,
        xgb_params_overrides=xgb_params_overrides,
    )

    # Build panel + holdout slice + strategy defaults
    wf_hold, model_cols_headline, strategy_signal_cols = build_wf_hold_panel(
        df,
        forecasts=forecasts,
        baseline_col=baseline_col,
        holdout_start_date=holdout_start_date,
    )

    # Data diagnostics
    data_diag = build_holdout_data_diagnostics(
        wf_hold=wf_hold,
        ret_col=COLS.RET,
        target_var_col=COLS.RVAR_FWD,
        baseline_var_col=baseline_col,
        model_var_cols=model_cols_headline,
        cash_col=COLS.CASH_R if run_strategy else None,
    )

    # Evaluation panels
    panels = compute_eval_panels(
        wf_hold,
        baseline_col=baseline_col,
        model_cols_headline=model_cols_headline,
        hac_lag_grid=hac_lag_grid,
    )

    # Strategy
    strat_df = None
    if run_strategy:
        strat_df = run_strategy_holdout_cost_grid(
            wf_hold,
            return_col=COLS.RET,
            cash_col = COLS.CASH_R,
            signal_var_cols=strategy_signal_cols,
            sigma_target=sigma_target,
            tcost_grid_bps=tcost_grid_bps,
            execution_lag_days=execution_lag_days,
            variants=strategy_variants,
            freq=freq,
        )

    # Meta
    meta: dict[str, Any] = {
        "horizon": horizon,
        "freq": freq,
        "holdout_start_date": holdout_start_date,
        "hac_lag_grid": list(hac_lag_grid),
        "wf_cfg": {
            "window_type": cfg.window_type,
            "initial_train_size": cfg.initial_train_size,
            "rolling_window_size": cfg.rolling_window_size,
            "refit_every": cfg.refit_every,
            "min_train_size": cfg.min_train_size
        },
    }

    return {
        "wf_hold": wf_hold,
        "baseline_col": baseline_col,
        "model_cols_headline": model_cols_headline,
        "strategy_signal_cols": strategy_signal_cols,
        "data_diag": data_diag,
        "availability": panels["availability"],
        "headline_full": panels["headline_full"],
        "headline_half1": panels["headline_half1"],
        "headline_half2": panels["headline_half2"],
        "split_mid": panels["split_mid"],
        "xgb_sanity": panels["xgb_sanity"],
        "calibration": panels["calibration"],
        "dm": panels["dm"],
        "strategy": strat_df,
        "meta": meta,
    }


def run_experiment(
    *,
    start_date: str,
    end_date: str | None,
    horizon: int,
    wf_cfg: WalkForwardConfig,
    sigma_target: float,
    freq: int  = 252,
    tcost_grid_bps: list[float] | None = None,
    garch_dist: str = 't',
    holdout_start_date: str,
    tuning_blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    hac_lag_grid: list[int] | None = None,
    run_strategy: bool,
    strategy_variants: list[str] | None = None,
    do_xgb_tuning: bool = False,
) -> dict[str, Any]:
    """
    End-to-end experiment entrypoint (recommended for notebooks).

    Builds the canonical experiment DataFrame (returns/targets/features/baselines), optionally
    tunes XGB parameters using pre-holdout data, then computes and returns the report dict.
    """
    # 1) Build canonical experiment dataframe
    base_df, build_meta = build_experiment_df(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        freq=freq,
    )

    # 2) Optional tuning (defaults live in tuning.py)
    xgb_params_overrides: dict[str, Any] | None = None
    xgb_tuning: dict[str, Any] = {"status": "not_run", "best": None, "table": None}

    if do_xgb_tuning:
        best_meta, tune_table = tune_xgb_params_pre_holdout(
            df=base_df,
            holdout_start=pd.Timestamp(holdout_start_date),
            horizon=horizon,
            cfg=wf_cfg,
            blocks = tuning_blocks,
        )
        xgb_params_overrides = best_meta["best_params_overrides"]

        xgb_tuning = {
            "status": "ran",
            "best": best_meta,
            "table": tune_table.to_dict(orient="records"),
        }

    # 3) Compute report (pure)
    report = compute_experiment_report(
        base_df,
        horizon=horizon,
        freq=freq,
        wf_cfg=wf_cfg,
        garch_dist=garch_dist,
        holdout_start_date=holdout_start_date,
        hac_lag_grid=hac_lag_grid,
        run_strategy=run_strategy,
        strategy_variants=strategy_variants,
        sigma_target=sigma_target,
        tcost_grid_bps=tcost_grid_bps,
        xgb_params_overrides=xgb_params_overrides,
    )

    report["build_meta"] = build_meta
    report["tuning"] = {"xgb": xgb_tuning}

    meta = report["meta"] 
    meta["final_xgb_params"] = xgb_params_overrides
    meta["xgb_tuning_status"] = xgb_tuning["status"]

    return report
