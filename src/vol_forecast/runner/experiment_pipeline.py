import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.data import (load_sp500_total_return_close,
                               compute_log_returns_from_series,
                               load_cash_daily_simple_act360,
                               load_vix_close_series)
from vol_forecast.features import build_core_features
from vol_forecast.models.garch import walk_forward_garch_family_var
from vol_forecast.models.har import walk_forward_log_har_var_generic
from vol_forecast.models.xgb import walk_forward_xgb_logtarget_var

from vol_forecast.eval.reporting import (availability_summary_holdout,
                                         pairwise_headline_table,
                                         split_holdout_into_halves,
                                         report_xgb_mean_median_sanity,
                                         calibration_spearman_holdout)
from vol_forecast.eval.dm import dm_panel_qlike_vs_baseline_holdout
from vol_forecast.eval.data_quality import build_core_data_diagnostics 
from vol_forecast.utils import safe_cols
from vol_forecast.schema import COLS
from typing import Any


def build_experiment_df(
    *,
    start_date: str = "1990-01-01",
    end_date: str | None = None,
    horizon: int = 20,
    freq: int = 252
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Builds the experiment DataFrame.

    Returns:
      df: canonical DataFrame indexed by trading dates, including:
          - COLS.RET
          - COLS.CASH_R
          - engineered columns from build_core_features(...)
      meta: lightweight provenance info (sources + labels) and the raw VIX close series (if loaded).
    """
    # 1) Returns
    label = "S&P 500 TOTAL RETURN"
    tr_close = load_sp500_total_return_close(start_date=start_date, end_date=end_date)
    ret = compute_log_returns_from_series(tr_close, out_name=COLS.RET, drop_nan=True)
    df = ret.to_frame()

    # 2) Cash proxy aligned to the same trading index
    cash_r, cash_source = load_cash_daily_simple_act360(
        start_date=str(df.index.min().date()),
        end_date=end_date,
        trading_index=df.index,
        lag_trading_days=1,
    )
    df[COLS.CASH_R] = cash_r.reindex(df.index)

    # 3) VIX close
    vix_close, vix_source = load_vix_close_series(
            start_date=str(df.index.min().date()),
            end_date=end_date)

    # 4) Feature/target engineering (does not drop rows)
    df = build_core_features(
        df,
        ret_col=COLS.RET,
        horizon=horizon,
        freq=freq,
        vix_close=vix_close,
    )
    df[COLS.VIX_CLOSE] = vix_close.reindex(df.index)

    # 5) Data diagnostics
    data_diag_core =  build_core_data_diagnostics(df=df,
        core_cols=COLS.EXPERIMENT_CORE_COLS,
        head_warmup=22,
        tail_cooldown=max(0, horizon - 1)
    )

    meta: dict[str, Any] = {
        "label": label,
        "start_date": start_date,
        "end_date": end_date,
        "cash_source": cash_source,
        "vix_source": vix_source,
        "data_diag_core": data_diag_core
    }
    return df, meta


def fit_forecasts(
    df: pd.DataFrame,
    *,
    cfg: WalkForwardConfig,
    horizon: int,
    active_ret_col: str,
    garch_dist: str,
    gjr_pneg_mode: str,
    holdout_start_date: str | None = None,
    xgb_params_overrides: dict[str, object] | None = None,
) -> dict[str, pd.Series]:
    """
    Fits walk-forward forecast models and returns a dict of forecast variance series
    aligned to `df.index` (HAR, XGB(HAR), XGB(HAR+VIX), GARCH, GJR for example).
    """
    forecasts: dict[str, pd.Series] = {}

    # HAR
    har_daily = walk_forward_log_har_var_generic(
        df=df,
        feature_cols=list(COLS.HAR_LOG_FEATURES),
        target_log_col=COLS.LOG_TARGET_VAR,
        target_var_col=COLS.RV20_FWD_VAR,
        horizon=horizon,
        out_name="har_daily_wf_forecast_var",
        cfg=cfg,
        start_date= pd.Timestamp(holdout_start_date) if holdout_start_date else None,
    ).reindex(df.index)
    forecasts["har_daily"] = har_daily

    # XGB (HAR)
    xgb_har_med, xgb_har_mean = walk_forward_xgb_logtarget_var(
        df=df,
        features=list(COLS.HAR_LOG_FEATURES),
        target_var_col=COLS.RV20_FWD_VAR,
        target_log_col=COLS.LOG_TARGET_VAR,
        horizon=horizon,
        cfg=cfg,
        early_stopping_rounds=50,
        name_prefix="xgb_har_wf",
        apply_lognormal_mean_correction=True,
        start_date= pd.Timestamp(holdout_start_date) if holdout_start_date else None,
        params_overrides=xgb_params_overrides, 
    )
    forecasts["xgb_har_med"] = xgb_har_med.reindex(df.index)
    forecasts["xgb_har_mean"] = xgb_har_mean.reindex(df.index)

    # XGB (HAR+VIX)
    # We use the same tuned parameters for both XGB variants for simplcity.
    xgb_feats_harvix = list(COLS.HAR_LOG_FEATURES + COLS.VIX_FEATURES)
    xgb_harvix_med, xgb_harvix_mean = walk_forward_xgb_logtarget_var(
            df=df,
            features=xgb_feats_harvix,
            target_var_col=COLS.RV20_FWD_VAR,
            target_log_col=COLS.LOG_TARGET_VAR,
            horizon=horizon,
            cfg=cfg,
            early_stopping_rounds=50,
            name_prefix="xgb_harvix_wf",
            apply_lognormal_mean_correction=True,
            embargo=horizon,
            start_date= pd.Timestamp(holdout_start_date) if holdout_start_date else None,
            params_overrides=xgb_params_overrides,
    )
    xgb_harvix_med = xgb_harvix_med.reindex(df.index)
    xgb_harvix_mean = xgb_harvix_mean.reindex(df.index)

    forecasts["xgb_harvix_med"] = xgb_harvix_med
    forecasts["xgb_harvix_mean"] = xgb_harvix_mean
   
    # GARCH
    garch_var, _, _ = walk_forward_garch_family_var(
            df=df,
            ret_col=active_ret_col,
            trailing_var_col=COLS.RV20_VAR,
            kind="garch",
            horizon=horizon,
            cfg=cfg,
            ret_scale=100.0,
            dist=garch_dist,
            start_date= pd.Timestamp(holdout_start_date) if holdout_start_date else None,
        )
    garch_var = garch_var.reindex(df.index)

   # GJR-GARCH
    gjr_var, _, _ = walk_forward_garch_family_var(
            df=df,
            ret_col=active_ret_col,
            trailing_var_col=COLS.RV20_VAR,
            kind="gjr",
            horizon=horizon,
            cfg=cfg,
            ret_scale=100.0,
            dist=garch_dist,
            gjr_pneg_mode=gjr_pneg_mode,
            start_date= pd.Timestamp(holdout_start_date) if holdout_start_date else None,
        )
    gjr_var = gjr_var.reindex(df.index)

    forecasts["garch_var"] = garch_var
    forecasts["gjr_var"] = gjr_var

    return forecasts


def build_wf_hold_panel(
    df: pd.DataFrame,
    *,
    forecasts: dict[str, pd.Series],
    baseline_col: str,
    holdout_start_date: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Constructs the target-available walk-forward panel, adds forecast columns,
    selects model columns for evaluation and default signal columns for strategy,
    then slices the holdout period starting at `holdout_start_date`.
    """
    # We restrict to timestamps where the forward target is available.
    wf_df = df.loc[df[COLS.RV20_FWD_VAR].notna()].copy()

    wf_df["har_daily_wf_forecast_var"] = forecasts["har_daily"].reindex(wf_df.index)
    wf_df["xgb_har_wf_mean_var"] = forecasts["xgb_har_mean"].reindex(wf_df.index)
    wf_df["xgb_har_wf_median_var"] = forecasts["xgb_har_med"].reindex(wf_df.index)
    wf_df["xgb_harvix_wf_mean_var"] = forecasts["xgb_harvix_mean"].reindex(wf_df.index)
    wf_df["xgb_harvix_wf_median_var"] = forecasts["xgb_harvix_med"].reindex(wf_df.index)
    
    wf_df["garch_wf_forecast_var"] = forecasts["garch_var"].reindex(wf_df.index)
    wf_df["gjr_wf_forecast_var"] = forecasts["gjr_var"].reindex(wf_df.index)

    model_cols_headline = safe_cols(
        wf_df,
        [
            COLS.RW_FORECAST_VAR,
            baseline_col,
            "har_daily_wf_forecast_var",
            "xgb_har_wf_mean_var",
            "xgb_harvix_wf_mean_var",
            "garch_wf_forecast_var",
            "gjr_wf_forecast_var",
        ],
    )

    strategy_signal_cols = safe_cols(
        wf_df,
        [
            baseline_col,
            "har_daily_wf_forecast_var",
            "xgb_har_wf_mean_var",
            "xgb_harvix_wf_mean_var",
            "garch_wf_forecast_var",
            "gjr_wf_forecast_var",
        ],
    )

    hold_ts = pd.Timestamp(holdout_start_date)
    wf_hold = wf_df.loc[wf_df.index >= hold_ts].copy()

    return wf_hold, model_cols_headline, strategy_signal_cols


def compute_eval_panels(
    wf_hold: pd.DataFrame,
    *,
    baseline_col: str,
    model_cols_headline: list[str],
    hac_lag_grid: list[int] | None,
) -> dict[str, Any]:
    """
    Computes holdout evaluation panels (availability, headline tables, calibration,
    XGB sanity checks, and DM vs baseline). Returned as a dict for stable access.
    """
    availability = availability_summary_holdout(
        wf_hold,
        target_var_col=COLS.RV20_FWD_VAR,
        baseline_var_col=baseline_col,
        model_var_cols=model_cols_headline,
    )

    headline_full = pairwise_headline_table(
        wf_hold,
        segment="HOLDOUT_full",
        target_var_col=COLS.RV20_FWD_VAR,
        baseline_var_col=baseline_col,
        model_var_cols=model_cols_headline,
        include_rmse_vol=True,
        min_n=60,
    )

    # Stability check: we rerun headline metrics on two holdout halves to detect regime sensitivity.
    h1, h2, mid = split_holdout_into_halves(wf_hold)

    headline_half1 = pairwise_headline_table(
        h1,
        segment="HOLDOUT_half1",
        target_var_col=COLS.RV20_FWD_VAR,
        baseline_var_col=baseline_col,
        model_var_cols=model_cols_headline,
        include_rmse_vol=True,
    )

    headline_half2 = pairwise_headline_table(
        h2,
        segment="HOLDOUT_half2",
        target_var_col=COLS.RV20_FWD_VAR,
        baseline_var_col=baseline_col,
        model_var_cols=model_cols_headline,
        include_rmse_vol=True,
    )

    xgb_pairs = [
        ("xgb_har_wf_median_var", "xgb_har_wf_mean_var", "XGB(HAR)"),
        ("xgb_harvix_wf_median_var", "xgb_harvix_wf_mean_var", "XGB(HAR+VIX)"),
    ]
    xgb_sanity = report_xgb_mean_median_sanity(
        wf_hold,
        target_var_col=COLS.RV20_FWD_VAR,
        baseline_var_col=baseline_col,
        pairs=xgb_pairs,
        min_n=150,
    )

    calibration = calibration_spearman_holdout(
        wf_hold,
        target_var_col=COLS.RV20_FWD_VAR,
        model_var_cols=model_cols_headline,
        min_n=200,
    )

    # DM policy: for each model, we compute DM vs baseline on its own available sample
    # (intersection of target/baseline/model). The DM panel reports sample size `n` 
    # which we expect to be consistent across models. A common-sample policy can be adopted otherwise.

    wf_hold_dm_base = wf_hold.dropna(subset=[COLS.RV20_FWD_VAR, baseline_col])

    dm = dm_panel_qlike_vs_baseline_holdout(
        wf_hold_dm_base,
        target_var_col=COLS.RV20_FWD_VAR,
        model_var_cols=[c for c in model_cols_headline if c != baseline_col],
        baseline_var_col=baseline_col,
        hac_lag_grid=hac_lag_grid,
        min_n=60,
    )

    return {
        "availability": availability,
        "headline_full": headline_full,
        "headline_half1": headline_half1,
        "headline_half2": headline_half2,
        "split_mid": mid,
        "xgb_sanity": xgb_sanity,
        "calibration": calibration,
        "dm": dm,
    }
