import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
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

from vol_forecast.utils import safe_cols
from vol_forecast.schema import COLS
from typing import Any

def fit_forecasts(
    df: pd.DataFrame,
    *,
    cfg: WalkForwardConfig,
    horizon: int,
    active_ret_col: str,
    garch_dist: str,
    gjr_pneg_mode: str,
) -> tuple[dict[str, pd.Series], bool]:
    """
    Fit walk-forward models and return a dict of forecast series aligned to df.index.
    Also returns have_vix_cols = whether engineered VIX feature columns exist in df.
    """
    forecasts: dict[str, pd.Series] = {}

    har_daily = walk_forward_log_har_var_generic(
        df=df,
        feature_cols=list(COLS.HAR_LOG_FEATURES),
        target_log_col=COLS.LOG_TARGET_VAR,
        target_var_col=COLS.RV20_FWD_VAR,
        horizon=horizon,
        out_name="har_daily_wf_forecast_var",
        cfg=cfg,
    ).reindex(df.index)
    forecasts["har_daily"] = har_daily

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
        embargo=horizon,
    )
    forecasts["xgb_har_med"] = xgb_har_med.reindex(df.index)
    forecasts["xgb_har_mean"] = xgb_har_mean.reindex(df.index)

    # HAR+VIX XGB is run iff build_core_features produced the engineered VIX columns.
    have_vix_cols = all(c in df.columns for c in COLS.VIX_FEATURES)

    xgb_harvix_med = pd.Series(index=df.index, dtype=float, name="xgb_harvix_wf_median_var")
    xgb_harvix_mean = pd.Series(index=df.index, dtype=float, name="xgb_harvix_wf_mean_var")

    if have_vix_cols:
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
        )
        xgb_harvix_med = xgb_harvix_med.reindex(df.index)
        xgb_harvix_mean = xgb_harvix_mean.reindex(df.index)

    forecasts["xgb_harvix_med"] = xgb_harvix_med
    forecasts["xgb_harvix_mean"] = xgb_harvix_mean

    garch_var = pd.Series(index=df.index, dtype=float, name="garch_wf_forecast_var")
    gjr_var = pd.Series(index=df.index, dtype=float, name="gjr_wf_forecast_var")

   
    garch_var, _, _ = walk_forward_garch_family_var(
            df=df,
            ret_col=active_ret_col,
            trailing_var_col=COLS.RV20_VAR,
            kind="garch",
            horizon=horizon,
            cfg=cfg,
            ret_scale=100.0,
            dist=garch_dist,
        )
    garch_var = garch_var.reindex(df.index)

   
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
        )
    gjr_var = gjr_var.reindex(df.index)

    forecasts["garch_var"] = garch_var
    forecasts["gjr_var"] = gjr_var

    return forecasts, have_vix_cols


def build_wf_hold_panel(
    df: pd.DataFrame,
    *,
    forecasts: dict[str, pd.Series],
    baseline_col: str,
    holdout_start_date: str,
) -> tuple[pd.DataFrame, list[str], list[str], int]:
    """
    Assemble the target-available panel, add forecast columns, pick headline cols,
    pick default signal for strategy, slice holdout, and compute VIX coverage diagnostic.
    """
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

    # Default strategy signal selection:
    # Use HAR+VIX XGB mean only if it has enough non-NaN coverage; else fallback to HAR-only XGB mean.
    # prefer_harvix = (
    #     "xgb_harvix_wf_mean_var" in wf_df.columns and wf_df["xgb_harvix_wf_mean_var"].notna().sum() > 200
    # )
    # default_xgb = "xgb_harvix_wf_mean_var" if prefer_harvix else "xgb_har_wf_mean_var"

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

    # Diagnostic only (no gating): engineered feature availability on holdout
    if COLS.LOG_VIX_LAG1 in wf_hold.columns:
        n_vix_feat_holdout = int(wf_hold[COLS.LOG_VIX_LAG1].notna().sum())
    else:
        n_vix_feat_holdout = 0

    return wf_hold, model_cols_headline, strategy_signal_cols, n_vix_feat_holdout


def compute_eval_panels(
    wf_hold: pd.DataFrame,
    *,
    baseline_col: str,
    model_cols_headline: list[str],
    hac_lag_grid: list[int] | None,
) -> dict[str, Any]:
    """
    Compute evaluation panels for holdout. Returns a dict to avoid tuple unpack fragility.
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

    effective_hac = hac_lag_grid if hac_lag_grid is not None else [20, 40, 60]
    dm = dm_panel_qlike_vs_baseline_holdout(
        wf_hold.dropna(subset=[COLS.RV20_FWD_VAR, baseline_col]),
        target_var_col=COLS.RV20_FWD_VAR,
        model_var_cols=[c for c in model_cols_headline if c != baseline_col],
        baseline_var_col=baseline_col,
        hac_lag_grid=effective_hac,
        min_n=60,
    )

    # Store effective grid for downstream meta recovery
    dm.attrs["hac_lag_grid"] = list(effective_hac)

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