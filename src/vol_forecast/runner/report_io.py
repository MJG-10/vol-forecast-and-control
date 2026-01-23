from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vol_forecast.utils import safe_cols
from vol_forecast.schema import COLS


def print_section(title: str) -> None:
    """Prints a visually separated console header section."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_df(df: pd.DataFrame, float_fmt: str = ".6g") -> None:
    """Prints a DataFrame to console with a compact float format."""
    if df is None or df.empty:
        print("empty df")
        return
    print(df.to_string(index=False, float_format=lambda x: format(x, float_fmt)))


def _print_kv(d: dict[str, object], *, keys: list[str]) -> None:
    for k in keys:
        if k in d:
            print(f"{k}: {d[k]}")


def console_build_meta(build_meta: dict[str, object]) -> None:
    """
    Prints build/provenance metadata returned by build_experiment_df().
    """
    print_section("BUILD META")
    _print_kv(build_meta, keys=["label", "start_date", "end_date", "cash_source", "vix_source"])

    dd = build_meta["data_diag_core"]
    print_section("CORE DATA DIAGNOSTICS")
    print("Core index:", dd["core_index"])
    print_section("Core coverage (key cols)")
    print_df(dd["core_coverage"])


def plot_tail_vol(
    df: pd.DataFrame,
    target_var_col: str,
    forecast_var_cols: list[str],
    n: int = 500,
    title: str = "",
) -> None:
    """Plots the last n rows in Vol space (sqrt of variance) for target and forecast columns."""
    cols = safe_cols(df, [target_var_col] + forecast_var_cols)
    if not cols:
        return

    tmp = df[cols].tail(n).copy()
    for c in cols:
        tmp[c] = np.sqrt(tmp[c].clip(lower=0.0))

    tmp.plot(figsize=(12, 6))
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def console_report(report: dict[str, object]) -> None:
    """
    Prints a compact holdout/run report.
    Expects a report dict as returned by vol_forecast.runner.experiment.compute_experiment_report().
    This function intentionally focuses on run + holdout content; build/core meta is printed via
    console_build_meta(build_meta) separately.
    """
    meta: dict[str, object] = report["meta"]
    wf_hold: pd.DataFrame = report["wf_hold"]
    baseline_col: str = report["baseline_col"]

    if "build_meta" in report:
        console_build_meta(report["build_meta"])

    print_section(f"EXPERIMENT: {meta['label']}")
    _print_kv(meta, keys=["horizon", "freq", "holdout_start_date", "hac_lag_grid"])
    
    # print(f"horizon={meta['horizon']}  holdout_start_date={meta['holdout_start_date']}  baseline=EWMA  var_proxy=close-to-close")
    # print(f"freq={meta['freq']}  hac_lag_grid={meta['hac_lag_grid']}")
    
    print("wf_cfg:", meta["wf_cfg"])
    print(f"baseline_col={baseline_col}")

    print_section("HOLDOUT REPORT")
    if len(wf_hold):
        print(f"HOLDOUT: {wf_hold.index.min().date()} -> {wf_hold.index.max().date()}  | rows={len(wf_hold)}")
    else:
        print("HOLDOUT is empty.")

    # Holdout diagnostics
    dd = report["data_diag"]
    print_section("HOLDOUT DATA DIAGNOSTICS")
    print("Holdout index:", dd["holdout_index"])
    print_section("Holdout coverage (target/baseline/models)")
    print_df(dd["holdout_coverage"])

    if "cash_coverage_holdout" in dd:
        print_section("Holdout cash coverage")
        print_df(dd["cash_coverage_holdout"])

    print_section("HOLDOUT AVAILABILITY SUMMARY (COMPACT)")
    print_df(report["availability"])

    print_section("HOLDOUT HEADLINE TABLE (PAIRWISE VS EWMA BASELINE)")
    print_df(report["headline_full"])

    mid = report["split_mid"]
    print_section(f"HOLDOUT SPLIT-HALF HEADLINE (CUT DATE ~ {mid.date()})")
    hh = pd.concat([report["headline_half1"], report["headline_half2"]], axis=0)
    print_df(hh)

    print_section("XGB MEAN VS MEDIAN SANITY (HOLDOUT)")
    print("Interpretation: mean-correction helps if QLIKE(mean) < QLIKE(median) and d_qlike(mean-median) < 0.")
    print_df(report["xgb_sanity"])

    print_section("CALIBRATION MONOTONICITY (HOLDOUT): SPEARMAN corr(forecast vol, realized vol)")
    print_df(report["calibration"])

    print_section(f"DM VS baseline='{baseline_col}' ON HOLDOUT (overlap daily) | HAC grid={meta["hac_lag_grid"]}")
    print("Note: DM is for context. Focus is on sign/magnitude stability across HAC lags (not on any single p-value).")
    print_df(report["dm"])

    strat = report["strategy"]
    if strat is not None:
        print_section("STRATEGY BACKTEST (HOLDOUT): COMPACT COST GRID")
        print_df(strat)


def plot_report(report: dict[str, object], *, n: int = 500) -> None:
    """Plots the holdout tail comparison in Vol space for headline columns."""
    wf_hold: pd.DataFrame = report["wf_hold"]
    meta: dict[str, object] = report["meta"]
    model_cols_headline: list[str] = report["model_cols_headline"]

    if len(wf_hold) == 0:
        return

    title = f"{meta["label"]}: HOLDOUT tail (VOL space)".strip(": ")

    plot_tail_vol(
        wf_hold,
        target_var_col=COLS.RV20_FWD_VAR,
        forecast_var_cols=safe_cols(wf_hold, model_cols_headline),
        n=n,
        title=title,
    )


def save_report_csvs(report: dict[str, object], *, out_dir: Path) -> None:
    """Writes key report tables to CSV files in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Forecast headline tables
    headline_full: pd.DataFrame = report["headline_full"]
    headline_h1: pd.DataFrame = report["headline_half1"]
    headline_h2: pd.DataFrame = report["headline_half2"]
    pd.concat([headline_full, headline_h1, headline_h2], axis=0).reset_index(drop=True).to_csv(
        out_dir / "forecast_holdout_headline.csv", index=False
    )

    # Holdout/run artifacts
    report["dm"].to_csv(out_dir / "dm_vs_baseline_holdout.csv", index=False)
    report["availability"].to_csv(out_dir / "availability_holdout.csv", index=False)
    report["calibration"].to_csv(out_dir / "calibration_holdout.csv", index=False)
    report["xgb_sanity"].to_csv(out_dir / "xgb_mean_vs_median_sanity_holdout.csv", index=False)

    # Holdout diagnostics (always present in current pipeline)
    dd = report["data_diag"]
    dd["holdout_coverage"].to_csv(out_dir / "data_diag_holdout_coverage.csv", index=False)
    if "cash_coverage_holdout" in dd:
        dd["cash_coverage_holdout"].to_csv(out_dir / "data_diag_holdout_cash_coverage.csv", index=False)

    # Core/build diagnostics ONLY if you attached build_meta right before saving
    if "build_meta" in report:
        report["build_meta"]["data_diag_core"]["core_coverage"].to_csv(
            out_dir / "data_diag_core_coverage.csv", index=False
        )

    # Strategy grid (optional)
    strat = report["strategy"]
    if strat is not None:
        strat.to_csv(out_dir / "strategy_stats_cost_grid.csv", index=False)
