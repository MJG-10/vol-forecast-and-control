import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from vol_forecast.utils import safe_cols

def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_df(df: pd.DataFrame, float_fmt: str = ".6g") -> None:
    if df is None or df.empty:
        print("empty df")
        return
    print(df.to_string(index=False, float_format=lambda x: format(x, float_fmt)))


def plot_tail_vol(df: pd.DataFrame, target_var_col: str, forecast_var_cols: list[str], n: int = 500, title: str = "") -> None:
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
    meta = report["meta"]
    wf_hold: pd.DataFrame = report["wf_hold"]

    print_section(f"EXPERIMENT: {meta['label']}")
    print(f"horizon={meta['horizon']}  holdout_start_date={meta['holdout_start_date']}  baseline=EWMA  var_proxy=close-to-close")
    
    if "data_diag" in report:
        dd = report["data_diag"]
        print_section("DATA diagnostics")
        print("Core index:", dd["core_index"])
        print("Holdout index:", dd["holdout_index"])

        print_section("Core coverage (key cols)")
        print_df(dd["core_coverage"]) 

        print_section("Holdout coverage (target/baseline/models)")
        print_df(dd["holdout_coverage"])

        print_section("Aux coverage (aligned)")
        print_df(dd["aux_coverage"])

    print_section("HOLDOUT report (minimal)")
    if len(wf_hold):
        print(f"HOLDOUT: {wf_hold.index.min().date()} -> {wf_hold.index.max().date()}  | rows={len(wf_hold)}")
    else:
        print("HOLDOUT is empty.")

    print_section("HOLDOUT availability summary (compact)")
    print_df(report["availability"])  # type: ignore[arg-type]

    print_section("HOLDOUT headline table (pairwise vs EWMA baseline)")
    print_df(report["headline_full"])  # type: ignore[arg-type]

    mid = report["split_mid"]
    print_section(f"HOLDOUT split-half headline (cut date ~ {mid.date()})")
    hh = pd.concat([report["headline_half1"], report["headline_half2"]], axis=0)  # type: ignore[list-item]
    print_df(hh)

    print_section("XGB mean vs median sanity (HOLDOUT)")
    print("Interpretation: mean-correction helps if QLIKE(mean) < QLIKE(median) and d_qlike(mean-median) < 0.")
    print_df(report["xgb_sanity"])  # type: ignore[arg-type]

    print_section("Calibration monotonicity (HOLDOUT): Spearman corr(forecast vol, realized vol)")
    print_df(report["calibration"])  # type: ignore[arg-type]

    baseline_col = report["baseline_col"]
    print_section(f"DM vs baseline='{baseline_col}' on HOLDOUT (overlap daily) | HAC grid={meta['hac_lag_grid']}")
    print("Note: DM is supporting. Focus on sign/magnitude stability across HAC lags.")
    print_df(report["dm"])  # type: ignore[arg-type]

    print_section("Strategy backtest (HOLDOUT): compact cost grid")
    print_df(report["strategy"])  # type: ignore[arg-type]


def plot_report(report: dict[str, object], *, n: int = 500) -> None:
    wf_hold: pd.DataFrame = report["wf_hold"]
    meta = report["meta"]
    model_cols_headline: list[str] = report["model_cols_headline"]

    if len(wf_hold) == 0:
        return

    plot_tail_vol(
        wf_hold,
        target_var_col="rv20_fwd_var",
        forecast_var_cols=safe_cols(wf_hold, model_cols_headline),
        n=n,
        title=f"{meta['label']}: HOLDOUT tail (VOL space)",
    )

def save_report_csvs(report: dict[str, object], *, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    headline_full: pd.DataFrame = report["headline_full"]
    headline_h1: pd.DataFrame = report["headline_half1"]
    headline_h2: pd.DataFrame = report["headline_half2"]
    strat_df: pd.DataFrame = report["strategy"]

    forecast_csv = pd.concat([headline_full, headline_h1, headline_h2], axis=0).reset_index(drop=True)
    forecast_csv.to_csv(out_dir / "forecast_holdout_headline.csv", index=False)
    strat_df.to_csv(out_dir / "strategy_stats_cost_grid.csv", index=False)
