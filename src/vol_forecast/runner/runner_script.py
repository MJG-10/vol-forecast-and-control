"""Runner script to load inputs, compute the experiment report and emit console output/CSVs/plots."""
import pandas as pd
from pathlib import Path
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.runner.report_io import (console_report, 
                                           save_report_csvs, plot_report)
from vol_forecast.runner.experiment import compute_experiment_report
from vol_forecast.experiment_pipeline import build_experiment_df


def run_experiment(
    *,
    start_date: str,
    end_date: str | None,
    horizon: int,
    freq: int,
    wf_cfg: WalkForwardConfig,
    sigma_target: float,
    tcost_grid_bps:list[float] | None,
    garch_dist: str,
    holdout_start_date: str ,
    gjr_pneg_mode: str,
    hac_lag_grid: list[int] | None,
    run_strategy: bool,
    strategy_variants: list[str] | None,
    do_plots: bool | None,
    out_dir: Path | None = None,
    ) -> None:
    """Builds inputs, computes an experiment report, and emits console output/CSVs/plots."""
    # Build canonical experiment dataframe (RET + CASH + FEATURES)
    base_df, build_meta = build_experiment_df(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        freq=freq,
    )

    report = compute_experiment_report(base_df,
                                       horizon=horizon,
                                       freq=freq,
                                       wf_cfg=wf_cfg,
                                       garch_dist=garch_dist,
                                       holdout_start_date=holdout_start_date,
                                       gjr_pneg_mode=gjr_pneg_mode,
                                       hac_lag_grid=hac_lag_grid,
                                       run_strategy=run_strategy,
                                       strategy_variants=strategy_variants,
                                       sigma_target=sigma_target,
                                       tcost_grid_bps=tcost_grid_bps)

    report["build_meta"] = build_meta
    console_report(report)

    if out_dir is not None:
        save_report_csvs(report, out_dir=out_dir)

    if do_plots:
        plot_report(report)


def main() -> None:
    """Entry point for running the experiment with the default parameters."""
    horizon = 20
    freq = 252

    wf_cfg = WalkForwardConfig(
        window_type="rolling",   # "rolling" or "expanding"
        rolling_window_size=1000,
        min_train_size = 500,
        refit_every=60,
    )

    sigma_target = 0.10
    garch_dist = "t"
    gjr_pneg_mode = "empirical" # "implied" or "empirical"  

    holdout_start_date = "2019-01-01"
    tcost_grid_bps = [0.0, 5.0, 10.0, 25.0]

    run_experiment(
        start_date="1990-01-01",
        end_date=None,
        horizon=horizon,
        freq=freq,
        wf_cfg=wf_cfg,
        sigma_target=sigma_target,
        tcost_grid_bps=tcost_grid_bps,
        garch_dist=garch_dist,
        holdout_start_date=holdout_start_date,
        gjr_pneg_mode=gjr_pneg_mode,
        hac_lag_grid=[20, 30, 40, 60],
        run_strategy = True, 
        strategy_variants=["daily", "tranche20"],
        do_plots=True,
    )



if __name__ == "__main__":
    main()