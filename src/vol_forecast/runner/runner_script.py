"""Runner script to load inputs, compute the experiment report and emit console output/CSVs/plots."""
from pathlib import Path
import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.runner.report_io import (console_report, 
                                           save_report_csvs, plot_report)
from vol_forecast.runner.experiment import run_experiment


def main_runner(
    *,
    horizon: int,
    wf_cfg: WalkForwardConfig,
    start_date: str | None,
    end_date: str | None,
    holdout_start_date: str | None,
    do_xgb_tuning: bool,
    hac_lag_grid: list[int],
    run_strategy: bool,
    strategy_variants: list[str],
    sigma_target: float,
    tcost_grid_bps: list[float],
    tuning_blocks: list[tuple[pd.Timestamp, pd.Timestamp]],
    out_dir: Path | None,
    do_plots: bool
) -> None:   
    """Script wrapper around `run_experiment` that prints/saves/plots report artifacts."""
    report = run_experiment(
        start_date=start_date,
        end_date=end_date,
        horizon=horizon,
        wf_cfg=wf_cfg,
        holdout_start_date=holdout_start_date,
        tuning_blocks=tuning_blocks,
        sigma_target=sigma_target,
        tcost_grid_bps=tcost_grid_bps,
        hac_lag_grid=hac_lag_grid,
        run_strategy=run_strategy,
        strategy_variants=strategy_variants,
        do_xgb_tuning=do_xgb_tuning
    )

    console_report(report)

    if out_dir is not None:
        save_report_csvs(report, out_dir=out_dir)

    if do_plots:
        plot_report(report)


def main() -> None:
    """Entry point for running the experiment with the default parameters."""
    
    horizon = 22

    wf_cfg = WalkForwardConfig(
        window_type="rolling",
        rolling_window_size=1000,
        min_train_size = 500,
        refit_every=60,
    )
  
    data_start_date = "2004-01-01"
    data_end_date = "2026-02-06"
    holdout_start_date = "2015-01-01"

    run_strategy = True
    do_xgb_tuning = True

    out_dir = None
    do_plots = True

    sigma_target = 0.1
    tcost_grid_bps = [0.0, 5.0, 10.0, 25.0]
    hac_lag_grid = [20, 30, 40, 60]

    strategy_variants = ["daily_reset", "band_no_trade"]

    # Pre-holdout tuning blocks
    xgb_tuning_block_dates = [
    ("2005-01-01", "2006-12-31"),
    ("2008-01-01", "2009-12-31"),
    ("2011-01-01", "2012-12-31"),
    ("2013-06-01", "2014-11-30"),
    ]

    xgb_tuning_blocks  = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in xgb_tuning_block_dates]

    main_runner(
        horizon=horizon,
        wf_cfg=wf_cfg,
        start_date=data_start_date,
        end_date=data_end_date,
        holdout_start_date=holdout_start_date,
        sigma_target=sigma_target,
        tcost_grid_bps=tcost_grid_bps,
        do_xgb_tuning=do_xgb_tuning,
        tuning_blocks=xgb_tuning_blocks,
        hac_lag_grid=hac_lag_grid,
        run_strategy=run_strategy,
        strategy_variants=strategy_variants,
        out_dir=out_dir,
        do_plots=do_plots
    )

if __name__ == "__main__":
    main()