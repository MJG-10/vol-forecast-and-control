import pandas as pd
import numpy as np
from pathlib import Path
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.runner.report_io import (console_report, 
                                           save_report_csvs, plot_report, print_section)

from vol_forecast.runner.experiment import compute_experiment_report
from vol_forecast.data import (load_sp500_ohlc, 
                          compute_log_returns,
                          load_sp500_total_return_close,
                          compute_log_returns_from_series,
                          load_cash_daily_simple_act360,
)
from vol_forecast.schema import COLS


def run_experiment(
    base: pd.DataFrame,
    *,
    do_plots: bool = True,
    out_dir: Path|None = None,
    **kwargs,
) -> dict[str, object]:
    report = compute_experiment_report(base, **kwargs)

    console_report(report)

    if out_dir is not None:
        save_report_csvs(report, out_dir=out_dir)

    if do_plots:
        plot_report(report)

    return report


def main() -> None:
    horizon = 20
    freq = 252

    wf_cfg = WalkForwardConfig(
        initial_train_frac=0.4,
        window_type="rolling",   # "rolling" or "expanding"
        rolling_w=2000,
        refit_every=60,
    )

    sigma_target = 0.10
    L_max = 1.0
    cash_rate_annual_fallback = 0.03

    garch_dist = "t"
    # gjr_pneg_mode = "implied"
    gjr_pneg_mode = "empirical"

    holdout_start_date = "2019-01-01"
    tcost_grid_bps = [0.0, 5.0, 10.0, 25.0]

    print_section("LOAD TOTAL RETURN SERIES")
    tr_close = load_sp500_total_return_close(start_date="1990-01-01", end_date=None)
    base = compute_log_returns_from_series(tr_close, out_name=COLS.RET).to_frame()
    print("Loaded ^SP500TR successfully.")

    label = ("S&P 500 TOTAL RETURN")

    print_section("LOAD CASH PROXY (ACT/360 per-period returns)")
    cash_daily_simple, cash_source = load_cash_daily_simple_act360(
        start_date=str(base.index.min().date()),
        end_date=None,
        trading_index=base.index,   # required for ACT/360 gap accrual
        prefer_fred=True
    )

    cash_daily_simple = cash_daily_simple.reindex(base.index)
    
    run_experiment(
        base=base,
        label=label,
        horizon=horizon,
        freq=freq,
        wf_cfg=wf_cfg,
        sigma_target=sigma_target,
        L_max=L_max,
        tcost_grid_bps=tcost_grid_bps,
        cash_daily_simple=cash_daily_simple,
        garch_dist=garch_dist,
        holdout_start_date=holdout_start_date,
        gjr_pneg_mode=gjr_pneg_mode,
        hac_lag_grid = [20, 30, 40, 60],
        do_plots=True,
    )


if __name__ == "__main__":
    main()
