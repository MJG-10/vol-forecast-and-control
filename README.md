# vol-forecast-and-control

In this project we implement a walk-forward experiment to forecast volatility for equity index returns (default here is S&P 500 Total Return). We evaluate multiple models using loss-based diagnostics during a holdout period and also feed these forecasts into an optional volatility-targeting backtest with transaction cost sensitivity.

The objective is to compare the volatility forecasts of econometric and machine learning models against simple baselines.

## What we test specifically 

- **Target**: forward realized variance over horizon $h$ (annualized; $h = 20$ trading days here).
$$\mathrm{RVar}_{\mathrm{fwd,ann}}(t;h) = \frac{f}{h}\sum_{k=0}^{h-1} r_{t+k}^2$$
where $r_t = \log(P_t / P_{t-1})$ is the daily log return and $f$ is the annualization factor (usually 252).
- **Timing convention**: at forecast origin **t**, predictors use information available by the close of **t-1** (no look-ahead).
  The target is aligned back onto **t** (forward window **t .. t+h-1**). Strategy execution can optionally apply an extra delay via `execution_lag_days`.
  The target $\mathrm{RVar}_{\mathrm{fwd,ann}}(t;h)$ aggregates realized returns $(r_t,\ldots,r_{t+h-1})$ and is aligned back onto date $t$.
  Strategy execution can optionally apply an additional delay via `execution_lag_days`.
- **Experiment design**: walk-forward training with a fixed holdout segment for headline comparisons.
- **Models**: HAR-type models, GARCH variants (including GJR-GARCH) and XGBoost-based forecasters (HAR-only and HAR+VIX variants).
- **Baseline**: Random-walk baseline on variance.
- **Diagnostics**: Loss-based evaluation (RMSE on volatility, QLIKE on variance) and Diebold–Mariano (DM) tests across multiple HAC lags. Headline comparisons are based on QLIKE and RMSE, and DM tests are added for context.
- **Backtest (optional)**: a volatility-targeting backtest driven by the forecasts and evaluated across a grid of transaction costs. This requires a cash return series and can report multiple execution variants (daily with turnover buffer or tranche-style rebalancing). Execution timing is controlled via `execution_lag_days` (an extra lag beyond the t-1 predictor alignment).


## Results

The primary output of this repository is the results notebook, which contains the tables, figures, and the narrative for the holdout evaluation and strategy sensitivity analysis.

* **Main results notebook:** `notebooks/01_results.ipynb`

The source code in `src/vol_forecast/` is structured to keep the notebook thin and make refactoring convenient. 

## Installation

From the repository root:

**Standard install:**

```bash
pip install -e .
```

**Full pipeline dependencies (data sources + models + plotting):**

```bash
pip install -e ".[full]"
```

**Notebook tooling:**

```bash
pip install -e ".[dev]"
```

**Full pipeline + notebook tooling**

```bash
pip install -e ".[full,dev]"
```

## How to run


1. **Notebook run (end-to-end, flexible)**

In the results notebook, we run the workflow step by step:

- Build the canonical experiment dataframe `df` (data loading + return construction + feature/target engineering).
- (Optional) Run XGB pre-holdout tuning to choose among a small fixed set of parameter configurations.
- Run `compute_experiment_report(df, ...)` to produce the report dictionary (tables/panels + metadata) used by the notebook.

The notebook-friendly wrapper `run_experiment(...)` performs the same workflow in one call and returns a report dictionary containing the evaluation panels/tables and metadata without plotting or printing anything.

Path: `src/vol_forecast/runner/experiment.py` (`run_experiment`, `compute_experiment_report`)

2. **Script run**

For an end-to-end run that produces console output and optional plots.

Path: `src/vol_forecast/runner/runner_script.py`

This script is a convenience wrapper around the compute pipeline and emits console output and optional plots.

## Repository layout

- `src/vol_forecast/`: Installable package (src layout)

  - `data.py`: Data loading and return calculation helpers.

  - `features.py`: Feature and target construction.

  - `models/`: Forecasting models (HAR, GARCH, XGB) and walk-forward utilities.

  - `models_tuning/`: Optional pre-holdout tuning utilities (candidate grids + selection logic for XGB parameter overrides).

  - `eval/`: Reporting and statistical comparisons (headline tables, DM panels, diagnostics).

  - `runner/`: Experiment orchestration (compute pipeline, script wrapper, report I/O).

  - `strategy.py`: Volatility targeting backtest utilities.

  - `wf_config.py`: Walk-forward configuration.

  - `schema.py`: Canonical column names (`COLS`) shared across modules.

- `notebooks/`: Results notebook


## Data sources

Equity index series: Loaded from Yahoo via `yfinance` (default: `^SP500TR`).

VIX: FRED `VIXCLS` (via `pandas_datareader`).

Cash proxy: FRED `DFF` spliced/overwritten with `EFFR` when available, converted to per-period simple returns using ACT/360 conventions and aligned with a 1-trading-day lag.


