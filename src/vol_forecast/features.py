import pandas as pd
import numpy as np
from vol_forecast.schema import COLS, require_cols


# def forward_mean(series: pd.Series, horizon: int) -> pd.Series:
#     """
#     Forward looking rolling mean aligned at time t.

#     Computes: out[t] = mean(series[t], ..., series[t+horizon-1]).
#     """
#     # rolling() is backward-looking by default; shift aligns the mean to start at t.
#     return series.rolling(window=horizon).mean().shift(-(horizon - 1))


def compute_daily_var_close(df: pd.DataFrame, ret_col: str) -> pd.Series:
    """Daily variance proxy from close-to-close returns: r_t^2."""
    return (df[ret_col].astype(float) ** 2).rename("daily_var")


def compute_trailing_annualized_var(daily_var: pd.Series, window: int = 20, freq: int = 252) -> pd.Series:
    """Trailing annualized variance: freq * mean(daily_var over last `window` observations)."""
    return float(freq) * daily_var.rolling(window=window).mean()


# def compute_forward_annualized_var(daily_var: pd.Series, horizon: int = 20, freq: int = 252) -> pd.Series:
#     """Forward annualized variance target: freq * mean(daily_var over next `horizon` observations)."""
#     return float(freq) * forward_mean(daily_var, horizon=horizon)

def compute_forward_annualized_var(daily_var: pd.Series, horizon: int = 20, freq: int = 252) -> pd.Series:
    """Forward annualized variance target aligned at t: freq * mean(daily_var[t..t+h-1])."""
    h = int(horizon)
    return float(freq) * daily_var.rolling(window=h).mean().shift(-(h - 1))


def add_vix_features_tminus1(df: pd.DataFrame, vix_close: pd.Series) -> pd.DataFrame:
    """
    Add minimal VIX features aligned to t-1 (no look-ahead).

    Adds:
      - log_{prefix}_lag1: log(VIX_{t-1})
      - dlog_{prefix}_5:   log(VIX_{t-1}) - log(VIX_{t-6})  (5 business-day log change)
    """
    df = df.copy()
    eps = 1e-12

    # Align VIX to the main index and forward-fill to handle VIX holidays / missing dates.
    v = vix_close.reindex(df.index).ffill()
    lv = np.log(v.astype(float).clip(lower=eps))

    # t-1 alignment ensures features are known at decision time t.
    df[COLS.LOG_VIX_LAG1] = lv.shift(1)
    df[COLS.DLOG_VIX_5] = lv.shift(1) - lv.shift(6)
    return df


def add_baseline_forecasts_var_tminus1(
    df: pd.DataFrame,
    trailing_var_col: str,
    ewma_alpha: float = 0.06,
) -> pd.DataFrame:
    """
    Add simple variance-forecast baselines (t-1 aligned): random-walk and EWMA.

    Uses trailing variance shifted by 1 to avoid using information from day t when forecasting at t.
    """
    df = df.copy()
    v_lag = df[trailing_var_col].shift(1)

    # Random-walk baseline: forecast equals last available trailing estimate.
    df[COLS.RW_FORECAST_VAR] = v_lag

    # EWMA baseline on the lagged trailing variance series (common, stable benchmark).
    df[COLS.EWMA_FORECAST_VAR] = v_lag.ewm(alpha=ewma_alpha, adjust=False).mean()
    return df


def add_har_features_from_daily_var_tminus1(
    df: pd.DataFrame,
    daily_var_col: str,
    *,
    freq: int = 252,
    prefix: str = "dvhar",
) -> pd.DataFrame:
    """
    Add HAR-style predictors from a daily variance proxy, aligned to t-1 and annualized.

    Features:
      COLS.DVHAR_1D  = freq * daily_var[t-1]
      COLS.DVHAR_5D  = freq * mean(daily_var[t-1..t-5])
      COLS.DVHAR_22D = freq * mean(daily_var[t-1..t-22])
    """
    df = df.copy()

    # Shift to t-1 to prevent look-ahead when predicting a target defined over t..t+h-1.
    v = df[daily_var_col].shift(1)

    df[COLS.DVHAR_1D] = float(freq) * v
    df[COLS.DVHAR_5D] = float(freq) * v.rolling(window=5).mean()
    df[COLS.DVHAR_22D] = float(freq) * v.rolling(window=22).mean()
    return df


def add_log_features_for_daily_har(
    df: pd.DataFrame,
    *,
    eps: float = 1e-18,
) -> pd.DataFrame:
    """Add log-transformed HAR features (with clipping) to stabilize scale and reduce skew."""
    df = df.copy()

    # Clip to avoid log(0) in calm regimes (variance proxy can be extremely small).
    df[COLS.LOG_DVHAR_1D] = np.log(df[COLS.DVHAR_1D].clip(lower=eps))
    df[COLS.LOG_DVHAR_5D] = np.log(df[COLS.DVHAR_5D].clip(lower=eps))
    df[COLS.LOG_DVHAR_22D] = np.log(df[COLS.DVHAR_22D].clip(lower=eps))
    return df


def add_log_target_var(df: pd.DataFrame, target_var_col: str, eps: float = 1e-18) -> pd.DataFrame:
    """Add `log_target_var` = log(target_var_col) with clipping to avoid log(0)."""
    df = df.copy()
    df[COLS.LOG_TARGET_VAR] = np.log(df[target_var_col].clip(lower=eps))
    return df


def build_core_features(
    df: pd.DataFrame,
    *,
    ret_col: str,
    horizon: int,
    freq: int = 252,
    trailing_window: int = 20,
    ewma_alpha: float = 0.06,
    vix_close: pd.Series = None,
    eps_log: float = 1e-18,
) -> pd.DataFrame:
    """
    Build the canonical feature/target table used by the forecasting pipeline.

    This wrapper composes the primitive feature functions and pins down the
    project-wide column names via `COLS`. It does not drop rows.
    """
    out = df.copy()
    # strict boundary check
    require_cols(df.columns, [ret_col], context="build_core_features")

    # 1) Variance proxy
    out[COLS.DAILY_VAR] = compute_daily_var_close(out, ret_col=ret_col)

    # 2) Trailing realized variance + forward target variance
    out[COLS.RV20_VAR] = compute_trailing_annualized_var(
        out[COLS.DAILY_VAR], window=int(trailing_window), freq=int(freq)
    )
    out[COLS.RV20_FWD_VAR] = compute_forward_annualized_var(
        out[COLS.DAILY_VAR], horizon=int(horizon), freq=int(freq)
    )

    # 3) Baselines (t-1 aligned)
    out = add_baseline_forecasts_var_tminus1(
        out, trailing_var_col=COLS.RV20_VAR, ewma_alpha=float(ewma_alpha)
    )

    # 4) HAR features (t-1 aligned) + log transforms
    out = add_har_features_from_daily_var_tminus1(
        out, daily_var_col=COLS.DAILY_VAR, freq=int(freq)
    )
    out = add_log_features_for_daily_har(out, eps=float(eps_log))

    # 5) Log target
    out = add_log_target_var(out, target_var_col=COLS.RV20_FWD_VAR, eps=float(eps_log))

    # 6) VIX features
    out = add_vix_features_tminus1(out, vix_close=vix_close)

    return out

