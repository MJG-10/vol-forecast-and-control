import pandas as pd
import numpy as np
from vol_forecast.schema import COLS, require_cols


def compute_daily_var_close(df: pd.DataFrame, ret_col: str) -> pd.Series:
    """Daily variance proxy from close-to-close returns: r_t^2."""
    return (df[ret_col].astype(float) ** 2).rename(COLS.DAILY_VAR)


def compute_trailing_annualized_var(daily_var: pd.Series, window: int = 20, freq: int = 252) -> pd.Series:
    """Trailing annualized variance: freq * mean(daily_var over last `window` observations)."""
    return float(freq) * daily_var.rolling(window=window).mean()


def compute_forward_annualized_var(daily_var: pd.Series, horizon: int = 20, freq: int = 252) -> pd.Series:
    """Forward annualized variance target aligned at origin t: freq * mean(daily_var[t..t+h-1]) (implemented via shift)."""
    h = int(horizon)
    return float(freq) * daily_var.rolling(window=h).mean().shift(-(h - 1))


def add_vix_features_tminus1(df: pd.DataFrame, vix_close: pd.Series) -> pd.DataFrame:
    """
    Adds VIX features (t-1 aligned).

    Expects a VIX close series; values are reindexed to df.index and ffilled,
    then shifted so predictors at origin t only use info available by close of t-1.
    """
    df = df.copy()
    eps = 1e-12

    v = vix_close.reindex(df.index).ffill()
    lv = np.log(v.astype(float).clip(lower=eps))

    df[COLS.LOG_VIX_LAG1] = lv.shift(1)
    df[COLS.DLOG_VIX_5] = lv.shift(1) - lv.shift(6)
    return df


def add_baseline_forecasts_var_tminus1(
    df: pd.DataFrame,
    trailing_var_col: str
) -> pd.DataFrame:
    """
    Adds t-1 aligned baseline variance forecasts.

    - RW baseline: trailing realized variance over the horizon, shifted by 1 day.
    """
    df = df.copy()
    v_lag = df[trailing_var_col].shift(1)
    df[COLS.RW_FORECAST_VAR] = v_lag
    return df


def add_har_features_from_daily_var_tminus1(
    df: pd.DataFrame,
    daily_var_col: str,
    *,
    freq: int = 252
) -> pd.DataFrame:
    """
    Adds HAR-style predictors from a daily variance proxy, aligned to t-1 and annualized.

    Features:
      COLS.DVHAR_1D  = freq * daily_var[t-1]
      COLS.DVHAR_5D  = freq * mean(daily_var[t-1..t-5])
      COLS.DVHAR_22D = freq * mean(daily_var[t-1..t-22])
    """
    df = df.copy()
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
    """Adds log-transformed HAR features (with clipping) to stabilize scale and reduce skew."""
    df = df.copy()

    df[COLS.LOG_DVHAR_1D] = np.log(df[COLS.DVHAR_1D].clip(lower=eps))
    df[COLS.LOG_DVHAR_5D] = np.log(df[COLS.DVHAR_5D].clip(lower=eps))
    df[COLS.LOG_DVHAR_22D] = np.log(df[COLS.DVHAR_22D].clip(lower=eps))
    return df


def add_log_target_var(df: pd.DataFrame, target_var_col: str, eps: float = 1e-18) -> pd.DataFrame:
    """Add `COLS.LOG_TARGET_VAR` = log(target_var_col) after clipping at `eps` for numerical stability."""
    df = df.copy()
    df[COLS.LOG_TARGET_VAR] = np.log(df[target_var_col].clip(lower=eps))
    return df


def build_core_features(
    df: pd.DataFrame,
    *,
    ret_col: str,
    horizon: int,
    vix_close: pd.Series,
    freq: int = 252,
    trailing_window: int = 20,
    eps_log: float = 1e-18
) -> pd.DataFrame:
    """
    Builds the canonical feature/target table used by the forecasting pipeline.

    Calls the lower-level feature builders and standardizes column names via `COLS`.
    Does not drop rows.
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
        out, trailing_var_col=COLS.RV20_VAR
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

