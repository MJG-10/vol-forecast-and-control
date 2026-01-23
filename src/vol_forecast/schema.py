"""
Time index and alignment convention (canonical)

Let t denote the DatetimeIndex label of a row.

1) Equity return:
   RET[t] = log(P_t / P_{t-1}), i.e. the close-to-close log return over (t-1 -> t).

2) Cash return:
   CASH_R[t] is the simple cash return accrued over the same holding interval (t-1 -> t),
   computed from an annualized rate series using ACT/360 on the calendar-day gap.
   The rate *level* is lagged by 1 trading day so that the fixing used for (t-1 -> t)
   is available at/ before t-1 (conservative).

3) Features used for forecasting at row t must be known without using information from day t.
   Therefore, predictors derived from market closes (e.g. VIX) are aligned to t-1.

4) Strategy execution timing:
   The leverage applied to RET[t] must be computed from information available by t-1 close.
   If inputs are already t-1 aligned, then no additional execution lag is required.
"""

from dataclasses import dataclass
from collections.abc import Iterable, Sequence


@dataclass(frozen=True)
class Cols:
    """
    Canonical column names used across modules.
    Keep these stable to avoid string drift between features/models/runner.
    """

    # Core return / variance / targets
    RET: str = "log_ret"
    DAILY_VAR: str = "daily_var"
    RV20_VAR: str = "rv20_var"
    RV20_FWD_VAR: str = "rv20_fwd_var"
    LOG_TARGET_VAR: str = "log_target_var"

    # Cash return
    CASH_R: str = "cash_r_act360"

    # Baselines
    RW_FORECAST_VAR: str = "rw_forecast_var"
    EWMA_FORECAST_VAR: str = "ewma_forecast_var"

    # HAR
    DVHAR_1D: str = "dvhar_1d"
    DVHAR_5D: str = "dvhar_5d"
    DVHAR_22D: str = "dvhar_22d"

    # HAR (log features)
    LOG_DVHAR_1D: str = "log_dvhar_1d"
    LOG_DVHAR_5D: str = "log_dvhar_5d"
    LOG_DVHAR_22D: str = "log_dvhar_22d"

    # raw VIX
    VIX_CLOSE: str = "vix_close"

    # VIX features
    LOG_VIX_LAG1: str = "log_vix_lag1"
    DLOG_VIX_5: str = "dlog_vix_5"


    @property
    def HAR_FEATURES(self) -> tuple[str, ...]:
        return (self.DVHAR_1D, self.DVHAR_5D, self.DVHAR_22D)

    @property
    def HAR_LOG_FEATURES(self) -> tuple[str, ...]:
        return (self.LOG_DVHAR_1D, self.LOG_DVHAR_5D, self.LOG_DVHAR_22D)

    @property
    def VIX_FEATURES(self) -> tuple[str, ...]:
        return (self.LOG_VIX_LAG1, self.DLOG_VIX_5)

    @property
    def EXPERIMENT_CORE_COLS(self) -> tuple[str, ...]:
        return (
            self.RET,
            self.DAILY_VAR,
            self.RV20_VAR,
            self.RV20_FWD_VAR,
            self.RW_FORECAST_VAR,
            self.EWMA_FORECAST_VAR,
            self.LOG_TARGET_VAR,
            *self.HAR_LOG_FEATURES,
            *self.VIX_FEATURES
        )

COLS = Cols()


def missing_cols(available: Sequence[str], required: Iterable[str]) -> list[str]:
    """Return missing column names (preserving order)."""
    avail = set(available)
    return [c for c in required if c not in avail]


def require_cols(available: Sequence[str], required: Iterable[str], *, context: str = "") -> None:
    """
    Strict-mode validator: raise if any required column is missing.
    Use this in runner/model code when you prefer hard failures over silent skipping.
    """
    miss = missing_cols(available, required)
    if miss:
        prefix = f"{context}: " if context else ""
        raise ValueError(prefix + f"Missing required columns: {miss}")


