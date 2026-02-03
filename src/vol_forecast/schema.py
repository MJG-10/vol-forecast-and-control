"""
Time index and alignment convention (canonical)

Let t denote the DatetimeIndex label of a row.

1) Equity return:
   RET[t] = log(P_t / P_{t-1}), i.e. close-to-close log return over (t-1 -> t).

2) Cash return:
   CASH_R[t] is the simple cash return accrued over (t-1 -> t). It is computed from an
   annualized overnight rate series using ACT/360 on the calendar-day gap. The accrual
   for (t-1 -> t) uses the rate level available by t-1 (i.e., the rate series is lagged
   by 1 trading day before conversion to returns).

3) Forecasting features:
   Features used at origin t must be known without using information from day t.
   Therefore, predictors derived from market closes (e.g., VIX, daily variance)
   are aligned to t-1.

4) Strategy execution timing:
   The leverage applied to RET[t] must be computable from information available
   by t-1 close. execution_lag_days is an extra conservative delay beyond t-1
   alignment; set to 0 if t-1 alignment is treated as sufficient.
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

    # Baseline
    RW_FORECAST_VAR: str = "rw_forecast_var"

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
            self.LOG_TARGET_VAR,
            *self.HAR_LOG_FEATURES,
            *self.VIX_FEATURES
        )

COLS = Cols()


def missing_cols(available: Sequence[str], required: Iterable[str]) -> list[str]:
    """Returns missing column names (preserving order)."""
    avail = set(available)
    return [c for c in required if c not in avail]


def require_cols(available: Sequence[str], required: Iterable[str], *, context: str = "") -> None:
    """
    Strict-mode validator: raises error if any required column is missing.
    """
    miss = missing_cols(available, required)
    if miss:
        prefix = f"{context}: " if context else ""
        raise ValueError(prefix + f"Missing required columns: {miss}")


