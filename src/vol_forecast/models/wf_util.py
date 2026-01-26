import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig


def get_train_slice(i: int, window_type: str, rolling_window_size: int | None) -> slice:
    if window_type == "expanding":
        return slice(0, i)
    if window_type == "rolling":
        if rolling_window_size is None:
            raise ValueError("rolling_window_size must be set in rolling mode")
        start = max(0, i - rolling_window_size)
        return slice(start, i)
    raise ValueError("window_type must be 'expanding' or 'rolling'")


def compute_start_pos(
    idx: pd.DatetimeIndex,
    *,
    cfg: WalkForwardConfig,
    n_rows: int,
    origin_start_date: pd.Timestamp | None = None,
) -> int:
    """
    First origin position to forecast.

    Combines:
      (1) feasibility: at least cfg.initial_train_size usable rows (after dropna),
      (2) optional boundary: do not forecast before origin_start_date.

    If origin_start_date is not an exact index member, uses searchsorted (next available date).
    """
    n = int(n_rows)
    if n <= 0:
        return 0

    start_pos = int(cfg.initial_train_size)

    if origin_start_date is not None:
        ts = pd.Timestamp(origin_start_date)
        start_pos = max(start_pos, int(idx.searchsorted(ts)))

    return start_pos


def compute_train_end_excl(pos: int, *, horizon: int) -> int:
    """
    Exclusive training end index for forecast at position `pos`, enforcing overlap safety:
    train indices i must satisfy i + horizon - 1 < pos  =>  i <= pos - horizon
    so train_end_excl = pos - horizon + 1 (exclusive).
    """
    return max(0, pos - horizon + 1)


def compute_purged_val_split(
    n_all: int,
    *,
    val_frac: float,
    min_val_size: int,
    embargo: int,
    min_train_size: int = 100,
    min_val_points: int = 50,
) -> tuple[int, int]|None:
    """
    Given a contiguous in-sample block of length n_all, compute a purged train/val split:

      [0 .. split_end) = train for fitting
      [split_end .. split_end+embargo) = embargo gap
      [split_end+embargo .. split_end+embargo+val_size) = validation

    Returns (split_end, val_size), or None if a feasible split cannot be formed.
    This matches your current intent: keep validation near the end, and enforce a gap
    to reduce leakage/overlap issues.

    Notes:
    - min_train_size applies to the fitting block length (split_end >= min_train_size).
    - min_val_points is the minimum size of validation to be meaningful.
    """
    if n_all < (min_train_size + embargo + min_val_points):
        return None

    val_size = max(int(min_val_size), int(val_frac * n_all))
    val_size = min(val_size, max(min_val_points, n_all // 4))

    split_end = n_all - (val_size + embargo)

    if split_end < min_train_size:
        # Try to salvage by shrinking validation while keeping min_train_size and embargo
        val_size2 = n_all - (min_train_size + embargo)
        if val_size2 < min_val_points:
            return None
        val_size = val_size2
        split_end = n_all - (val_size + embargo)
        if split_end < min_train_size:
            return None

    return int(split_end), int(val_size)