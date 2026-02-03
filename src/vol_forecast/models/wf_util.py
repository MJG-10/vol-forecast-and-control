import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig


def get_train_slice(i: int, window_type: str, rolling_window_size: int | None) -> slice:
    """Returns the training slice ending at i (exclusive) for expanding or rolling windows."""
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
    First feasible forecast-origin position (row index).

    Enforces minimum training history (cfg.initial_train_size or cfg.min_train_size),
    and optionally moves the start forward to origin_start_date (via searchsorted).
    """
    n = int(n_rows)
    if n <= 0:
        return 0

    start_pos = cfg.initial_train_size if cfg.window_type == "expanding" else cfg.min_train_size

    if origin_start_date is not None:
        ts = pd.Timestamp(origin_start_date)
        start_pos = max(start_pos, int(idx.searchsorted(ts)))

    return start_pos

def compute_train_end_excl(pos: int, *, horizon: int) -> int:
    """
    Exclusive end index for training origins when the label uses a forward window.

    For a horizon h label at origin i that depends on rows [i .. i+h-1], requiring
    i+h-1 < pos implies i <= pos-h. Therefore the training slice should end at
    pos-h+1 (exclusive).
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
) -> tuple[int, int] | None:
    """
    Purged train/val split with an embargo gap to prevent overlap of forward-window labels.

      train:   [0 .. split_end)
      embargo: [split_end .. split_end+embargo)
      val:     [split_end+embargo .. split_end+embargo+val_size)

    Returns (split_end, val_size), or None if infeasible.
    """
    if n_all < (min_train_size + embargo + min_val_points):
        return None

    val_size = max(int(min_val_size), int(val_frac * n_all))
    val_size = min(val_size, max(min_val_points, n_all // 4))

    split_end = n_all - (val_size + embargo)

    if split_end < min_train_size:
        val_size2 = n_all - (min_train_size + embargo)
        if val_size2 < min_val_points:
            return None
        val_size = val_size2
        split_end = n_all - (val_size + embargo)
        if split_end < min_train_size:
            return None

    return int(split_end), int(val_size)