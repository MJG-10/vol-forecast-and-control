def get_train_slice(i: int, window_type: str, rolling_w: int) -> slice:
    if window_type == "expanding":
        return slice(0, i)
    if window_type == "rolling":
        start = max(0, i - rolling_w)
        return slice(start, i)
    raise ValueError("window_type must be 'expanding' or 'rolling'")


def compute_train_end_excl(
    pos: int,
    *,
    horizon: int,
    min_train_end_excl: int = 51
) -> int|None:
    """
    Returns the exclusive training end index for a forecast at integer position `pos`.

    Only allow training rows i whose forward label
    window [i .. i+horizon-1] fully ends before `pos`. This implies:
        train_end_excl = pos - horizon + 1

    If not enough history (train_end_excl <= min_train_end_excl-1), return None (caller continues).
    """
    train_end_excl = pos - horizon + 1

    if train_end_excl < min_train_end_excl:
        return None

    return train_end_excl


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