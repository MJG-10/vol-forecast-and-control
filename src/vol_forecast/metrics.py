import numpy as np


def qlike_series_var(v_true, v_pred, eps: float = 1e-12) -> np.ndarray:
    """
    QLIKE per observation for variance forecasts.

    Inputs are variances. Returns an array aligned with inputs.

    Definition (normalized form):
        x = v_true / v_pred
        qlike = x - log(x) - 1
    This is >= 0 with minimum 0 at perfect forecasts (v_pred == v_true).
    """
    vt = np.asarray(v_true, dtype=float)
    vp = np.asarray(v_pred, dtype=float)

    out = np.full_like(vt, np.nan, dtype=float)

    m = np.isfinite(vt) & np.isfinite(vp)
    if not np.any(m):
        return out

    vt_m = np.clip(vt[m], eps, np.inf)
    vp_m = np.clip(vp[m], eps, np.inf)
    x = vt_m / vp_m

    out[m] = x - np.log(x) - 1.0
    return out


def qlike_loss_var(v_true, v_pred, eps: float = 1e-12) -> float:
    """
    Mean QLIKE over finite observations (lower is better).
    Returns NaN if no finite observations exist after filtering.
    """
    s = qlike_series_var(v_true, v_pred, eps=eps)
    s = s[np.isfinite(s)]
    return float(np.mean(s)) if s.size else float("nan")


def qlike_loss_var(v_true, v_pred, eps: float = 1e-12) -> float:
    v_true = np.asarray(v_true, dtype=float)
    v_pred = np.asarray(v_pred, dtype=float)
    v_true = np.clip(v_true, eps, None)
    v_pred = np.clip(v_pred, eps, None)
    ratio = v_true / v_pred
    return float(np.mean(ratio - np.log(ratio) - 1.0))
