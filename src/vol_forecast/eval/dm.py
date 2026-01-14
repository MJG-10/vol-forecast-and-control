import math
import numpy as np
import pandas as pd
from vol_forecast.metrics import qlike_series_var


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def newey_west_long_run_var(x: np.ndarray, lag: int) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    x = x - x.mean()
    gamma0 = np.dot(x, x) / n
    lr_var = gamma0
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - (L / (lag + 1.0))
        gammaL = np.dot(x[L:], x[:-L]) / n
        lr_var += 2.0 * w * gammaL
    return float(lr_var)


def dm_test_qlike_var_vs_baseline_holdout(
    df_hold: pd.DataFrame,
    *,
    target_var_col: str,
    model_var_col: str,
    baseline_var_col: str,
    hac_lag: int,
    min_n: int = 60,
) -> dict[str, float]:
    """
    Minimal DM test on HOLDOUT only (OVERLAPPING daily sample):
      - loss: QLIKE on variance, per time step
      - compare: model_var_col vs baseline_var_col
      - HAC: Newey-West long-run variance on the loss differential

    Returns dict with:
      n, dm_stat, p_value, mean_d
    Convention:
      d_t = loss_model_t - loss_baseline_t
      Negative mean_d => model has LOWER loss (better) than baseline.
    """
    needed = [target_var_col, model_var_col, baseline_var_col]
    if any(c not in df_hold.columns for c in needed):
        return {"n": 0.0, "dm_stat": float("nan"), "p_value": float("nan"), "mean_d": float("nan")}

    sub = df_hold[needed].dropna()
    n = int(len(sub))
    if n < int(min_n):
        return {"n": float(n), "dm_stat": float("nan"), "p_value": float("nan"), "mean_d": float("nan")}

    v = sub[target_var_col].values.astype(float)
    m = sub[model_var_col].values.astype(float)
    b = sub[baseline_var_col].values.astype(float)

    loss_m = qlike_series_var(v, m)
    loss_b = qlike_series_var(v, b)

    d = (loss_m - loss_b).astype(float)
    d = d[np.isfinite(d)]
    n2 = int(len(d))
    if n2 < int(min_n):
        return {"n": float(n2), "dm_stat": float("nan"), "p_value": float("nan"), "mean_d": float(np.mean(d))}

    d_mean = float(np.mean(d))
    lr_var = newey_west_long_run_var(d, lag=int(hac_lag))
    if (not np.isfinite(lr_var)) or (lr_var <= 0.0):
        return {"n": float(n2), "dm_stat": float("nan"), "p_value": float("nan"), "mean_d": float(d_mean)}

    dm_stat = d_mean / math.sqrt(lr_var / float(n2))
    p = 2.0 * (1.0 - _normal_cdf(abs(dm_stat)))

    return {"n": float(n2), "dm_stat": float(dm_stat), "p_value": float(p), "mean_d": float(d_mean)}


def dm_panel_qlike_vs_baseline_holdout(
    df_hold: pd.DataFrame,
    *,
    target_var_col: str,
    model_var_cols: list[str],
    baseline_var_col: str,
    hac_lag_grid: list[int],
    min_n: int = 60,
) -> pd.DataFrame:
    cols = [c for c in model_var_cols if c in df_hold.columns and c != baseline_var_col]
    if baseline_var_col not in df_hold.columns or target_var_col not in df_hold.columns or len(cols) == 0:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    for lag in hac_lag_grid:
        for col in cols:
            res = dm_test_qlike_var_vs_baseline_holdout(
                df_hold,
                target_var_col=target_var_col,
                model_var_col=col,
                baseline_var_col=baseline_var_col,
                hac_lag=int(lag),
                min_n=min_n,
            )
            rows.append({
                "model": col,
                "hac_lag": float(lag),
                "n": float(res.get("n", np.nan)),
                "mean_d": float(res.get("mean_d", np.nan)),
                "dm_stat": float(res.get("dm_stat", np.nan)),
                "p_value": float(res.get("p_value", np.nan)),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["better_than_baseline"] = out["mean_d"] < 0.0
    out = out.sort_values(["hac_lag", "p_value", "mean_d"], ascending=[True, True, True]).reset_index(drop=True)
    return out

