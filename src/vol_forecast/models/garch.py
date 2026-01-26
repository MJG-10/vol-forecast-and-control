import numpy as np
import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
from arch import arch_model
from .wf_util import get_train_slice, compute_start_pos


def _is_symmetric_dist(dist: str) -> bool:
    d = dist.lower().strip()
    return d in ("t", "studentst", "student's t", "normal", "gaussian")


def walk_forward_garch_family_var(
    df: pd.DataFrame,
    ret_col: str,
    trailing_var_col: str,
    kind: str,  # "garch" or "gjr"
    horizon: int = 20,
    *,
    cfg: WalkForwardConfig|None = None,
    ret_scale: float = 100.0,
    dist: str = "t",
    gjr_pneg_mode: str = "implied",
    start_date: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series, dict]:
    """
    Walk-forward GARCH-family variance forecasts (GARCH(1,1) or GJR-GARCH(1,1)).

    - Fits on returns scaled by `ret_scale` (e.g., 100 for percent returns) for numerical stability.
    - Produces an annualized variance forecast over `horizon` trading days by propagating the
      conditional variance recursion and averaging the forward variance path.
    - On fit failure / non-convergence, falls back to lagged trailing variance (`trailing_var_col`.shift(1))
      and marks the timestamp as failed.

    GJR asymmetry expectation:
      - `gjr_pneg_mode="implied"` uses p_neg=0.5 for symmetric innovation distributions.
      - `gjr_pneg_mode="empirical"` estimates p_neg from the training window (clipped) to reflect sign imbalance.
    """
    kind = kind.lower()
    if kind not in ("garch", "gjr"):
        raise ValueError("kind must be 'garch' or 'gjr'")
    if gjr_pneg_mode not in ("implied", "empirical"):
        raise ValueError("gjr_pneg_mode must be 'implied' or 'empirical'")
    cfg = cfg or WalkForwardConfig()

    out = pd.Series(index=df.index, dtype=float, name=f"{kind}_wf_forecast_var")
    # failed = pd.Series(index=df.index, dtype=bool, name=f"{kind}_failed")
    failed = pd.Series(index=df.index, dtype="boolean", name=f"{kind}_failed")

    df2 = df.dropna(subset=[ret_col, trailing_var_col]).copy()
    n2 = len(df2)

    diag: dict = {
        "kind": kind,
        "dist": dist,
        "gjr_pneg_mode": gjr_pneg_mode,
        "n_rows": int(n2),
        "horizon": int(horizon),
        "ret_scale": float(ret_scale),
        # updated later
        "n_refits": 0,
        "forecast_points": 0,
        "n_failures": 0,
        "failure_rate": float("nan"),
    }

    # if n2 < cfg.min_total_size:
    #     return out, failed, diag

    # start_pos = compute_start_pos(n2, cfg)
    start_pos = compute_start_pos(
        df2.index,
        cfg=cfg,
        n_rows=n2,
        origin_start_date=start_date,
    )
    diag["start_pos"] = int(start_pos)
    # start_pos = max(start_pos, initial_train_idx)

    # We scale returns for stable estimation; forecasts are scaled back to variance units via (ret_scale**2).
    returns = df2[ret_col] * float(ret_scale)
    fallback_var = df2[trailing_var_col].shift(1)

    fitted = False
    omega = alpha = beta = gamma = None
    h_prev = None
    eps_prev = None
    p_neg = 0.5

    n_fail = 0
    n_fit = 0
    sym = _is_symmetric_dist(dist)

    for pos in range(start_pos, n2):
        do_refit = (not fitted) or ((pos - start_pos) % cfg.refit_every == 0)

        if do_refit:
            train_slice = get_train_slice(pos, cfg.window_type, cfg.rolling_window_size)
            train = returns.iloc[train_slice]
            # to review, maybe best to remove
            # if len(train_ret) < cfg.min_train_rows:
            #     continue

            if cfg.window_type == "expanding":
                required_min = cfg.initial_train_size 
            else:
                required_min = cfg.min_train_size

            if len(train) < required_min:
                continue

            am = arch_model(
                train,
                mean="Zero",
                vol="GARCH",
                p=1,
                o=(1 if kind == "gjr" else 0),
                q=1,
                dist=dist,
                rescale=False,
            )
            try:
                res = am.fit(disp="off")
            except Exception:
                idx = df2.index[pos]
                out.loc[idx] = float(fallback_var.loc[idx]) if np.isfinite(fallback_var.loc[idx]) else float("nan")
                failed.loc[idx] = True
                n_fail += 1
                fitted = False
                continue

            if getattr(res, "convergence_flag", 0) != 0:
                idx = df2.index[pos]
                out.loc[idx] = float(fallback_var.loc[idx]) if np.isfinite(fallback_var.loc[idx]) else float("nan")
                failed.loc[idx] = True
                n_fail += 1
                fitted = False
                continue

            omega = float(res.params.get("omega", np.nan))
            alpha = float(res.params.get("alpha[1]", np.nan))
            beta = float(res.params.get("beta[1]", np.nan))
            gamma = float(res.params.get("gamma[1]", 0.0)) if kind == "gjr" else 0.0

            ok = np.isfinite(omega) and np.isfinite(alpha) and np.isfinite(beta) and (np.isfinite(gamma) if kind == "gjr" else True)
            if not ok:
                idx = df2.index[pos]
                out.loc[idx] = float(fallback_var.loc[idx]) if np.isfinite(fallback_var.loc[idx]) else float("nan")
                failed.loc[idx] = True
                n_fail += 1
                fitted = False
                continue

            h_prev = float(res.conditional_volatility.iloc[-1] ** 2)
            eps_prev = float(train.iloc[-1])

            if kind == "gjr":
                if gjr_pneg_mode == "implied" and sym:
                    p_neg = 0.5
                else:
                    vals = train.values.astype(float)
                    vals = vals[np.isfinite(vals)]
                    # We clip empirical p_neg to avoid extreme gamma * p_neg effects in short/noisy samples.
                    p_neg = float(np.mean(vals < 0.0)) if len(vals) else 0.5
                    p_neg = float(np.clip(p_neg, 0.05, 0.95))

            fitted = True
            n_fit += 1

        idx = df2.index[pos]
        if (not fitted) or (omega is None) or (alpha is None) or (beta is None) or (h_prev is None) or (eps_prev is None):
            out.loc[idx] = float(fallback_var.loc[idx]) if np.isfinite(fallback_var.loc[idx]) else float("nan")
            failed.loc[idx] = True
            n_fail += 1
            continue

        Ineg = 1.0 if eps_prev < 0.0 else 0.0
        if kind == "garch":
            h_t = omega + alpha * (eps_prev ** 2) + beta * h_prev
            ab = alpha + beta
        else:
            h_t = omega + (alpha + gamma * Ineg) * (eps_prev ** 2) + beta * h_prev
            ab = alpha + beta + gamma * p_neg

        h_path = np.empty(horizon, dtype=float)
        h_path[0] = h_t
        for k in range(1, horizon):
            h_path[k] = omega + ab * h_path[k - 1]

        ann_var_forecast = float(252.0 * (h_path.mean() / (ret_scale ** 2)))
        out.loc[idx] = ann_var_forecast
        failed.loc[idx] = False

        if pos < n2 - 1:
            eps_prev = float(returns.iloc[pos])
            h_prev = float(h_t)

    denom = max(1, (n2 - start_pos))
    diag.update({
        "n_refits": int(n_fit),
        "forecast_points": int(denom),
        "n_failures": int(n_fail),
        "failure_rate": float(n_fail / denom),
    })
    return out, failed, diag