import numpy as np
import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
from arch import arch_model
from .wf_util import get_train_slice


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
) -> tuple[pd.Series, pd.Series, dict]:
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
        "initial_train_frac": float(cfg.initial_train_frac),
        "n_rows": int(n2),
        "horizon": int(horizon),
        "window_type": cfg.window_type,
        "rolling_w": int(cfg.rolling_w),
        "refit_every": int(cfg.refit_every),
        "ret_scale": float(ret_scale),
        # filled later:
        "initial_train_idx": None,
        "n_refits": 0,
        "forecast_points": 0,
        "n_failures": 0,
        "failure_rate": float("nan"),
    }

    # if n2 < 300:
    if n2 < cfg.min_rows_total:
        return out, failed, diag

    initial_train_idx = int(cfg.initial_train_frac * n2)
    diag["initial_train_idx"] = int(initial_train_idx)

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

    for pos in range(initial_train_idx, n2):
        do_refit = (not fitted) or ((pos - initial_train_idx) % cfg.refit_every == 0)

        if do_refit:
            train_slice = get_train_slice(pos, cfg.window_type, cfg.rolling_w)
            train_ret = returns.iloc[train_slice]
            # to review, maybe best to remove
            if len(train_ret) < cfg.min_train_rows:
                continue

            am = arch_model(
                train_ret,
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
            eps_prev = float(train_ret.iloc[-1])

            if kind == "gjr":
                if gjr_pneg_mode == "implied" and sym:
                    p_neg = 0.5
                else:
                    vals = train_ret.values.astype(float)
                    vals = vals[np.isfinite(vals)]
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

    denom = max(1, (n2 - initial_train_idx))
    diag.update({
        "n_refits": int(n_fit),
        "forecast_points": int(denom),
        "n_failures": int(n_fail),
        "failure_rate": float(n_fail / denom),
    })
    return out, failed, diag