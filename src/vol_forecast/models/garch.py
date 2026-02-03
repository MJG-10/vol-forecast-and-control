import numpy as np
import pandas as pd
from vol_forecast.wf_config import WalkForwardConfig
from arch import arch_model
from .wf_util import get_train_slice, compute_start_pos
from typing import Literal


RefitCandidate = tuple[float, float, float, float, float, float]

def _build_refit_candidate(
    *,
    res,
    train: pd.Series,
    kind: Literal["garch", "gjr"],
    p_neg: float,
    tol_ab: float,
) -> RefitCandidate | None:
    """Validates a refit result and returns a commit-ready candidate, else None."""
    omega_new = float(res.params.get("omega", np.nan))
    alpha_new = float(res.params.get("alpha[1]", np.nan))
    beta_new  = float(res.params.get("beta[1]", np.nan))
    gamma_new = float(res.params.get("gamma[1]", 0.0)) if kind == "gjr" else 0.0

    if not (np.isfinite(omega_new) and np.isfinite(alpha_new) and np.isfinite(beta_new)):
        return None
    if kind == "gjr" and not np.isfinite(gamma_new):
        return None

    h_prev_new = float(res.conditional_volatility.iloc[-1] ** 2)
    eps_prev_new = float(train.iloc[-1])
    if not (np.isfinite(h_prev_new) and h_prev_new > 0.0 and np.isfinite(eps_prev_new)):
        return None

    ab_new = (alpha_new + beta_new) if kind == "garch" else (alpha_new + beta_new + gamma_new * p_neg)
    if not (np.isfinite(ab_new) and ab_new < 1.0 - tol_ab):
        return None

    return (omega_new, alpha_new, beta_new, gamma_new, h_prev_new, eps_prev_new)


def walk_forward_garch_family_var(
    df: pd.DataFrame,
    ret_col: str,
    kind: Literal["garch", "gjr"],
    horizon: int = 20,
    *,
    cfg: WalkForwardConfig | None = None,
    ret_scale: float = 100.0,
    dist: Literal["normal", "t"] = "t",
    start_date: pd.Timestamp | None = None,
) -> pd.Series:
    """
    Walk-forward variance forecasts from GARCH(1,1) / GJR-GARCH(1,1).

    Forecasts are aligned at origin t using returns observed through t-1 (scaled by ret_scale).
    This is a returns-only model, so `horizon` affects only the forecast aggregation: we average
    the next `horizon` steps of the variance recursion to produce an annualized horizon-mean forecast.

    Refitting / robustness:
      - Refits every cfg.refit_every origins on a rolling/expanding window.
      - On refit failure, retains the last valid fitted state and continues.
      - If the one-step recursion is invalid (non-finite/non-positive) or fails the persistence check
        (ab >= 1 - tol_ab), writes NaN and does not update state for that step.

    GJR multi-step recursion assumes symmetric innovations, using P(eps < 0) = 0.5 in ab.
    """
    if dist not in ("t", "normal"):
        raise ValueError(f"dist must be 'normal' or 't'.")
    if kind not in ("garch", "gjr"):
        raise ValueError("kind must be 'garch' or 'gjr'")

    cfg = cfg or WalkForwardConfig()

    out = pd.Series(index=df.index, dtype=float, name=f"{kind}_wf_forecast_var")

    df2 = df.dropna(subset=[ret_col]).copy()
    n2 = len(df2)

    start_pos = compute_start_pos(
        df2.index,
        cfg=cfg,
        n_rows=n2,
        origin_start_date=start_date,
    )

    # Scaled returns for stable estimation; forecasts scaled back via (ret_scale**2).
    returns = df2[ret_col] * float(ret_scale)

    tol_ab = 1e-6 

    fitted = False
    omega = alpha = beta = gamma = None
    h_prev = None
    eps_prev = None

    # assumes symmetric innovations for GJR horizon recursion (E[I{eps<0}])
    P_NEG_IMPLIED = 0.5

    for pos in range(start_pos, n2):
        idx = df2.index[pos]
        do_refit = (not fitted) or ((pos - start_pos) % cfg.refit_every == 0)

        if do_refit:
            train_slice = get_train_slice(pos, cfg.window_type, cfg.rolling_window_size)
            train = returns.iloc[train_slice]

            required_min = cfg.initial_train_size if cfg.window_type == "expanding" else cfg.min_train_size

            if len(train) >= required_min:
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
                    res = None
                else:
                    if getattr(res, "convergence_flag", 0) != 0:
                        res = None

                if res is not None:
                    cand = _build_refit_candidate(
                        res=res,
                        train=train,
                        kind=kind,
                        p_neg=P_NEG_IMPLIED,
                        tol_ab=tol_ab,
                    ) 
                    if cand is not None:
                        omega, alpha, beta, gamma, h_prev, eps_prev = cand
                        fitted = True
                                        
        if not fitted:
            continue

        Ineg = 1.0 if eps_prev < 0.0 else 0.0
        if kind == "garch":
            h_t = omega + alpha * (eps_prev ** 2) + beta * h_prev
            ab = alpha + beta
        else:
            h_t = omega + (alpha + gamma * Ineg) * (eps_prev ** 2) + beta * h_prev
            ab = alpha + beta + gamma * P_NEG_IMPLIED

        if (not np.isfinite(h_t)) or (h_t <= 0.0) or (not np.isfinite(ab)) or (ab >= 1.0 - tol_ab):
            out.loc[idx] = np.nan
            continue

        h_path = np.empty(horizon, dtype=float)
        h_path[0] = h_t
        for k in range(1, horizon):
            h_path[k] = omega + ab * h_path[k - 1]

        out.loc[idx] = float(252.0 * (h_path.mean() / (ret_scale ** 2)))

        if pos < n2 - 1:
            eps_prev = float(returns.iloc[pos])
            h_prev = float(h_t)
    return out