import pandas as pd 
import numpy as np
import math


def compute_leverage_from_vol_forecast(
    sigma_hat: pd.Series,
    sigma_target: float,
    L_max: float = 1.0,
    eps: float = 1e-8,
) -> pd.Series:
    sigma = sigma_hat.clip(lower=eps)
    L = (sigma_target / sigma).clip(lower=0.0, upper=L_max)
    L.name = "leverage"
    return L


def leverage_hold_k_days(L_daily: pd.Series, k: int = 20, offset: int = 0) -> pd.Series:
    """Hold the daily target leverage constant except on every k-th day (with phase offset)."""
    L = L_daily.copy()
    n = len(L)
    mask = np.zeros(n, dtype=bool)
    mask[offset::k] = True
    return L.where(mask).ffill()


def leverage_tranche_k_days(L_daily: pd.Series, k: int = 20) -> pd.Series:
    """Stagger k 'hold-k-days' schedules (offset 0..k-1) and average them to smooth turnover."""
    Ls = [leverage_hold_k_days(L_daily, k=k, offset=off) for off in range(k)]
    out = pd.concat(Ls, axis=1).mean(axis=1)
    out.name = f"leverage_tranche_{k}"
    return out


def leverage_daily_turnover_buffer(
    L_daily: pd.Series,
    *,
    buffer_pct: float = 0.05,
) -> pd.Series:
    """
    Deadband rule: only update leverage when the relative change from the last
    implemented leverage exceeds buffer_pct.

    If buffer_pct=0.05, we ignore moves <= 5% and keep prior leverage.
    """
    if buffer_pct <= 0:
        out = L_daily.copy()
        out.name = "leverage_daily"
        return out

    L_star = L_daily.astype(float)
    out = L_star.copy()

    # Iterate once; O(n) and extremely cheap relative to the rest of the pipeline.
    prev = float(out.iloc[0])
    for i in range(1, len(out)):
        cur = float(L_star.iloc[i])

        # Relative change; handle prev==0 safely
        if prev > 0:
            rel = abs(cur / prev - 1.0)
        else:
            rel = abs(cur - prev)

        if rel <= buffer_pct:
            out.iloc[i] = prev
        else:
            prev = cur
            out.iloc[i] = cur

    out.name = "leverage_daily_buffer"
    return out


def simulate_vol_control_strategy(
    log_returns: pd.Series,
    sigma_hat: pd.Series,
    sigma_target: float,
    cash_daily_simple: pd.Series,
    variant: str = "daily",
    k: int = 20,
    tcost_bps: float = 0.0,
    execution_lag_days: int = 1,
) -> tuple[pd.Series, pd.Series]:
    """
    Returns:
      strat_log_ret: strategy log returns
      leverage_exec: executed leverage series
    """

    df = pd.concat({"ret": log_returns, "sigma_hat": sigma_hat}, axis=1).dropna(subset=["ret", "sigma_hat"])
    df = df.join(cash_daily_simple.rename("cash_r"), how="left")
    df = df.dropna(subset=["cash_r"])

    L_max = 1.0
    L_daily = compute_leverage_from_vol_forecast(df["sigma_hat"], sigma_target=sigma_target, L_max=L_max)

    DAILY_TURNOVER_BUFFER = 0.05

    if variant == "daily":
        L = leverage_daily_turnover_buffer(L_daily, buffer_pct=DAILY_TURNOVER_BUFFER)
    elif variant == "tranche20":
        L = leverage_tranche_k_days(L_daily, k=k)
    else:
        raise ValueError("variant must be 'daily' or 'tranche20'")

    df["L"] = L
    df["L_exec"] = df["L"].shift(int(execution_lag_days))
    df = df.dropna(subset=["L_exec"])

    r_sp = np.expm1(df["ret"])

    # r_cash = df["cash_r"].values.astype(float)
    r_cash = df["cash_r"].to_numpy(dtype=float)
    
    strat_simple = df["L_exec"] * r_sp + (1.0 - df["L_exec"]) * r_cash

    if tcost_bps > 0:
        cost = (tcost_bps / 10000.0) * df["L_exec"].diff().abs().fillna(0.0)
        strat_simple = strat_simple - cost

    strat_simple = strat_simple.clip(lower=-0.999999)
    strat_log = np.log1p(strat_simple)
    return strat_log.rename("strat_log_ret"), df["L_exec"].rename("leverage_exec")


def compute_strategy_stats(log_rets: pd.Series, freq: int = 252) -> dict[str, float]:
    log_rets = log_rets.dropna()
    n = int(len(log_rets))
    if n == 0:
        return {
            "n": 0.0,
            "ann_log_ret": float("nan"),
            "ann_simple_ret": float("nan"),
            "ann_vol": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }

    mean = float(log_rets.mean())
    std = float(log_rets.std(ddof=1))

    ann_log_ret = freq * mean
    ann_simple_ret = float(np.expm1(ann_log_ret))
    ann_vol = math.sqrt(freq) * std
    sharpe = ann_log_ret / ann_vol if ann_vol > 0 else float("nan")

    equity = np.exp(log_rets.cumsum())
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())

    return {
        "n": float(n),
        "ann_log_ret": float(ann_log_ret),
        "ann_simple_ret": float(ann_simple_ret),
        "ann_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
    }


def run_strategy_holdout_cost_grid(
    wf_hold: pd.DataFrame,
    *,
    return_col: str,
    signal_var_cols: list[str],
    cash_col: str,
    sigma_target: float,
    horizon: int,
    tcost_grid_bps: list[float],
    execution_lag_days: int,
    variants: list[str]|None = None,
    freq: int = 252,
) -> pd.DataFrame:

    # require_cols(wf_hold.columns, [cash_col], context="run_strategy_holdout_cost_grid")

    if variants is None:
        variants = ["daily", "tranche20"]

    rows: list[dict[str, float]] = []

    # buy & hold baseline for each cost (cost doesn't apply to BH here; keep one row per cost for consistent table)
    for cost in tcost_grid_bps:
        # sub_bh = wf_hold[[return_col]].dropna()
        sub_bh = wf_hold[[return_col, cash_col]].dropna()

        bh = sub_bh[return_col]
        bh_stats = compute_strategy_stats(bh, freq=freq)
      
        bh_stats["vol_ratio"] = float(bh_stats["ann_vol"] / sigma_target) if sigma_target > 0 else float("nan")
        bh_stats["avg_abs_dL"] = 0.0  # BH leverage is constant at 1, so no churn
        # cash_use = wf_hold.loc[sub_bh.index, cash_col]
        rows.append({
            "tcost_bps": float(cost),
            "strategy": "buy_and_hold",
            **bh_stats,
            "avg_leverage": 1.0,
        })

        for sig in signal_var_cols:
            if sig not in wf_hold.columns:
                continue

            # per-signal alignment (return + this signal only)
            sub = wf_hold[[return_col, sig]].dropna()
            if len(sub) < 200:
                continue

            sigma_hat = np.sqrt(sub[sig].clip(lower=0.0))
            cash_sig = wf_hold.loc[sub.index, cash_col]

            for variant in variants:
                strat_rets, L_exec = simulate_vol_control_strategy(
                    sub[return_col],
                    sigma_hat,
                    sigma_target=sigma_target,
                    variant=variant,
                    k=horizon,
                    tcost_bps=float(cost),
                    cash_daily_simple=cash_sig,
                    execution_lag_days=execution_lag_days,
                )
                stats = compute_strategy_stats(strat_rets, freq=freq)

                stats["vol_ratio"] = float(stats["ann_vol"] / sigma_target) if sigma_target > 0 else float("nan")
                dL = L_exec.diff().abs().dropna()
                stats["avg_abs_dL"] = float(dL.mean()) if len(dL) else float("nan")

                rows.append({
                    "tcost_bps": float(cost),
                    "strategy": f"{sig}__{variant}",
                    **stats,
                    "avg_leverage": float(L_exec.mean()) if len(L_exec) else float("nan"),
                })

    out = pd.DataFrame(rows)
    # keep only key columns
    keep = ["tcost_bps", "strategy", "n", "ann_log_ret", "ann_simple_ret", "ann_vol", "vol_ratio", "sharpe", "max_drawdown", 
            "avg_leverage", "avg_abs_dL"]
    out = out[keep].sort_values(["tcost_bps", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    return out
