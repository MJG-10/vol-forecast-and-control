import pandas as pd 
import numpy as np
import math


def compute_risky_weight_from_vol_forecast(
    sigma_hat: pd.Series,
    sigma_target: float,
    w_max: float = 1.0,
    eps: float = 1e-8,
) -> pd.Series:
    """Converts a volatility forecast into a bounded target risky weight series."""
    sigma = sigma_hat.clip(lower=eps)
    w = (sigma_target / sigma).clip(lower=0.0, upper=w_max)
    w.name = "risky_weight"
    return w


def _drift_weight_one_step(w: float, r_sp: float, r_cash: float, w_max: float = 1.0) -> float:
    """Updates the risky weight after one period if no rebalance is performed."""
    w = float(np.clip(w, 0.0, w_max))
    a = w * (1.0 + float(r_sp))
    b = (1.0 - w) * (1.0 + float(r_cash))
    denom = a + b
    if denom <= 0:
        return w

    w_next = float(a / denom)
    return float(np.clip(w_next, 0.0, w_max))


def simulate_vol_control_strategy(
    log_returns: pd.Series,
    sigma_hat: pd.Series,
    sigma_target: float,
    cash_daily_simple: pd.Series,
    variant: str = "daily_reset",
    tcost_bps: float = 0.0,
    execution_lag_days: int = 0,
    *,
    w_max: float = 1.0,
    tiny_eps: float = 1e-12,
) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    """
    Volatility-targeting / risk-control backtest driven by volatility forecasts.

    Builds an executed target risky weight from `sigma_hat` (capped at w_max, with w_max <= 1.0), allocates
    the remainder to `cash_daily_simple`, and simulates strategy log returns with an
    explicit transaction cost model.

    Conventions
    - Input risky returns are log returns; portfolio arithmetic is done in simple returns
      (to combine risky + cash), then converted back to log returns.
    - Risky weight is capped at `w_max` (w_max <= 1.0 enforced, no leverage).
    - `execution_lag_days` delays the target schedule beyond upstream alignment.

    Variants
    - "daily_reset": rebalance at the start of each day to the executed target. Costs are
      charged on drift-aware traded notional (move from drifted pre-trade weight to target).
    - "band_no_trade": maintain a drifting risky weight and trade only when deviation from
      target exceeds a fixed band; costs are charged only on those trades.

    Transaction costs
    - tcost_bps is applied to daily turnover abs(tgt - w_pre) as a fraction of NAV (trade_frac).

    Returns
    - (strat_log_ret, risky_weight_used, trade_frac, pct_capped)
    """
    if w_max > 1.0:
        raise ValueError("w_max must be <= 1.0 (no leverage).")

    df = pd.concat({"ret": log_returns, "sigma_hat": sigma_hat}, axis=1).dropna(subset=["ret", "sigma_hat"])
    df = df.join(cash_daily_simple.rename("cash_r"), how="left").dropna(subset=["cash_r"])

    if variant not in ("daily_reset", "band_no_trade"):
        raise ValueError("variant must be 'daily_reset' or 'band_no_trade'")

    w_target = compute_risky_weight_from_vol_forecast(df["sigma_hat"], sigma_target=sigma_target, w_max=w_max)
    df["w_exec_target"] = w_target.shift(int(execution_lag_days))
    df = df.dropna(subset=["w_exec_target"])

    r_sp = np.expm1(df["ret"]).astype(float).to_numpy()
    r_cash = df["cash_r"].astype(float).to_numpy()
    w_exec = df["w_exec_target"].astype(float).to_numpy()

    n = len(df)
    w_used = np.empty(n, dtype=float)
    trade = np.zeros(n, dtype=float)
    strat_simple = np.empty(n, dtype=float)

    BAND_ABS = 0.05  # absolute risky-weight band for "band_no_trade"

    # Convention: start at the first executable target (no initial entry cost).
    w_act = float(np.clip(w_exec[0], 0.0, w_max))
    w_used[0] = w_act
    strat_simple[0] = w_act * r_sp[0] + (1.0 - w_act) * r_cash[0]

    for t in range(1, n):
        w_act = _drift_weight_one_step(w_act, r_sp[t - 1], r_cash[t - 1], w_max=w_max)
        tgt = float(np.clip(w_exec[t], 0.0, w_max))

        if variant == "daily_reset" or abs(tgt - w_act) > BAND_ABS:
            trade[t] = abs(tgt - w_act)
            w_act = tgt

        w_used[t] = w_act
        strat_simple[t] = w_act * r_sp[t] + (1.0 - w_act) * r_cash[t]

    if tcost_bps > 0:
        strat_simple = strat_simple - (float(tcost_bps) / 10000.0) * trade

    strat_simple = np.clip(strat_simple, -0.999999, np.inf)
    strat_log = np.log1p(strat_simple)

    idx = df.index

    cap_level = w_max - float(tiny_eps)
    pct_capped = float(np.mean(w_used >= cap_level)) if n else float("nan")
    return (
        pd.Series(strat_log, index=idx, name="strat_log_ret"),
        pd.Series(w_used, index=idx, name="risky_weight_used"),
        pd.Series(trade, index=idx, name="trade_frac"),
        pct_capped
    )


def compute_strategy_stats(log_rets: pd.Series, freq: int = 252) -> dict[str, float]:
    """Computes annualized performance and drawdown stats from log returns."""
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
    tcost_grid_bps: list[float],
    execution_lag_days: int,
    variants: list[str]|None = None,
    freq: int = 252,
) -> pd.DataFrame:
    """Backtests vol-control variants over signals and cost levels on the holdout window."""
    if variants is None:
        variants = ["daily_reset", "band_no_trade"]

    rows: list[dict[str, float]] = []

    # Keep one BH row per cost for table comparability (no cost applied).
    for cost in tcost_grid_bps:
        sub_bh = wf_hold[[return_col, cash_col]].dropna()

        bh = sub_bh[return_col]
        bh_stats = compute_strategy_stats(bh, freq=freq)
      
        bh_stats["vol_ratio"] = float(bh_stats["ann_vol"] / sigma_target) if sigma_target > 0 else float("nan")
        bh_stats["avg_trade"] = 0.0
       
        rows.append({
            "tcost_bps": float(cost),
            "strategy": "buy_and_hold",
            **bh_stats,
            "avg_risky_weight": 1.0,
            "pct_capped": 1.0  # fully invested every day
        })

        for sig in signal_var_cols:
            if sig not in wf_hold.columns:
                continue

            sub = wf_hold[[return_col, sig]].dropna()
            if len(sub) < 200:
                continue

            sigma_hat = np.sqrt(sub[sig].clip(lower=0.0))
            cash_sig = wf_hold.loc[sub.index, cash_col]

            for variant in variants:
                strat_rets, w_used, trade_frac, pct_capped = simulate_vol_control_strategy(
                    sub[return_col],
                    sigma_hat,
                    sigma_target=sigma_target,
                    variant=variant,
                    tcost_bps=float(cost),
                    cash_daily_simple=cash_sig,
                    execution_lag_days=execution_lag_days,
                )
                stats = compute_strategy_stats(strat_rets, freq=freq)

                stats["vol_ratio"] = float(stats["ann_vol"] / sigma_target) if sigma_target > 0 else float("nan")
                stats["avg_trade"] = float(trade_frac.mean()) if len(trade_frac) else float("nan")

                rows.append({
                    "tcost_bps": float(cost),
                    "strategy": f"{sig}__{variant}",
                    **stats,
                    "avg_risky_weight": float(w_used.mean()) if len(w_used) else float("nan"),
                    "pct_capped": float(pct_capped)
                })

    out = pd.DataFrame(rows)
    # keep only key columns
    keep = ["tcost_bps", "strategy", "n", "ann_log_ret", "ann_simple_ret", "ann_vol", "vol_ratio", "sharpe", "max_drawdown", 
            "avg_risky_weight", "avg_trade", "pct_capped"]
    out = out[keep].sort_values(["tcost_bps", "sharpe"], ascending=[True, False]).reset_index(drop=True)
    return out
