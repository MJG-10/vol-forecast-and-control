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
    L = L_daily.copy()
    n = len(L)
    mask = np.zeros(n, dtype=bool)
    mask[offset::k] = True
    return L.where(mask).ffill()


def leverage_tranche_k_days(L_daily: pd.Series, k: int = 20) -> pd.Series:
    Ls = [leverage_hold_k_days(L_daily, k=k, offset=off) for off in range(k)]
    out = pd.concat(Ls, axis=1).mean(axis=1)
    out.name = f"leverage_tranche_{k}"
    return out


def simulate_vol_control_strategy(
    log_returns: pd.Series,
    sigma_hat: pd.Series,
    sigma_target: float,
    cash_daily_simple: pd.Series,
    L_max: float = 1.0,
    variant: str = "daily",
    k: int = 20,
    tcost_bps: float = 0.0,
    execution_lag_days: int = 1,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      strat_log_ret: strategy log returns
      leverage_exec: executed leverage series
      cash_log_ret: cash log return series aligned to strategy dates (for excess Sharpe)
    """
    # df = pd.concat({"ret": log_returns, "sigma_hat": sigma_hat}, axis=1)
   
    # df = df.join(cash_daily_simple.rename("cash_r"), how="left")
    # # df["cash_r"] = df["cash_r"].fillna(0.0)
    # # df = df.dropna(subset=["ret", "sigma_hat"])
    # df = df.dropna(subset=["ret", "sigma_hat", "cash_r"])

    df = pd.concat({"ret": log_returns, "sigma_hat": sigma_hat}, axis=1).dropna(subset=["ret", "sigma_hat"])
    df = df.join(cash_daily_simple.rename("cash_r"), how="left")
    df = df.dropna(subset=["cash_r"])

    L_daily = compute_leverage_from_vol_forecast(df["sigma_hat"], sigma_target=sigma_target, L_max=L_max)

    if variant == "daily":
        L = L_daily
    elif variant == "hold20":
        L = leverage_hold_k_days(L_daily, k=k, offset=0)
    elif variant == "tranche20":
        L = leverage_tranche_k_days(L_daily, k=k)
    else:
        raise ValueError("variant must be 'daily', 'hold20', or 'tranche20'")

    df["L"] = L
    df["L_exec"] = df["L"].shift(int(execution_lag_days))
    df = df.dropna(subset=["L_exec"])

    r_sp = np.expm1(df["ret"])

    r_cash = df["cash_r"].values.astype(float)

    cash_log_ret = np.log1p(np.clip(r_cash, -0.999999, None))
    strat_simple = df["L_exec"] * r_sp + (1.0 - df["L_exec"]) * r_cash

    if tcost_bps > 0:
        cost = (tcost_bps / 10000.0) * df["L_exec"].diff().abs().fillna(0.0)
        strat_simple = strat_simple - cost

    strat_simple = strat_simple.clip(lower=-0.999999)
    strat_log = np.log1p(strat_simple)
    return strat_log.rename("strat_log_ret"), df["L_exec"].rename("leverage_exec"), pd.Series(cash_log_ret, index=df.index, name="cash_log_ret")


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


def compute_excess_sharpe(strat_log: pd.Series, cash_log: pd.Series, *, freq: int = 252) -> float:
    df = pd.concat({"s": strat_log, "c": cash_log}, axis=1).dropna()
    if len(df) < 50:
        return float("nan")
    excess = df["s"] - df["c"]
    m = float(excess.mean())
    sd = float(excess.std(ddof=1))
    if sd <= 0:
        return float("nan")
    return float((freq * m) / (math.sqrt(freq) * sd))


def run_strategy_holdout_cost_grid(
    wf_hold: pd.DataFrame,
    *,
    return_col: str,
    signal_var_cols: list[str],
    cash_daily_simple: pd.Series,
    sigma_target: float,
    L_max: float,
    horizon: int,
    tcost_grid_bps: list[float],
    execution_lag_days: int,
    variants: list[str]|None = None,
    freq: int = 252,
) -> pd.DataFrame:
    if variants is None:
        variants = ["hold20", "tranche20"]

    rows: list[dict[str, float]] = []

    # buy & hold baseline for each cost (cost doesn't apply to BH here; keep one row per cost for consistent table)
    for cost in tcost_grid_bps:
        sub_bh = wf_hold[[return_col]].dropna()
        bh = sub_bh[return_col]
        bh_stats = compute_strategy_stats(bh, freq=freq)

        cash_use = cash_daily_simple.reindex(sub_bh.index)
        cash_lr = pd.Series(np.log1p(np.clip(cash_use.values.astype(float), -0.999999, None)), index=sub_bh.index)
        
        bh_stats["sharpe_excess"] = compute_excess_sharpe(bh, cash_lr, freq=freq)
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
            cash_sig = cash_daily_simple.reindex(sub.index)

            for variant in variants:
                strat_rets, L_exec, cash_log = simulate_vol_control_strategy(
                    sub[return_col],
                    sigma_hat,
                    sigma_target=sigma_target,
                    L_max=L_max,
                    variant=variant,
                    k=horizon,
                    tcost_bps=float(cost),
                    cash_daily_simple=cash_sig,
                    execution_lag_days=execution_lag_days,
                )
                stats = compute_strategy_stats(strat_rets, freq=freq)
                stats["sharpe_excess"] = compute_excess_sharpe(strat_rets, cash_log, freq=freq)

                rows.append({
                    "tcost_bps": float(cost),
                    "strategy": f"{sig}__{variant}",
                    **stats,
                    "avg_leverage": float(L_exec.mean()) if len(L_exec) else float("nan"),
                })

    out = pd.DataFrame(rows)
    # keep only key columns
    keep = ["tcost_bps", "strategy", "n", "ann_log_ret", "ann_simple_ret", "ann_vol", "sharpe", "sharpe_excess", "max_drawdown", "avg_leverage"]
    out = out[keep].sort_values(["tcost_bps", "sharpe_excess"], ascending=[True, False]).reset_index(drop=True)
    return out
