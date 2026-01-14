import pandas as pd


def _index_stats(df: pd.DataFrame) -> dict[str, object]:
    n = int(len(df))
    out: dict[str, object] = {"n": n}
    if n == 0:
        return out

    idx = df.index
    out["start"] = idx.min()
    out["end"] = idx.max()

    if isinstance(idx, pd.DatetimeIndex):
        out["n_dups"] = int(idx.duplicated().sum())
        gaps = idx.to_series().diff().dt.days.dropna().astype(float)
        out["gap_days_mean"] = float(gaps.mean()) if len(gaps) else float("nan")
        out["gap_days_p95"] = float(gaps.quantile(0.95)) if len(gaps) else float("nan")
        out["gap_days_max"] = float(gaps.max()) if len(gaps) else float("nan")
    else:
        out["n_dups"] = 0

    return out


def _coverage_table(df: pd.DataFrame, *, cols: list[str], warmup: int) -> pd.DataFrame:
    n = int(len(df))
    warm = int(min(max(warmup, 0), n))
    present = [c for c in cols if c in df.columns]

    rows: list[dict[str, object]] = []
    for c in present:
        s = df[c]
        non = int(s.notna().sum())
        pct = (non / n) if n else float("nan")

        if warm >= n:
            pct_after = float("nan")
            n_after = 0
        else:
            tail = s.iloc[warm:]
            n_after = int(len(tail))
            non_after = int(tail.notna().sum())
            pct_after = (non_after / n_after) if n_after else float("nan")

        rows.append(
            {
                "col": c,
                "nonNaN": float(non),
                "pct_nonNaN": float(pct),
                "pct_nonNaN_after_warmup": float(pct_after),
                "n_after_warmup": float(n_after),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["pct_nonNaN_after_warmup", "pct_nonNaN"], ascending=[True, True]
        ).reset_index(drop=True)
    return out


def _aligned_series_row(
    s: pd.Series | None,
    *,
    trading_index: pd.DatetimeIndex,
    name: str,
) -> dict[str, object]:
    if s is None:
        return {"name": name, "available": False}
    x = s.reindex(trading_index)
    return {
        "name": name,
        "available": True,
        "n": int(len(x)),
        "nonNaN": int(x.notna().sum()),
        "pct_nonNaN": float(x.notna().mean()) if len(x) else float("nan"),
    }


def build_data_diagnostics(
    *,
    df: pd.DataFrame,
    wf_hold: pd.DataFrame,
    ret_col: str,
    target_var_col: str,
    baseline_var_col: str,
    model_var_cols: list[str],
    vix_close: pd.Series | None,
    cash_daily_simple: pd.Series | None,
    core_cols: list[str],
    warmup_core: int = 200,
) -> dict[str, object]:
    """
    Single entry point: compute data completeness and alignment diagnostics.

    Returns a dict with:
      - core_index: dict
      - holdout_index: dict
      - core_coverage: DataFrame
      - holdout_coverage: DataFrame
      - aux_coverage: DataFrame (rows for vix/cash on df and holdout indices)
    """
    core_index = _index_stats(df)
    hold_index = _index_stats(wf_hold)

    core_coverage = _coverage_table(df, cols=core_cols, warmup=warmup_core)

    hold_cols = [ret_col, target_var_col, baseline_var_col] + list(model_var_cols)
    holdout_coverage = _coverage_table(wf_hold, cols=hold_cols, warmup=0)

    aux_rows = [
        _aligned_series_row(vix_close, trading_index=df.index, name="vix_close@df"),
        _aligned_series_row(cash_daily_simple, trading_index=df.index, name="cash@df"),
        _aligned_series_row(vix_close, trading_index=wf_hold.index, name="vix_close@holdout"),
        _aligned_series_row(cash_daily_simple, trading_index=wf_hold.index, name="cash@holdout"),
    ]
    aux_coverage = pd.DataFrame(aux_rows)

    return {
        "core_index": core_index,
        "holdout_index": hold_index,
        "core_coverage": core_coverage,
        "holdout_coverage": holdout_coverage,
        "aux_coverage": aux_coverage,
    }
