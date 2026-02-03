from collections.abc import Sequence
import pandas as pd


def _index_stats(df: pd.DataFrame) -> dict[str, object]:
    """Basic index diagnostics: sample size, start/end, duplicates, and (DatetimeIndex) gap stats."""
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


def _coverage_table(
    df: pd.DataFrame,
    *,
    cols: Sequence[str],
    head_warmup: int = 0,
    tail_cooldown: int = 0,
) -> pd.DataFrame:
    """
    Coverage table for selected columns.

    Computes coverage on:
      - full sample
      - interior slice that skips `head_warmup` rows at the start and `tail_cooldown` rows at the end

    This avoids structural NaNs from rolling windows/feature lags (head) and forward targets (tail).
    """
    n = int(len(df))
    start = int(min(max(head_warmup, 0), n))
    end = int(min(max(n - int(tail_cooldown), start), n))

    present = [c for c in cols if c in df.columns]
    rows: list[dict[str, object]] = []

    for c in present:
        s = df[c]

        # Full sample
        non_full = int(s.notna().sum())
        pct_full = (non_full / n) if n else float("nan")

        # Interior slice
        s_in = s.iloc[start:end]
        n_in = int(len(s_in))
        non_in = int(s_in.notna().sum())
        pct_in = (non_in / n_in) if n_in else float("nan")

        rows.append(
            {
                "col": c,
                "nonNaN_full": float(non_full),
                "pct_nonNaN_full": float(pct_full),
                "pct_nonNaN_interior": float(pct_in),
                "n_interior": float(n_in),
                "head_warmup": float(start),
                "tail_cooldown": float(n - end),
            }
        )

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(
            ["pct_nonNaN_interior", "pct_nonNaN_full"], ascending=[True, True]
        ).reset_index(drop=True)
    return out


def build_core_data_diagnostics(
    *,
    df: pd.DataFrame,
    core_cols: Sequence[str],
    head_warmup: int = 22,
    tail_cooldown: int = 0,
) -> dict[str, object]:
    """Index + NaN coverage diagnostics for the canonical core dataframe."""
    return {
        "core_index": _index_stats(df),
        "core_coverage": _coverage_table(
            df,
            cols=core_cols,
            head_warmup=head_warmup,
            tail_cooldown=tail_cooldown,
        )
    }


def build_holdout_data_diagnostics(
    *,
    wf_hold: pd.DataFrame,
    ret_col: str,
    target_var_col: str,
    baseline_var_col: str,
    model_var_cols: list[str],
    cash_col: str | None = None,
) -> dict[str, object]:
    """Index + NaN coverage diagnostics for the holdout walk-forward output table."""
    hold_cols = [ret_col, target_var_col, baseline_var_col] + model_var_cols

    out: dict[str, object] = {
        "holdout_index": _index_stats(wf_hold),
        "holdout_coverage": _coverage_table(
            wf_hold,
            cols=hold_cols,
            head_warmup=0,
            tail_cooldown=0,
        ),
    }

    if cash_col is not None:
        out["cash_coverage_holdout"] = _coverage_table(
            wf_hold,
            cols=[cash_col],
            head_warmup=0,
            tail_cooldown=0,
        )

    return out

