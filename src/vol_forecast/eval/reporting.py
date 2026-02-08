import pandas as pd
import numpy as np
from vol_forecast.metrics import qlike_loss_var


def compressed_dropna_diagnostic(df: pd.DataFrame, needed: list[str], warmup: int = 200) -> dict[str, float | int]:
    """
    Minimal diagnostic for the 'compressed df2 = dropna(subset=needed)' approach.

    Returns:
      - drop_frac_total: fraction of rows dropped by dropna(subset=needed)
      - drop_frac_after_warmup: fraction of rows dropped AFTER the first `warmup` rows
      - n, n_kept
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "n_kept": 0, "drop_frac_total": float("nan"), "drop_frac_after_warmup": float("nan")}

    valid = df[needed].notna().all(axis=1).to_numpy()
    n_kept = int(valid.sum())
    drop_frac_total = float(1.0 - (n_kept / n))

    warmup = int(min(max(warmup, 0), n))
    if warmup >= n:
        drop_frac_after = 0.0
    else:
        valid_after = valid[warmup:]
        drop_frac_after = float(1.0 - (valid_after.mean()))

    return {
        "n": n,
        "n_kept": n_kept,
        "drop_frac_total": drop_frac_total,
        "drop_frac_after_warmup": drop_frac_after,
    }


def availability_summary_holdout(
    wf_hold: pd.DataFrame,
    *,
    target_var_col: str,
    baseline_var_col: str,
    model_var_cols: list[str],
) -> pd.DataFrame:
    """
    Holdout availability summary per model column.

    Reports per-model sample sizes under:
      - intersection(target, model)
      - intersection(target, model, baseline)
    in addition to baseline-only sample size and the common intersection across all listed models.
    """
    rows = []
    n_base = int(wf_hold[[target_var_col, baseline_var_col]].dropna().shape[0]) if baseline_var_col in wf_hold.columns else 0
    cols_present = [c for c in model_var_cols if c in wf_hold.columns]

    # intersection over models (plus target+baseline)
    inter_cols = [target_var_col, baseline_var_col] + cols_present
    n_inter = int(wf_hold[inter_cols].dropna().shape[0]) if all(c in wf_hold.columns for c in inter_cols) else 0

    for c in cols_present:
        n_tm = int(wf_hold[[target_var_col, c]].dropna().shape[0])
        n_tmb = int(wf_hold[[target_var_col, c, baseline_var_col]].dropna().shape[0]) if baseline_var_col in wf_hold.columns else 0
        miss_pct = float(100.0 * wf_hold[c].isna().mean())
        rows.append({
            "model": c,
            "n(target+model)": n_tm,
            "n(target+model+baseline)": n_tmb,
            "missing_%_in_holdout": miss_pct,
            "baseline_n(target+baseline)": n_base,
            "intersection_n(all_models+baseline)": n_inter,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["missing_%_in_holdout", "n(target+model+baseline)"], ascending=[True, False]).reset_index(drop=True)
    return out


def split_holdout_into_halves(df_hold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Splits a holdout panel into two time-ordered halves and returns (half1, half2, midpoint_timestamp)."""
    if df_hold.empty:
        return df_hold.copy(), df_hold.copy(), pd.Timestamp("1970-01-01")
    idx = df_hold.index.sort_values()
    mid = idx[len(idx) // 2]
    a = df_hold.loc[df_hold.index <= mid].copy()
    b = df_hold.loc[df_hold.index > mid].copy()
    return a, b, mid


def pairwise_headline_table(
    df: pd.DataFrame,
    *,
    segment: str,
    target_var_col: str,
    baseline_var_col: str,
    model_var_cols: list[str],
    include_rmse_vol: bool = True,
    min_n: int = 60,
) -> pd.DataFrame:
    """
    Headline loss table on a given segment.

    For each model column, computes:
      - QLIKE on variance scale
      - delta_qlike_vs_baseline = qlike(model) - qlike(baseline)
      - (optional) RMSE on volatility scale = RMSE(sqrt(var_true), sqrt(var_pred))

    Each row is computed on the available intersection of required columns (dropna).
    Includes a baseline self-row with delta_qlike_vs_baseline = 0.
    """
    rows: list[dict[str, object]] = []  

    for m in model_var_cols:
        if m not in df.columns:
            continue

        if m == baseline_var_col:
            sub = df[[target_var_col, baseline_var_col]].dropna()
            n = int(len(sub))
            if n < min_n:
                rows.append({
                    "segment": segment,
                    "model": m,
                    "n": n,
                    "qlike": float("nan"),
                    "delta_qlike_vs_baseline": float("nan"),
                    "rmse_vol": float("nan"),
                })
                continue

            v = sub[target_var_col].values.astype(float)
            b = sub[baseline_var_col].values.astype(float)

            q_b = qlike_loss_var(v, b)
            rmse_vol = float("nan")
            if include_rmse_vol:
                vt = np.sqrt(np.clip(v, 0.0, None))
                vb = np.sqrt(np.clip(b, 0.0, None))
                rmse_vol = float(np.sqrt(np.mean((vt - vb) ** 2)))

            rows.append({
                "segment": segment,
                "model": m,
                "n": n,
                "qlike": float(q_b),
                "delta_qlike_vs_baseline": 0.0,
                "rmse_vol": float(rmse_vol),
            })
            continue
       

        # Normal case: model != baseline
        sub = df[[target_var_col, baseline_var_col, m]].dropna()
        n = int(len(sub))
        if n < min_n:
            rows.append({
                "segment": segment,
                "model": m,
                "n": n,
                "qlike": float("nan"),
                "delta_qlike_vs_baseline": float("nan"),
                "rmse_vol": float("nan"),
            })
            continue

        v = sub[target_var_col].values.astype(float)
        b = sub[baseline_var_col].values.astype(float)
        p = sub[m].values.astype(float)

        q_m = qlike_loss_var(v, p)
        q_b = qlike_loss_var(v, b)
        d = float(q_m - q_b)

        rmse_vol = float("nan")
        if include_rmse_vol:
            vt = np.sqrt(np.clip(v, 0.0, None))
            vp = np.sqrt(np.clip(p, 0.0, None))
            rmse_vol = float(np.sqrt(np.mean((vt - vp) ** 2)))


        rows.append({
            "segment": segment,
            "model": m,
            "n": n,
            "qlike": float(q_m),
            "delta_qlike_vs_baseline": float(d),
            "rmse_vol": float(rmse_vol),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["delta_qlike_vs_baseline", "qlike"], ascending=[True, True]).reset_index(drop=True)
    return out


def report_xgb_mean_median_sanity(
    df_hold: pd.DataFrame,
    *,
    target_var_col: str,
    baseline_var_col: str,
    pairs: list[tuple[str, str, str]],  # (median_col, mean_col, label)
    min_n: int = 100,
) -> pd.DataFrame:
    """
    Sanity check: compares XGB median vs mean-corrected forecasts on holdout.

    For each (median_col, mean_col) pair, reports QLIKE(mean), QLIKE(median),
    their difference, and the average volatility ratio sqrt(mean)/sqrt(median).
    """
    rows = []
    for med_col, mean_col, label in pairs:
        if med_col not in df_hold.columns or mean_col not in df_hold.columns:
            continue
        sub = df_hold[[target_var_col, baseline_var_col, med_col, mean_col]].dropna()
        n = int(len(sub))
        if n < min_n:
            rows.append({"label": label, "n": n, "qlike_mean": np.nan, "qlike_median": np.nan, "d_qlike(mean-median)": np.nan, "avg_vol_ratio(mean/median)": np.nan})
            continue

        v = sub[target_var_col].values.astype(float)
        q_mean = qlike_loss_var(v, sub[mean_col].values.astype(float))
        q_med = qlike_loss_var(v, sub[med_col].values.astype(float))
        d = float(q_mean - q_med)

        vol_ratio = np.sqrt(np.clip(sub[mean_col].values.astype(float), 0, None)) / np.sqrt(np.clip(sub[med_col].values.astype(float), 1e-18, None))
        vol_ratio = vol_ratio[np.isfinite(vol_ratio)]
        avg_ratio = float(np.mean(vol_ratio)) if len(vol_ratio) else float("nan")

        rows.append({
            "label": label,
            "n": n,
            "qlike_mean": float(q_mean),
            "qlike_median": float(q_med),
            "d_qlike(mean-median)": float(d),
            "avg_vol_ratio(mean/median)": float(avg_ratio),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["d_qlike(mean-median)"], ascending=[True]).reset_index(drop=True)
    return out


def spearman_rank_vol_holdout(
    df_hold: pd.DataFrame,
    *,
    target_var_col: str,
    model_var_cols: list[str],
    min_n: int = 200,
) -> pd.DataFrame:
    """
    Ranks models on HOLDOUT by Spearman rank correlation on the volatility scale.
    Computes Spearman corr(sqrt(target_var), sqrt(model_var)) per model column.
    """
    rows = []
    for c in model_var_cols:
        if c not in df_hold.columns:
            continue
        sub = df_hold[[target_var_col, c]].dropna()
        n = int(len(sub))
        if n < min_n:
            rows.append({"model": c, "n": n, "spearman_rank_vol": np.nan})
            continue
        r_vol = np.sqrt(np.clip(sub[target_var_col].astype(float).values, 0.0, None))
        f_vol = np.sqrt(np.clip(sub[c].astype(float).values, 0.0, None))
        spearman = float(pd.Series(f_vol).corr(pd.Series(r_vol), method="spearman"))
        rows.append({"model": c, "n": n, "spearman_rank_vol": spearman})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["spearman_rank_vol"], ascending=[False]).reset_index(drop=True)
    return out
