import pandas as pd
import numpy as np
from vol_forecast.metrics import qlike_loss_var


def compressed_dropna_diagnostic(df: pd.DataFrame, needed: list[str], warmup: int = 200) -> dict[str, float]:
    """
    Minimal diagnostic for the 'compressed df2 = dropna(subset=needed)' approach.

    Returns:
      - drop_frac_total: fraction of rows dropped by dropna(subset=needed)
      - drop_frac_after_warmup: fraction of rows dropped AFTER the first `warmup` rows
      - n, n_kept
    """
    n = len(df)
    if n == 0:
        return {"n": 0.0, "n_kept": 0.0, "drop_frac_total": float("nan"), "drop_frac_after_warmup": float("nan")}

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
        "n": float(n),
        "n_kept": float(n_kept),
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
            "n(target+model+ewma)": n_tmb,
            "missing_%_in_holdout": miss_pct,
            "baseline_n(target+ewma)": n_base,
            "intersection_n(all_models+ewma)": n_inter,
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["missing_%_in_holdout", "n(target+model+ewma)"], ascending=[True, False]).reset_index(drop=True)
    return out


def split_holdout_into_halves(df_hold: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
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
    rows: list[dict[str, float]] = []

    for m in model_var_cols:
        if m not in df.columns:
            continue

        # --- Fix 2: baseline row special-case to avoid selecting baseline twice ---
        if m == baseline_var_col:
            sub = df[[target_var_col, baseline_var_col]].dropna()
            n = int(len(sub))
            if n < min_n:
                rows.append({
                    "segment": segment,
                    "model": m,
                    "n": float(n),
                    "qlike": float("nan"),
                    "delta_qlike_vs_ewma": float("nan"),
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
                # rmse_vol = float(np.sqrt(mean_squared_error(vt, vb)))
                rmse_vol = float(np.sqrt(np.mean((vt - vb) ** 2)))


            rows.append({
                "segment": segment,
                "model": m,
                "n": float(n),
                "qlike": float(q_b),
                "delta_qlike_vs_ewma": 0.0,
                "rmse_vol": float(rmse_vol),
            })
            continue
        # --- end Fix 2 ---

        # Normal case: model != baseline
        sub = df[[target_var_col, baseline_var_col, m]].dropna()
        n = int(len(sub))
        if n < min_n:
            rows.append({
                "segment": segment,
                "model": m,
                "n": float(n),
                "qlike": float("nan"),
                "delta_qlike_vs_ewma": float("nan"),
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
            # rmse_vol = float(np.sqrt(mean_squared_error(vt, vp)))
            rmse_vol = float(np.sqrt(np.mean((vt - vp) ** 2)))


        rows.append({
            "segment": segment,
            "model": m,
            "n": float(n),
            "qlike": float(q_m),
            "delta_qlike_vs_ewma": float(d),
            "rmse_vol": float(rmse_vol),
        })

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["delta_qlike_vs_ewma", "qlike"], ascending=[True, True]).reset_index(drop=True)
    return out


def report_xgb_mean_median_sanity(
    df_hold: pd.DataFrame,
    *,
    target_var_col: str,
    baseline_var_col: str,
    pairs: list[tuple[str, str, str]],  # (median_col, mean_col, label)
    min_n: int = 100,
) -> pd.DataFrame:
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


def calibration_spearman_holdout(
    df_hold: pd.DataFrame,
    *,
    target_var_col: str,
    model_var_cols: list[str],
    min_n: int = 200,
) -> pd.DataFrame:
    rows = []
    for c in model_var_cols:
        if c not in df_hold.columns:
            continue
        sub = df_hold[[target_var_col, c]].dropna()
        n = int(len(sub))
        if n < min_n:
            rows.append({"model": c, "n": n, "spearman_vol": np.nan})
            continue
        r_vol = np.sqrt(np.clip(sub[target_var_col].astype(float).values, 0.0, None))
        f_vol = np.sqrt(np.clip(sub[c].astype(float).values, 0.0, None))
        spearman = float(pd.Series(f_vol).corr(pd.Series(r_vol), method="spearman"))
        rows.append({"model": c, "n": n, "spearman_vol": spearman})

    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values(["spearman_vol"], ascending=[False]).reset_index(drop=True)
    return out
