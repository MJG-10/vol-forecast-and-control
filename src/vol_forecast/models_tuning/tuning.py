"""
Pre-holdout tuning utilities.

We choose among a small, fixed set of candidate XGB parameter overrides using only
data strictly before the holdout start (with an embargo to avoid label overlap).
Candidates are scored by QLIKE on one or more pre-holdout validation blocks (default: one window, ending at the embargo boundary).
Selection is by (median block QLIKE, worst-block QLIKE). The tuner returns the selected override dict,
window metadata, and a compact score table for audit.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from vol_forecast.metrics import qlike_loss_var
from vol_forecast.wf_config import WalkForwardConfig
from vol_forecast.schema import COLS
from vol_forecast.models.xgb import walk_forward_xgb_logtarget_var
from .tuning_config import DEFAULT_XGB_CANDIDATE_OVERRIDES, DEFAULT_XGB_TUNE_VAL_DAYS


@dataclass(frozen=True)
class TuneWindow:
    holdout_start: pd.Timestamp
    horizon: int
    tune_val_days: int = 252  # ~1y trading days

    def compute_dates(self, idx: pd.DatetimeIndex) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Defines a tuning validation window that is strictly pre-holdout with an embargo
        that prevents label overlap with holdout.

        We assume holdout is defined on forecast origin dates t >= holdout_start.
        Because target is forward-looking over horizon, we embargo the boundary using horizon 
        so that any tuning origin's target window ends before holdout_start.
        """
        if not isinstance(idx, pd.DatetimeIndex):
            raise TypeError("idx must be a DatetimeIndex")

        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")

        # position of holdout start in the trading calendar
        pos_H = int(idx.searchsorted(self.holdout_start))
        if pos_H >= len(idx):
            raise ValueError(
                f"holdout_start={pd.Timestamp(self.holdout_start).date()} is after the last index date."
            )

        # Tuning cutoff with horizon embargo: we choose last tuning origin so its forward target window ends before holdout_start.
        pos_tune_end = pos_H - (self.horizon - 1) - 1
        if pos_tune_end <= 0:
            holdout_eff = idx[pos_H]
            raise ValueError(
                f"Not enough pre-holdout history for embargo: effective holdout boundary is {holdout_eff.date()} "
                f"with horizon={self.horizon}. Need more history before holdout_start."
            )

        tune_val_end = idx[pos_tune_end]

        # Validation block ending at tune_end
        pos_val_start = max(0, pos_tune_end - self.tune_val_days + 1)
        tune_val_start = idx[pos_val_start]

        return tune_val_start, tune_val_end


def forecast_xgb_mean_var(
    *,
    df: pd.DataFrame,
    cfg: WalkForwardConfig,
    horizon: int,
    overrides: dict[str, Any],
    apply_lognormal_mean_correction: bool = True,
) -> pd.Series:
    """
    Adapter: produces an XGB mean variance forecast series aligned to df.index for a given overrides dict.
    """
    _med, mean = walk_forward_xgb_logtarget_var(
        df=df,
        features=list(COLS.HAR_LOG_FEATURES),
        target_var_col=COLS.RVAR_FWD,
        target_log_col=COLS.LOG_RVAR_FWD,
        horizon=horizon,
        cfg=cfg,
        start_date=None,
        params_overrides=overrides,
        name_prefix="xgb_tune",
        apply_lognormal_mean_correction=apply_lognormal_mean_correction,
    )
    return mean


def tune_xgb_params_pre_holdout(
    df: pd.DataFrame,
    *,
    holdout_start: pd.Timestamp,
    horizon: int,
    cfg: WalkForwardConfig,
    tune_val_days: int = DEFAULT_XGB_TUNE_VAL_DAYS,
    apply_lognormal_mean_correction: bool = True,
    blocks: list[tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    min_block_n: int = 200,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """
    Selects among a fixed set of XGB parameter candidates using pre-holdout data only.
    Applies an embargo around holdout_start to avoid forward-window overlap.

    Scoring:
      - Default: mean daily QLIKE over the tuning validation window [tune_val_start, tune_val_end].
      - If blocks are provided: median block QLIKE (tie-breaker: worst-block QLIKE).
      - Blocks with n < min_block_n are assigned inf.
    """

    tw = TuneWindow(
        holdout_start=holdout_start,
        horizon=horizon,
        tune_val_days=int(tune_val_days),
    )

    tune_val_start, tune_val_end = tw.compute_dates(df.index)
    if blocks is not None:
        idx = df.index
        for s, e in blocks:
            s_ts = pd.Timestamp(s)
            e_ts = pd.Timestamp(e)
            if s_ts > e_ts:
                raise ValueError(f"Invalid block: start {s_ts.date()} > end {e_ts.date()}.")

            pos_e = int(idx.searchsorted(e_ts, side="right")) - 1
            if pos_e < 0:
                raise ValueError(
                    f"Block end {e_ts.date()} is before the first index date {idx[0].date()} "
                    f"(block is entirely out of range)."
                )
            e_eff = idx[pos_e]

            if e_eff > tune_val_end:
                raise ValueError(
                    f"Invalid tuning block ({pd.Timestamp(s).date()}..{e_ts.date()}): "
                    f"must be strictly pre-holdout under horizon embargo (end>{tune_val_end.date()})."
                )


    candidates = DEFAULT_XGB_CANDIDATE_OVERRIDES
    rows: list[dict[str, Any]] = []

    best_id: int | None = None
    best_key = (float("inf"), float("inf"))  # (median, worst)
    best_overrides: dict[str, Any] | None = None

    if blocks is None:
        blocks = [(tune_val_start, tune_val_end)]
        block_names = ["tune_window"]
    else:
        block_names = [f"block_{k}" for k in range(len(blocks))]

    for i, overrides in enumerate(candidates):
        mean_for = forecast_xgb_mean_var(
            df=df,
            cfg=cfg,
            horizon=horizon,
            overrides=overrides,
            apply_lognormal_mean_correction=apply_lognormal_mean_correction,
        )

        block_scores: list[float] = []
        block_ns: list[int] = []

        for (start, end), bname in zip(blocks, block_names):
            sub = pd.concat(
                [
                    df.loc[(df.index >= start) & (df.index <= end), COLS.RVAR_FWD].rename("y"),
                    mean_for.loc[(mean_for.index >= start) & (mean_for.index <= end)].rename("p"),
                ],
                axis=1,
            ).dropna()

            n_b = int(len(sub))
            block_ns.append(n_b)

            if n_b < int(min_block_n):
                score_b = float("inf")
            else:
                score_b = qlike_loss_var(sub["y"], sub["p"])
                if not np.isfinite(score_b):
                    score_b = float("inf")
            block_scores.append(score_b)

        med = float(np.median(block_scores))
        worst = float(np.max(block_scores))
        key = (med, worst)

        row = {
            "candidate_id": i,
            "median_block_qlike": med,
            "worst_block_qlike": worst,
            "overrides": overrides,
        }

        for bname, sc, nn in zip(block_names, block_scores, block_ns):
            row[f"{bname}_qlike"] = float(sc)
            row[f"{bname}_n"] = int(nn)

        rows.append(row)

        if key < best_key:
            best_key = key
            best_id = i
            best_overrides = overrides

    assert best_id is not None and best_overrides is not None

    best_meta = {
        "best_id": int(best_id),
        "best_score_median": float(best_key[0]),
        "best_score_worst": float(best_key[1]),
        "best_params_overrides": best_overrides,
        "holdout_start": holdout_start,
        "horizon": int(horizon),
        "used_blocks": [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in blocks],
        "min_block_n": int(min_block_n),
        "tune_val_start": tune_val_start,
        "tune_val_end": tune_val_end,
        "tune_val_days": int(tune_val_days),
    }

    table = (
        pd.DataFrame(rows)
        .sort_values(["median_block_qlike", "worst_block_qlike"], ascending=[True, True])
        .reset_index(drop=True)
    )
    return best_meta, table
