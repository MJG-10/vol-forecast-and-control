"""Default tuning policy"""
from typing import Any


DEFAULT_XGB_TUNE_VAL_DAYS: int = 252  # trading days

# Candidate override grid for lightweight XGB tuning: each dict overrides a subset of base params.
# Probes depth/capacity, min_child_weight (smoothing), subsample/colsample (bagging), reg_lambda (L2).
DEFAULT_XGB_CANDIDATE_OVERRIDES: list[dict[str, Any]] = [
    {},

    {"max_depth": 2},
    {"max_depth": 4},

    {"min_child_weight": 5},
    {"min_child_weight": 10},

    {"subsample": 0.7, "colsample_bytree": 0.7},

    {"max_depth": 4, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 4, "min_child_weight": 10},
    {"min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7},

    {"reg_lambda": 2.0},
    {"reg_lambda": 5.0},
]
