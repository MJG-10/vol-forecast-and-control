from typing import Any

# -----------------------------
# Default tuning policy
# -----------------------------

DEFAULT_XGB_TUNE_VAL_DAYS: int = 252


# Candidate override grid for lightweight xgb tuning.
# Probes a few high leverage axes: depth (capacity), min_child_weight (smoothing),
# subsample/colsample (bagging), reg_lambda (L2). Each dict overrides some of the base params.
DEFAULT_XGB_CANDIDATE_OVERRIDES = [
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
