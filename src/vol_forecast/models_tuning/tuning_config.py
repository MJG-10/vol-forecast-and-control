from typing import Any

# -----------------------------
# Default tuning policy
# -----------------------------

DEFAULT_XGB_TUNE_VAL_DAYS: int = 252  # ~1 trading year of validation

DEFAULT_XGB_CANDIDATE_OVERRIDES: list[dict[str, Any]] = [
    {},  # baseline: use model defaults
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 800, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 4, "learning_rate": 0.03, "n_estimators": 1200, "subsample": 0.8, "colsample_bytree": 0.8},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.7, "colsample_bytree": 0.8},
]

from typing import Any

DEFAULT_XGB_CANDIDATE_OVERRIDES: list[dict[str, Any]] = [
    {},  # baseline

    # Depth axis (capacity)
    {"max_depth": 2},
    {"max_depth": 4},

    # min_child_weight axis (smoothing / regularization)
    {"min_child_weight": 5},
    {"min_child_weight": 10},

    # Bagging axis
    {"subsample": 0.7, "colsample_bytree": 0.7},

    # Key interactions: high capacity + regularization
    {"max_depth": 4, "subsample": 0.7, "colsample_bytree": 0.7},
    {"max_depth": 4, "min_child_weight": 5},
    {"max_depth": 4, "min_child_weight": 10},
    {"max_depth": 4, "min_child_weight": 10, "subsample": 0.7, "colsample_bytree": 0.7},

    # L2 probes (one or two)
    {"reg_lambda": 5.0},
    {"max_depth": 4, "reg_lambda": 5.0},
]


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
