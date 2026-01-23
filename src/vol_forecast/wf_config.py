from dataclasses import dataclass, replace
from typing import Literal

@dataclass(frozen=True)
class WalkForwardConfig:
    # Training-window policy for walk-forward refits.
    # - "rolling": uses only the most recent `rolling_window_size` observations at each refit.
    # - "expanding": uses all available history up to the leakage safe cutoff.
    window_type: Literal["rolling", "expanding"] = "rolling" 


    # shared: first-fit gate (there are no forecasts until first successful fit)
    initial_train_size: int = 500
    refit_every: int = 60

    # rolling-only (must be set in rolling)
    rolling_window_size: int | None = 1000
    min_train_size: int | None = 500
    rolling_calendar_cap: int | None = None


def validate_wf_config(cfg: WalkForwardConfig) -> None:
    wt = str(cfg.window_type).lower().strip()
    if wt not in ("rolling", "expanding"):
        raise ValueError(f"window_type must be 'rolling' or 'expanding', got {cfg.window_type!r}")

    if int(cfg.refit_every) <= 0:
        raise ValueError(f"refit_every must be > 0, got {cfg.refit_every}")

    if int(cfg.initial_train_size) < 0:
        raise ValueError(f"initial_train_size must be >= 0, got {cfg.initial_train_size}")

    if wt == "rolling":
        if cfg.rolling_window_size is None or int(cfg.rolling_window_size) <= 0:
            raise ValueError(f"rolling_window_size must be set and > 0 in rolling mode, got {cfg.rolling_window_size}")

        if cfg.min_train_size is None or int(cfg.min_train_size) <= 0:
            raise ValueError(f"min_train_size must be set and > 0 in rolling mode, got {cfg.min_train_size}")

        if cfg.initial_train_size > cfg.rolling_window_size:
            raise ValueError(
                "initial_train_size must be <= rolling_window_size in rolling mode "
                f"(got initial_train_size={cfg.initial_train_size}, rolling_window_size={cfg.rolling_window_size})"
            )

        if cfg.min_train_size > cfg.rolling_window_size:
            raise ValueError(
                "min_train_size must be <= rolling_window_size in rolling mode "
                f"(got min_train_size={cfg.min_train_size}, rolling_window_size={cfg.rolling_window_size})"
            )

        if cfg.rolling_calendar_cap is not None and int(cfg.rolling_calendar_cap) <= 0:
            raise ValueError(f"rolling_calendar_cap must be > 0 when set, got {cfg.rolling_calendar_cap}")


def effective_wf_config(cfg: WalkForwardConfig) -> WalkForwardConfig:
    """
    Optional internal normalization for clarity.
    In expanding mode, rolling-only fields are ignored; this returns a copy with them set to None.
    """
    validate_wf_config(cfg)
    if cfg.window_type == "expanding":
        return replace(cfg, rolling_window_size=None, min_train_size=None, rolling_calendar_cap=None)
    return cfg
