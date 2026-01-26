from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        # Normalize window_type consistently
        wt = str(self.window_type).lower().strip()
        object.__setattr__(self, "window_type", wt)

        # Validate core invariants
        if int(self.refit_every) <= 0:
            raise ValueError(f"refit_every must be > 0, got {self.refit_every}")
        if int(self.initial_train_size) < 0:
            raise ValueError(f"initial_train_size must be >= 0, got {self.initial_train_size}")

        if wt not in ("rolling", "expanding"):
            raise ValueError(f"window_type must be 'rolling' or 'expanding', got {self.window_type!r}")

        # Normalize + validate mode-specific constraints
        if wt == "expanding":
            # In expanding mode, rolling-only fields are meaningless: force them to None
            object.__setattr__(self, "rolling_window_size", None)
            object.__setattr__(self, "min_train_size", None)
            return

        # rolling mode validations
        if self.rolling_window_size is None or int(self.rolling_window_size) <= 0:
            raise ValueError(
                f"rolling_window_size must be set and > 0 in rolling mode, got {self.rolling_window_size}"
            )
        if self.min_train_size is None or int(self.min_train_size) <= 0:
            raise ValueError(f"min_train_size must be set and > 0 in rolling mode, got {self.min_train_size}")

        if self.initial_train_size > self.rolling_window_size:
            raise ValueError(
                "initial_train_size must be <= rolling_window_size in rolling mode "
                f"(got initial_train_size={self.initial_train_size}, rolling_window_size={self.rolling_window_size})"
            )

        if self.min_train_size > self.rolling_window_size:
            raise ValueError(
                "min_train_size must be <= rolling_window_size in rolling mode "
                f"(got min_train_size={self.min_train_size}, rolling_window_size={self.rolling_window_size})"
            )

        if self.min_train_size > self.initial_train_size:
            raise ValueError(
                "min_train_size must be <= initial_train_size "
                f"(got min_train_size={self.min_train_size}, initial_train_size={self.initial_train_size})"
            )


    

