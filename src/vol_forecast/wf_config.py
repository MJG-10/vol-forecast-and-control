from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class WalkForwardConfig:
    """
    Walk-forward refit configuration.

    window_type:
      - "rolling": fixed-length window ending at each origin (uses rolling_window_size).
      - "expanding": growing window from start (uses initial_train_size).

    refit_every: number of rows between refits (typically trading days).
    """
    window_type: Literal["rolling", "expanding"] = "rolling" 
   
    # shared: refit cadence (in origin rows)
    refit_every: int = 60

    # expanding-only
    initial_train_size: int | None = None

    # rolling-only
    rolling_window_size: int | None = 1000
    min_train_size: int | None = 500

    def __post_init__(self) -> None:
        # Expanding mode requires initial_train_size and does not use rolling_window_size.
        wt = self.window_type
        if int(self.refit_every) <= 0:
            raise ValueError(f"refit_every must be > 0, got {self.refit_every}")

        if wt == "expanding":
            # expanding validations
            if self.initial_train_size is None or int(self.initial_train_size) <= 0:
                raise ValueError(
                    f"initial_train_size must be set and > 0 in expanding mode, got {self.initial_train_size}"
                )

            # rolling only fields set to None
            object.__setattr__(self, "rolling_window_size", None)
            object.__setattr__(self, "min_train_size", None)
            return

        # rolling validations
        if self.rolling_window_size is None or int(self.rolling_window_size) <= 0:
            raise ValueError(
                f"rolling_window_size must be set and > 0 in rolling mode, got {self.rolling_window_size}"
            )
        if self.min_train_size is None or int(self.min_train_size) <= 0:
            raise ValueError(f"min_train_size must be set and > 0 in rolling mode, got {self.min_train_size}")

        if int(self.min_train_size) > int(self.rolling_window_size):
            raise ValueError(
                "min_train_size must be <= rolling_window_size in rolling mode "
                f"(got min_train_size={self.min_train_size}, rolling_window_size={self.rolling_window_size})"
            )

        # expanding only set to None
        object.__setattr__(self, "initial_train_size", None)




    

