from dataclasses import dataclass

@dataclass(frozen=True)
class WalkForwardConfig:
    initial_train_frac: float = 0.4
    window_type: str = "rolling"   # "rolling" or "expanding"
    rolling_w: int = 2000
    refit_every: int = 60

    min_rows_total: int = 300          # minimum rows after dropna
    min_train_rows: int = 200          # minimum rows in training slice for fit
    min_train_end_excl: int = 51       # minimum usable train_end_excl (post leakage adjustment)