import pandas as pd 

def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Returns the subset of cols that exist in df.columns, preserving input order."""
    return [c for c in cols if c in df.columns]
