import pandas as pd 
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np


def _flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens yfinance MultiIndex columns to one level (single-ticker downloads)."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _standardize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Coerces index to DatetimeIndex, drops invalid timestamps, sorts and de-duplicates."""
    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    df = df[~df.index.isna()]
    df = df.sort_index()
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")]
    return df


def _download_yahoo(
    ticker: str,
    start_date: str,
    end_date: str | None,
    auto_adjust: bool,
) -> pd.DataFrame:
    """Downloads OHLCV from Yahoo via yfinance and returns a standardized DataFrame."""
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        group_by="column",
        progress=False,
    )
    data = _flatten_yahoo_columns(data)
    data = _standardize_time_index(data)

    if data.empty:
        raise ValueError(f"Yahoo returned no data for {ticker} in [{start_date}, {end_date}]")

    return data


def load_yahoo_close(
    ticker: str,
    start_date: str = "1990-01-01",
    end_date: str | None = None,
    auto_adjust: bool = True,
) -> pd.Series:
    """Loads Yahoo close prices for `ticker` as a float Series (drops missing)."""
    data = _download_yahoo(ticker, start_date, end_date, auto_adjust)

    if "Close" not in data.columns:
        raise ValueError(f"No Close found for {ticker}. Got: {list(data.columns)}")

    s = pd.to_numeric(data["Close"], errors="coerce").astype(float).dropna()
    s.name = f"{ticker}_close"
    return s


def load_yahoo_ohlc(
    ticker: str,
    start_date: str = "1990-01-01",
    end_date: str | None = None,
    auto_adjust: bool = True,
    require_complete_ohlc: bool = True,
) -> pd.DataFrame:
    """Loads Yahoo OHLC for `ticker` as float columns; optionally requires complete OHLC rows."""
    data = _download_yahoo(ticker, start_date, end_date, auto_adjust)

    needed = ["Open", "High", "Low", "Close"]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns from yfinance for {ticker}: {missing}. Got: {list(data.columns)}")

    out = data[needed].copy()
    out = out.apply(pd.to_numeric, errors="coerce").astype(float)

    how = "any" if require_complete_ohlc else "all"
    out = out.dropna(subset=needed, how=how)
    return out


def load_sp500_ohlc(start_date: str = "1990-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Convenience wrapper for S&P 500 OHLC (ticker: ^GSPC)."""
    return load_yahoo_ohlc("^GSPC", start_date=start_date, end_date=end_date)


def load_sp500_total_return_close(start_date: str = "1990-01-01", end_date: str | None = None) -> pd.Series:
    """Convenience wrapper for S&P 500 total return close series (ticker: ^SP500TR)."""
    return load_yahoo_close("^SP500TR", start_date=start_date, end_date=end_date)


def compute_log_returns_from_series(price: pd.Series, out_name: str = "log_ret", drop_nan: bool = True) -> pd.Series:
    """Computes 1-period log returns log(P_t/P_{t-1}). 
    Invalid prices (<= 0 or non-numeric) and non-finite returns are treated as missing; dropped if `drop_nan=True`.
    """
    p = pd.to_numeric(price, errors="coerce").astype(float)
    p = p.where(p > 0)
    r = np.log(p / p.shift(1))
    r = r.replace([np.inf, -np.inf], np.nan)
    r.name = out_name
    return r.dropna() if drop_nan else r


def load_fred_series(series_id: str, start_date: str, end_date: str|None = None) -> pd.Series:
    """Loads a FRED series as a float Series indexed by date (drops missing observations)."""
    end = end_date if end_date is not None else pd.Timestamp.today().strftime("%Y-%m-%d")
    df = pdr.DataReader(series_id, "fred", start_date, end)
    df = _standardize_time_index(df)

    if series_id not in df.columns:
        raise ValueError(
            f"FRED returned unexpected columns for {series_id!r}: {list(df.columns)}"
        )

    s = pd.to_numeric(df[series_id], errors="coerce").astype(float).dropna()
    s.name = series_id
    return s


def load_cash_rate_percent_fred(*, start_date: str, end_date: str | None) -> tuple[pd.Series, str]:
    """Loads cash-rate level (%) from FRED: DFF backbone, overwritten with EFFR where available; returns (series, source_label)."""
    dff = load_fred_series("DFF", start_date=start_date, end_date=end_date)
    out = dff.copy()
    label = "FRED:DFF"

    try:
        effr = load_fred_series("EFFR", start_date=start_date, end_date=end_date)
        if not effr.empty:
            out = out.reindex(out.index.union(effr.index)).sort_index()
            out.loc[effr.index] = effr
            label = "spliced(FRED:DFF->FRED:EFFR)"
    except Exception:
        pass

    out.name = "cash_rate_percent"
    return out, label


def cash_rate_percent_to_period_return_act360(
    rate_percent: pd.Series,
    index: pd.DatetimeIndex,
    lag_trading_days: int = 1,
) -> pd.Series:
    """
    Converts an annualized rate level (%) to per-period simple returns on `index` using ACT/360.

    Return at t applies to the period from the previous index timestamp to t (calendar-day gap).
    """
    rp = rate_percent.reindex(index).ffill()

    if lag_trading_days:
        rp = rp.shift(int(lag_trading_days))

    gap_days = index.to_series().diff().dt.days.astype(float)

    r = (rp / 100.0) * (gap_days / 360.0)
    r = r.replace([np.inf, -np.inf], np.nan)
    r.name = "cash_r_act360"
    return r

def load_cash_daily_simple_act360(
    *,
    start_date: str,
    end_date: str | None,
    trading_index: pd.DatetimeIndex,
    lag_trading_days: int = 1,
) -> tuple[pd.Series, str]:
    """Loads cash rate (FRED) and converts to ACT/360 simple returns aligned to `trading_index`."""
    rate_pct, src = load_cash_rate_percent_fred(start_date=start_date, end_date=end_date)
    cash_r = cash_rate_percent_to_period_return_act360(rate_pct, trading_index, lag_trading_days=lag_trading_days)
    return cash_r, src


def load_vix_close_series(
    start_date: str,
    end_date: str | None = None,
) -> pd.Series:
    """
    Loads VIX close level series from FRED (VIXCLS).
    """
    s = load_fred_series("VIXCLS", start_date=start_date, end_date=end_date)

    if s.empty:
        raise RuntimeError("FRED:VIXCLS returned empty series.")

    s.name = "vix_close"
    return s




