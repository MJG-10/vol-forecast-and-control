import pandas as pd 
from pandas_datareader import data as pdr
import yfinance as yf
import numpy as np


def _flatten_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flattens yfinance MultiIndex columns to single level names (single ticker convenience).
       This function assumes group_by="column" was used when downloading the data.
    """
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def _standardize_time_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures DatetimeIndex is sorted ascending with unique timestamps."""
    df = df.copy()

    # yfinance typically returns a DatetimeIndex but we still enforce it.
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")

    # we drop rows with invalid timestamps.
    df = df[~df.index.isna()]

    # Sort and de-duplicate.
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
    Invalid prices (<= 0 or non-numeric) and non-finite returns are treated as missing and dropped.
    """
    p = price.where(price > 0)
    r = np.log(p / p.shift(1))
    r = r.replace([np.inf, -np.inf], np.nan)
    r.name = out_name
    return r.dropna() if drop_nan else r


def compute_log_returns(df: pd.DataFrame, price_col: str = "Close", out_col: str = "log_ret") -> pd.DataFrame:
    """Adds 1-period log returns log(P_t/P_{t-1}) to `df` as `out_col` and uses `price_col` as the price series.
    
    Invalid prices (<= 0 or non-numeric) and non-finite returns are treated as missing, and their rows are dropped.
    Returns a copy of the input DataFrame (does not modify it).
    """
    out = df.copy()
    out[out_col] = compute_log_returns_from_series(out[price_col], out_name=out_col, drop_nan=False)
    return out.dropna(subset=[out_col])


def load_fred_series(series_id: str, start_date: str, end_date: str|None = None) -> pd.Series:
    """Loads a FRED time series via pandas_datareader and returns it as a float Series named according to
    the series_id parameter.
    """
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


def load_cash_yield_irx(start_date: str = "1990-01-01", end_date: str | None = None) -> pd.Series:
    """Loads the 13-week T-bill yield proxy (^IRX) from Yahoo as a float Series."""
    s = load_yahoo_close("^IRX", start_date=start_date, end_date=end_date, auto_adjust=False)
    s.name = "irx_yield_percent"
    return s

def load_available_cash_rate_percent(
    start_date: str,
    end_date: str | None
) -> tuple[pd.Series, str]:
    """
    Load an annualized cash-rate proxy as a percent *level* series (not returns).

    Priority: FRED SOFR -> EFFR, else Yahoo ^IRX.
    Returns (series, source_label).
    """
    for sid in ["SOFR", "EFFR"]:
        try:
            s = load_fred_series(sid, start_date=start_date, end_date=end_date).astype(float)
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            s.name = f"{sid}_percent"
            return s, f"FRED:{sid}"
        except Exception as e:
            last_err = f"FRED:{sid} failed: {type(e).__name__}: {e}"

    try:
        s = load_cash_yield_irx(start_date=start_date, end_date=end_date)
        s.name = "IRX_percent"
        return s, "Yahoo:^IRX"
    except Exception as e:
        yahoo_err = f"Yahoo:^IRX failed: {type(e).__name__}: {e}"
        msg = (
            "Failed to load cash rate percent series from all sources. "
            f"{last_err + '; ' if last_err else ''}{yahoo_err}"
        )
        raise RuntimeError(msg) from e



def cash_rate_percent_to_period_return_act360(
    rate_percent: pd.Series,
    index: pd.DatetimeIndex,
    lag_trading_days: int = 1,
) -> pd.Series:
    """
    Convert annualized rate (%) into per-period simple returns aligned to `index`,
    accruing over calendar-day gaps using ACT/360.

    Alignment matches your log returns convention: return at t corresponds to t-1 -> t.
    """
    # First element is NaN because there is no prior period; caller may drop/fill as desired.

    # if not isinstance(index, pd.DatetimeIndex):
    #     raise TypeError("index must be a DatetimeIndex")
    # if not isinstance(rate_percent.index, pd.DatetimeIndex):
    #     raise TypeError("rate_percent must have a DatetimeIndex")

    # 1) Align the *rate level* to trading dates and forward-fill missing fixings/quotes.
    rp = rate_percent.astype(float).replace([np.inf, -np.inf], np.nan)
    rp = rp.reindex(index).ffill()

    if lag_trading_days:
        rp = rp.shift(int(lag_trading_days))

    # 2) Compute calendar-day gaps of the trading index (t-1 -> t).
    gap_days = index.to_series().diff().dt.days.astype(float)

    # 3) Simple ACT/360 accrual for each holding period.
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
    """
    Load an annualized cash-rate proxy as a percent level series (SOFR/EFFR or ^IRX),
    then convert it into per-period simple returns aligned to `trading_index` using ACT/360
    with actual calendar-day gaps.

    Returns: (cash_daily_simple, source_label)
    """
    rate_pct, src = load_available_cash_rate_percent(
        start_date=start_date,
        end_date=end_date
    )

    cash_r = cash_rate_percent_to_period_return_act360(rate_pct, trading_index, 
                                                       lag_trading_days=lag_trading_days)

    cash_r.name = "cash_r_act360"
    return cash_r, src


def load_vix_close_series(start_date: str, end_date: str | None = None) -> tuple[pd.Series, str]:
    """VIX close level: prefers FRED (VIXCLS), otherwise falls back to Yahoo (^VIX).
    Returns series and source and raises RuntimeError if both sources fail.
    """
    try:
        s = load_fred_series("VIXCLS", start_date=start_date, end_date=end_date)
        s.name = "vix_close"
        return s, "FRED:VIXCLS"
    except Exception as e:
         fred_err = f"FRED:VIXCLS failed: {type(e).__name__}: {e}"

    try:
        s2 = load_yahoo_close("^VIX", start_date=start_date, end_date=end_date, auto_adjust=False)
        s2.name = "vix_close"
        return s2, "Yahoo:^VIX"
    except Exception as e2:
        yahoo_err = f"Yahoo:^VIX failed: {type(e2).__name__}: {e2}"
        msg = (
            "Failed to load VIX close series from both FRED and Yahoo. "
            f"{fred_err}; {yahoo_err}"
        )
        raise RuntimeError(msg) from e2



