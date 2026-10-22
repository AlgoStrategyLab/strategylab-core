"""Data loading with a local parquet cache so backtests are reproducible offline."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(os.environ.get("STRATEGYLAB_DATA", Path.cwd() / "data"))


def load_prices(tickers: list[str], start: str = "2000-01-01", end: str | None = None,
                field: str = "Close", refresh: bool = False) -> pd.DataFrame:
    """Return a wide DataFrame of adjusted prices (dividends reinvested) indexed by date.

    Uses yfinance with ``auto_adjust=True`` so ``Close`` is total-return adjusted.
    Each ticker is cached to ``data/<TICKER>.parquet``.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = {}
    for t in tickers:
        path = CACHE_DIR / f"{t}.parquet"
        if path.exists() and not refresh:
            s = pd.read_parquet(path)[field]
        else:
            import yfinance as yf
            raw = yf.download(t, start="1990-01-01", end=end, progress=False, auto_adjust=True)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
            raw.to_parquet(path)
            s = raw[field]
        frames[t] = s
    px = pd.DataFrame(frames).sort_index()
    px = px.loc[start:end] if end else px.loc[start:]
    return px.dropna(how="all")
