"""Download Yahoo Finance adjusted close for all tickers in holdings."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import polars as pl
import structlog
import yfinance as yf

from src.ticker_map import TICKER_MAP

logger = structlog.get_logger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PRICES_PATH = DATA_DIR / "prices.parquet"


def download_prices(
    start: dt.date = dt.date(2024, 12, 27),
    end: dt.date | None = None,
) -> pl.DataFrame:
    end = end or dt.date.today()
    tickers = sorted(set(TICKER_MAP.values()))
    logger.info("downloading", n=len(tickers), start=str(start), end=str(end))

    data = yf.download(
        tickers,
        start=start,
        end=end + dt.timedelta(days=1),
        auto_adjust=True,  # `Close` is already adjusted for splits & dividends
        progress=False,
        group_by="ticker",
        threads=True,
    )

    rows: list[dict] = []
    for t in tickers:
        if t not in data.columns.get_level_values(0):
            logger.warning("missing_ticker", ticker=t)
            continue
        sub = data[t][["Close"]].dropna().reset_index()
        for d, c in zip(sub["Date"], sub["Close"]):
            rows.append({"date": d.date(), "ticker": t, "adj_close": float(c)})

    df = pl.DataFrame(rows).sort("ticker", "date")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(PRICES_PATH)
    logger.info("prices_saved", path=str(PRICES_PATH), rows=df.height,
                tickers=df["ticker"].n_unique())
    return df


if __name__ == "__main__":
    df = download_prices()
    print(df.group_by("ticker").agg(
        pl.col("date").min().alias("first"),
        pl.col("date").max().alias("last"),
        pl.len().alias("n"),
    ).sort("first"))
