"""Compute period-end PnL curve for Situational Awareness LP.

Convention: holdings reported for period P are assumed effective from the
first trading day AFTER P (perfect-information). Daily $-PnL for ticker T on
day D = shares(P, T) * (adj_close[D] - adj_close[D-1]) where P is the latest
period <= D-1.

Puts are subtracted (treated as short the underlying); Calls and SH are
added. This matches "replace options by commons" with sign preserved.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import structlog

from src.ticker_map import TICKER_MAP

logger = structlog.get_logger(__name__)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


_RAW_ROWS_CACHE: list[dict] | None = None


def _all_raw_rows() -> list[dict]:
    """Cache the result of parsing every 13F infoTable (cheap: ~1s)."""
    global _RAW_ROWS_CACHE
    if _RAW_ROWS_CACHE is None:
        from src.holdings import (
            _find_infotable_xml, _normalize_name, list_13f_filings, parse_infotable,
        )
        rows: list[dict] = []
        for f in list_13f_filings():
            url = _find_infotable_xml(f["accession"])
            for r in parse_infotable(url):
                r["period"] = f["period"]
                r["filing_date"] = f["filing_date"]
                r["issuer"] = _normalize_name(r["name"])
                rows.append(r)
        _RAW_ROWS_CACHE = rows
    return _RAW_ROWS_CACHE


def load_holdings_with_tickers(option_delta: float = 1.0) -> pl.DataFrame:
    """Load aggregated holdings with tickers.

    option_delta controls how Call / Put lines are interpreted:
      1.0 → full notional (1 contract = 100 share-equivalents); replica view
      0.5 → 50-delta proxy (Calls and Puts scaled by ½)
      0.0 → drop all Put / Call rows; commons only

    Sign: SH and Call rows are long, Put rows are short. Re-parses raw 13F
    rows each call (cached) so the aggregation reflects `option_delta`.
    """
    raw_rows = _all_raw_rows()
    rows: list[dict] = []
    for r0 in raw_rows:
        r = dict(r0)
        pc = r["put_call"]
        if pc == "SH":
            scale, sign = 1.0, 1
        elif pc == "Call":
            if option_delta == 0:
                continue
            scale, sign = option_delta, 1
        elif pc == "Put":
            if option_delta == 0:
                continue
            scale, sign = option_delta, -1
        else:
            continue
        r["net_shares"] = int(round(sign * scale * r["shares"]))
        r["net_value"] = int(round(sign * scale * r["value"]))
        r["abs_value"] = int(round(scale * r["value"]))
        rows.append(r)

    raw = pl.DataFrame(rows)
    df = (
        raw.group_by("period", "issuer")
        .agg(
            pl.col("name").first().alias("raw_name"),
            pl.col("cusip").unique().sort().str.join(",").alias("cusips"),
            pl.col("net_shares").sum().alias("shares"),
            pl.col("net_value").sum().alias("value"),
            pl.col("abs_value").sum().alias("gross_value"),
            pl.col("put_call").unique().sort().str.join("+").alias("put_call_mix"),
            pl.col("filing_date").first(),
        )
        .with_columns(pl.col("period").str.to_date())
        .sort("period", "value", descending=[False, True])
    )
    return df.with_columns(
        pl.col("issuer").replace_strict(TICKER_MAP, default=None).alias("ticker")
    )


def signed_shares(df: pl.DataFrame) -> pl.DataFrame:
    """Compatibility shim — shares are already signed at the aggregation step."""
    return df.with_columns(pl.col("shares").alias("signed_shares"))


def _global_next_period_map(periods: list) -> dict:
    """For each period, return the next period overall (not per ticker).

    Used so a ticker dropped from period P+1 has its position closed at P+1d+1,
    not silently carried until the next period it reappears in.
    """
    sorted_p = sorted(periods)
    out = {p: q for p, q in zip(sorted_p, sorted_p[1:])}
    import datetime as _dt
    out[sorted_p[-1]] = _dt.date(2099, 1, 1)
    return out


def _position_windows(
    holdings: pl.DataFrame, use_filing_date: bool = False
) -> pl.DataFrame:
    """Attach (effective_from, effective_until) windows using GLOBAL next-period.

    Tickers absent from period P+1 are cleanly closed at P+1.

    use_filing_date=True → effective_from = filing_date + 1 (realistic copycat
    timing — no lookahead). effective_until = the next period's filing date.
    """
    if use_filing_date:
        if "filing_date" not in holdings.columns:
            raise ValueError("holdings must include filing_date for copycat mode")
        # Build period → next_filing_date map (need a join, not just next-period)
        fdates = (
            holdings.select("period", "filing_date")
            .unique()
            .with_columns(pl.col("filing_date").str.to_date())
            .sort("period")
        )
        next_fdates = fdates.with_columns(
            pl.col("filing_date").shift(-1)
            .fill_null(pl.date(2099, 1, 1))
            .alias("next_filing_date")
        ).select("period", "next_filing_date")
        return (
            holdings.with_columns(pl.col("filing_date").str.to_date())
            .join(next_fdates, on="period", how="left")
            .with_columns(
                (pl.col("filing_date") + pl.duration(days=1)).alias("effective_from"),
                pl.col("next_filing_date").alias("effective_until"),
            )
            .drop("next_filing_date", "filing_date")
        )

    next_map = _global_next_period_map(holdings["period"].unique().to_list())
    return (
        holdings.with_columns(
            (pl.col("period") + pl.duration(days=1)).alias("effective_from"),
            pl.col("period").replace_strict(next_map).alias("effective_until"),
        )
    )


def _blended_position_windows(holdings: pl.DataFrame) -> pl.DataFrame:
    """Between two consecutive 13F period-ends, hold 0.5 * earlier + 0.5 * later.

    After the final reported period, hold the full latest snapshot
    (no future to average with).
    """
    import datetime as _dt

    periods = sorted(holdings["period"].unique().to_list())
    shares_by_pt = {
        (r["period"], r["ticker"]): r["signed_shares"]
        for r in holdings.select("period", "ticker", "signed_shares")
        .iter_rows(named=True)
    }
    all_tickers = sorted({t for (_, t) in shares_by_pt.keys()})

    rows: list[dict] = []
    for i in range(len(periods) - 1):
        p_a, p_b = periods[i], periods[i + 1]
        for t in all_tickers:
            blended = 0.5 * shares_by_pt.get((p_a, t), 0) + 0.5 * shares_by_pt.get((p_b, t), 0)
            if blended == 0:
                continue
            rows.append({
                "ticker": t,
                "effective_from": p_a + _dt.timedelta(days=1),
                "effective_until": p_b,  # exclusive — matches existing convention
                "signed_shares": blended,
            })
    # Final period: full position from P_N + 1 onwards
    p_last = periods[-1]
    for t in all_tickers:
        s = shares_by_pt.get((p_last, t), 0)
        if s == 0:
            continue
        rows.append({
            "ticker": t,
            "effective_from": p_last + _dt.timedelta(days=1),
            "effective_until": _dt.date(2099, 1, 1),
            "signed_shares": s,
        })
    return pl.DataFrame(rows)


def build_pnl(
    option_delta: float = 1.0,
    use_filing_date: bool = False,
    blended: bool = False,
) -> pl.DataFrame:
    """Daily portfolio $-PnL and cumulative curve.

    use_filing_date=True swaps perfect-information for realistic copycat timing.
    blended=True overrides use_filing_date: in each inter-period window the
    position is 0.5 × earlier-snapshot + 0.5 × later-snapshot.
    """
    holdings = signed_shares(load_holdings_with_tickers(option_delta))
    prices = pl.read_parquet(DATA_DIR / "prices.parquet")

    prices = prices.sort("ticker", "date").with_columns(
        pl.col("adj_close").diff().over("ticker").alias("d_price")
    )

    if blended:
        pos = _blended_position_windows(
            holdings.select("period", "ticker", "signed_shares")
        )
    else:
        pos = (
            _position_windows(
                holdings.select("period", "ticker", "signed_shares", "filing_date"),
                use_filing_date=use_filing_date,
            )
            .select("ticker", "effective_from", "effective_until", "signed_shares")
        )

    # Join prices to positions: for each price row, find the position window covering it
    px_with_pos = prices.join(pos, on="ticker", how="left").filter(
        (pl.col("date") >= pl.col("effective_from"))
        & (pl.col("date") < pl.col("effective_until"))
    )

    # Daily $-PnL per ticker
    px_with_pos = px_with_pos.with_columns(
        (pl.col("signed_shares") * pl.col("d_price")).alias("pnl_usd")
    )

    # Aggregate to daily portfolio PnL
    daily = (
        px_with_pos.group_by("date")
        .agg(pl.col("pnl_usd").sum().alias("pnl_usd"))
        .sort("date")
        .with_columns(pl.col("pnl_usd").cum_sum().alias("cum_pnl_usd"))
    )

    # Gross notional per period (|value| of each line). Puts contribute as
    # absolute exposure so the denominator never goes negative.
    notional_col = "gross_value" if "gross_value" in holdings.columns else "value"
    period_notional = (
        holdings.group_by("period")
        .agg(
            pl.col(notional_col).abs().sum().alias("notional_usd"),
            pl.col("filing_date").first().alias("filing_date"),
        )
        .sort("period")
    )
    if blended:
        # Blended notional: in window [P_a+1, P_b), notional = 0.5*(N_a + N_b);
        # after the final period, full latest notional.
        import datetime as _dt
        pn = period_notional.sort("period")
        n_vals = pn["notional_usd"].to_list()
        p_vals = pn["period"].to_list()
        rows = []
        for i in range(len(p_vals) - 1):
            rows.append({
                "effective_from": p_vals[i] + _dt.timedelta(days=1),
                "notional_usd": 0.5 * (n_vals[i] + n_vals[i + 1]),
            })
        rows.append({
            "effective_from": p_vals[-1] + _dt.timedelta(days=1),
            "notional_usd": n_vals[-1],
        })
        period_notional = pl.DataFrame(rows)
    elif use_filing_date:
        period_notional = period_notional.with_columns(
            (pl.col("filing_date").str.to_date() + pl.duration(days=1))
            .alias("effective_from")
        )
    else:
        period_notional = period_notional.with_columns(
            (pl.col("period") + pl.duration(days=1)).alias("effective_from")
        )

    # As-of join: which period's notional is active on each date
    daily = daily.join_asof(
        period_notional.select("effective_from", "notional_usd").sort("effective_from"),
        left_on="date",
        right_on="effective_from",
        strategy="backward",
    )

    # Time-weighted return: each day, daily_ret = daily_pnl / notional_for_that_day
    daily = daily.with_columns(
        (pl.col("pnl_usd") / pl.col("notional_usd")).alias("daily_return"),
    )
    # Cumulative geometric return (start at 1.0)
    daily = daily.with_columns(
        ((pl.col("daily_return").fill_null(0.0) + 1.0).cum_prod() - 1.0)
        .alias("cum_return_pct")
    )

    return daily


def smh_benchmark(start: "object", end: "object") -> pl.DataFrame:
    """SMH daily return series over [start, end], aligned to SALP's calendar."""
    prices = pl.read_parquet(DATA_DIR / "prices.parquet")
    smh = (
        prices.filter(pl.col("ticker") == "SMH")
        .filter((pl.col("date") >= start) & (pl.col("date") <= end))
        .sort("date")
        .with_columns(
            pl.col("adj_close").pct_change().alias("daily_return"),
        )
        .with_columns(
            ((pl.col("daily_return").fill_null(0.0) + 1.0).cum_prod() - 1.0)
            .alias("cum_return_pct"),
        )
        .select("date", "daily_return", "cum_return_pct")
    )
    return smh


def sharpe(daily_returns: pl.Series, rf_annual: float = 0.0,
           periods_per_year: int = 252) -> float:
    """Annualized Sharpe ratio. rf_annual is the risk-free rate (e.g. 0.04 for 4%)."""
    r = daily_returns.drop_nulls()
    if r.is_empty():
        return float("nan")
    daily_rf = rf_annual / periods_per_year
    excess = r - daily_rf
    mu = float(excess.mean())
    sigma = float(excess.std())
    if sigma == 0:
        return float("nan")
    return (mu / sigma) * (periods_per_year ** 0.5)


def scale_to_vol(daily_returns: pl.Series, target_vol_daily: float) -> pl.Series:
    """Scale a return series multiplicatively so its std matches target_vol_daily."""
    sigma = float(daily_returns.drop_nulls().std())
    if sigma == 0:
        return daily_returns
    k = target_vol_daily / sigma
    return daily_returns * k


def build_ew_voltargeted_pnl(
    vol_window: int = 30,
    target_daily_vol: float = 0.01,
) -> pl.DataFrame:
    """Daily portfolio return for an equal-weighted, vol-targeted replication.

    Construction (per trading day D, given the latest period P ≤ D-1):
      universe       = set of (ticker) reported in P, with signs from Put/Call
      sigma_i(D)     = trailing `vol_window`-day std-dev of daily returns of i
                       (computed on data strictly before D — no lookahead)
      weight_i(D)    = sign_i * (target_daily_vol / sigma_i(D)) / N_active(D)
      portfolio_r(D) = sum_i weight_i(D) * daily_return_i(D)

    Each name therefore contributes (target_daily_vol / N) of vol on average,
    independent of its raw price level or volatility.

    Commons-only universe: Call/Put rows are dropped entirely (option_delta=0).
    """
    holdings = load_holdings_with_tickers(option_delta=0.0)
    # Sign: net_shares already carries the sign; collapse to +1/-1 per position
    holdings = holdings.with_columns(
        pl.when(pl.col("shares") < 0).then(-1).otherwise(1).alias("sign")
    ).select("period", "ticker", "sign")

    prices = (
        pl.read_parquet(DATA_DIR / "prices.parquet")
        .sort("ticker", "date")
        .with_columns(
            pl.col("adj_close").pct_change().over("ticker").alias("ret"),
        )
        .with_columns(
            # rolling std computed on returns strictly BEFORE current day:
            pl.col("ret")
            .shift(1)
            .rolling_std(window_size=vol_window, min_samples=vol_window // 2)
            .over("ticker")
            .alias("vol30"),
        )
    )

    pos = _position_windows(holdings)

    joined = (
        prices.join(pos, on="ticker", how="left")
        .filter(
            (pl.col("date") >= pl.col("effective_from"))
            & (pl.col("date") < pl.col("effective_until"))
            & pl.col("vol30").is_not_null()
            & (pl.col("vol30") > 0)
        )
    )

    # Per-name vol-targeted contribution; equal weighting happens by averaging
    # across the N active tickers each day.
    contrib = joined.with_columns(
        (pl.col("sign") * pl.col("ret") * (target_daily_vol / pl.col("vol30")))
        .alias("name_contrib"),
    )

    daily = (
        contrib.group_by("date")
        .agg(
            pl.col("name_contrib").mean().alias("daily_return"),  # mean = (1/N)*sum
            pl.col("ticker").n_unique().alias("n_active"),
        )
        .sort("date")
        .with_columns(
            ((pl.col("daily_return").fill_null(0.0) + 1.0).cum_prod() - 1.0)
            .alias("cum_return_pct")
        )
    )
    return daily


def per_ticker_contribution(
    option_delta: float = 1.0,
    use_filing_date: bool = False,
    blended: bool = False,
) -> pl.DataFrame:
    """Each ticker's contribution to the portfolio's cumulative return.

    Per day, ticker contributes `pnl_ticker_usd / portfolio_notional_for_that_day`
    to the portfolio return. Summed across all days = ticker's arithmetic
    contribution to the cumulative (arithmetic-sum) portfolio return.
    """
    holdings = signed_shares(load_holdings_with_tickers(option_delta))
    prices = (
        pl.read_parquet(DATA_DIR / "prices.parquet")
        .sort("ticker", "date")
        .with_columns(pl.col("adj_close").diff().over("ticker").alias("d_price"))
    )

    if blended:
        pos = _blended_position_windows(
            holdings.select("period", "ticker", "signed_shares")
        )
    else:
        pos = (
            _position_windows(
                holdings.select("period", "ticker", "signed_shares", "filing_date"),
                use_filing_date=use_filing_date,
            )
            .select("ticker", "effective_from", "effective_until", "signed_shares")
        )

    # Per-period notional, matching build_pnl's logic
    notional_col = "gross_value" if "gross_value" in holdings.columns else "value"
    period_notional = (
        holdings.group_by("period")
        .agg(
            pl.col(notional_col).abs().sum().alias("notional_usd"),
            pl.col("filing_date").first().alias("filing_date"),
        )
        .sort("period")
    )
    if blended:
        import datetime as _dt
        pn = period_notional.sort("period")
        n_vals = pn["notional_usd"].to_list()
        p_vals = pn["period"].to_list()
        rows = []
        for i in range(len(p_vals) - 1):
            rows.append({"effective_from": p_vals[i] + _dt.timedelta(days=1),
                         "notional_usd": 0.5 * (n_vals[i] + n_vals[i + 1])})
        rows.append({"effective_from": p_vals[-1] + _dt.timedelta(days=1),
                     "notional_usd": n_vals[-1]})
        period_notional = pl.DataFrame(rows)
    elif use_filing_date:
        period_notional = period_notional.with_columns(
            (pl.col("filing_date").str.to_date() + pl.duration(days=1))
            .alias("effective_from")
        )
    else:
        period_notional = period_notional.with_columns(
            (pl.col("period") + pl.duration(days=1)).alias("effective_from")
        )

    px = (
        prices.join(pos, on="ticker", how="left")
        .filter(
            (pl.col("date") >= pl.col("effective_from"))
            & (pl.col("date") < pl.col("effective_until"))
        )
        .with_columns((pl.col("signed_shares") * pl.col("d_price")).alias("pnl_usd"))
        .sort("date")
        .join_asof(
            period_notional.select("effective_from", "notional_usd").sort("effective_from"),
            left_on="date", right_on="effective_from", strategy="backward",
        )
        .with_columns(
            (pl.col("pnl_usd") / pl.col("notional_usd")).alias("ret_contrib")
        )
    )
    return (
        px.group_by("ticker")
        .agg(
            pl.col("pnl_usd").sum().alias("total_pnl_usd"),
            pl.col("ret_contrib").sum().alias("contribution_pct"),
        )
        .sort("contribution_pct", descending=True)
    )


def per_ticker_pnl(option_delta: float = 1.0) -> pl.DataFrame:
    """Cumulative PnL by ticker for attribution."""
    holdings = signed_shares(load_holdings_with_tickers(option_delta))
    prices = pl.read_parquet(DATA_DIR / "prices.parquet").sort("ticker", "date")
    prices = prices.with_columns(
        pl.col("adj_close").diff().over("ticker").alias("d_price")
    )
    pos = _position_windows(holdings.select("period", "ticker", "signed_shares"))
    pnl_rows = (
        prices.join(pos, on="ticker", how="left")
        .filter(
            (pl.col("date") >= pl.col("effective_from"))
            & (pl.col("date") < pl.col("effective_until"))
        )
        .with_columns((pl.col("signed_shares") * pl.col("d_price")).alias("pnl_usd"))
    )
    return (
        pnl_rows.group_by("ticker")
        .agg(pl.col("pnl_usd").sum().alias("total_pnl_usd"))
        .sort("total_pnl_usd", descending=True)
    )


if __name__ == "__main__":
    daily = build_pnl()
    print(daily.tail(10))
    print("\nFinal cumulative $:", f"${daily['cum_pnl_usd'][-1]:,.0f}")
    print("Final cumulative %:", f"{daily['cum_return_pct'][-1] * 100:.2f}%")
    print("\nPer-ticker total PnL:")
    print(per_ticker_pnl())
