"""Streamlit dashboard: PnL curve of Situational Awareness LP from 13F holdings."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.pnl import (
    build_ew_voltargeted_pnl,
    build_pnl,
    load_holdings_with_tickers,
    per_ticker_pnl,
    scale_to_vol,
    sharpe,
    signed_shares,
    smh_benchmark,
)

DATA_DIR = Path(__file__).resolve().parent / "data"

SA_ICON = str(Path(__file__).resolve().parent / "assets" / "sa_icon_light.svg")
st.set_page_config(
    page_title="Situational Awareness LP — 13F PnL",
    page_icon=SA_ICON,
    layout="wide",
)
st.title("Situational Awareness LP — 13F-based PnL")
st.caption(
    "Daily return = shares × Δ adjusted-close (Yahoo Finance). "
    "Holdings from each 13F-HR take effect the first trading day after the "
    "reported period (perfect-information). Tickers dropped between periods "
    "are closed cleanly. Sharpe quoted at risk-free = 0."
)

MODE_HALF = "50-Δ proxy options (Δ=0.5)"
MODE_COMMONS = "Commons only — period-end (perfect info)"
MODE_COPYCAT = "Copy-trade commons — filing date (realistic)"
MODE_BLENDED = "Commons blended (50/50 between two snapshots)"
MODE_EWVOL = "Equal-weight, 30d vol-targeted (commons only)"

_modes = [MODE_HALF, MODE_COMMONS, MODE_COPYCAT, MODE_BLENDED, MODE_EWVOL]
mode = st.radio(
    "Position handling", _modes, horizontal=True,
    index=_modes.index(MODE_COMMONS),
)
is_ew = mode == MODE_EWVOL
is_copycat = mode == MODE_COPYCAT
is_blended = mode == MODE_BLENDED
option_delta = {
    MODE_HALF: 0.5,
    MODE_COMMONS: 0.0,
    MODE_COPYCAT: 0.0,
    MODE_BLENDED: 0.0,
    MODE_EWVOL: 0.0,
}[mode]

if is_ew:
    st.caption(
        "**EW vol-targeted:** every common-share ticker in the latest 13F is "
        "held with equal weight, scaled by `1 / 30d-realised-vol` so each "
        "position contributes equal vol. Calls and Puts are excluded — pure "
        "stock-picking view stripped of position sizing."
    )
elif is_copycat:
    st.caption(
        "**Copy-trade:** holdings applied from `filing_date + 1` (realistic — "
        "no lookahead). Calls / Puts dropped. The 45-day reporting delay means "
        "some early Q1 2025 PnL is missed, but no hindsight is used."
    )
elif is_blended:
    st.caption(
        "**Blended:** in the window between two consecutive 13F period-ends, "
        "hold 0.5 × earlier-snapshot + 0.5 × later-snapshot per ticker. Models "
        "a gradual transition between quarters. Uses foresight of the next "
        "snapshot. Calls / Puts dropped."
    )
elif option_delta == 0.5:
    st.caption("Each Call / Put line scaled by 0.5 (50-Δ proxy). Cheap "
               "approximation that respects sign but assumes ATM-ish delta.")
else:
    st.caption("Option lines discarded entirely; only `SH` rows count. "
               "Applied at period-end + 1 (perfect-information).")

# ─── Cached loaders ──────────────────────────────────────────────────────────
@st.cache_data
def _daily_long_short(
    option_delta: float, use_filing_date: bool = False, blended: bool = False
) -> pl.DataFrame:
    return build_pnl(
        option_delta=option_delta,
        use_filing_date=use_filing_date,
        blended=blended,
    )

@st.cache_data
def _daily_ew() -> pl.DataFrame:
    return build_ew_voltargeted_pnl(vol_window=30, target_daily_vol=0.01)

@st.cache_data
def _holdings(option_delta: float) -> pl.DataFrame:
    return signed_shares(load_holdings_with_tickers(option_delta=option_delta))

@st.cache_data
def _per_ticker(option_delta: float) -> pl.DataFrame:
    return per_ticker_pnl(option_delta=option_delta)

holdings = _holdings(option_delta)
per_tk = _per_ticker(option_delta)
if is_ew:
    daily = _daily_ew()
else:
    daily = _daily_long_short(
        option_delta, use_filing_date=is_copycat, blended=is_blended,
    )

# ─── Headline metrics ────────────────────────────────────────────────────────
final_pct = float(daily["cum_return_pct"][-1])
start_date = daily["date"].min()
end_date = daily["date"].max()
peak_dd = float((daily["cum_return_pct"] - daily["cum_return_pct"].cum_max()).min())
ann_vol = float(daily["daily_return"].std()) * (252 ** 0.5)
ann_ret = float(daily["daily_return"].mean()) * 252
sh = sharpe(daily["daily_return"], rf_annual=0.0)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Time-weighted return", f"{final_pct * 100:.1f}%")
c2.metric("Annualised return", f"{ann_ret * 100:.1f}%")
c3.metric("Annualised vol", f"{ann_vol * 100:.1f}%")
c4.metric("Sharpe (rf=0)", f"{sh:.2f}")
c5.metric("Max drawdown", f"{peak_dd * 100:.1f}%")

# ─── Return curve ────────────────────────────────────────────────────────────
fig = go.Figure()
color = "#9467bd" if is_ew else "#1f77b4"
fillc = "rgba(148,103,189,0.12)" if is_ew else "rgba(31,119,180,0.12)"
fig.add_trace(go.Scatter(
    x=daily["date"], y=daily["cum_return_pct"] * 100,
    mode="lines", line=dict(color=color, width=2),
    name="Return %", fill="tozeroy", fillcolor=fillc,
))
y_max = float((daily["cum_return_pct"] * 100).max())
for p in sorted(holdings["period"].unique().to_list()):
    x = dt.datetime.combine(p, dt.time())
    fig.add_shape(
        type="line", xref="x", yref="paper", x0=x, x1=x, y0=0, y1=1,
        line=dict(color="rgba(128,128,128,0.35)", width=1, dash="dot"),
    )
    fig.add_annotation(
        x=x, y=y_max, xref="x", yref="y",
        text=p.isoformat(), showarrow=False,
        yshift=10, font=dict(size=10, color="gray"),
    )
fig.update_layout(height=480, yaxis_title="Cumulative return (%)",
                  margin=dict(t=20, b=30))
st.plotly_chart(fig, use_container_width=True)

# ─── Benchmark vs SMH ────────────────────────────────────────────────────────
st.subheader("Benchmark vs SMH (VanEck Semiconductor ETF)")

@st.cache_data
def _benchmark(mode_key: str, option_delta: float):
    if mode_key == MODE_EWVOL:
        d = _daily_ew()
    else:
        d = _daily_long_short(
            option_delta,
            use_filing_date=(mode_key == MODE_COPYCAT),
            blended=(mode_key == MODE_BLENDED),
        )
    smh = smh_benchmark(d["date"].min(), d["date"].max())
    return (
        d.select("date", "daily_return")
        .rename({"daily_return": "salp"})
        .join(smh.select("date", "daily_return").rename({"daily_return": "smh"}),
              on="date", how="inner")
        .drop_nulls()
    )

bench = _benchmark(mode, option_delta)
salp_r = bench["salp"]
smh_r = bench["smh"]

salp_sharpe = sharpe(salp_r, rf_annual=0.0)
smh_sharpe = sharpe(smh_r, rf_annual=0.0)
salp_vol = float(salp_r.std()) * (252 ** 0.5)
smh_vol = float(smh_r.std()) * (252 ** 0.5)
salp_ann = float(salp_r.mean()) * 252
smh_ann = float(smh_r.mean()) * 252

# Correlation + beta
import numpy as _np
_s = _np.asarray(salp_r.to_list(), dtype=float)
_m = _np.asarray(smh_r.to_list(), dtype=float)
corr = float(_np.corrcoef(_s, _m)[0, 1])
beta = float(_np.cov(_s, _m, ddof=1)[0, 1] / _np.var(_m, ddof=1))

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("SALP Sharpe", f"{salp_sharpe:.2f}",
          delta=f"{salp_sharpe - smh_sharpe:+.2f} vs SMH")
m2.metric("SMH Sharpe", f"{smh_sharpe:.2f}")
m3.metric("SALP ann.vol / return", f"{salp_vol*100:.1f}% / {salp_ann*100:.1f}%")
m4.metric("SMH ann.vol / return", f"{smh_vol*100:.1f}% / {smh_ann*100:.1f}%")
m5.metric("Correlation to SMH", f"{corr:+.2f}")
m6.metric("Beta to SMH", f"{beta:+.2f}")

smh_scaled = scale_to_vol(smh_r, float(salp_r.std()))
bench = bench.with_columns([
    ((salp_r.fill_null(0.0) + 1.0).cum_prod() - 1.0).alias("salp_cum"),
    ((smh_r.fill_null(0.0) + 1.0).cum_prod() - 1.0).alias("smh_cum"),
    ((smh_scaled.fill_null(0.0) + 1.0).cum_prod() - 1.0).alias("smh_scaled_cum"),
])

show_raw_smh = st.checkbox("Show raw (unscaled) SMH", value=True)

bfig = go.Figure()
bfig.add_trace(go.Scatter(
    x=bench["date"], y=bench["salp_cum"] * 100,
    name=f"SALP (σ={salp_vol*100:.1f}%)",
    line=dict(color="#1f77b4", width=2),
))
bfig.add_trace(go.Scatter(
    x=bench["date"], y=bench["smh_scaled_cum"] * 100,
    name=f"SMH scaled to SALP vol (×{salp_vol/smh_vol:.2f})",
    line=dict(color="#ff7f0e", width=2),
))
if show_raw_smh:
    bfig.add_trace(go.Scatter(
        x=bench["date"], y=bench["smh_cum"] * 100,
        name=f"SMH raw (σ={smh_vol*100:.1f}%)",
        line=dict(color="#ff7f0e", width=1.2, dash="dot"),
        opacity=0.6,
    ))
bfig.update_layout(
    height=480, yaxis_title="Cumulative return (%)",
    hovermode="x unified", margin=dict(t=20, b=40),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(bfig, use_container_width=True)

st.caption(
    f"Vol-matching multiplies SMH daily returns by **{salp_vol/smh_vol:.2f}× "
    f"(={salp_vol*100:.1f}% / {smh_vol*100:.1f}%)** so both series have "
    f"the same realised volatility."
)

# ─── SA fund PnL minus SMH exposure ──────────────────────────────────────────
st.subheader(f"SA fund PnL minus its SMH exposure (β = {beta:.2f})")

salp_cum = bench["salp_cum"].to_numpy()
smh_cum = bench["smh_cum"].to_numpy()
diff_cum = salp_cum - beta * smh_cum

# Daily residual return series (beta-hedged). Sharpe / ann.return / ann.vol
# computed on this daily series — what you'd get from a beta-neutral version
# of the strategy (short β×SMH against long SA).
resid_daily = _s - beta * _m
resid_sharpe = float(_np.mean(resid_daily) / _np.std(resid_daily, ddof=1) * (252 ** 0.5))
resid_ann_ret = float(_np.mean(resid_daily) * 252)
resid_ann_vol = float(_np.std(resid_daily, ddof=1) * (252 ** 0.5))

r1, r2, r3, r4 = st.columns(4)
r1.metric("Cumulative", f"{float(diff_cum[-1])*100:+.1f}%")
r2.metric("Annualised return", f"{resid_ann_ret*100:+.1f}%")
r3.metric("Annualised vol", f"{resid_ann_vol*100:.1f}%")
r4.metric("Sharpe (rf=0)", f"{resid_sharpe:+.2f}")

dfig = go.Figure()
dfig.add_trace(go.Scatter(
    x=bench["date"], y=diff_cum * 100,
    mode="lines", line=dict(color="#2ca02c", width=2),
    name="SA PnL − β × SMH", fill="tozeroy",
    fillcolor="rgba(44,160,44,0.12)",
))
dfig.update_layout(
    height=420, yaxis_title="Cumulative return (%)",
    margin=dict(t=20, b=30),
    showlegend=False,
)
st.plotly_chart(dfig, use_container_width=True)
st.caption(
    "Cumulative SA-fund return minus β × (cumulative SMH return). "
    "Sharpe / ann.return / ann.vol are computed on the **daily residual** "
    "`r_SA − β·r_SMH` (the daily PnL of a beta-neutral version)."
)

# ─── Ticker attribution ──────────────────────────────────────────────────────
if not is_ew:
    st.subheader("Per-ticker $-PnL attribution")
    bar = go.Figure(go.Bar(
        x=per_tk["ticker"],
        y=per_tk["total_pnl_usd"] / 1e6,
        marker_color=[
            "#2ca02c" if v >= 0 else "#d62728"
            for v in per_tk["total_pnl_usd"]
        ],
    ))
    bar.update_layout(height=400, yaxis_title="Total PnL ($M)",
                      margin=dict(t=20, b=40))
    st.plotly_chart(bar, use_container_width=True)

# ─── Reported holdings — single bar chart, commons only ─────────────────────
st.subheader("Reported holdings — common shares only ($ notional)")
holdings_commons = _holdings(0.0)
periods = sorted(holdings_commons["period"].unique().to_list(), reverse=True)
sel_period = st.selectbox("13F period", periods, index=0,
                          format_func=lambda d: d.isoformat())
sub = (
    holdings_commons.filter(pl.col("period") == sel_period)
    .select("ticker", "value")
    .sort("value", descending=True)
)
total = sub["value"].sum()
st.markdown(f"**Period:** {sel_period}  ·  **Total notional:** "
            f"${total/1e9:.2f}B  ·  **Positions:** {sub.height}")
hbar = go.Figure(go.Bar(
    x=sub["ticker"],
    y=sub["value"] / 1e6,
    marker_color="#1f77b4",
    text=[f"${v/1e6:,.0f}M" for v in sub["value"]],
    textposition="outside",
))
hbar.update_layout(
    height=500, yaxis_title="Notional ($M)", xaxis_title="",
    margin=dict(t=10, b=40),
    xaxis=dict(tickangle=-45),
)
st.plotly_chart(hbar, use_container_width=True)
