# Situational Awareness LP — 13F-based PnL dashboard

Streamlit app that reconstructs the performance of [Situational Awareness LP](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0002045724)
(Leopold Aschenbrenner's AI-infra hedge fund) from its public 13F-HR filings.

## Modes

- **50-Δ proxy options** — Calls / Puts scaled by 0.5
- **Commons only, period-end** — perfect-information, holdings effective day after each quarter end
- **Commons only, filing date** — realistic copycat, holdings effective day after each 13F is filed
- **Commons blended** — between two snapshots, hold 0.5 × earlier + 0.5 × later (smoothed perfect-info)
- **Equal-weight, 30-day vol-targeted** — every stock equal-weighted, scaled by `1 / 30d realised vol`

Each mode is compared against SMH (VanEck Semiconductor ETF), with the benchmark vol-matched to the strategy.

## Run locally

```bash
uv sync
uv run streamlit run app.py
```

## Refresh data

```bash
uv run python -m src.holdings   # re-fetch 13F filings
uv run python -m src.prices     # re-download Yahoo adjusted close
```

## Data sources

- 13F-HR filings: SEC EDGAR (live HTTP)
- Adjusted close prices: Yahoo Finance via `yfinance`
- CUSIP → ticker mapping: manual (`src/ticker_map.py`)
