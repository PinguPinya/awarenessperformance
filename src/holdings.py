"""Fetch and parse 13F filings for Situational Awareness LP.

Calls/Puts are treated as if they were the underlying common (shares summed
per CUSIP per period).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import httpx
import polars as pl
import structlog

logger = structlog.get_logger(__name__)

CIK = "0002045724"
ENTITY = "Situational Awareness LP"
UA = "PnL Research barre.ant@gmail.com"
SUBMISSIONS_URL = f"https://data.sec.gov/submissions/CIK{CIK}.json"
ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
NS = {"n": "http://www.sec.gov/edgar/document/thirteenf/informationtable"}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HOLDINGS_PATH = DATA_DIR / "holdings.parquet"


def list_13f_filings() -> list[dict]:
    r = httpx.get(SUBMISSIONS_URL, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    recent = r.json()["filings"]["recent"]
    out = []
    for form, fdate, acc, period in zip(
        recent["form"], recent["filingDate"], recent["accessionNumber"], recent["reportDate"]
    ):
        if form.startswith("13F"):
            out.append({"form": form, "filing_date": fdate, "period": period, "accession": acc})
    return out


def _find_infotable_xml(accession: str) -> str:
    no_dash = accession.replace("-", "")
    base = f"{ARCHIVES}/{int(CIK)}/{no_dash}"
    idx = httpx.get(f"{base}/", headers={"User-Agent": UA}, timeout=20).text
    import re
    xmls = re.findall(r'href="(/Archives[^"]+\.xml)"', idx)
    for x in xmls:
        if "primary_doc" not in x:
            return "https://www.sec.gov" + x
    raise RuntimeError(f"no infoTable xml in {accession}")


def parse_infotable(url: str) -> list[dict]:
    xml = httpx.get(url, headers={"User-Agent": UA}, timeout=30).text
    root = ET.fromstring(xml)
    rows = []
    for it in root.findall("n:infoTable", NS):
        rows.append({
            "name": it.findtext("n:nameOfIssuer", "", NS).strip(),
            "class": it.findtext("n:titleOfClass", "", NS).strip(),
            "cusip": it.findtext("n:cusip", "", NS).strip(),
            "value": int(it.findtext("n:value", "0", NS)),
            "shares": int(it.find("n:shrsOrPrnAmt/n:sshPrnamt", NS).text),
            "shares_type": it.find("n:shrsOrPrnAmt/n:sshPrnamtType", NS).text,
            "put_call": (it.findtext("n:putCall", "", NS) or "").strip() or "SH",
        })
    return rows


def _normalize_name(name: str) -> str:
    s = name.upper().strip()
    # Strip trailing legal suffixes and noise
    for suffix in [" CORPORATION", " CORP", " INCORPORATED", " INC", " LTD",
                   " LIMITED", " HOLDINGS", " HLDGS", " HLDG", " PLC", " CO",
                   " GROUP", " TECHNOLOGIES", " TECHNOLOGY", " ENTERPRISES",
                   " RLTY", " MFG"]:
        if s.endswith(suffix):
            s = s[: -len(suffix)].strip()
    return s.replace("&", "AND").replace(".", "").replace(",", "").strip()


def build_holdings() -> pl.DataFrame:
    """All quarters, aggregated by normalized issuer per period.

    Options (Call/Put) and convertible-class CUSIPs are collapsed into the
    underlying common: SH and Call rows count positively, Put rows count
    negatively (net signed shares per period/issuer).
    """
    filings = list_13f_filings()
    logger.info("filings_found", n=len(filings))
    all_rows: list[dict] = []
    for f in filings:
        url = _find_infotable_xml(f["accession"])
        rows = parse_infotable(url)
        for r in rows:
            r["period"] = f["period"]
            r["filing_date"] = f["filing_date"]
            r["issuer"] = _normalize_name(r["name"])
            # Sign: Put = short, Call/SH = long
            sign = -1 if r["put_call"] == "Put" else 1
            r["net_shares"] = sign * r["shares"]
            r["net_value"] = sign * r["value"]
        all_rows.extend(rows)
        logger.info("parsed", period=f["period"], n_rows=len(rows))

    df = pl.DataFrame(all_rows)
    agg = (
        df.group_by("period", "issuer")
        .agg(
            pl.col("name").first().alias("raw_name"),
            pl.col("cusip").unique().sort().str.join(",").alias("cusips"),
            # Net (signed) and gross views of position
            pl.col("net_shares").sum().alias("shares"),
            pl.col("net_value").sum().alias("value"),
            pl.col("value").sum().alias("gross_value"),
            pl.col("put_call").unique().sort().str.join("+").alias("put_call_mix"),
            pl.col("filing_date").first(),
        )
        .with_columns(pl.col("period").str.to_date())
        .sort("period", "value", descending=[False, True])
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(HOLDINGS_PATH)
    logger.info("holdings_saved", path=str(HOLDINGS_PATH), rows=agg.height)
    return agg


if __name__ == "__main__":
    df = build_holdings()
    print(df)
    print("\nPer-period totals:")
    print(df.group_by("period").agg(pl.len(), pl.col("value").sum()).sort("period"))
