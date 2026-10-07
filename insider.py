"""Form 4 cluster-buy score (Cohen/Malloy/Pomorski style) from SEC EDGAR, free.

score = (#distinct insiders with open-market buys in window) * (total $ bought) / (avg daily insider $ over 90d)
Best-effort: parses Form 4 XML, code 'P' (open-market purchase) only. Cached CIK map.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)
STATE = Path(__file__).resolve().parent / "state"
CIK_FILE = STATE / "cik_map.json"
UA = os.getenv("SEC_USER_AGENT", "optscan contact@example.com")
H = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}


def _get(url, **kw):
    r = requests.get(url, headers=H, timeout=20, **kw)
    r.raise_for_status()
    time.sleep(0.15)  # SEC rate limit ~10 req/s
    return r


def cik_for(ticker: str) -> str | None:
    if CIK_FILE.exists():
        m = json.loads(CIK_FILE.read_text())
    else:
        data = _get("https://www.sec.gov/files/company_tickers.json").json()
        m = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in data.values()}
        CIK_FILE.write_text(json.dumps(m))
    return m.get(ticker.upper())


def _form4_docs(cik: str, since: dt.date) -> list[tuple[str, str, str]]:
    """[(accession, primary_doc, filing_date)] for Form 4 filings since `since`."""
    sub = _get(f"https://data.sec.gov/submissions/CIK{cik}.json").json()
    rf = sub["filings"]["recent"]
    out = []
    for form, acc, doc, fdate in zip(rf["form"], rf["accessionNumber"], rf["primaryDocument"], rf["filingDate"]):
        if form != "4":
            continue
        if dt.date.fromisoformat(fdate) < since:
            break
        out.append((acc, doc, fdate))
    return out


def _parse_form4(cik: str, acc: str, doc: str) -> list[dict]:
    """Return open-market purchase transactions: [{owner, shares, price, usd, date}]."""
    acc_nodash = acc.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/"
    # primaryDocument may be .xml or an html wrapper; find the xml
    xml_name = doc if doc.endswith(".xml") else None
    if xml_name is None:
        idx = _get(base).text
        m = re.findall(r'href="[^"]*?([^"/]+\.xml)"', idx)
        xml_name = next((x for x in m if "primary" not in x.lower() and not x.startswith("0")), m[0] if m else None)
    if not xml_name:
        return []
    x = _get(base + xml_name).text
    owner = re.search(r"<rptOwnerName>(.*?)</rptOwnerName>", x)
    owner = owner.group(1).strip() if owner else "?"
    buys = []
    for blk in re.findall(r"<nonDerivativeTransaction>(.*?)</nonDerivativeTransaction>", x, re.S):
        code = re.search(r"<transactionCode>(\w)</transactionCode>", blk)
        if not code or code.group(1) != "P":
            continue
        ad = re.search(r"<transactionAcquiredDisposedCode>\s*<value>(\w)</value>", blk)
        if ad and ad.group(1) != "A":
            continue
        sh = re.search(r"<transactionShares>\s*<value>([\d.]+)</value>", blk)
        px = re.search(r"<transactionPricePerShare>\s*<value>([\d.]+)</value>", blk)
        d = re.search(r"<transactionDate>\s*<value>([\d-]+)</value>", blk)
        if not sh:
            continue
        shares = float(sh.group(1))
        price = float(px.group(1)) if px else 0.0
        buys.append({"owner": owner, "shares": shares, "price": price, "usd": shares * price,
                     "date": d.group(1) if d else ""})
    return buys


def cluster_score(ticker: str, window_days: int = 5, baseline_days: int = 90) -> dict:
    """Returns {score, n_insiders, usd_window, usd_daily_baseline, buys}."""
    empty = {"score": 0.0, "n_insiders": 0, "usd_window": 0.0, "usd_daily_baseline": 0.0, "buys": []}
    try:
        cik = cik_for(ticker)
        if not cik:
            return empty
        today = dt.date.today()
        docs = _form4_docs(cik, today - dt.timedelta(days=baseline_days))
        all_buys = []
        for acc, doc, fdate in docs:
            try:
                for b in _parse_form4(cik, acc, doc):
                    b["filed"] = fdate
                    all_buys.append(b)
            except Exception as e:
                log.debug("%s %s parse fail: %s", ticker, acc, e)
        if not all_buys:
            return empty
        win_start = today - dt.timedelta(days=window_days)
        win = [b for b in all_buys if dt.date.fromisoformat(b["filed"]) >= win_start]
        usd_win = sum(b["usd"] for b in win)
        n_ins = len({b["owner"] for b in win})
        base_daily = sum(b["usd"] for b in all_buys) / baseline_days
        score = (n_ins * usd_win / base_daily) if base_daily > 0 else 0.0
        return {"score": round(score, 1), "n_insiders": n_ins, "usd_window": round(usd_win, 0),
                "usd_daily_baseline": round(base_daily, 0), "buys": win}
    except Exception as e:
        log.warning("insider %s failed: %s", ticker, e)
        return empty
