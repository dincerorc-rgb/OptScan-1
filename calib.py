"""
calib.py — shared core for the OptScan calibration add-on.

  * parse_signals(text, source)  -> list of signal dicts pulled from a Telegram message
  * gate(signal, book)           -> list of (level, reason) flags   [level: BLOCK / WARN / OK]
  * invalidation(signal)         -> price that kills the thesis
  * log_signals(signals)         -> append (de-duplicated) to state/signals.csv
  * send_telegram(text)          -> plain sendMessage using the repo's existing secrets

Nothing here places orders. It only records, grades and warns.
"""
from __future__ import annotations

import csv
import datetime as dt
import os
import re
from pathlib import Path

STATE = Path(os.environ.get("CALIB_STATE", "state"))
SIGNALS = STATE / "signals.csv"
UNPARSED = STATE / "unparsed.txt"
BOOK = Path(os.environ.get("CALIB_BOOK", "book.csv"))

# Prior probability that a signal of this grade "works" (hits target before stop,
# or is in profit at +5 sessions when no target/stop was given).
# These are ASSUMPTIONS used for the Brier score — the weekly review shows the
# realised hit rate next to them so you can see whether the grades mean anything.
GRADE_P = {"A": 0.70, "B": 0.60, "C": 0.52, "D": 0.50, "S": 0.55}

# ---- risk-gate limits (edit here) -------------------------------------------
MAX_PREMIUM_PCT = 0.03    # single new position premium <= 3% of net liq
MAX_TICKER_PCT = 0.25     # existing exposure in one ticker > 25% -> BLOCK adds
MAX_OPEN_PCT = 0.60       # total open premium > 60% of net liq -> BLOCK new debits
LONG_SKEW_WARN = 0.80     # >=80% of book premium bullish and signal is CALL -> WARN
ATR_MULT = 1.5            # fallback invalidation = entry -/+ 1.5 x ATR14

FIELDS = ["ts_utc", "date", "source", "ticker", "direction", "grade", "setup",
          "spot", "entry", "target", "stop", "invalid", "p", "raw"]

SETUPS = ("BREAKOUT", "COIL", "PULLBACK", "EXTENDED", "BEARDOWN")
_NUM = r"\$?(\d{1,6}(?:\.\d{1,4})?)"
_KV = {
    "entry": re.compile(r"\b(?:entry|ent|buy|@)\s*[:=]?\s*" + _NUM, re.I),
    "target": re.compile(r"\b(?:target|tgt|tp1?|t1)\s*[:=]?\s*" + _NUM, re.I),
    "stop": re.compile(r"\b(?:stop|sl|stp|inval(?:id)?)\s*[:=<]?\s*" + _NUM, re.I),
    "spot": re.compile(r"\b(?:spot|px|last|close)\s*[:=]?\s*" + _NUM, re.I),
}
_TICKER = re.compile(r"(?:^|[\s*_`•\-|▶►#])\$?([A-Z]{1,5}(?:\.[A-Z])?)\b")
_VERDICT = re.compile(r"\b(CALL|PUT|NONE)\b\s*[:\-–(]?\s*(?:grade\s*)?\(?([ABCD])?\b", re.I)
_STOPWORDS = {"CALL", "PUT", "NONE", "GEX", "OI", "PC", "IV", "ATM", "OTM", "ITM", "DTE",
              "ET", "UTC", "AEST", "TP", "SL", "BUY", "SELL", "MAP", "MAX", "PAIN", "FLIP",
              "WALL", "A", "B", "C", "D", "S", "US", "USD", "NEW", "IVR", "RSI", "EMA",
              "ATR", "BE", "R", "OK", "WARN", "BLOCK", "GRADE", "TOP", "LOW", "HIGH"} | set(SETUPS)


def _today() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


def _first_ticker(line: str) -> str | None:
    for m in _TICKER.finditer(line):
        t = m.group(1)
        if t not in _STOPWORDS:
            return t
    return None


def parse_signals(text: str, source: str) -> list[dict]:
    """Pull CALL/PUT (options-map) or setup (scanner) signals out of one message.

    Works line by line: a line needs a ticker plus either a CALL/PUT verdict or a
    scanner setup keyword. Numbers for entry/target/stop/spot are optional and are
    picked up from the same line or the next two lines (multi-line layouts).
    """
    out: list[dict] = []
    lines = [l for l in text.splitlines()]
    for i, line in enumerate(lines):
        up = line.upper()
        direction = grade = setup = None
        v = _VERDICT.search(line)
        if v and v.group(1).upper() in ("CALL", "PUT"):
            direction, grade = v.group(1).upper(), (v.group(2) or "").upper() or None
        else:
            for s in SETUPS:
                if re.search(rf"\b{s}\b", up):
                    setup = s
                    direction = "PUT" if s == "BEARDOWN" else "CALL"
                    if s == "EXTENDED":          # extended = don't chase; log but no edge claim
                        direction = None
                    break
        if not direction:
            continue
        ticker = _first_ticker(line)
        # ticker may sit on a header line just above ("PLTR 174.2\n → CALL B ...")
        j = i - 1
        while ticker is None and j >= max(0, i - 2):
            ticker = _first_ticker(lines[j]); j -= 1
        if not ticker:
            continue
        ctx = " ".join(lines[i:i + 3])
        # stop scanning context at the next ticker line so levels don't bleed across tickers
        nums = {}
        for k, rx in _KV.items():
            m = rx.search(line) or rx.search(ctx)
            if m:
                nums[k] = float(m.group(1))
        g = grade or ("S" if setup else "C")
        out.append({
            "ts_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
            "date": _today(), "source": source, "ticker": ticker, "direction": direction,
            "grade": g, "setup": setup or "", "spot": nums.get("spot", ""),
            "entry": nums.get("entry", ""), "target": nums.get("target", ""),
            "stop": nums.get("stop", ""), "invalid": "", "p": GRADE_P.get(g, 0.5),
            "raw": line.strip()[:160],
        })
    # de-dup within a message (same ticker+direction)
    seen, uniq = set(), []
    for s in out:
        k = (s["ticker"], s["direction"])
        if k not in seen:
            seen.add(k); uniq.append(s)
    return uniq


# ---- invalidation -----------------------------------------------------------
def invalidation(sig: dict, atr: float | None = None) -> tuple[float | None, str]:
    """Stop from the message if given, else entry/spot -/+ ATR_MULT x ATR14."""
    if sig.get("stop") not in ("", None):
        return float(sig["stop"]), "stop"
    ref = sig.get("entry") or sig.get("spot")
    if ref in ("", None) or not atr:
        return None, "n/a"
    ref = float(ref)
    lvl = ref - ATR_MULT * atr if sig["direction"] == "CALL" else ref + ATR_MULT * atr
    return round(lvl, 2), f"{ATR_MULT}xATR"


# ---- book + gate ------------------------------------------------------------
def load_book(path: Path = BOOK) -> dict | None:
    """book.csv:
         netliq,25000
         ticker,bias,premium_usd
         AAPL,long,1450
         QQQ,short,2800
       bias = long (bullish delta) / short (bearish delta / hedge)
    """
    if not path.exists():
        return None
    netliq, pos = None, []
    with path.open(newline="") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip() != ""]
            if not row or row[0].startswith("#"):
                continue
            if row[0].lower() == "netliq":
                netliq = float(row[1].replace(",", "").replace("$", ""))
            elif row[0].lower() == "ticker":
                continue
            elif len(row) >= 3:
                pos.append({"ticker": row[0].upper(), "bias": row[1].lower(),
                            "premium": float(row[2].replace(",", "").replace("$", ""))})
    if not netliq:
        return None
    return {"netliq": netliq, "positions": pos}


def gate(sig: dict, book: dict | None) -> list[tuple[str, str]]:
    flags: list[tuple[str, str]] = []
    g = sig.get("grade", "C")
    if g == "D":
        flags.append(("BLOCK", "grade D — skip"))
    elif g == "C":
        flags.append(("WARN", "grade C — half size max"))
    if sig.get("target") not in ("", None) and sig.get("stop") not in ("", None):
        ref = sig.get("entry") or sig.get("spot")
        if ref not in ("", None):
            ref, tg, st = float(ref), float(sig["target"]), float(sig["stop"])
            risk, reward = abs(ref - st), abs(tg - ref)
            if risk > 0 and reward / risk < 1.5:
                flags.append(("WARN", f"R:R {reward/risk:.1f} < 1.5"))
    if not book:
        flags.append(("WARN", "book.csv not set — only grade checks applied"))
        return flags
    nl = book["netliq"]
    pos = book["positions"]
    cap = nl * MAX_PREMIUM_PCT
    open_prem = sum(p["premium"] for p in pos)
    tick = sum(p["premium"] for p in pos if p["ticker"] == sig["ticker"])
    longp = sum(p["premium"] for p in pos if p["bias"] == "long")
    if open_prem > nl * MAX_OPEN_PCT:
        flags.append(("BLOCK", f"open premium {open_prem/nl:.0%} of net liq > {MAX_OPEN_PCT:.0%}"))
    if tick > nl * MAX_TICKER_PCT:
        flags.append(("BLOCK", f"{sig['ticker']} already {tick/nl:.0%} of net liq"))
    elif tick > 0:
        flags.append(("WARN", f"already hold {sig['ticker']} (${tick:,.0f})"))
    if sig["direction"] == "CALL" and open_prem and longp / open_prem >= LONG_SKEW_WARN:
        flags.append(("WARN", f"book {longp/open_prem:.0%} bullish — no hedge"))
    flags.append(("OK", f"max premium ${cap:,.0f} ({MAX_PREMIUM_PCT:.0%})"))
    return flags


def verdict(flags) -> str:
    lv = {f[0] for f in flags}
    return "⛔" if "BLOCK" in lv else ("⚠️" if "WARN" in lv else "✅")


# ---- storage ----------------------------------------------------------------
def _read(path: Path = SIGNALS) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def log_signals(sigs: list[dict], path: Path = SIGNALS) -> list[dict]:
    """Append; skip anything already logged today for same source+ticker+direction."""
    STATE.mkdir(parents=True, exist_ok=True)
    have = {(r["date"], r["source"], r["ticker"], r["direction"]) for r in _read(path)}
    new = [s for s in sigs if (s["date"], s["source"], s["ticker"], s["direction"]) not in have]
    if not new:
        return []
    exists = path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerows(new)
    return new


def send_telegram(text: str) -> bool:
    tok, chat = os.environ.get("TELEGRAM_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        print("[calib] no Telegram secrets — message follows\n" + text)
        return False
    import json
    import urllib.request
    ok = True
    for i in range(0, len(text), 3900):            # Telegram limit 4096
        body = json.dumps({"chat_id": chat, "text": text[i:i + 3900],
                           "disable_web_page_preview": True}).encode()
        req = urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage",
                                     data=body, headers={"Content-Type": "application/json"})
        req._calib_skip = True                      # don't let the wrapper re-capture it
        try:
            urllib.request.urlopen(req, timeout=20).read()
        except Exception as e:                      # never break the run over a message
            print("[calib] telegram send failed:", e); ok = False
    return ok
