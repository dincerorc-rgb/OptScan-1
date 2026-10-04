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
BOOK = Path(os.environ.get("CALIB_BOOK", Path(__file__).resolve().parent / "book.csv"))

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
RR_BLOCK = 2.0            # Minervini: reward:risk below 2 -> BLOCK
RR_WARN = 3.0             #            below 3 -> WARN (aim for 3:1+)
MAX_STOP_PCT = 8.0        # stop more than 8% under entry -> BLOCK

FIELDS = ["ts_utc", "date", "source", "ticker", "direction", "grade", "setup",
          "spot", "entry", "target", "stop", "invalid", "p", "raw"]

SETUPS = ("BREAKOUT", "COIL", "PULLBACK", "EXTENDED", "BEARDOWN")
_NUM = r"\$?(\d{1,6}(?:\.\d{1,4})?)"
_KV = {
    "entry": re.compile(r"\b(?:entry|ent|buy|pivot|@)\s*[:=≤≥<>]?\s*" + _NUM, re.I),
    "target": re.compile(r"\b(?:target|tgt|tp1?|t1)\s*[:=≤≥<>]?\s*" + _NUM, re.I),
    "stop": re.compile(r"\b(?:stop|sl|stp|inval(?:id)?)\s*[:=≤≥<>]?\s*" + _NUM, re.I),
    "spot": re.compile(r"\b(?:spot|px|last|close)\s*[:=]?\s*" + _NUM, re.I),
}
_TICKER = re.compile(r"(?:^|[\s*_`•\-|▶►#])\$?([A-Z]{1,5}(?:\.[A-Z])?)\b")
_VERDICT = re.compile(r"\b(CALL|PUT|NONE)\b\s*[:\-–(]?\s*(?:grade\s*)?\(?([ABCD])?\b", re.I)
_STOPWORDS = {"CALL", "PUT", "NONE", "GEX", "OI", "PC", "IV", "ATM", "OTM", "ITM", "DTE",
              "ET", "UTC", "AEST", "TP", "SL", "BUY", "SELL", "MAP", "MAX", "PAIN", "FLIP",
              "WALL", "A", "B", "C", "D", "S", "US", "USD", "NEW", "IVR", "RSI", "EMA",
              "ATR", "BE", "R", "OK", "WARN", "BLOCK", "GRADE", "TOP", "LOW", "HIGH", "PCR", "BT", "ADX", "RV", "CW", "PW", "MP", "P", "SPOT", "UTC", "OPTIONS", "WATCH", "LONG", "SHORT", "BULL", "BEAR", "NET", "VOL", "AND", "SEPA", "VCP", "TT", "RS", "SPY", "QM", "STOP", "MOVE", "EXIT", "TRIM", "TGT", "E"} | set(SETUPS)


def _today() -> str:
    return dt.datetime.utcnow().strftime("%Y-%m-%d")


_PREFIX = re.compile(r"^[^\w$]*(?:BUY(?:\s+PUTS)?|SELL/[A-Z]+)\s+")
_BAYES = re.compile(r"Bayes\s+\d{1,3}%→(\d{1,3})%")
_HEADER = re.compile(r"^[\s*_`•\-|▶►#>\d.)]*\$?([A-Z]{1,5}(?:\.[A-Z])?)\b[\s*_`:]*(?:spot\s*)?\$?(\d{1,6}(?:\.\d{1,4})?)?")


def _header(line: str):
    """A line that STARTS with a ticker (e.g. '*PLTR*  spot 174.2', 'MU 1012.00 BREAKOUT')."""
    line = _PREFIX.sub("", line)   # scanner lines start '🟢 BUY NVDA ...' / '🔴 BUY PUTS ...'
    m = _HEADER.match(line)
    if not m or m.group(1) in _STOPWORDS:
        return None, None
    return m.group(1), (float(m.group(2)) if m.group(2) else None)


def _clean(text: str) -> str:
    import html
    return re.sub(r"[*_`]", "", html.unescape(re.sub(r"</?(?:b|i|u|s|code|pre|a|strong|em)(?:\s[^<>\n]*)?>", "", text)))


def parse_signals(text: str, source: str) -> list[dict]:
    """Pull CALL/PUT (options-map) or setup (scanner) signals out of one message.

    Block-based: a header line that starts with a ticker opens a block; a CALL/PUT
    verdict or scanner setup keyword inside that block becomes a signal for that
    ticker. entry/target/stop/spot numbers are read from the rest of the block.
    """
    lines = _clean(text).splitlines()
    blocks, cur = [], None
    for line in lines:
        if re.match(r"^[^\w$]*SELL/", line):     # SELL/EXIT/TRIM lines are exits, not new signals
            cur = None
            continue
        t, sp = _header(line)
        if t:
            cur = {"ticker": t, "spot": sp, "lines": [line]}
            blocks.append(cur)
        elif cur is not None:
            cur["lines"].append(line)
    out: list[dict] = []
    for blk in blocks:
        body = "\n".join(blk["lines"])
        direction = grade = setup = None
        for line in blk["lines"]:
            up = line.upper()
            for s in SETUPS:
                if re.search(rf"\b{s}\b", up) and not setup:
                    setup = s
                    if s == "EXTENDED":
                        continue
                    direction = "PUT" if (s == "BEARDOWN" or re.search(r"\b(short|bear)", line, re.I)) else "CALL"
            v = _VERDICT.search(line)
            if v and v.group(1).upper() in ("CALL", "PUT") and not direction:
                direction, grade = v.group(1).upper(), (v.group(2) or "").upper() or None
        if not direction:
            continue
        nums = {}
        for k, rx in _KV.items():
            m = rx.search(body)
            if m:
                nums[k] = float(m.group(1))
        spot = nums.get("spot", blk["spot"])
        g = grade or ("S" if setup else "C")
        if setup and setup != "BEARDOWN":
            setup += "/SEPA" if re.search(r"SEPA ✅|VCP/SEPA", body) else "/QM"
        bm = _BAYES.search(body)          # scanner posterior beats the grade-assumed p
        out.append({
            "ts_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%MZ"),
            "date": _today(), "source": source, "ticker": blk["ticker"], "direction": direction,
            "grade": g, "setup": setup or "", "spot": spot if spot is not None else "",
            "entry": nums.get("entry", ""), "target": nums.get("target", ""),
            "stop": nums.get("stop", ""), "invalid": "",
            "p": round(int(bm.group(1)) / 100, 2) if bm else GRADE_P.get(g, 0.5),
            "raw": blk["lines"][0].strip()[:160],
        })
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
def _num(x):
    try:
        return float(str(x).replace(",", "").replace("$", ""))
    except ValueError:
        return None


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
                            "premium": float(row[2].replace(",", "").replace("$", "")),
                            "label": row[3] if len(row) > 3 else row[0].upper(),
                            "entry": _num(row[4]) if len(row) > 5 else None,
                            "stop": _num(row[5]) if len(row) > 5 else None})
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
            if risk > 0:
                rr = reward / risk
                rr = round(rr, 2)
                if rr < RR_BLOCK:
                    flags.append(("BLOCK", f"R:R {rr:.1f} < {RR_BLOCK:g}"))
                elif rr < RR_WARN:
                    flags.append(("WARN", f"R:R {rr:.1f} < {RR_WARN:g}"))
    if sig.get("setup") and sig.get("stop") not in ("", None):
        ref = sig.get("entry") or sig.get("spot")
        if ref not in ("", None) and float(ref) > 0:
            w = 100 * abs(float(ref) - float(sig["stop"])) / float(ref)
            if w > MAX_STOP_PCT:
                flags.append(("BLOCK", f"stop {w:.1f}% away > {MAX_STOP_PCT:g}%"))
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


def git_save(paths: list[str], msg: str) -> None:
    """Commit + push the log from inside GitHub Actions (no extra workflow step needed).
    Needs repo Settings -> Actions -> General -> Workflow permissions = Read and write."""
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    import subprocess as sp

    def run(*a):
        return sp.run(list(a), capture_output=True, text=True)
    run("git", "config", "user.name", "optscan-bot")
    run("git", "config", "user.email", "optscan-bot@users.noreply.github.com")
    for p in paths:
        if Path(p).exists():
            run("git", "add", "-f", p)
    if run("git", "diff", "--cached", "--quiet").returncode == 0:
        return
    run("git", "commit", "-m", f"{msg} [skip ci]")
    for _ in range(3):
        run("git", "pull", "--rebase", "-X", "theirs")
        r = run("git", "push")
        if r.returncode == 0:
            print("[calib] log saved to repo"); return
        import time; time.sleep(5)
    print("[calib] could not push log:", r.stderr.strip()[:300],
          "\n -> Settings > Actions > General > Workflow permissions = Read and write")
