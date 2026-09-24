"""
calib_run.py — run an existing OptScan script unchanged and calibrate its output.

    python calib_run.py options_map.py [args...]
    python calib_run.py optscan.py scan

What happens:
  1. Telegram sends made by the script (requests or urllib) are captured — they
     still go out exactly as before.
  2. Every CALL/PUT (options-map) or setup (scanner) line is parsed and logged to
     state/signals.csv with a grade-implied probability.
  3. One extra Telegram message: "🛡 Gate" — per new signal, the invalidation level
     and the risk-gate verdict against book.csv.
The script's exit code is preserved. Any calib error is printed, never raised.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import traceback
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402

CAPTURED: list[str] = []


def _grab(url: str, payload) -> None:
    if "api.telegram.org" not in str(url):
        return
    if not any(m in str(url) for m in ("sendMessage", "sendPhoto", "sendDocument", "editMessageText")):
        return
    try:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8", "ignore")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = dict(urllib.parse.parse_qsl(payload))
        if isinstance(payload, dict):
            txt = payload.get("text") or payload.get("caption")
            if txt:
                CAPTURED.append(str(txt))
    except Exception:
        pass


def _patch() -> None:
    # requests (covers requests.post / requests.get / Session)
    try:
        import requests

        orig = requests.Session.request

        def request(self, method, url, *a, **kw):
            _grab(url, kw.get("json") or kw.get("data") or kw.get("params"))
            return orig(self, method, url, *a, **kw)

        requests.Session.request = request
    except ImportError:
        pass
    # urllib
    import urllib.request as ur

    orig_open = ur.urlopen

    def urlopen(req, *a, **kw):
        if not getattr(req, "_calib_skip", False):
            if isinstance(req, ur.Request):
                _grab(req.full_url, req.data)
            else:
                _grab(req, kw.get("data") or (a[0] if a else None))
        return orig_open(req, *a, **kw)

    ur.urlopen = urlopen


def _atr(ticker: str) -> float | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period="2mo", interval="1d", progress=False, auto_adjust=False)
        if df is None or len(df) < 15:
            return None
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        h, l, c = df["High"], df["Low"], df["Close"]
        tr = (h - l).combine((h - c.shift()).abs(), max).combine((l - c.shift()).abs(), max)
        return float(tr.tail(14).mean())
    except Exception:
        return None


def _spot(ticker: str) -> float | None:
    try:
        import yfinance as yf
        df = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
        if hasattr(df.columns, "levels"):
            df.columns = df.columns.get_level_values(0)
        return round(float(df["Close"].iloc[-1]), 2)
    except Exception:
        return None


def post_process(source: str, atr_fn=_atr, spot_fn=_spot) -> str | None:
    sigs = []
    for msg in CAPTURED:
        sigs += calib.parse_signals(msg, source)
    if not sigs:
        if any(k in m for m in CAPTURED for k in ("CALL", "PUT", "BREAKOUT", "BEARDOWN")):
            calib.STATE.mkdir(parents=True, exist_ok=True)
            with calib.UNPARSED.open("a") as f:
                f.write(f"\n=== {calib._today()} {source}\n" + "\n---\n".join(CAPTURED))
            print("[calib] signals seen but not parsed — saved to", calib.UNPARSED)
        return None
    # de-dup across messages
    seen, uniq = set(), []
    for s in sigs:
        k = (s["ticker"], s["direction"])
        if k not in seen:
            seen.add(k); uniq.append(s)
    book = calib.load_book()
    lines = []
    for s in uniq:
        if s["spot"] in ("", None):
            sp = spot_fn(s["ticker"])
            if sp:
                s["spot"] = sp
        lvl, how = calib.invalidation(s, None if s["stop"] not in ("", None) else atr_fn(s["ticker"]))
        s["invalid"] = lvl if lvl is not None else ""
        flags = calib.gate(s, book)
        side = "<" if s["direction"] == "CALL" else ">"
        inv = f"invalid {side}{lvl} ({how})" if lvl is not None else "invalid n/a"
        tag = s["setup"] or s["grade"]
        worst = [r for lv, r in flags if lv != "OK"]
        okcap = [r for lv, r in flags if lv == "OK"]
        lines.append(f"{calib.verdict(flags)} {s['ticker']} {s['direction']} {tag} | {inv}"
                     + (f"\n   " + "; ".join(worst) if worst else "")
                     + (f"\n   {okcap[0]}" if okcap and "BLOCK" not in {f[0] for f in flags} else ""))
    new = calib.log_signals(uniq)
    head = f"🛡 Gate — {source} — {len(uniq)} signal(s), {len(new)} new logged"
    return head + "\n" + "\n".join(lines)


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__); return 2
    script = sys.argv[1]
    source = os.environ.get("CALIB_SOURCE") or Path(script).stem
    _patch()
    sys.argv = [script] + sys.argv[2:]
    code = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception:
        traceback.print_exc(); code = 1
    try:
        msg = post_process(source)
        if msg:
            print(msg)
            calib.send_telegram(msg)
    except Exception:
        print("[calib] post-process failed (scanner output unaffected):")
        traceback.print_exc()
    return code


if __name__ == "__main__":
    sys.exit(main())
