"""breakout_watch.py — Qullamaggie-style intraday breakout alerts -> Telegram.

Polls 1-min bars every POLL_SEC during the US regular session and fires:
  ✅ BREAKOUT      price > pivot AND volume pace >= VOL_PACE_MIN
  ⚠️ LOW-VOL BREAK price > pivot but pace below threshold (upgrades to ✅ if volume arrives)
  ❌ FAILED        after a break alert, price back below pivot by > FAIL_PCT
Run with --once from a 15/30-min cron (cheap on Actions minutes); without it, loops all session.
Each alert fires once per ticker per day (state kept in state/bw_state.json).

Config: watchlist.json  {"expires":"YYYY-MM-DD","tickers":{"CRWD":{"pivot":268.95},...}}
Env:    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID   (same secrets as the scanner)
"""
import json, os, sys, time
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import requests
import yfinance as yf

ET = ZoneInfo("America/New_York")
POLL_SEC = int(os.getenv("POLL_SEC", "60"))
MAX_RUN_MIN = int(os.getenv("MAX_RUN_MIN", "345"))       # stay under GitHub's 6h job cap
VOL_PACE_MIN = float(os.getenv("VOL_PACE_MIN", "1.5"))
FAIL_PCT = float(os.getenv("FAIL_PCT", "0.5"))
STATE = "bwstate/bw_state.json"

# cumulative share of a normal day's volume by minute-of-session (U-shaped curve)
CURVE = [(0, 0.0), (30, 0.13), (60, 0.21), (120, 0.33), (240, 0.55), (360, 0.84), (390, 1.0)]


def expected_frac(m):
    m = max(1, min(390, m))
    for (x0, y0), (x1, y1) in zip(CURVE, CURVE[1:]):
        if m <= x1:
            return y0 + (y1 - y0) * (m - x0) / (x1 - x0)
    return 1.0


def tg(text):
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    print(text, flush=True)
    if not tok or not chat:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                      json={"chat_id": chat, "text": text}, timeout=15)
    except Exception as e:
        print("telegram error:", e, flush=True)


def load_state(today):
    try:
        s = json.load(open(STATE))
        if s.get("date") == today:
            return s
    except Exception:
        pass
    return {"date": today, "fired": {}}


def save_state(s):
    os.makedirs("bwstate", exist_ok=True)
    json.dump(s, open(STATE, "w"), indent=1)


def avg_daily_vol(t):
    d = yf.Ticker(t).history(period="3mo", interval="1d", auto_adjust=False)
    d = d[d.index.date < datetime.now(ET).date()]
    return float(d["Volume"].tail(20).mean())


def snapshot(t):
    b = yf.Ticker(t).history(period="1d", interval="1m", prepost=False, auto_adjust=False)
    if b.empty:
        return None
    b = b[b.index.date == datetime.now(ET).date()]
    if b.empty:
        return None
    first5 = b.iloc[:5]
    return {"last": float(b["Close"].iloc[-1]), "high": float(b["High"].max()),
            "lod": float(b["Low"].min()), "cumvol": float(b["Volume"].sum()),
            "orh": float(first5["High"].max()), "bar": b.index[-1].strftime("%H:%M")}


def check(t, cfg, avgv, st, mins):
    s = snapshot(t)
    if not s:
        return
    pivot = cfg["pivot"]
    pace = s["cumvol"] / (avgv * expected_frac(mins)) if avgv else 0
    f = st["fired"].setdefault(t, [])
    over = (s["last"] / pivot - 1) * 100
    risk = (s["last"] / s["lod"] - 1) * 100
    core = (f"{t} {s['last']:.2f}  (pivot {pivot:.2f}, {over:+.2f}%)\n"
            f"Vol pace {pace:.1f}x  |  ORH {s['orh']:.2f}  |  LOD/stop {s['lod']:.2f} (risk {risk:.1f}%)\n"
            f"{s['bar']} ET")

    if s["last"] > pivot:
        if pace >= VOL_PACE_MIN and "confirmed" not in f:
            tg(f"✅ BREAKOUT — {core}\nEntry: hold above ORH; stop = LOD. Premium ≤1.5–2%.")
            f.append("confirmed")
        elif pace < VOL_PACE_MIN and "lowvol" not in f and "confirmed" not in f:
            tg(f"⚠️ LOW-VOL BREAK — {core}\nNot confirmed (need ≥{VOL_PACE_MIN}x). Wait.")
            f.append("lowvol")
    elif ("confirmed" in f or "lowvol" in f) and "failed" not in f \
            and s["last"] < pivot * (1 - FAIL_PCT / 100):
        tg(f"❌ FAILED — back below pivot\n{core}")
        f.append("failed")


def main():
    cfg = json.load(open("watchlist.json"))
    now = datetime.now(ET)
    if cfg.get("expires") and now.date() > date.fromisoformat(cfg["expires"]):
        print("watchlist expired — nothing to do"); return
    if now.weekday() >= 5:
        print("weekend"); return

    tickers = cfg["tickers"]
    if "--test" in sys.argv:
        tg("🧪 breakout-watch test OK — " + ", ".join(f"{k}>{v['pivot']}" for k, v in tickers.items()))
        return

    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    stop_t = min(close_t, now + timedelta(minutes=MAX_RUN_MIN))
    once = "--once" in sys.argv   # one pass and exit: cron polls, no paid minutes spent sleeping
    if now >= close_t:
        print("session over"); return
    if now < open_t:
        if once:
            print("pre-market — nothing to do"); return
        time.sleep((open_t - now).total_seconds())

    st = load_state(now.date().isoformat())
    avg = {}
    for t in tickers:
        try:
            avg[t] = avg_daily_vol(t)
        except Exception as e:
            print(t, "avg vol error", e); avg[t] = 0
    if not st.get("hello"):
        tg("👀 Watching: " + " | ".join(
            f"{t} > {c['pivot']:.2f} (avg vol {avg[t]/1e6:.1f}M)" for t, c in tickers.items()))
        st["hello"] = True; save_state(st)

    while datetime.now(ET) < stop_t:
        mins = (datetime.now(ET) - open_t).total_seconds() / 60
        for t, c in tickers.items():
            if "confirmed" in st["fired"].get(t, []) and "failed" in st["fired"][t]:
                continue
            try:
                check(t, c, avg[t], st, mins)
            except Exception as e:
                print(t, "error", e, flush=True)
        save_state(st)
        if once:
            break
        time.sleep(POLL_SEC)
    print("loop end", datetime.now(ET))


if __name__ == "__main__":
    main()
