"""
calib_score.py — score logged signals against what price actually did.

    python calib_score.py score     # daily: (re)score every signal -> state/scored.csv
    python calib_score.py review    # weekly: score + send the calibration report to Telegram
    python calib_score.py review --dry   # print only

Outcome per signal (1 = worked, 0 = failed):
  * target AND stop known -> 1 if target is touched before stop within 20 sessions
    (both in one day counts as stop — conservative); if neither by session 20,
    1 if the close is in the signal's favour.
  * otherwise              -> 1 if the directional return at +5 sessions is > 0.
Brier = mean((p - outcome)^2). A coin flip scores 0.25; lower is better.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib  # noqa: E402

SCORED = calib.STATE / "scored.csv"
HORIZON = 20
SFIELDS = ["key", "date", "source", "ticker", "direction", "grade", "setup", "p",
           "ref", "r5", "r20", "invalidated", "outcome", "status", "sessions"]


def fetch_yf(ticker: str, start: str):
    import yfinance as yf
    df = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    return [(i.strftime("%Y-%m-%d"), float(r.Open), float(r.High), float(r.Low), float(r.Close))
            for i, r in df.iterrows()]


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def score_one(sig: dict, bars: list) -> dict:
    d = sig["direction"]
    sgn = 1 if d == "CALL" else -1
    after = [b for b in bars if b[0] > sig["date"]][:HORIZON]
    ref = _f(sig.get("entry")) or _f(sig.get("spot"))
    if ref is None:
        same = [b for b in bars if b[0] <= sig["date"]]
        ref = same[-1][4] if same else (after[0][1] if after else None)
    out = {"key": "|".join([sig["date"], sig["source"], sig["ticker"], d]),
           **{k: sig.get(k, "") for k in ("date", "source", "ticker", "direction", "grade", "setup", "p")},
           "ref": round(ref, 2) if ref else ref, "r5": "", "r20": "", "invalidated": "", "outcome": "",
           "status": "open", "sessions": len(after)}
    if ref is None or not after:
        return out
    if len(after) >= 5:
        out["r5"] = round(sgn * (after[4][4] / ref - 1) * 100, 2)
    if len(after) >= HORIZON:
        out["r20"] = round(sgn * (after[HORIZON - 1][4] / ref - 1) * 100, 2)
    inv = _f(sig.get("invalid"))
    if inv is not None:
        crossed = any((b[3] <= inv) if d == "CALL" else (b[2] >= inv) for b in after)
        out["invalidated"] = int(crossed)
    tgt, stp = _f(sig.get("target")), _f(sig.get("stop"))
    if tgt is not None and stp is not None:
        for b in after:
            hit_t = b[2] >= tgt if d == "CALL" else b[3] <= tgt
            hit_s = b[3] <= stp if d == "CALL" else b[2] >= stp
            if hit_s:
                out.update(outcome=0, status="stop"); return out
            if hit_t:
                out.update(outcome=1, status="target"); return out
        if len(after) >= HORIZON:
            out.update(outcome=int(sgn * (after[-1][4] - ref) > 0), status="time")
        return out
    if len(after) >= 5:
        out.update(outcome=int(out["r5"] > 0), status="r5")
    return out


def score_all(fetch=fetch_yf) -> list[dict]:
    sigs = calib._read()
    if not sigs:
        return []
    by_t = defaultdict(list)
    for s in sigs:
        by_t[s["ticker"]].append(s)
    rows = []
    for t, ss in by_t.items():
        start = (dt.date.fromisoformat(min(s["date"] for s in ss)) - dt.timedelta(days=7)).isoformat()
        try:
            bars = fetch(t, start)
        except Exception as e:
            print(f"[score] {t}: fetch failed {e}"); bars = []
        rows += [score_one(s, bars) for s in ss]
    calib.STATE.mkdir(parents=True, exist_ok=True)
    with SCORED.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SFIELDS); w.writeheader(); w.writerows(rows)
    return rows


def _stats(rows):
    res = [r for r in rows if r["outcome"] != ""]
    n = len(res)
    if not n:
        return {"n": 0}
    hits = sum(int(r["outcome"]) for r in res)
    p = sum(float(r["p"]) for r in res) / n
    brier = sum((float(r["p"]) - int(r["outcome"])) ** 2 for r in res) / n
    r5 = [float(r["r5"]) for r in res if r["r5"] != ""]
    return {"n": n, "hit": hits / n, "p": p, "brier": brier,
            "r5": sum(r5) / len(r5) if r5 else None}


def _fmt(label, s):
    if not s["n"]:
        return f"{label:<9} n=0"
    r5 = f"{s['r5']:+.1f}%" if s["r5"] is not None else "  —"
    return f"{label:<9} n={s['n']:<3} hit {s['hit']:.0%}  p {s['p']:.2f}  Brier {s['brier']:.3f}  r5 {r5}"


def review(rows: list[dict]) -> str:
    today = dt.date.today()
    wk = (today - dt.timedelta(days=7)).isoformat()
    L = [f"📊 Calibration review — {today.isoformat()}"]
    week = [r for r in rows if r["date"] >= wk]
    L.append(f"\nThis week: {len(week)} signals")
    for r in sorted(week, key=lambda r: (r["grade"], r["ticker"])):
        res = {"target": "✅ tgt", "stop": "❌ stop", "time": "⏱ time", "r5": "r5",
               "open": "… open"}[r["status"]]
        oc = "" if r["outcome"] == "" else (" ✅" if int(r["outcome"]) else " ❌")
        r5 = f" {float(r['r5']):+.1f}%" if r["r5"] != "" else ""
        inv = " ⚠inv" if str(r["invalidated"]) == "1" else ""
        L.append(f" {r['ticker']} {r['direction'][0]} {r['setup'] or r['grade']} {res}{r5}{oc}{inv}")
    L.append("\nAll-time, by grade (hit vs assumed p):")
    g = defaultdict(list)
    for r in rows:
        g[r["grade"]].append(r)
    for k in sorted(g):
        L.append(" " + _fmt(k, _stats(g[k])))
    su = defaultdict(list)
    for r in rows:
        if r["setup"]:
            su[r["setup"]].append(r)
    if su:
        L.append("\nScanner setups:")
        for k in sorted(su):
            L.append(" " + _fmt(k, _stats(su[k])))
    L.append("\nBy source:")
    so = defaultdict(list)
    for r in rows:
        so[r["source"]].append(r)
    for k in sorted(so):
        L.append(" " + _fmt(k[:9], _stats(so[k])))
    tot = _stats(rows)
    L.append(" " + _fmt("ALL", tot))
    # verdicts
    notes = []
    sa, sc = _stats(g.get("A", [])), _stats(g.get("C", []))
    if sa["n"] >= 10 and sc["n"] >= 10 and sa["hit"] <= sc["hit"]:
        notes.append("A grades are NOT beating C — grading isn't separating edge.")
    for k in sorted(g):
        s = _stats(g[k])
        if s["n"] >= 15 and s["hit"] < s["p"] - 0.10:
            notes.append(f"Grade {k} overconfident: hit {s['hit']:.0%} vs p {s['p']:.0%}.")
    if tot["n"] >= 20 and tot["brier"] >= 0.25:
        notes.append("Overall Brier ≥ 0.25 — no better than a coin flip so far.")
    inv = [r for r in rows if str(r["invalidated"]) == "1" and str(r["outcome"]) == "1"]
    if inv:
        notes.append(f"{len(inv)} winner(s) crossed invalidation first — stops may be too tight.")
    if tot["n"] < 20:
        notes.append(f"Only {tot['n']} resolved — too early to judge (need ~30+/grade).")
    L.append("\nVerdict:\n " + ("\n ".join(notes) if notes else "No red flags."))
    return "\n".join(L)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "score"
    rows = score_all()
    print(f"[score] {len(rows)} signals scored -> {SCORED}")
    calib.git_save([str(SCORED)], "calib: score")
    if mode == "review":
        msg = review(rows)
        print(msg)
        if "--dry" not in sys.argv:
            calib.send_telegram(msg)


if __name__ == "__main__":
    main()
