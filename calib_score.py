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

R multiple (Minervini expectancy): 1R = |entry - stop|. Stop hit = -1R (or worse if the
open gapped through it); target hit = +reward/risk; otherwise the close at session 20.
Expectancy = mean R per trade = win% x avg win R - loss% x avg loss R. This is the headline.
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
           "ref", "r5", "r20", "invalidated", "outcome", "status", "sessions", "r_mult"]


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
           "status": "open", "sessions": len(after), "r_mult": ""}
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
    # buy-stop entries (entry beyond spot, e.g. PULLBACK pivot): only a trade once price trades through entry
    spot = _f(sig.get("spot"))
    if spot and _f(sig.get("entry")) and sgn * (ref - spot) > 0.001 * spot:
        trig = next((i for i, b in enumerate(after) if (b[2] >= ref if d == "CALL" else b[3] <= ref)), None)
        if trig is None:
            out["status"] = "untriggered" if len(after) < HORIZON else "never"
            return out
        after = after[trig:]
    risk = abs(ref - stp) if stp is not None else 0
    R = (lambda px: round(sgn * (px - ref) / risk, 2)) if risk > 0 else (lambda px: "")
    if stp is not None and risk > 0:
        for b in after:
            gap_s = b[1] <= stp if d == "CALL" else b[1] >= stp
            hit_t = tgt is not None and (b[2] >= tgt if d == "CALL" else b[3] <= tgt)
            hit_s = b[3] <= stp if d == "CALL" else b[2] >= stp
            if hit_s:
                out.update(outcome=0, status="stop", r_mult=R(b[1]) if gap_s else -1.0); return out
            if hit_t:
                out.update(outcome=1, status="target", r_mult=R(tgt)); return out
        if len(after) >= HORIZON:
            out.update(outcome=int(sgn * (after[-1][4] - ref) > 0), status="time", r_mult=R(after[-1][4]))
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


def _exp(rows):
    rs = [float(r["r_mult"]) for r in rows if r.get("r_mult") not in ("", None)]
    if not rs:
        return {"n": 0}
    w = [x for x in rs if x > 0]
    l = [x for x in rs if x <= 0]
    aw = sum(w) / len(w) if w else 0.0
    al = -sum(l) / len(l) if l else 0.0
    return {"n": len(rs), "win": len(w) / len(rs), "aw": aw, "al": al,
            "payoff": aw / al if al else float("inf"), "E": sum(rs) / len(rs), "tot": sum(rs)}


def _fmt_e(label, e):
    if not e["n"]:
        return f"{label:<16} n=0"
    po = "∞" if e["payoff"] == float("inf") else f"{e['payoff']:.1f}"
    return (f"{label:<16} n={e['n']:<3} win {e['win']:.0%} · +{e['aw']:.1f}R/−{e['al']:.1f}R"
            f" · payoff {po} · E {e['E']:+.2f}R · Σ {e['tot']:+.1f}R")


def _fmt(label, s):
    if not s["n"]:
        return f"{label:<9} n=0"
    r5 = f"{s['r5']:+.1f}%" if s["r5"] is not None else "  —"
    return f"{label:<9} n={s['n']:<3} hit {s['hit']:.0%}  p {s['p']:.2f}  Brier {s['brier']:.3f}  r5 {r5}"


def review(rows: list[dict]) -> str:
    today = dt.date.today()
    wk = (today - dt.timedelta(days=7)).isoformat()
    L = [f"📊 Calibration review — {today.isoformat()}"]
    # --- headline: expectancy (Minervini) ---
    L.append("\nEXPECTANCY (R per trade) — the number that pays:")
    tot_e = _exp(rows)
    L.append(" " + _fmt_e("ALL", tot_e))
    strat = defaultdict(list)
    for r in rows:
        su = r.get("setup") or ""
        strat[su.split("/")[1] if "/" in su else (su or r["source"])].append(r)
    for k in sorted(strat):
        if _exp(strat[k])["n"]:
            L.append(" " + _fmt_e(k[:16], _exp(strat[k])))
    for k in sorted({r.get("setup") for r in rows if "/" in (r.get("setup") or "")}):
        e = _exp([r for r in rows if r.get("setup") == k])
        if e["n"]:
            L.append("   " + _fmt_e(k[:16], e))
    if tot_e["n"]:
        be = 1 / (1 + tot_e["payoff"]) if tot_e["payoff"] not in (0, float("inf")) else 0
        L.append(f" break-even win rate at this payoff: {be:.0%} (you have {tot_e['win']:.0%})")
        L.append(f" → set config bayes.avg_win_r: {tot_e['aw']:.1f}  avg_loss_r: {max(tot_e['al'], 1):.1f} once n ≥ 30")
    week = [r for r in rows if r["date"] >= wk]
    L.append(f"\nThis week: {len(week)} signals")
    for r in sorted(week, key=lambda r: (r["grade"], r["ticker"])):
        res = {"target": "✅ tgt", "stop": "❌ stop", "time": "⏱ time", "r5": "r5",
               "open": "… open", "untriggered": "… waiting entry", "never": "– never triggered"}[r["status"]]
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
    if tot_e["n"] >= 20 and tot_e["E"] <= 0:
        notes.append(f"Expectancy {tot_e['E']:+.2f}R ≤ 0 — system loses money regardless of hit rate.")
    if tot_e["n"] >= 20 and tot_e["al"] > 1.2:
        notes.append(f"Avg loss {tot_e['al']:.1f}R > 1.2R — stops gapping/not honoured; tighten or avoid earnings.")
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
