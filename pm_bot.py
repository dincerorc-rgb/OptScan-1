"""pm_bot — prediction-market edge alerts (Polymarket + Kalshi). Alerts only, never trades.

For every market in pm_markets.yaml:
  fair      = Bayes posterior  (prior x signal likelihood ratios, bayes.revise_probability)
  entry     = average fill price for your stake, walked through the live order book
  usable    = fair - entry - execution_costs - safety_margin       (calculate_trade_edge)
Telegram alert only when usable edge > 0 on YES or NO, with quarter-Kelly size.

    python pm_bot.py scan            # check all enabled markets -> Telegram on edge
    python pm_bot.py scan --dry      # print only
    python pm_bot.py discover btc fed   # top Polymarket markets matching words (finds slugs)
    python pm_bot.py review          # score resolved alerts: your Brier vs the market's
    python pm_bot.py demo            # worked example, no network
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

import requests
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bayes  # noqa: E402
import calib  # noqa: E402  (send_telegram, git_save)

STATE = ROOT / "state"
PM_STATE = STATE / "pm_state.json"
PM_LOG = STATE / "pm_log.csv"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
N = NormalDist()
UA = {"User-Agent": "optscan-pm/1.0"}


# ---------- the two formulas ----------

def calculate_trade_edge(fair_value, average_entry, execution_costs, safety_margin):
    price_advantage = fair_value - average_entry
    return price_advantage - execution_costs - safety_margin


def kelly_stake(fair: float, entry: float, usable: float, bankroll: float, frac: float, cap: float) -> float:
    """Fractional Kelly on a binary contract paying 1. Uses usable edge, not raw edge."""
    if usable <= 0 or entry >= 1:
        return 0.0
    f = usable / (1 - entry)
    return round(min(bankroll * f * frac, cap), 0)


# ---------- order books ----------

def vwap(levels: list[tuple[float, float]], usd: float) -> tuple[float | None, float]:
    """Average price buying `usd` worth from asks [(price, size_in_contracts)], best first.
    Returns (avg_price, usd_filled)."""
    spent = qty = 0.0
    for px, sz in levels:
        if px <= 0 or px >= 1:
            continue
        take = min(sz, (usd - spent) / px)
        spent += take * px
        qty += take
        if spent >= usd - 1e-9:
            break
    return (spent / qty if qty else None), round(spent, 2)


def _get(url, **params):
    r = requests.get(url, params=params or None, headers=UA, timeout=20)
    r.raise_for_status()
    return r.json()


def _jl(x):
    return json.loads(x) if isinstance(x, str) else (x or [])


def poly_market(slug: str) -> dict:
    data = _get(f"{GAMMA}/markets", slug=slug)
    if not data:
        raise ValueError(f"polymarket slug not found: {slug}")
    m = data[0]
    outs, prices, toks = _jl(m.get("outcomes")), _jl(m.get("outcomePrices")), _jl(m.get("clobTokenIds"))
    yi = next((i for i, o in enumerate(outs) if str(o).lower() == "yes"), 0)
    ni = 1 - yi if len(outs) == 2 else None
    out = {"platform": "polymarket", "id": slug, "title": m.get("question", slug),
           "end": (m.get("endDate") or "")[:10], "closed": bool(m.get("closed")),
           "mid_yes": float(prices[yi]) if prices else None, "books": {}}
    if out["closed"] and prices:
        out["result"] = 1 if float(prices[yi]) > 0.99 else (0 if float(prices[yi]) < 0.01 else None)
    for side, idx in (("yes", yi), ("no", ni)):
        if idx is None or idx >= len(toks):
            continue
        try:
            bk = _get(f"{CLOB}/book", token_id=toks[idx])
            asks = sorted(((float(a["price"]), float(a["size"])) for a in bk.get("asks", [])), key=lambda x: x[0])
            out["books"][side] = asks
        except Exception as e:
            print(f"[pm] {slug} {side} book: {e}")
    if out["mid_yes"] is None and out["books"].get("yes"):
        out["mid_yes"] = out["books"]["yes"][0][0]
    return out


def kalshi_market(ticker: str) -> dict:
    m = _get(f"{KALSHI}/markets/{ticker}")["market"]

    def cents(k):
        v = m.get(f"{k}_dollars")
        if v not in (None, ""):
            return float(v)
        v = m.get(k)
        return None if v in (None, "") else float(v) / 100

    yb, ya = cents("yes_bid"), cents("yes_ask")
    out = {"platform": "kalshi", "id": ticker, "title": m.get("title", ticker),
           "end": (m.get("close_time") or m.get("expiration_time") or "")[:10],
           "closed": m.get("status") in ("closed", "settled", "finalized"),
           "mid_yes": (yb + ya) / 2 if yb is not None and ya is not None else cents("last_price"), "books": {}}
    if m.get("result") in ("yes", "no"):
        out["result"] = 1 if m["result"] == "yes" else 0
    try:
        ob = _get(f"{KALSHI}/markets/{ticker}/orderbook").get("orderbook") or {}

        def lv(key):  # bids on one side = asks on the other at (1 - price)
            raw = ob.get(f"{key}_dollars") or ob.get(key) or []
            res = []
            for p, q in raw:
                p = float(p)
                res.append((p if p <= 1 else p / 100, float(q)))
            return res
        yes_bids, no_bids = lv("yes"), lv("no")
        out["books"]["yes"] = sorted(((round(1 - p, 4), q) for p, q in no_bids), key=lambda x: x[0])
        out["books"]["no"] = sorted(((round(1 - p, 4), q) for p, q in yes_bids), key=lambda x: x[0])
    except Exception as e:
        print(f"[pm] {ticker} book: {e}")
        if ya is not None:
            out["books"]["yes"] = [(ya, 1e9)]
        if yb is not None:
            out["books"]["no"] = [(round(1 - yb, 4), 1e9)]
    return out


def fetch_market(spec: dict) -> dict:
    return kalshi_market(spec["kalshi"]) if spec.get("kalshi") else poly_market(spec["polymarket"])


# ---------- fair value ----------

def threshold_prob(spot: float, strike: float, vol: float, days: float, above: bool) -> float:
    """Driftless lognormal P(S_T > strike) — the 'no view' probability for a price-level market."""
    T = max(days, 0.5) / 365
    z = (math.log(spot / strike) - vol * vol / 2 * T) / (vol * math.sqrt(T))
    return N.cdf(z) if above else N.cdf(-z)


def _underlying(sym: str, vol_mode) -> tuple[float, float, str]:
    import numpy as np
    import data
    bars = data.daily_bars(sym, 1)
    spot = float(bars.close.iloc[-1])
    if isinstance(vol_mode, (int, float)):
        return spot, float(vol_mode), "fixed"
    ann = 365 if sym.endswith("-USD") else 252
    rv = float(np.log(bars.close).diff().tail(60).std() * math.sqrt(ann))
    if vol_mode == "iv" and not sym.endswith("-USD"):
        try:
            import options
            ch = data.option_chain(sym, 7, 60)
            iv, _, _ = options.atm_iv(ch, spot)
            if iv and math.isfinite(iv) and iv > 0.01:
                return spot, iv, "ATM IV"
        except Exception as e:
            print(f"[pm] {sym} IV failed, using RV: {e}")
    return spot, rv, "60d RV"


def fair_value(spec: dict, mkt: dict, defaults: dict) -> dict:
    prior_spec = spec.get("prior", "market")
    note = ""
    if isinstance(prior_spec, (int, float)):
        prior, src = float(prior_spec), "your prior"
    elif prior_spec == "threshold":
        th = spec["threshold"]
        end = th.get("date") or mkt.get("end")
        days = (dt.date.fromisoformat(end) - dt.date.today()).days + 0.5
        spot, vol, vsrc = _underlying(th["symbol"], th.get("vol", "iv"))
        prior = threshold_prob(spot, float(th["strike"]), vol, days, th.get("direction", "above") == "above")
        src = "lognormal"
        note = f"{th['symbol']} {spot:,.2f} vs {th['strike']:,} {th.get('direction', 'above')} · {vsrc} {vol:.0%} · {days:.0f}d"
    else:
        if mkt.get("mid_yes") is None:
            raise ValueError("no market price for prior")
        prior, src = float(mkt["mid_yes"]), "market mid"
    lr = 1.0
    used = []
    for s in spec.get("signals") or []:
        if s.get("active", True):
            lr *= float(s["lr"])
            used.append(f"{s['name']}×{float(s['lr']):g}")
    cap = defaults.get("lr_cap", 4.0)
    lr = max(1 / cap, min(cap, lr))
    fair = bayes.revise_probability(prior, lr)
    return {"prior": prior, "prior_src": src, "lr": lr, "fair": fair, "signals": used, "note": note}


# ---------- edge ----------

def evaluate(spec: dict, mkt: dict, fv: dict, d: dict) -> dict:
    stake = float(spec.get("stake_usd", d["stake_usd"]))
    margin = float(spec.get("safety_margin", d["safety_margin"]))
    plat = d["platforms"][mkt["platform"]]
    best = None
    for side, fair in (("YES", fv["fair"]), ("NO", 1 - fv["fair"])):
        asks = mkt["books"].get(side.lower()) or []
        entry, filled = vwap(asks, stake)
        if entry is None:
            continue
        top = asks[0][0]
        # costs per $1 contract: platform fee + (optional) per-trade slippage allowance
        fee = plat.get("fee_coef", 0.0) * entry * (1 - entry) + plat.get("fee_flat", 0.0)
        costs = fee + float(spec.get("slippage", d.get("slippage", 0.0)))
        usable = calculate_trade_edge(fair, entry, costs, margin)
        row = {"side": side, "fair": fair, "top": top, "entry": entry, "filled": filled, "costs": costs,
               "margin": margin, "raw": fair - entry, "usable": usable,
               "stake": kelly_stake(fair, entry, usable, d["bankroll_usd"], d["kelly_frac"], stake)}
        if best is None or row["usable"] > best["usable"]:
            best = row
    return best or {}


def fmt_alert(spec, mkt, fv, ev) -> str:
    pct = lambda x: f"{100 * x:.1f}¢"
    L = [f"🎯 <b>{ev['side']}</b> {mkt['title'][:90]}",
         f"{mkt['platform']} · ends {mkt.get('end') or '?'} · mid {pct(mkt['mid_yes']) if mkt.get('mid_yes') is not None else '?'}",
         f"fair {fv['fair']:.1%} ({fv['prior_src']} {fv['prior']:.1%} ×{fv['lr']:.2f}"
         + (": " + ", ".join(fv["signals"]) if fv["signals"] else "") + ")"]
    if fv["note"]:
        L.append("   " + fv["note"])
    L.append(f"entry {pct(ev['entry'])} avg for ${ev['filled']:,.0f} (top {pct(ev['top'])})")
    L.append(f"edge {ev['raw']:+.1%} − costs {ev['costs']:.1%} − margin {ev['margin']:.1%} = <b>{ev['usable']:+.1%}</b>")
    L.append(f"→ buy {ev['side']} ≤{pct(ev['entry'])} · ¼-Kelly ${ev['stake']:,.0f}")
    return "\n".join(L)


# ---------- runs ----------

def _load():
    cfg = yaml.safe_load((ROOT / "pm_markets.yaml").read_text()) or {}
    d = {"stake_usd": 100, "safety_margin": 0.02, "slippage": 0.0, "bankroll_usd": 2000,
         "kelly_frac": 0.25, "realert_pp": 0.02, "lr_cap": 4.0,
         "platforms": {"polymarket": {"fee_coef": 0.0}, "kalshi": {"fee_coef": 0.07}}}
    user = cfg.get("defaults") or {}
    plats = {**d["platforms"], **(user.get("platforms") or {})}
    d.update(user)
    d["platforms"] = plats
    return d, [m for m in (cfg.get("markets") or []) if m.get("enabled", True)]


def _key(spec):
    return f"kalshi:{spec['kalshi']}" if spec.get("kalshi") else f"poly:{spec['polymarket']}"


def scan(dry: bool = False, fetch=fetch_market) -> str | None:
    d, specs = _load()
    st = json.loads(PM_STATE.read_text()) if PM_STATE.exists() else {}
    alerts, gone, errors, lines = [], [], [], []
    for spec in specs:
        k = _key(spec)
        try:
            mkt = fetch(spec)
            if mkt.get("closed"):
                lines.append(f"{k}: closed"); continue
            fv = fair_value(spec, mkt, d)
            ev = evaluate(spec, mkt, fv, d)
        except Exception as e:
            errors.append(f"{spec.get('name', k)}: {e}"); continue
        if not ev:
            errors.append(f"{spec.get('name', k)}: empty book"); continue
        lines.append(f"{spec.get('name', k)}: fair {fv['fair']:.1%} {ev['side']} entry {ev['entry']:.3f} usable {ev['usable']:+.1%}")
        prev = st.get(k)
        if ev["usable"] > 0:
            changed = (not prev or not prev.get("live") or prev["side"] != ev["side"]
                       or abs(ev["usable"] - prev["usable"]) >= d["realert_pp"])
            if changed:
                alerts.append(fmt_alert(spec, mkt, fv, ev))
                _log(k, spec, mkt, fv, ev, dry)
            st[k] = {"live": True, "side": ev["side"], "usable": round(ev["usable"], 4),
                     "ts": dt.datetime.utcnow().isoformat(timespec="minutes")}
        else:
            if prev and prev.get("live"):
                gone.append(f"⚪ edge gone: {spec.get('name', mkt['title'][:60])} ({ev['side']} now {ev['usable']:+.1%})")
            st[k] = {"live": False, "side": ev["side"], "usable": round(ev["usable"], 4),
                     "ts": dt.datetime.utcnow().isoformat(timespec="minutes")}
    print("\n".join(lines + [f"error: {e}" for e in errors]) or "no markets enabled")
    if not dry:
        STATE.mkdir(exist_ok=True)
        PM_STATE.write_text(json.dumps(st, indent=1))
        calib.git_save([str(PM_STATE), str(PM_LOG)], "pm: state")
    if not alerts and not gone:
        return None
    msg = "\n\n".join([f"<b>pm_bot</b> {len(alerts)} edge · {len(specs)} markets"] + alerts + gone
                      + ([f"errors: " + "; ".join(errors)] if errors else []))
    if dry:
        print("\n" + msg)
    else:
        import notify
        notify.send(msg)
    return msg


LOG_FIELDS = ["ts", "key", "name", "side", "fair_yes", "mid_yes", "entry", "usable", "stake"]


def _log(k, spec, mkt, fv, ev, dry):
    if dry:
        return
    STATE.mkdir(exist_ok=True)
    new = not PM_LOG.exists()
    with PM_LOG.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"ts": dt.datetime.utcnow().isoformat(timespec="minutes"), "key": k,
                    "name": spec.get("name", mkt["title"][:60]), "side": ev["side"],
                    "fair_yes": round(fv["fair"], 4), "mid_yes": mkt.get("mid_yes"),
                    "entry": round(ev["entry"], 4), "usable": round(ev["usable"], 4), "stake": ev["stake"]})


def review(dry: bool = False, fetch=fetch_market) -> str:
    if not PM_LOG.exists():
        msg = "🎯 pm_bot review — no alerts logged yet."
    else:
        with PM_LOG.open(newline="") as f:
            rows = list(csv.DictReader(f))
        res, pnl = [], 0.0
        cache = {}
        for r in rows:
            plat, ident = r["key"].split(":", 1)
            if r["key"] not in cache:
                try:
                    cache[r["key"]] = fetch({"kalshi": ident} if plat == "kalshi" else {"polymarket": ident})
                except Exception as e:
                    print(f"[pm] {r['key']}: {e}"); cache[r["key"]] = {}
            out = cache[r["key"]].get("result")
            if out is None:
                continue
            won = (out == 1) == (r["side"] == "YES")
            e = float(r["entry"])
            pnl += float(r["stake"] or 0) * ((1 - e) / e if won else -1)
            res.append((float(r["fair_yes"]), float(r["mid_yes"] or 0.5), out, won))
        L = [f"🎯 pm_bot review — {dt.date.today()}", f"{len(rows)} alerts, {len(res)} resolved"]
        if res:
            bm = sum((f - o) ** 2 for f, _, o, _ in res) / len(res)
            bk = sum((m - o) ** 2 for _, m, o, _ in res) / len(res)
            L.append(f"Brier you {bm:.3f} vs market {bk:.3f} " + ("✅ beating market" if bm < bk else "⚠ market sharper"))
            L.append(f"hit {sum(w for *_, w in res) / len(res):.0%} · paper P&L at ¼-Kelly ${pnl:+,.0f}")
            if len(res) < 30:
                L.append("Too few resolved to judge (need ~30+).")
        msg = "\n".join(L)
    print(msg)
    if not dry:
        calib.send_telegram(msg)
    return msg


def discover(words: list[str], dry: bool = False) -> str:
    data = _get(f"{GAMMA}/markets", active="true", closed="false", limit=500,
                order="volume24hr", ascending="false")
    ws = [w.lower() for w in words]
    hits = [m for m in data if all(w in (m.get("question", "") + " " + m.get("slug", "")).lower() for w in ws)][:15]
    L = [f"🔎 Polymarket: {' '.join(words) or 'top volume'} — {len(hits)} found"]
    for m in hits:
        p = _jl(m.get("outcomePrices"))
        L.append(f"• {m.get('question', '')[:80]}\n  slug: {m.get('slug')} · YES {float(p[0]):.0%} · ends {(m.get('endDate') or '')[:10]}"
                 if p else f"• {m.get('question', '')[:80]}\n  slug: {m.get('slug')}")
    msg = "\n".join(L)
    print(msg)
    if not dry:
        calib.send_telegram(msg)
    return msg


def demo():
    edge = calculate_trade_edge(fair_value=0.55, average_entry=0.49, execution_costs=0.012, safety_margin=0.008)
    print(f"Tradable edge: {edge:.2%}")
    asks = [(0.48, 50), (0.49, 100), (0.51, 400)]
    avg, filled = vwap(asks, 100)
    print(f"$100 through book {asks} -> avg entry {avg:.4f}")
    fair = bayes.revise_probability(0.42, 1.62)
    u = calculate_trade_edge(fair, avg, 0.0, 0.02)
    print(f"fair {fair:.2%} (0.42 x LR 1.62) - entry {avg:.2%} - margin 2% = {u:+.2%}; "
          f"1/4-Kelly on $2,000 = ${kelly_stake(fair, avg, u, 2000, 0.25, 500):,.0f}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    dry = "--dry" in sys.argv
    args = [a for a in sys.argv[2:] if a != "--dry"]
    if cmd == "scan":
        scan(dry)
    elif cmd == "review":
        review(dry)
    elif cmd == "discover":
        discover(args, dry)
    else:
        demo()
