#!/usr/bin/env python3
"""Stockbee lane (Pradeep Bonde) — stocks only, alerts to Telegram.

Scans the whole US listed universe once per day after the close:
  * Market Monitor   — 4% breakouts vs 4% breakdowns, 5d/10d ratios, 25%-in-a-quarter breadth → regime
  * MB  (momentum burst)  — today up >=4% on volume > yesterday, scored on 2LYNCH (0-6)
  * EP  (episodic pivot)  — 9M+ share day with a gap up, Cat / Dog / Lava tag
  * ANTS watch       — momentum name + 2 tight, quiet days → day 3 checked by the pre-close run
Pre-close run (~15:30 ET) re-checks the ANTS watch list on today's partial bar and alerts
ANTS-READY names to buy at the close (MOC/LOC order before 15:50 ET).

  python stockbee.py eod      [--dry] [--tickers A B C]
  python stockbee.py preclose [--dry] [--force]
  python stockbee.py test

Thresholds live in config.yaml -> stockbee:. Rules: RULES.md -> "Stockbee lane".
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import json
import logging
import math
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yaml
import yfinance as yf

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
STATE.mkdir(exist_ok=True)
log = logging.getLogger("stockbee")
ET = ZoneInfo("America/New_York")

DEFAULTS = {
    "min_price": 3.0,
    "min_avg_vol": 300_000,          # 20d average shares — tradeability floor
    "mb_pct": 4.0,                   # 4% breakout
    "mb_min_vol": 100_000,
    "mb_min_score": 5,               # 2LYNCH score needed to alert (of 6)
    "mb_min_vx": 1.2,                # breakout volume vs 50d avg
    "mb_max_alerts": 10,
    "c_tight_pct": 15.0,             # prior 10-bar high-low range / close
    "y_max_prior_bursts": 2,         # 4% bursts in prior 40 bars
    "h_close_pos": 0.70,             # close in top 30% of day range
    "ep_min_vol": 9_000_000,
    "ep_gap_pct": 4.0,
    "ep_vol_mult": 3.0,              # vs 50d avg; below this a 9M day is just "liquid lava"
    "ants_mom_pct": 20.0,            # up >= 20% over the prior 20 bars (momentum first)
    "ants_tight_atr": 0.70,          # each tight day range <= 0.7 x ATR20
    "ants_max_chg_pct": 2.0,         # each tight day close-to-close move within ±2%
    "ants_max_alerts": 12,
    "stop_max_pct": 2.5,             # MB/ANTS hard stop
    "ep_stop_max_pct": 8.0,
    "risk_pct": 0.5,                 # % of net liq risked per trade at the stop
    "max_pos_pct": 20.0,             # notional cap per position
    "net_liq": 50_000,               # fallback when book.csv is absent
}

UNIVERSE_URLS = [
    "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
    "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
]
SKIP_NAME = ("warrant", "right", " unit", "units", "preferred", "depositary share", "notes due",
             "debenture", "% ", "acquisition corp", "trust units")


# ---------------------------------------------------------------- config / book
def load_cfg() -> dict:
    cfg = dict(DEFAULTS)
    p = ROOT / "config.yaml"
    if p.exists():
        cfg.update((yaml.safe_load(p.read_text()) or {}).get("stockbee") or {})
    try:
        import calib
        bk = calib.load_book()
        if bk and bk.get("netliq"):
            cfg["net_liq"] = float(bk["netliq"])
    except Exception as e:  # book is optional
        log.info("book not loaded (%s) — using net_liq %s", e, cfg["net_liq"])
    return cfg


# ---------------------------------------------------------------- universe
def load_universe() -> list[str]:
    cache = STATE / "stockbee_universe.txt"
    syms: set[str] = set()
    try:
        for url in UNIVERSE_URLS:
            txt = requests.get(url, timeout=30, headers={"User-Agent": "optscan"}).text
            df = pd.read_csv(io.StringIO(txt), sep="|", dtype=str).fillna("")
            sym_col = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            df = df[~df[sym_col].str.startswith("File Creation")]
            if "Test Issue" in df.columns:
                df = df[df["Test Issue"] != "Y"]
            if "ETF" in df.columns:
                df = df[df["ETF"] != "Y"]
            if "Financial Status" in df.columns:   # nasdaq: D=deficient, E=delinquent...
                df = df[df["Financial Status"].isin(["", "N"])]
            name = df["Security Name"].str.lower()
            df = df[~name.apply(lambda n: any(k in n for k in SKIP_NAME))]
            for s in df[sym_col]:
                s = s.strip()
                if not s or not s.replace(".", "").isalpha() or len(s) > 5:
                    continue
                syms.add(s.replace(".", "-"))
        if len(syms) > 2000:
            cache.write_text("\n".join(sorted(syms)))
    except Exception as e:
        log.warning("universe download failed (%s) — using cache", e)
    if len(syms) < 2000 and cache.exists():
        syms = set(cache.read_text().split())
    log.info("universe: %d symbols", len(syms))
    return sorted(syms)


# ---------------------------------------------------------------- data
def download(tickers: list[str], period: str = "6mo", chunk: int = 400) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(tickers), chunk):
        part = tickers[i:i + chunk]
        try:
            raw = yf.download(part, period=period, interval="1d", auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
        except Exception as e:
            log.warning("chunk %d failed: %s", i, e)
            continue
        for t in part:
            try:
                d = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
            except KeyError:
                continue
            d = d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
            if len(d) >= 60:
                d.index = pd.to_datetime(d.index).tz_localize(None)
                out[t] = d
        log.info("downloaded %d/%d (have %d)", min(i + chunk, len(tickers)), len(tickers), len(out))
    return out


def completed(d: pd.DataFrame) -> pd.DataFrame:
    """Drop today's in-progress bar if the US session is still open."""
    now = dt.datetime.now(ET)
    if d.index[-1].date() == now.date() and now.time() < dt.time(16, 5):
        return d.iloc[:-1]
    return d


# ---------------------------------------------------------------- per-ticker features
def feats(d: pd.DataFrame) -> dict:
    c, o, h, l, v = (d[k].to_numpy(float) for k in ("close", "open", "high", "low", "volume"))
    rng = h - l
    chg = np.r_[np.nan, c[1:] / c[:-1] - 1] * 100
    return {"c": c, "o": o, "h": h, "l": l, "v": v, "rng": rng, "chg": chg}


def burst_mask(f: dict, pct: float, min_vol: float) -> np.ndarray:
    v_prev = np.r_[np.nan, f["v"][:-1]]
    return (f["chg"] >= pct) & (f["v"] > v_prev) & (f["v"] >= min_vol)


def drop_mask(f: dict, pct: float, min_vol: float) -> np.ndarray:
    v_prev = np.r_[np.nan, f["v"][:-1]]
    return (f["chg"] <= -pct) & (f["v"] > v_prev) & (f["v"] >= min_vol)


def lynch(f: dict, i: int, cfg: dict) -> tuple[int, str]:
    """2LYNCH on breakout bar i. Returns (score 0-6, letters passed)."""
    c, h, l, v, rng, chg = f["c"], f["h"], f["l"], f["v"], f["rng"], f["chg"]
    avg_rng = np.nanmean(rng[i - 20:i])
    passed = []
    # 2: not already up 2 days in a row before the breakout
    if not (chg[i - 1] > 0 and chg[i - 2] > 0):
        passed.append("2")
    # L: orderly — at most one wide/jagged bar in the prior 20
    if np.sum(rng[i - 20:i] > 2 * avg_rng) <= 1:
        passed.append("L")
    # Y: young — few prior bursts in the last 40 bars
    prior = burst_mask(f, cfg["mb_pct"], cfg["mb_min_vol"])[i - 40:i]
    if np.nansum(prior) <= cfg["y_max_prior_bursts"]:
        passed.append("Y")
    # N: narrow or negative day before
    if rng[i - 1] < 0.8 * avg_rng or chg[i - 1] < 0:
        passed.append("N")
    # C: tight consolidation + quiet volume before
    base = (np.max(h[i - 10:i]) - np.min(l[i - 10:i])) / c[i - 1] * 100
    if base <= cfg["c_tight_pct"] and np.mean(v[i - 5:i]) < np.mean(v[i - 50:i]):
        passed.append("C")
    # H: close near the high of the day
    if rng[i] > 0 and (c[i] - l[i]) / rng[i] >= cfg["h_close_pos"]:
        passed.append("H")
    return len(passed), "".join(passed)


def size(entry: float, stop: float, cfg: dict) -> int:
    risk = entry - stop
    if risk <= 0:
        return 0
    sh = math.floor(cfg["net_liq"] * cfg["risk_pct"] / 100 / risk)
    cap = math.floor(cfg["net_liq"] * cfg["max_pos_pct"] / 100 / entry)
    return max(0, min(sh, cap))


# ---------------------------------------------------------------- fundamentals (hits only)
def tag(t: str) -> dict:
    """Best-effort bucket / basket tag from yfinance. Never raises."""
    out = {"mcap": None, "rev_g": [], "eps": None, "age_y": None, "bucket": "-", "basket": "-", "name": t}
    try:
        tk = yf.Ticker(t)
        info = tk.info or {}
        out["name"] = (info.get("shortName") or t)[:28]
        out["mcap"] = info.get("marketCap")
        if not out["mcap"]:
            try:
                out["mcap"] = float(tk.fast_info["market_cap"]) or None
            except Exception:
                pass
        out["eps"] = info.get("trailingEps")
        ftd = info.get("firstTradeDateEpochUtc") or info.get("firstTradeDateMilliseconds")
        if ftd:
            ftd = ftd / 1000 if ftd > 1e11 else ftd
            out["age_y"] = (dt.datetime.utcnow() - dt.datetime.utcfromtimestamp(ftd)).days / 365.25
        try:
            q = tk.quarterly_income_stmt
            if q is not None and "Total Revenue" in q.index:
                rev = q.loc["Total Revenue"].dropna().sort_index()
                vals = rev.to_numpy(float)
                last_q = float(vals[-1]) if len(vals) else None
                for k in (1, 2):   # yoy growth for the latest two quarters where available
                    if len(vals) >= 4 + k and vals[-4 - k] > 0:
                        out["rev_g"].append((vals[-k] / vals[-4 - k] - 1) * 100)
                out["last_q_rev"] = last_q
        except Exception:
            pass
        if not out["rev_g"] and info.get("revenueGrowth") is not None:
            out["rev_g"] = [info["revenueGrowth"] * 100]
        g = out["rev_g"]
        two_q39 = len(g) >= 2 and all(x >= 39 for x in g[:2])
        one_q100 = len(g) >= 1 and g[0] >= 100
        young = out["age_y"] is not None and out["age_y"] <= 10
        small = out["mcap"] is not None and out["mcap"] < 10e9
        big_q = (out.get("last_q_rev") or 0) >= 25e6
        if two_q39 and young and small and big_q:
            out["bucket"] = "B0"
        elif two_q39 or one_q100:
            out["bucket"] = "B1"
        if (g and g[0] >= 39) or one_q100:
            out["basket"] = "CAT"
        elif out["eps"] is not None and out["eps"] <= 0:
            out["basket"] = "DOG"
    except Exception as e:
        log.info("tag %s failed: %s", t, e)
    return out


def fmt_tag(tg: dict) -> str:
    g = ("/".join(f"{x:+.0f}" for x in tg["rev_g"][:2]) + "%") if tg["rev_g"] else "n/a"
    mc = f"${tg['mcap']/1e9:.1f}B" if tg.get("mcap") else "mcap n/a"
    lab = " ".join(x for x in (tg["basket"], tg["bucket"]) if x != "-")
    return f"{lab + ' · ' if lab else ''}sales yoy {g} · {mc}"


# ---------------------------------------------------------------- market monitor
def monitor(bars: dict[str, pd.DataFrame], cfg: dict) -> dict:
    up = dn = None
    q_up = q_dn = 0
    days = None
    for d in bars.values():
        d = d.iloc[-70:]
        f = feats(d)
        u = burst_mask(f, 4.0, 100_000)[-10:].astype(float)
        w = drop_mask(f, 4.0, 100_000)[-10:].astype(float)
        if len(u) < 10:
            continue
        if up is None:
            up, dn, days = np.zeros(10), np.zeros(10), d.index[-10:]
        up += np.nan_to_num(u)
        dn += np.nan_to_num(w)
        if len(f["c"]) >= 66 and f["c"][-66] > 0:
            r = f["c"][-1] / f["c"][-66] - 1
            q_up += r >= 0.25
            q_dn += r <= -0.25
    if up is None:
        return {}
    r5 = up[-5:].sum() / max(dn[-5:].sum(), 1)
    r10 = up.sum() / max(dn.sum(), 1)
    if r10 >= 2 and up[-1] >= 300:
        regime = "THRUST — let bursts run (15–25%), full size"
    elif r10 >= 1.2:
        regime = "CONSTRUCTIVE — normal MB, sell ⅓ at +8%"
    elif r10 >= 0.8:
        regime = "CHOPPY — singles only: +5–8%, tight stops, half size"
    else:
        regime = "WEAK — few MBs; watch selling-exhaustion reversals, quarter size"
    return {"date": str(days[-1].date()), "up4": int(up[-1]), "dn4": int(dn[-1]), "r5": round(float(r5), 2),
            "r10": round(float(r10), 2), "q_up25": int(q_up), "q_dn25": int(q_dn), "regime": regime}


# ---------------------------------------------------------------- scans
def scan_eod(bars: dict[str, pd.DataFrame], cfg: dict) -> dict:
    mb, ep, ants_watch, ants_ready = [], [], [], []
    for t, d in bars.items():
        f = feats(d)
        i = len(d) - 1
        c, v = f["c"], f["v"]
        if c[i] < cfg["min_price"] or np.mean(v[i - 20:i]) < cfg["min_avg_vol"]:
            continue
        avg50 = np.mean(v[i - 50:i])
        atr20 = np.mean(f["rng"][i - 20:i])
        # --- EP: 9M+ day with a gap
        gap = (f["o"][i] / c[i - 1] - 1) * 100
        if v[i] >= cfg["ep_min_vol"] and gap >= cfg["ep_gap_pct"] and c[i] >= f["o"][i] * 0.98:
            kind = "EP" if v[i] >= cfg["ep_vol_mult"] * avg50 else "LAVA"
            if gap >= 20 and f["rng"][i] / c[i] < 0.02:
                kind = "M&A?"   # huge gap + pinned range = likely takeover, no momentum to trade
            ep.append({"t": t, "kind": kind, "close": c[i], "gap": gap, "chg": f["chg"][i],
                       "vol": v[i], "vx": v[i] / avg50, "low": f["l"][i], "high": f["h"][i]})
        # --- MB: 4% breakout + 2LYNCH
        if burst_mask(f, cfg["mb_pct"], cfg["mb_min_vol"])[i]:
            s, letters = lynch(f, i, cfg)
            if s >= cfg["mb_min_score"] and v[i] >= cfg["mb_min_vx"] * avg50:
                mb.append({"t": t, "close": c[i], "chg": f["chg"][i], "vol": v[i], "vx": v[i] / avg50,
                           "low": f["l"][i], "high": f["h"][i], "score": s, "letters": letters})
        # --- ANTS: momentum + tight quiet days
        if c[i - 22] > 0:
            mom = (np.max(c[i - 20:i + 1]) / c[i - 22] - 1) * 100
        else:
            mom = 0
        if mom >= cfg["ants_mom_pct"]:
            def tight(k):
                return (f["rng"][k] <= cfg["ants_tight_atr"] * atr20 and abs(f["chg"][k]) <= cfg["ants_max_chg_pct"])
            t2 = tight(i) and tight(i - 1) and v[i] < v[i - 1]
            t3 = t2 and tight(i - 2) and v[i - 1] < v[i - 2]
            rec = {"t": t, "close": c[i], "mom": mom, "lo3": float(np.min(f["l"][i - 2:i + 1])),
                   "hi3": float(np.max(f["h"][i - 2:i + 1])), "atr20": atr20, "v_last": v[i],
                   "lo2": float(np.min(f["l"][i - 1:i + 1]))}
            if t3:
                ants_ready.append(rec)
            elif t2:
                ants_watch.append(rec)
    mb.sort(key=lambda x: (-x["score"], -x["vx"]))
    ep.sort(key=lambda x: (x["kind"] != "EP", -x["vx"]))
    ants_ready.sort(key=lambda x: -x["mom"])
    ants_watch.sort(key=lambda x: -x["mom"])
    return {"mb": mb, "ep": ep, "ants_watch": ants_watch, "ants_ready": ants_ready}


# ---------------------------------------------------------------- formatting
def fmt_mon(m: dict) -> str:
    if not m:
        return "Market monitor: n/a"
    return (f"<b>Market monitor {m['date']}</b>\n"
            f"4%↑ {m['up4']} · 4%↓ {m['dn4']} · 5d {m['r5']} · 10d {m['r10']} · Q±25% {m['q_up25']}/{m['q_dn25']}\n"
            f"→ {m['regime']}")


def mb_line(x: dict, cfg: dict) -> str:
    buy_max = x["close"] * 1.03
    stop = x["close"] * (1 - cfg["stop_max_pct"] / 100)
    sh = size(x["close"], stop, cfg)
    hold = "5–10d" if x["close"] >= 100 else "3–5d"
    return (f"<b>{x['t']}</b> {x['close']:.2f} {x['chg']:+.1f}% vol {x['vx']:.1f}x · 2LYNCH {x['score']}/6 [{x['letters']}]\n"
            f"   D2 buy ≤{buy_max:.2f} (else skip) · stop LOD, max {stop:.2f} (−{cfg['stop_max_pct']}%) · "
            f"{sh} sh · sell ⅓ +8% · hold {hold}")


def ep_line(x: dict, tg: dict | None, cfg: dict) -> str:
    stop = max(x["low"], x["close"] * (1 - cfg["ep_stop_max_pct"] / 100))
    sh = size(x["close"], stop, cfg)
    if tg and tg["basket"] == "DOG":
        sh = min(sh, math.floor(cfg["net_liq"] * 0.05 / x["close"]))   # story stock: ≤5% notional
    head = f"<b>{x['t']}</b> [{x['kind']}] {x['close']:.2f} gap {x['gap']:+.1f}% day {x['chg']:+.1f}% "
    head += f"vol {x['vol']/1e6:.1f}M ({x['vx']:.1f}x)"
    lines = [head]
    if tg:
        lines.append(f"   {tg['name']} · {fmt_tag(tg)}")
    lines.append(f"   D2: buy if holds > EP low {x['low']:.2f} · stop {stop:.2f} · {sh} sh · peel ⅓ into strength")
    return "\n".join(lines)


def ants_line(x: dict, cfg: dict, live: dict | None = None) -> str:
    px = live["px"] if live else x["close"]
    stop = max(x["lo3"], px * (1 - cfg["stop_max_pct"] / 100))
    sh = size(px, stop, cfg)
    return (f"<b>{x['t']}</b> {px:.2f} · 1M mom +{x['mom']:.0f}% · 3d box {x['lo3']:.2f}–{x['hi3']:.2f}\n"
            f"   buy at close (MOC/LOC before 15:50 ET) · stop {stop:.2f} · {sh} sh · "
            f"sell into gap / cut next AM if no follow-through")


# ---------------------------------------------------------------- runs
def run_eod(cfg: dict, tickers: list[str] | None, send: bool) -> str:
    import notify
    uni = tickers or load_universe()
    bars = download(uni)
    bars = {t: completed(d) for t, d in bars.items()}
    bars = {t: d for t, d in bars.items() if len(d) >= 60}
    asof = max(d.index[-1] for d in bars.values()) if bars else None
    bars = {t: d for t, d in bars.items() if d.index[-1] == asof}   # drop stale/halted names
    mon = monitor(bars, cfg)
    res = scan_eod(bars, cfg)

    tags = {}
    for x in res["ep"][:10]:
        tags[x["t"]] = tag(x["t"])
    # tag a wider MB pool, then rank growth names (Bonde buckets) first
    for x in res["mb"][:3 * cfg["mb_max_alerts"]]:
        tags.setdefault(x["t"], tag(x["t"]))
    pri = {"B0": 0, "B1": 1}
    res["mb"].sort(key=lambda x: (pri.get(tags.get(x["t"], {}).get("bucket"), 2), -x["score"], -x["vx"]))

    msg = [f"🐝 <b>Stockbee EOD</b> {asof.date() if asof is not None else ''} · {len(bars)} stocks · net liq ${cfg['net_liq']:,.0f}",
           fmt_mon(mon), ""]
    eps = [x for x in res["ep"] if x["kind"] == "EP"]
    lava = [x for x in res["ep"] if x["kind"] == "LAVA"]
    mna = [x for x in res["ep"] if x["kind"] == "M&A?"]
    msg.append(f"<b>EP — 9M+ gap ups ({len(eps)})</b>")
    msg += [ep_line(x, tags.get(x["t"]), cfg) for x in eps[:8]] or ["   none"]
    if lava:
        msg.append(f"Liquid lava (9M, normal for them): " + ", ".join(f"{x['t']} {x['chg']:+.1f}%" for x in lava[:8]))
    if mna:
        msg.append("Skipped, likely takeover (pinned after gap): " + ", ".join(f"{x['t']} {x['gap']:+.0f}%" for x in mna))
    msg.append("")
    msg.append(f"<b>MB — 4% bursts, 2LYNCH ≥{cfg['mb_min_score']} ({len(res['mb'])})</b>")
    for x in res["mb"][:cfg["mb_max_alerts"]]:
        line = mb_line(x, cfg)
        if x["t"] in tags:
            line += f"\n   {fmt_tag(tags[x['t']])}"
        msg.append(line)
    if not res["mb"]:
        msg.append("   none")
    msg.append("")
    msg.append(f"<b>ANTS ready — TTT done ({len(res['ants_ready'])})</b> (buy next close if day 4 stays tight)")
    msg += [ants_line(x, cfg) for x in res["ants_ready"][:cfg["ants_max_alerts"]]] or ["   none"]
    msg.append(f"<b>ANTS watch — 2 tight days ({len(res['ants_watch'])})</b>: "
               + (", ".join(x["t"] for x in res["ants_watch"][:25]) or "none")
               + "\n   → pre-close run (~6:30am Melb) confirms day 3")
    msg.append("\nRules: MB/ANTS stop ≤2.5% · cut fast if no follow-through · sell into strength · log every trade in R")
    text = "\n".join(msg)

    # persist for the pre-close run + history
    (STATE / "stockbee_last.json").write_text(json.dumps(
        {"asof": str(asof.date()) if asof is not None else None, "monitor": mon,
         "ants_watch": res["ants_watch"] + res["ants_ready"],
         "mb": [x["t"] for x in res["mb"]], "ep": [x["t"] for x in eps]}, default=float, indent=1))
    hist = STATE / "stockbee_monitor.csv"
    if mon:
        row = pd.DataFrame([mon]).drop(columns=["regime"])
        if hist.exists():
            old = pd.read_csv(hist)
            row = pd.concat([old[old["date"] != mon["date"]], row])
        row.to_csv(hist, index=False)
    sig = STATE / "stockbee_signals.csv"
    rows = [{"date": mon.get("date"), "setup": "MB", "t": x["t"], "close": round(x["close"], 2), "score": x["score"]}
            for x in res["mb"][:cfg["mb_max_alerts"]]]
    rows += [{"date": mon.get("date"), "setup": x["kind"], "t": x["t"], "close": round(x["close"], 2), "score": ""}
             for x in res["ep"][:10]]
    rows += [{"date": mon.get("date"), "setup": "ANTS3", "t": x["t"], "close": round(x["close"], 2), "score": ""}
             for x in res["ants_ready"][:cfg["ants_max_alerts"]]]
    if rows:
        new = pd.DataFrame(rows)
        if sig.exists():
            old = pd.read_csv(sig)
            new = pd.concat([old[old["date"] != mon.get("date")], new])
        new.to_csv(sig, index=False)

    print(text)
    if send:
        notify.send(text)
    return text


def run_preclose(cfg: dict, send: bool, force: bool) -> str:
    import notify
    now = dt.datetime.now(ET)
    if not force and not (dt.time(15, 10) <= now.time() <= dt.time(15, 55)):
        print(f"outside pre-close window ({now:%H:%M} ET) — exit")
        return ""
    last = STATE / "stockbee_last.json"
    if not last.exists():
        print("no EOD state yet")
        return ""
    st = json.loads(last.read_text())
    watch = st.get("ants_watch", [])
    if not watch:
        print("ANTS watch empty")
        return ""
    bars = download([x["t"] for x in watch], period="1mo")
    ready = []
    for x in watch:
        d = bars.get(x["t"])
        if d is None or d.index[-1].date() != now.date():
            continue
        f = feats(d)
        i = len(d) - 1
        px = f["c"][i]
        # today must stay inside the tight box, quiet range, small move, volume pacing below yesterday
        frac = max(0.1, min(1.0, ((now.hour - 9) * 60 + now.minute - 30) / 390))
        pace = f["v"][i] / frac
        ok = (f["rng"][i] <= cfg["ants_tight_atr"] * x["atr20"] and abs(f["chg"][i]) <= cfg["ants_max_chg_pct"]
              and pace < f["v"][i - 1] and px >= x["lo2"] * 0.995)
        if ok:
            ready.append((x, {"px": px}))
    ready.sort(key=lambda r: -r[0]["mom"])
    lines = [f"🐜 <b>ANTS pre-close</b> {now:%a %d %b %H:%M} ET · {len(ready)}/{len(watch)} still tight"]
    mon = st.get("monitor") or {}
    if mon:
        lines.append(f"Regime: {mon.get('regime')}")
    lines += [ants_line(x, cfg, live) for x, live in ready[:cfg["ants_max_alerts"]]] or ["   none held tight today"]
    text = "\n".join(lines)
    print(text)
    if send and ready:
        notify.send(text)
    return text


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["eod", "preclose", "test"])
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tickers", nargs="*")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.cmd == "eod":
        run_eod(cfg, a.tickers, send=not a.dry)
    elif a.cmd == "preclose":
        run_preclose(cfg, send=not a.dry, force=a.force)
    else:
        import notify
        print("sent" if notify.send("🐝 stockbee lane online ✅") else "not sent")


if __name__ == "__main__":
    sys.exit(main())
