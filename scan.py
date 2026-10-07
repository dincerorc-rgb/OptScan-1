"""Orchestrator: bars -> setup -> options -> insider -> backtest gate -> alert."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import pandas as pd

import backtest, bayes, data, insider, notify, options, signals

log = logging.getLogger(__name__)
STATE = Path(__file__).resolve().parent / "state"


def scan_ticker(t: str, cfg: dict) -> dict | None:
    mcfg, ocfg, acct = cfg["momentum"], cfg["options"], cfg["account"]
    bars = data.daily_bars(t, 3)  # 3y: TT needs ~270 warm-up bars
    st = signals.latest_setup(t, bars, mcfg)
    rec = st.as_dict()

    # options layer (always pulled so IV history accrues daily)
    spot = float(bars["close"].iloc[-1])
    rv = signals.realised_vol(bars["close"], 20)
    oa = options.analyse(t, spot, rv, ocfg)
    rec.update({k: v for k, v in oa.items() if k != "chain"})

    if st.setup == "NONE":
        return rec  # summary only

    # backtest gate (this ticker, this setup)
    bt = backtest.run(t, bars, mcfg)
    rec["bt"] = backtest.verdict(bt, st.setup)
    r = bt[bt.setup == st.setup]
    rec["bt_detail"] = (f"n={int(r.n.iloc[0])} win10 {r.win10.iloc[0]:.0f}% avg10 {r.avg10.iloc[0]:+.1f}% t={r.tstat10.iloc[0]}"
                        if not r.empty else "n<8")

    # insider
    rec["insider_score"], rec["insider_n"], rec["insider_usd"] = 0.0, 0, 0.0
    if cfg["insider"]["enabled"] and st.bias == "LONG":
        ins = insider.cluster_score(t, cfg["insider"]["window_days"])
        rec["insider_score"], rec["insider_n"], rec["insider_usd"] = ins["score"], ins["n_insiders"], ins["usd_window"]
        if ins["score"] >= cfg["insider"]["cluster_score_min"]:
            rec["score"] = min(rec["score"] + 1, 5)
            rec["reason"] += f"; insider cluster {ins['score']}"

    sepa = (st.bias == "LONG" and st.setup in ("COIL", "BREAKOUT")
            and st.tt_n >= mcfg.get("tt_min_rules", 8) and t not in (mcfg.get("tt_exempt") or []))
    if sepa:
        horizon = "sepa"
    elif st.setup in ("PULLBACK", "COIL") and rec["iv_rank"] <= ocfg["iv_cheap_rank"]:
        horizon = "leaps"
    else:
        horizon = "swing"
    rec["horizon"] = horizon
    rec["tt_exempt"] = t in (mcfg.get("tt_exempt") or []) or not mcfg.get("tt_enabled", True)
    rec["trade"] = options.suggest(t, st.bias, st.setup, spot, rec["iv_rank"], oa["chain"], ocfg, acct, horizon).as_dict()

    if st.bias == "LONG":
        d_ = signals.enrich(bars, mcfg).dropna().iloc[-1]
        cl = climax(bars, d_, (cfg.get("oneil") or {}).get("climax") or {})
        if cl:   # O'Neil: climax is where you SELL, never a new entry
            rec["score"] = max(rec["score"] - 2, 0)
            rec["reason"] += f"; ⚠ climax run ({cl}) — no new entry"
    rec["plan"] = r_plan(rec, cfg)

    # Bayes layer: backtest prior x evidence LRs -> posterior + fair P(breakeven) vs market
    try:
        rec["tape_up"] = _tape_up(cfg)
        bt_row = r.iloc[0].to_dict() if not r.empty else None
        rec["bayes"] = bayes.assess(rec, cfg, bt_row)
    except Exception as e:
        log.warning("%s bayes failed: %s", t, e)
    return rec


def r_plan(rec: dict, cfg: dict) -> dict | None:
    """Minervini R plan for long setups: risk = pivot - stop = 1R.
    Size from the stop (lose risk_pct of the account if the stop hits), not from premium."""
    piv, stp = rec.get("pivot"), rec.get("stop")
    if rec.get("bias") != "LONG" or not (piv == piv and stp == stp) or not piv or piv <= stp:
        return None
    rk = cfg.get("risk") or {}
    acct = cfg["account"]["net_liq"]
    risk_usd = acct * rk.get("risk_pct", 1.0) / 100
    R = piv - stp
    plan = {"entry": round(piv, 2), "stop": round(stp, 2), "R": round(R, 2), "risk_pct": round(100 * R / piv, 1),
            "be_at": round(piv + rk.get("breakeven_at_r", 2) * R, 2),
            "t1": round(piv + rk.get("scale_at_r", 3) * R, 2), "risk_usd": round(risk_usd, 0),
            "shares": int(risk_usd // R)}
    pyr = (cfg.get("oneil") or {}).get("pyramid", [0.5, 0.3, 0.2])
    plan["pyr"] = [max(int(plan["shares"] * f), 0) for f in pyr]
    t = rec.get("trade") or {}
    if t.get("structure") not in (None, "WATCH", "NONE") and t.get("delta"):
        # option loss if the stock falls to the stop ~ delta x R per share (ignores theta/vega)
        per = abs(t["delta"]) * R * 100
        if "SPREAD" in t["structure"]:
            per = min(per, t["debit"] * 100)
        n_risk = int(risk_usd // per) if per > 0 else 0
        plan["opt_loss_1x"] = round(per, 0)
        plan["contracts_risk"] = n_risk
        t["contracts"] = min(t.get("contracts", 0), n_risk)
        t["premium_usd"] = round(t["contracts"] * t["debit"] * 100, 0)
    return plan


_TAPE: dict = {}


def _tape_up(cfg: dict) -> bool | None:
    """SPY above its EMA50 (risk-on tape). Cached per run."""
    if "v" not in _TAPE:
        try:
            c = signals.bench_close(cfg["momentum"].get("tt_bench", "SPY"))
            _TAPE["v"] = None if c is None else bool(c.iloc[-1] > signals.ema(c, 50).iloc[-1])
        except Exception:
            _TAPE["v"] = None
    return _TAPE["v"]


def climax(bars, d, ccfg: dict) -> str | None:
    """O'Neil climax run: big fast gain late in an extended move, on volume -> sell into strength."""
    c = bars["close"]
    if len(c) < 20:
        return None
    gain = c.iloc[-1] / c.iloc[-1 - ccfg.get("days", 15)] - 1
    if gain < ccfg.get("gain", 0.25):
        return None
    ext = c.iloc[-1] / d.sma200 - 1 if d.sma200 == d.sma200 and d.sma200 else 0
    rng = (bars["high"] - bars["low"]) / c
    widest = rng.iloc[-1] >= rng.tail(63).max() * 0.999
    gap = bars["open"].iloc[-1] > bars["high"].iloc[-2] * 1.02
    marks = [f"+{gain:.0%} in {ccfg.get('days', 15)}d"]
    if ext >= ccfg.get("ext_200", 0.70):
        marks.append(f"{ext:.0%} above 200d")
    if d.vol_ratio >= 2:
        marks.append(f"vol {d.vol_ratio:.1f}x")
    if widest:
        marks.append("widest range in 3m")
    if gap:
        marks.append("exhaustion gap")
    return ", ".join(marks) if len(marks) >= 3 else None


def check_position(p: dict, cfg: dict) -> dict | None:
    """SELL/TRIM/ADD/HOLD for an open position: R stop, O'Neil 8-week hold, climax, pyramid, trend breaks."""
    import datetime as _dt
    bars = data.daily_bars(p["ticker"], 2)
    d = signals.enrich(bars, cfg["momentum"]).dropna().iloc[-1]
    c, side = float(d.close), p["side"]
    act, why = None, ""
    rk = cfg.get("risk") or {}
    on = cfg.get("oneil") or {}
    out = lambda a, w: {"ticker": p["ticker"], "label": p.get("label", ""), "side": side, "action": a,
                        "close": round(c, 2), "rsi": round(float(d.rsi), 0), "why": w}
    e, st = p.get("entry"), p.get("stop")
    if e and st and e != st:
        sg = 1 if side == "LONG" else -1
        r = sg * (c - e) / abs(e - st)
        if sg * (c - st) <= 0:
            return out("EXIT", f"STOP {st:g} hit ({r:+.1f}R) — no hesitation")
        if side == "LONG":
            gain = c / e - 1
            # O'Neil 8-week hold: +20% within 3 weeks of entry = potential leader -> hold 8 weeks
            ed = p.get("entry_date")
            if ed:
                ed = _dt.date.fromisoformat(ed)
                since = bars[bars.index.date >= ed]
                wk = (_dt.date.today() - ed).days / 7
                fast = len(since) and (since["high"].iloc[:15].max() / e - 1) >= on.get("hold_gain", 0.20)
                if fast and wk < on.get("hold_weeks", 8):
                    if c <= e:
                        return out("EXIT", f"8-wk hold candidate fell back to cost {e:g} — a +20% winner never becomes a loss")
                    cl = climax(bars, d, on.get("climax") or {})
                    if cl:
                        return out("SELL INTO STRENGTH", f"climax run ({cl}) — overrides 8-wk hold")
                    return out("HOLD", f"8-WEEK HOLD to {ed + _dt.timedelta(weeks=on.get('hold_weeks', 8))}"
                                       f" (+{gain:.0%}, hit +20% in <3wk). Ignore TRIM/RSI; exit only below cost {e:g} or on climax")
            cl = climax(bars, d, on.get("climax") or {})
            if cl:
                return out("SELL INTO STRENGTH", f"climax run: {cl} — sell most into the buying")
            # pyramid: add only while it proves itself, inside the buy zone
            lo, hi = on.get("add_zone", [0.02, 0.05])
            if lo <= gain <= hi:
                act, why = "ADD", (f"+{gain:.1%} above entry {e:g} (zone +{lo:.0%}–{hi:.0%}) — pyramid "
                                   f"30% then 20% of planned size; never add below entry")
        if not act and r >= rk.get("scale_at_r", 3):
            act, why = "TRIM", f"{r:+.1f}R — sell ⅓, trail rest on EMA50 {d.ema_s:.2f}"
        elif not act and r >= rk.get("breakeven_at_r", 2) and sg * (st - e) < 0:
            act, why = "MOVE STOP", f"{r:+.1f}R — raise stop to breakeven {e:g}"
    if act:
        pass
    elif side == "LONG":
        if c < d.ema_s:
            act, why = "EXIT", f"close < EMA50 {d.ema_s:.2f}"
        elif c < d.ema_f and d.rsi < 45:
            act, why = "EXIT", f"lost EMA20 {d.ema_f:.2f}, RSI {d.rsi:.0f}"
        elif c < d.ema_f:
            act, why = "TRIM", f"below EMA20 {d.ema_f:.2f}"
        elif d.rsi >= cfg["momentum"]["rsi_max"]:
            act, why = "TRIM", f"RSI {d.rsi:.0f} extended - scale/trail"
    else:  # SHORT (puts)
        if c > d.ema_s:
            act, why = "EXIT", f"close > EMA50 {d.ema_s:.2f}"
        elif c > d.ema_f and d.rsi > 55:
            act, why = "EXIT", f"reclaimed EMA20 {d.ema_f:.2f}, RSI {d.rsi:.0f}"
        elif c > d.ema_f:
            act, why = "TRIM", f"above EMA20 {d.ema_f:.2f}"
        elif d.rsi <= 25:
            act, why = "TRIM", f"RSI {d.rsi:.0f} washed out - take some"
    if not act:
        return None
    return {"ticker": p["ticker"], "label": p.get("label", ""), "side": side, "action": act,
            "close": round(c, 2), "rsi": round(float(d.rsi), 0), "why": why}


def book_positions(cfg: dict) -> list[dict]:
    """book.csv is the single book: sets net liq for sizing and the SELL/TRIM list.
    Falls back to config.yaml `positions:` only if book.csv is missing."""
    import calib
    bk = calib.load_book()
    if not bk:
        return cfg.get("positions") or []
    cfg["account"]["net_liq"] = bk["netliq"]
    merged: dict[tuple, dict] = {}
    for p in bk["positions"]:
        side = "SHORT" if p["bias"] == "short" else "LONG"
        k = (p["ticker"], side)
        if k in merged:
            merged[k]["label"] += " + " + p["label"]
        else:
            merged[k] = {"ticker": p["ticker"], "side": side, "label": p["label"],
                         "entry": p.get("entry"), "stop": p.get("stop"), "entry_date": p.get("entry_date")}
    return list(merged.values())


def run(cfg: dict, tickers: list[str] | None = None, send: bool = True, min_score: int = 2) -> list[dict]:
    tickers = tickers or cfg["watchlist"]
    positions = book_positions(cfg)
    hits, flat, errors = [], [], []
    for t in tickers:
        try:
            rec = scan_ticker(t, cfg)
        except Exception as e:
            log.exception("%s failed", t)
            errors.append(f"{t}: {e}")
            continue
        if rec is None:
            continue
        if rec["setup"] == "NONE":
            flat.append(f"{t} {rec['close']:.2f} RSI{rec['rsi']:.0f} IVR{rec['iv_rank']:.0f}")
        else:
            hits.append(rec)

    hits.sort(key=lambda r: (-r["score"], r["bt"] != "PASS"))
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    (STATE / f"scan_{dt.date.today()}.json").write_text(json.dumps(hits, default=str, indent=1))

    alert = [h for h in hits if h["score"] >= min_score and h["bt"] != "FAIL"]
    bmin = (cfg.get("bayes") or {}).get("alert_min_post", 0) or 0
    if bmin:
        alert = [h for h in alert if h.get("bayes", {}).get("post", 1) >= bmin]
    rmin = (cfg.get("bayes") or {}).get("alert_min_exp_r")
    if rmin is not None:
        alert = [h for h in alert if h.get("bayes", {}).get("exp_r", rmin) >= rmin]
    emin = (cfg.get("bayes") or {}).get("alert_min_usable_pp")
    if emin is not None:
        alert = [h for h in alert if h.get("bayes", {}).get("usable_pp", emin) >= emin]
    for h in alert:
        if h.get("bayes") and send:
            try:
                bayes.log(h, h["bayes"])
            except Exception as e:
                log.warning("bayes log: %s", e)

    exits = []
    for p in positions:
        try:
            x = check_position(p, cfg)
            if x:
                exits.append(x)
        except Exception as e:
            errors.append(f"{p['ticker']} pos: {e}")

    if cfg["alerts"].get("only_on_signal") and not alert and not exits:
        log.info("no signals; nothing sent")
        return hits

    parts = [f"<b>optscan {stamp}</b>  BUY {len(alert)} · SELL {len(exits)} · {len(tickers)} names"]
    for x in exits:
        parts.append(f"🔻 <b>SELL/{x['action']}</b> {x['ticker']} {x['close']:.2f} — {x['label']}\n   {x['why']}")
    for h in alert:
        parts.append(("🟢 <b>BUY</b> " if h["bias"] == "LONG" else "🔴 <b>BUY PUTS</b> ") + notify.fmt_alert(h))
    low = [f"{h['ticker']} {h['setup']} ★{h['score']} BT:{h['bt']}" for h in hits if h not in alert]
    if low:
        parts.append("<i>below bar:</i> " + ", ".join(low))
    if cfg["alerts"].get("send_no_setup_summary") and flat:
        parts.append("<i>no setup:</i> " + " | ".join(flat))
    if errors:
        parts.append("<i>errors:</i> " + "; ".join(errors))
    msg = "\n\n".join(parts)
    if send:
        notify.send(msg)
    else:
        print(msg)
    return hits
