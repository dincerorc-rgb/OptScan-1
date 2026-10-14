"""Bayesian probability layer for optscan.

    prior  = this ticker's backtest win-rate for this setup (+10 sessions),
             shrunk toward 50% when the sample is small
    signals = independent-ish evidence, each a likelihood ratio (LR)
    post   = revise_probability(prior, product of LRs)      <- odds x LR

Then the posterior is turned into a fair probability that the suggested option
finishes past breakeven, and compared with what the option price implies
(risk-neutral N(d2)). Positive edge = you think the contract is cheap.

    python bayes.py calibrate         # score bayes_log, learn LRs, Telegram report
    python bayes.py calibrate --dry   # print only
    python bayes.py demo              # worked example, no network

Nothing here places orders.
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

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "state"
LOG = STATE / "bayes_log.csv"
LEARNED = STATE / "bayes_lr.json"
N = NormalDist()
EPS = 1e-4

DEFAULTS = {
    "prior_shrink_k": 20,      # pseudo-obs at 50% blended into the backtest win-rate
    "damp": 0.75,              # x total log-LR: signals overlap, don't fully stack
    "lr_cap": 4.0,             # combined LR clamped to [1/cap, cap]
    "drift_days": 10,          # posterior edge applies over this many sessions only
    "rate": 0.045,
    "learn_k": 30,             # pseudo-obs on the default LR when learning
    "horizon": 10,             # sessions; matches backtest win10
    "margin_pp": 3.0,          # safety margin (pp of probability) for model error
    "avg_win_r": 2.0,          # assumed avg winner in R until the Sunday review measures it
    "avg_loss_r": 1.0,         # loser = stop = -1R (gaps make it worse; review shows the real number)
    "lr": {
        "tt_full": 1.25,       # Minervini TT 8/8
        "tt_weak": 0.85,       # TT <= 5/8
        "rs_with": 1.15,       # RS vs SPY agrees with direction
        "rs_against": 0.90,
        "vol_surge": 1.20,     # volume >= 2x 20d avg
        "adx_strong": 1.10,    # ADX >= 30
        "rsi_hot": 0.85,       # long with RSI >= 72 / short with RSI <= 28
        "insider": 1.40,       # Form 4 cluster score >= threshold
        "pcr_against": 0.95,   # long with P/C >= 1.5, short with P/C <= 0.5
        "tape_with": 1.10,     # SPY vs its EMA50 agrees with direction
        "tape_against": 0.85,
        "nash_with": 1.10,     # Nash A on a long / Nash X on a put
        "nash_against": 0.90,  # Nash X on a long / Nash A on a put
    },
}


# ---------- core ----------

def revise_probability(previous_probability: float, signal_multiplier: float) -> float:
    """Odds-form Bayes update. Clamped so p=0/1 can't divide by zero or lock."""
    p = min(max(float(previous_probability), EPS), 1 - EPS)
    if signal_multiplier <= 0:
        raise ValueError("signal_multiplier must be > 0")
    odds = p / (1 - p) * signal_multiplier
    return odds / (1 + odds)


def _cfg(cfg: dict | None) -> dict:
    b = json.loads(json.dumps(DEFAULTS))
    user = (cfg or {}).get("bayes") or {}
    b.update({k: v for k, v in user.items() if k != "lr"})
    b["lr"].update(user.get("lr") or {})
    if LEARNED.exists():
        try:
            b["lr"].update(json.loads(LEARNED.read_text()).get("lr", {}))
            b["learned"] = True
        except Exception:
            pass
    return b


def prior_from_backtest(win_pct: float | None, n: int | None, k: float) -> float:
    if win_pct is None or n is None or not math.isfinite(win_pct) or n <= 0:
        return 0.5
    return (n * win_pct / 100 + k * 0.5) / (n + k)


def evidence(rec: dict, cfg: dict) -> list[str]:
    """Names of the features that fired for this alert."""
    long = rec.get("bias") == "LONG"
    f = []
    ins_min = (cfg.get("insider") or {}).get("cluster_score_min", 3.0)
    if long and not rec.get("tt_exempt"):
        tt = rec.get("tt_n", 0) or 0
        if tt >= 8:
            f.append("tt_full")
        elif tt <= 5:
            f.append("tt_weak")
    rs = rec.get("rs_excess")
    if rs is not None and math.isfinite(rs) and rs != 0:
        f.append("rs_with" if (rs > 0) == long else "rs_against")
    if (rec.get("vol_ratio") or 0) >= 2.0:
        f.append("vol_surge")
    if (rec.get("adx") or 0) >= 30:
        f.append("adx_strong")
    r = rec.get("rsi") or 50
    if (long and r >= 72) or (not long and r <= 28):
        f.append("rsi_hot")
    if (rec.get("insider_score") or 0) >= ins_min:
        f.append("insider")
    pcr = rec.get("pcr")
    if pcr is not None and isinstance(pcr, (int, float)) and math.isfinite(pcr):
        if (long and pcr >= 1.5) or (not long and pcr <= 0.5):
            f.append("pcr_against")
    ng = (rec.get("nash") or {}).get("grade")
    if ng in ("A", "B", "C", "X"):
        if ng == "A":
            f.append("nash_with" if long else "nash_against")
        elif ng == "X":
            f.append("nash_against" if long else "nash_with")
    tape = rec.get("tape_up")
    if tape is not None:
        f.append("tape_with" if bool(tape) == long else "tape_against")
    return f


def combine(feats: list[str], b: dict) -> float:
    s = sum(math.log(b["lr"].get(x, 1.0)) for x in feats) * b["damp"]
    cap = math.log(b["lr_cap"])
    return math.exp(max(-cap, min(cap, s)))


def p_beyond(spot: float, be: float, iv: float, dte: int, side: str, r: float,
             shift_sd: float = 0.0, drift_t: float = 0.0) -> float:
    """P(S_T past breakeven). shift_sd = extra drift in 10-day std units, applied over drift_t years."""
    T = max(dte, 1) / 365
    if not (spot > 0 and be > 0 and iv > 0):
        return float("nan")
    a = shift_sd * iv / math.sqrt(10 / 252)   # annualised extra drift
    mu = math.log(spot / be) + (r - iv * iv / 2) * T + a * min(T, drift_t)
    z = mu / (iv * math.sqrt(T))
    return N.cdf(z) if side == "call" else N.cdf(-z)


def assess(rec: dict, cfg: dict, bt_row: dict | None = None) -> dict:
    """Prior -> posterior -> fair P(breakeven) vs market. Adds nothing to rec on failure."""
    b = _cfg(cfg)
    n = int(bt_row["n"]) if bt_row else None
    win = float(bt_row[f"win{b['horizon']}"]) if bt_row else None
    prior = prior_from_backtest(win, n, b["prior_shrink_k"])
    feats = evidence(rec, cfg)
    lr = combine(feats, b)
    post = revise_probability(prior, lr)
    exp_r = post * b["avg_win_r"] - (1 - post) * b["avg_loss_r"]
    out = {"prior": round(prior, 3), "lr": round(lr, 2), "post": round(post, 3), "feats": feats,
           "exp_r": round(exp_r, 2),
           "prior_src": f"BT n={n}" if n else "flat 50%", "learned": b.get("learned", False)}

    t = rec.get("trade") or {}
    if t.get("structure") not in (None, "WATCH", "NONE") and t.get("breakeven"):
        side = "call" if rec["bias"] == "LONG" else "put"
        iv = (t.get("iv") or rec.get("atm_iv") or 0) / 100
        r, h = b["rate"], b["horizon"]
        sgn = 1 if side == "call" else -1
        # risk-neutral P(directional move > 0 over the horizon)
        p_rn = N.cdf(sgn * (r - iv * iv / 2) * (h / 252) / (iv * math.sqrt(h / 252))) if iv > 0 else 0.5
        shift = sgn * (N.inv_cdf(min(max(post, EPS), 1 - EPS)) - N.inv_cdf(p_rn))
        kw = dict(spot=rec["close"], be=t["breakeven"], iv=iv, dte=t["dte"], side=side, r=r)
        mkt = p_beyond(**kw)
        mdl = p_beyond(**kw, shift_sd=shift, drift_t=b["drift_days"] / 252)
        if math.isfinite(mkt) and math.isfinite(mdl):
            # execution: paying spread + commission moves your breakeven further away
            be_x = t["breakeven"] + sgn * float(t.get("slip") or 0)
            mdl_x = p_beyond(**{**kw, "be": be_x}, shift_sd=shift, drift_t=b["drift_days"] / 252)
            raw = 100 * (mdl - mkt)
            cost = 100 * (mdl - mdl_x) if math.isfinite(mdl_x) else 0.0
            margin = float(b["margin_pp"])
            out.update(p_be_mkt=round(mkt, 3), p_be_mdl=round(mdl, 3), edge_pp=round(raw, 1),
                       cost_pp=round(cost, 1), margin_pp=margin,
                       usable_pp=round(raw - cost - margin, 1))
    return out


SHORT = {"tt_full": "TT8", "tt_weak": "TT≤5", "rs_with": "RS+", "rs_against": "RS−",
         "vol_surge": "vol≥2x", "adx_strong": "ADX30", "rsi_hot": "RSIhot", "insider": "F4",
         "pcr_against": "PCR−", "tape_with": "tape+", "tape_against": "tape−",
         "nash_with": "Nash+", "nash_against": "Nash−"}


def fmt(bz: dict) -> str:
    f = " ".join(SHORT.get(x, x) for x in bz["feats"]) or "no extra signals"
    s = f"Bayes {bz['prior']:.0%}→{bz['post']:.0%} (×{bz['lr']:.2f}: {f})"
    if "exp_r" in bz:
        s += f" · E[R] {bz['exp_r']:+.2f}R"
    if "usable_pp" in bz:
        ok = "✅" if bz["usable_pp"] > 0 else "✖"
        s += (f"\n   P(BE) {bz['p_be_mdl']:.0%} vs mkt {bz['p_be_mkt']:.0%}: {bz['edge_pp']:+.1f}pp"
              f" − cost {bz['cost_pp']:.1f} − margin {bz['margin_pp']:.0f} = {bz['usable_pp']:+.1f}pp {ok}")
    elif "edge_pp" in bz:
        s += f"\n   P(BE) {bz['p_be_mdl']:.0%} vs mkt {bz['p_be_mkt']:.0%} → edge {bz['edge_pp']:+.0f}pp"
    return s


# ---------- log + learn ----------

LOG_FIELDS = ["date", "ticker", "setup", "bias", "spot", "prior", "lr", "post", "feats",
              "p_be_mdl", "p_be_mkt", "edge_pp", "usable_pp"]


def log(rec: dict, bz: dict, path: Path = LOG) -> None:
    STATE.mkdir(exist_ok=True)
    today = dt.date.today().isoformat()
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if any(r["date"] == today and r["ticker"] == rec["ticker"] and r["bias"] == rec["bias"] for r in rows):
            return
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"date": today, "ticker": rec["ticker"], "setup": rec["setup"], "bias": rec["bias"],
                    "spot": rec["close"], "prior": bz["prior"], "lr": bz["lr"], "post": bz["post"],
                    "feats": ";".join(bz["feats"]), "p_be_mdl": bz.get("p_be_mdl", ""),
                    "p_be_mkt": bz.get("p_be_mkt", ""), "edge_pp": bz.get("edge_pp", ""),
                    "usable_pp": bz.get("usable_pp", "")})


def _closes(ticker: str, start: str) -> list[tuple[str, float]]:
    import yfinance as yf
    df = yf.download(ticker, start=start, interval="1d", progress=False, auto_adjust=True)
    if df is None or df.empty:
        return []
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)
    return [(i.strftime("%Y-%m-%d"), float(c)) for i, c in df["Close"].items()]


def score(rows: list[dict], fetch=_closes, horizon: int = 10) -> list[dict]:
    """outcome = 1 if price moved in the signal's direction after `horizon` sessions."""
    by_t: dict[str, list[dict]] = {}
    for r in rows:
        by_t.setdefault(r["ticker"], []).append(r)
    out = []
    for t, rs in by_t.items():
        start = (dt.date.fromisoformat(min(r["date"] for r in rs)) - dt.timedelta(days=5)).isoformat()
        try:
            bars = fetch(t, start)
        except Exception as e:
            print(f"[bayes] {t}: {e}"); bars = []
        for r in rs:
            after = [c for d, c in bars if d > r["date"]]
            r = dict(r)
            if len(after) >= horizon:
                sgn = 1 if r["bias"] == "LONG" else -1
                r["outcome"] = int(sgn * (after[horizon - 1] - float(r["spot"])) > 0)
            out.append(r)
    return out


def learn(scored: list[dict], b: dict) -> tuple[dict, list[str]]:
    res = [r for r in scored if "outcome" in r]
    W = sum(r["outcome"] for r in res)
    L = len(res) - W
    lr, notes = {}, []
    for k, d in b["lr"].items():
        w_with = sum(1 for r in res if r["outcome"] == 1 and k in r["feats"].split(";"))
        l_with = sum(1 for r in res if r["outcome"] == 0 and k in r["feats"].split(";"))
        n_with = w_with + l_with
        if not n_with:
            continue
        emp = ((w_with + 1) / (W + 2)) / ((l_with + 1) / (L + 2))
        blended = math.exp((n_with * math.log(emp) + b["learn_k"] * math.log(d)) / (n_with + b["learn_k"]))
        lr[k] = round(blended, 3)
        notes.append(f" {SHORT.get(k, k):<7} n={n_with:<3} emp ×{emp:.2f} → use ×{blended:.2f} (default {d})")
    return lr, notes


def brier(rows: list[dict], key: str) -> float | None:
    res = [r for r in rows if "outcome" in r]
    return sum((float(r[key]) - r["outcome"]) ** 2 for r in res) / len(res) if res else None


def calibrate(dry: bool = False) -> str:
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    if LEARNED.exists():
        LEARNED.unlink()                 # learn against defaults, not last week's output
    b = _cfg(cfg)
    if not LOG.exists():
        return "🧮 Bayes calibration — no alerts logged yet."
    with LOG.open(newline="") as f:
        rows = list(csv.DictReader(f))
    sc = score(rows, horizon=b["horizon"])
    res = [r for r in sc if "outcome" in r]
    L = [f"🧮 Bayes calibration — {dt.date.today()}", f"{len(rows)} alerts logged, {len(res)} resolved (+{b['horizon']}d)"]
    if res:
        bp, bq = brier(res, "prior"), brier(res, "post")
        hit = sum(r["outcome"] for r in res) / len(res)
        L.append(f"hit {hit:.0%} · avg prior {sum(float(r['prior']) for r in res)/len(res):.0%}"
                 f" · avg post {sum(float(r['post']) for r in res)/len(res):.0%}")
        L.append(f"Brier prior {bp:.3f} → post {bq:.3f} " + ("✅ signals add info" if bq < bp else "⚠ signals not helping yet"))
        e = [r for r in res if r.get("usable_pp") not in ("", None)]
        pos = [r for r in e if float(r["usable_pp"]) > 0]
        neg = [r for r in e if float(r["usable_pp"]) <= 0]
        if pos:
            L.append(f"usable edge >0: {len(pos)} → hit {sum(r['outcome'] for r in pos)/len(pos):.0%}"
                     + (f" · ≤0: {len(neg)} → hit {sum(r['outcome'] for r in neg)/len(neg):.0%}" if neg else ""))
    min_n = (cfg.get("bayes") or {}).get("learn_min_resolved", 30)
    if len(res) >= min_n:
        lr, notes = learn(sc, b)
        STATE.mkdir(exist_ok=True)
        LEARNED.write_text(json.dumps({"date": dt.date.today().isoformat(), "n": len(res), "lr": lr}, indent=1))
        L += ["Learned LRs (applied from next scan):"] + notes
    else:
        L.append(f"Learning LRs after {min_n} resolved — using defaults.")
    msg = "\n".join(L)
    if not dry:
        sys.path.insert(0, str(ROOT))
        import calib
        calib.send_telegram(msg)
        calib.git_save([str(LEARNED), str(LOG)], "bayes: calibrate")
    return msg


def demo() -> None:
    p = revise_probability(0.42, 1.62)
    print(f"revise_probability(0.42, 1.62) = {p:.2%}")
    rec = {"ticker": "DEMO", "bias": "LONG", "setup": "BREAKOUT", "close": 100.0, "tt_n": 8, "rs_excess": 12,
           "vol_ratio": 2.4, "adx": 31, "rsi": 66, "insider_score": 0, "pcr": 0.8, "tape_up": True,
           "atm_iv": 45, "trade": {"structure": "LONG_CALL", "breakeven": 108.0, "dte": 60, "iv": 45, "slip": 0.20}}
    bz = assess(rec, {}, {"n": 22, "win10": 59})
    print(fmt(bz))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "calibrate":
        print(calibrate(dry="--dry" in sys.argv))
    else:
        demo()
