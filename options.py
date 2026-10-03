"""Options layer: IV vs RV, IV rank, structure choice, strike/expiry picker, sizing.

IV rank needs history. yfinance gives no IV history, so we log ATM IV per scan
to state/iv_history.csv and rank against it. Until >=60 obs exist we fall back to
IV percentile vs 1y realised-vol distribution (flagged as 'est').
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from data import option_chain

STATE = Path(__file__).resolve().parent / "state"
STATE.mkdir(exist_ok=True)
IV_HIST = STATE / "iv_history.csv"


# ---------- BS delta (chain delta not supplied by yfinance) ----------

def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_delta(spot, strike, dte, iv, side, r=0.045):
    t = max(dte, 1) / 365
    if iv <= 0 or spot <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (r + 0.5 * iv**2) * t) / (iv * math.sqrt(t))
    return _ncdf(d1) if side == "call" else _ncdf(d1) - 1


# ---------- IV metrics ----------

def atm_iv(chain: pd.DataFrame, spot: float) -> tuple[float, int, str]:
    """ATM IV from nearest-expiry call with strike closest to spot."""
    if chain.empty:
        return float("nan"), 0, ""
    calls = chain[chain.side == "call"].copy()
    calls["dist"] = (calls.strike - spot).abs()
    row = calls.sort_values(["dte", "dist"]).iloc[0]
    return float(row.iv), int(row.dte), str(row.expiry)


def put_call_ratio(chain: pd.DataFrame) -> float:
    if chain.empty:
        return float("nan")
    pv = chain[chain.side == "put"].vol.sum()
    cv = chain[chain.side == "call"].vol.sum()
    return round(pv / cv, 2) if cv else float("nan")


def log_iv(ticker: str, iv: float):
    if not np.isfinite(iv):
        return
    today = dt.date.today().isoformat()
    line = f"{today},{ticker},{iv:.4f}\n"
    if IV_HIST.exists():
        hist = pd.read_csv(IV_HIST, names=["date", "ticker", "iv"])
        if ((hist.date == today) & (hist.ticker == ticker)).any():
            return
    with IV_HIST.open("a") as f:
        f.write(line)


def iv_rank(ticker: str, iv: float, rv_series: pd.Series) -> tuple[float, str]:
    """(rank 0-100, source). Own IV history if >=60 obs, else RV-distribution estimate."""
    if IV_HIST.exists():
        hist = pd.read_csv(IV_HIST, names=["date", "ticker", "iv"])
        h = hist[hist.ticker == ticker].iv.tail(252)
        if len(h) >= 60:
            lo, hi = h.min(), h.max()
            return (round(100 * (iv - lo) / (hi - lo), 0) if hi > lo else 50.0), "hist"
    rv = rv_series.dropna().tail(252)
    if rv.empty or not np.isfinite(iv):
        return float("nan"), "n/a"
    return round(float((rv < iv).mean() * 100), 0), "est"


# ---------- structure + contract selection ----------

@dataclass
class Trade:
    structure: str        # LONG_CALL / BULL_CALL_SPREAD / LONG_PUT / BEAR_PUT_SPREAD / WATCH / NONE
    expiry: str
    dte: int
    legs: str             # human legs e.g. "+320C / -335C"
    debit: float          # per spread / contract ($ per share)
    breakeven: float
    delta: float
    iv: float
    contracts: int
    premium_usd: float
    note: str

    def as_dict(self):
        return asdict(self)


def _pick(chain: pd.DataFrame, side: str, spot: float, dte_lo: int, dte_hi: int,
          d_lo: float, d_hi: float, ocfg: dict) -> pd.Series | None:
    c = chain[(chain.side == side) & (chain.dte.between(dte_lo, dte_hi))].copy()
    if c.empty:
        return None
    c["delta"] = [bs_delta(spot, k, d, v, side) for k, d, v in zip(c.strike, c.dte, c.iv)]
    c = c[c.oi >= ocfg["min_open_interest"]]
    c = c[c.spread_pct <= ocfg["max_spread_pct"]]
    c = c[c.mid > 0.05]
    band = c[c.delta.abs().between(d_lo, d_hi)]
    if band.empty:
        return None
    # prefer middle of DTE window, then delta closest to mid-band
    tgt_dte = (dte_lo + dte_hi) / 2
    band = band.assign(_k=(band.dte - tgt_dte).abs() + 30 * (band.delta.abs() - (d_lo + d_hi) / 2).abs())
    return band.sort_values("_k").iloc[0]


def _size(debit: float, acct: dict) -> tuple[int, float]:
    cap = acct["net_liq"] * acct["max_premium_pct"] / 100
    per = debit * 100
    n = int(cap // per) if per > 0 else 0
    return max(n, 0), round(n * per, 0)


def suggest(ticker: str, bias: str, setup: str, spot: float, ivr: float,
            chain: pd.DataFrame, ocfg: dict, acct: dict, horizon: str = "swing") -> Trade:
    if bias == "FLAT" or setup in ("NONE", "EXTENDED") or chain.empty:
        return Trade("WATCH", "", 0, "", 0, 0, 0, float("nan"), 0, 0, "no entry / manage existing")

    if horizon == "sepa":
        dte_lo, dte_hi = ocfg.get("sepa_dte", [60, 120])
        d_lo, d_hi = ocfg.get("sepa_delta", [0.65, 0.75])
    else:
        dte_lo, dte_hi = ocfg["leaps_dte"] if horizon == "leaps" else ocfg["swing_dte"]
        d_lo, d_hi = ocfg["long_call_delta"]
    side = "call" if bias == "LONG" else "put"
    long_leg = _pick(chain, side, spot, dte_lo, dte_hi, d_lo, d_hi, ocfg)
    if long_leg is None:
        return Trade("NONE", "", 0, "", 0, 0, 0, float("nan"), 0, 0,
                     "no liquid contract in DTE/delta window")

    rich = np.isfinite(ivr) and ivr >= ocfg["iv_rich_rank"]
    cheap = np.isfinite(ivr) and ivr <= ocfg["iv_cheap_rank"]

    # Outright long premium only when IV is not rich; otherwise debit spread.
    if not rich and (cheap or setup == "BREAKOUT" or horizon == "sepa"):
        structure = "LONG_CALL" if side == "call" else "LONG_PUT"
        debit = float(long_leg.mid)
        be = long_leg.strike + debit if side == "call" else long_leg.strike - debit
        legs = f"+{long_leg.strike:g}{'C' if side=='call' else 'P'}"
        note = "IV cheap - buy premium" if cheap else "IV mid - outright ok on breakout"
        if horizon == "sepa":
            note = "SEPA: stop on underlying, not option price"
    else:
        short = chain[(chain.side == side) & (chain.expiry == long_leg.expiry)].copy()
        short["delta"] = [bs_delta(spot, k, d, v, side) for k, d, v in zip(short.strike, short.dte, short.iv)]
        short = short[short.mid > 0]
        if side == "call":
            short = short[short.strike > long_leg.strike]
        else:
            short = short[short.strike < long_leg.strike]
        if short.empty:
            return Trade("NONE", "", 0, "", 0, 0, 0, float("nan"), 0, 0, "no short leg")
        sl = short.iloc[(short.delta.abs() - ocfg["spread_short_delta"]).abs().argsort().iloc[0]]
        debit = float(long_leg.mid - sl.mid)
        if debit <= 0:
            return Trade("NONE", "", 0, "", 0, 0, 0, float("nan"), 0, 0, "spread priced <=0, skip")
        structure = "BULL_CALL_SPREAD" if side == "call" else "BEAR_PUT_SPREAD"
        be = long_leg.strike + debit if side == "call" else long_leg.strike - debit
        sym = "C" if side == "call" else "P"
        legs = f"+{long_leg.strike:g}{sym} / -{sl.strike:g}{sym}"
        width = abs(sl.strike - long_leg.strike)
        note = f"IV rich ({ivr:.0f}) - spread, max {width/debit:.1f}x" if rich else f"spread, max {width/debit:.1f}x"

    n, usd = _size(debit, acct)
    if n == 0:
        note += f" | 1x = ${debit*100:,.0f} > {acct['max_premium_pct']}% cap"
    return Trade(structure, str(long_leg.expiry), int(long_leg.dte), legs, round(debit, 2),
                 round(float(be), 2), round(float(long_leg.delta), 2), round(float(long_leg.iv) * 100, 1),
                 n, usd, note)


def analyse(ticker: str, spot: float, rv_series: pd.Series, ocfg: dict) -> dict:
    """Pull chain once; return iv metrics + chain for suggest()."""
    lo = min(ocfg["swing_dte"][0], 7)
    hi = ocfg["leaps_dte"][1]
    chain = option_chain(ticker, lo, hi)
    iv, dte, exp = atm_iv(chain[chain.dte.between(*ocfg["swing_dte"])] if not chain.empty else chain, spot)
    log_iv(ticker, iv)
    ivr, src = iv_rank(ticker, iv, rv_series)
    rv20 = float(rv_series.dropna().iloc[-1]) if not rv_series.dropna().empty else float("nan")
    return {
        "chain": chain,
        "atm_iv": round(iv * 100, 1) if np.isfinite(iv) else float("nan"),
        "atm_dte": dte,
        "iv_rank": ivr,
        "iv_rank_src": src,
        "iv_rv": round(iv / rv20, 2) if rv20 and np.isfinite(iv) else float("nan"),
        "pcr": put_call_ratio(chain),
    }
