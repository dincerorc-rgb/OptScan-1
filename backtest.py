"""Signal-only validation: forward equity returns after each setup.

Not an options P&L backtest (needs paid chain history). Answers: does this setup
produce directional edge over 5/10/20 sessions on THIS ticker? That's what a
long-premium buyer needs to know before paying theta.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals import all_setups

HORIZONS = (5, 10, 20)


def run(ticker: str, bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    s = all_setups(ticker, bars, cfg)
    close = bars["close"].reindex(s.index)
    for h in HORIZONS:
        s[f"fwd{h}"] = close.shift(-h) / close - 1
    s = s[s.setup != "NONE"].dropna(subset=[f"fwd{HORIZONS[-1]}"])
    # de-cluster: keep first signal, skip repeats within 5 sessions of same setup
    keep, last = [], {}
    for d, r in s.iterrows():
        k = r.setup
        if k in last and (d - last[k]).days < 7:
            keep.append(False)
        else:
            keep.append(True)
            last[k] = d
    s = s[keep]
    rows = []
    for setup, g in s.groupby("setup"):
        sign = -1 if g.bias.iloc[0] == "SHORT" else 1
        row = {"ticker": ticker, "setup": setup, "n": len(g)}
        for h in HORIZONS:
            r = sign * g[f"fwd{h}"]
            row[f"win{h}"] = round(100 * (r > 0).mean(), 0)
            row[f"avg{h}"] = round(100 * r.mean(), 2)
            row[f"tstat{h}"] = round(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))), 2) if len(r) > 2 and r.std() > 0 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(bt: pd.DataFrame, setup: str, h: int = 10) -> str:
    """PASS / WEAK / FAIL / NA gate used by the scanner."""
    r = bt[bt.setup == setup]
    if r.empty or r.n.iloc[0] < 8:
        return "NA"
    t, w = r[f"tstat{h}"].iloc[0], r[f"win{h}"].iloc[0]
    if t >= 1.5 and w >= 55:
        return "PASS"
    if t >= 0.8 and w >= 50:
        return "WEAK"
    return "FAIL"
