"""Momentum / breakout setups (Qullamaggie + Stockbee flavoured) on daily bars.

Setup types emitted:
  BREAKOUT   close > N-day high, volume expansion, trend aligned
  COIL       ATR contraction inside an uptrend, close near highs (pre-breakout watch)
  PULLBACK   uptrend, price tagging EMA20, RSI reset, not extended
  EXTENDED   trend intact but RSI > max -> no new entry, manage/scale
  BEARDOWN   below both EMAs, EMA20<EMA50, ADX rising -> put candidate
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ---------- indicators (pure pandas, no TA-lib dependency) ----------

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l = df["high"], df["low"]
    up = h.diff()
    dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, n)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def realised_vol(close: pd.Series, n: int = 20) -> pd.Series:
    """Annualised close-to-close realised vol."""
    return np.log(close).diff().rolling(n).std() * np.sqrt(252)


def enrich(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    d = df.copy()
    d["ema_f"] = ema(d["close"], cfg["ema_fast"])
    d["ema_s"] = ema(d["close"], cfg["ema_slow"])
    d["rsi"] = rsi(d["close"])
    d["adx"] = adx(d)
    d["atr5"] = atr(d, 5)
    d["atr20"] = atr(d, 20)
    d["atr_ratio"] = d["atr5"] / d["atr20"]
    d["hi_n"] = d["high"].rolling(cfg["breakout_lookback"]).max().shift(1)
    d["vol20"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / d["vol20"]
    d["rv20"] = realised_vol(d["close"], 20)
    d["rv60"] = realised_vol(d["close"], 60)
    d["ret5"] = d["close"].pct_change(5)
    d["ret20"] = d["close"].pct_change(20)
    return d


# ---------- setup classification ----------

@dataclass
class Setup:
    ticker: str
    date: str
    close: float
    setup: str          # BREAKOUT / COIL / PULLBACK / EXTENDED / BEARDOWN / NONE
    bias: str           # LONG / SHORT / FLAT
    score: int          # 0-5 conviction
    rsi: float
    adx: float
    vol_ratio: float
    atr_ratio: float
    ret20: float
    rv20: float
    ema_f: float
    ema_s: float
    hi_n: float
    reason: str

    def as_dict(self):
        return asdict(self)


def classify_row(ticker: str, row: pd.Series, cfg: dict) -> Setup:
    c = row["close"]
    up_trend = c > row["ema_f"] > row["ema_s"]
    down_trend = c < row["ema_f"] < row["ema_s"]
    strong = row["adx"] >= cfg["adx_min"]
    vol_x = row["vol_ratio"] >= cfg["vol_mult"]
    broke = c > row["hi_n"]
    coiled = row["atr_ratio"] <= cfg["atr_contraction"]
    near_ema = abs(c - row["ema_f"]) / row["ema_f"] <= cfg["pullback_ema_tol"]
    extended = row["rsi"] >= cfg["rsi_max"]

    setup, bias, score, why = "NONE", "FLAT", 0, []

    if up_trend and broke and vol_x and not extended:
        setup, bias, score = "BREAKOUT", "LONG", 3
        why.append(f"close>{cfg['breakout_lookback']}dH {row['hi_n']:.2f}")
        why.append(f"vol {row['vol_ratio']:.1f}x")
        if strong: score += 1; why.append(f"ADX {row['adx']:.0f}")
        if row["rsi"] >= 60: score += 1
    elif up_trend and extended:
        setup, bias, score = "EXTENDED", "LONG", 1
        why.append(f"RSI {row['rsi']:.0f} - no new entry, trail/scale")
    elif up_trend and coiled and c >= 0.95 * row["hi_n"]:
        setup, bias, score = "COIL", "LONG", 2
        why.append(f"ATR5/20 {row['atr_ratio']:.2f} tight, {100*(c/row['hi_n']-1):+.1f}% from {cfg['breakout_lookback']}dH")
        if strong: score += 1
    elif up_trend and near_ema and cfg["rsi_min"] <= row["rsi"] < cfg["rsi_max"]:
        setup, bias, score = "PULLBACK", "LONG", 2
        why.append(f"tag EMA{cfg['ema_fast']} {row['ema_f']:.2f}, RSI {row['rsi']:.0f}")
        if strong: score += 1
    elif down_trend and strong and row["rsi"] < 45:
        setup, bias, score = "BEARDOWN", "SHORT", 2
        why.append(f"below EMAs, ADX {row['adx']:.0f}, RSI {row['rsi']:.0f}")
        if vol_x: score += 1; why.append(f"vol {row['vol_ratio']:.1f}x")

    return Setup(
        ticker=ticker,
        date=str(row.name.date()),
        close=round(float(c), 2),
        setup=setup,
        bias=bias,
        score=int(min(score, 5)),
        rsi=round(float(row["rsi"]), 1),
        adx=round(float(row["adx"]), 1),
        vol_ratio=round(float(row["vol_ratio"]), 2),
        atr_ratio=round(float(row["atr_ratio"]), 2),
        ret20=round(float(row["ret20"]) * 100, 1),
        rv20=round(float(row["rv20"]) * 100, 1),
        ema_f=round(float(row["ema_f"]), 2),
        ema_s=round(float(row["ema_s"]), 2),
        hi_n=round(float(row["hi_n"]), 2),
        reason="; ".join(why),
    )


def latest_setup(ticker: str, bars: pd.DataFrame, cfg: dict) -> Setup:
    d = enrich(bars, cfg).dropna()
    return classify_row(ticker, d.iloc[-1], cfg)


def all_setups(ticker: str, bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Vectorised-ish: classify every row (for backtest)."""
    d = enrich(bars, cfg).dropna()
    rows = [classify_row(ticker, r, cfg).as_dict() for _, r in d.iterrows()]
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")
