"""Momentum / breakout setups (Qullamaggie + Stockbee flavoured) on daily bars.

Setup types emitted:
  BREAKOUT   close > N-day high, volume expansion, trend aligned
  COIL       ATR contraction inside an uptrend, close near highs (pre-breakout watch)
  PULLBACK   uptrend, price tagging EMA20, RSI reset, not extended
  EXTENDED   trend intact but RSI > max -> no new entry, manage/scale
  BEARDOWN   below both EMAs, EMA20<EMA50, ADX rising -> put candidate

Minervini SEPA layer (momentum.tt_* in config.yaml):
  Trend Template, 8 rules -> tt_n (0-8)
    1 close > SMA150 and SMA200      5 close > SMA50
    2 SMA150 > SMA200                6 close >= 1.30 x 52w low
    3 SMA200 rising vs 21 bars ago   7 close >= 0.75 x 52w high
    4 SMA50 > SMA150 and SMA200      8 RS: weighted 3/6/9/12m return beats SPY
                                       and RS line (close/SPY) > its 50d SMA
  COIL only fires when TT passes (= VCP-style SEPA setup); BREAKOUT/PULLBACK
  get +1 score when TT passes. Every long setup carries pivot + stop
  (stop = 10-bar low, never more than tt_max_stop_pct below the pivot).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
_BENCH: dict[str, pd.Series | None] = {}


def bench_close(symbol: str = "SPY") -> pd.Series | None:
    """Benchmark closes for RS, fetched once per run. None if unavailable."""
    if symbol not in _BENCH:
        try:
            import data
            _BENCH[symbol] = data.daily_bars(symbol, 3)["close"]
        except Exception as e:  # RS rule then counts as not met
            log.warning("benchmark %s unavailable: %s", symbol, e)
            _BENCH[symbol] = None
    return _BENCH[symbol]


def _weighted_ret(c: pd.Series) -> pd.Series:
    """IBD-style RS weighting: 40% last 3m, 20% each of 6/9/12m."""
    return (2 * c.pct_change(63) + c.pct_change(126) + c.pct_change(189) + c.pct_change(252)) / 5


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


def enrich(df: pd.DataFrame, cfg: dict, bench: pd.Series | None = None) -> pd.DataFrame:
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
    # --- Minervini Trend Template inputs ---
    c = d["close"]
    d["sma50"] = c.rolling(50).mean()
    d["sma150"] = c.rolling(150).mean()
    d["sma200"] = c.rolling(200).mean()
    d["sma200_prev"] = d["sma200"].shift(cfg.get("tt_sma200_rising_bars", 21))
    d["low52"] = d["low"].rolling(252, min_periods=200).min()
    d["high52"] = d["high"].rolling(252, min_periods=200).max()
    d["low10"] = d["low"].rolling(10).min()
    d["vol50"] = d["volume"].rolling(50).mean()
    d["vol_dry"] = d["volume"].rolling(5).mean() / d["vol50"]
    if bench is not None:
        b = bench.reindex(d.index).ffill()
        d["rs_excess"] = _weighted_ret(c) - _weighted_ret(b)
        rs_line = c / b
        d["rs_line_up"] = (rs_line > rs_line.rolling(50).mean()).astype(float)
        d.loc[rs_line.isna(), "rs_line_up"] = 0.0
    else:
        d["rs_excess"] = 0.0
        d["rs_line_up"] = 0.0
    d["rs_excess"] = d["rs_excess"].fillna(-1.0)
    return d


def trend_template(row: pd.Series) -> tuple[int, list[str]]:
    """Minervini Trend Template. Returns (rules met 0-8, names of failed rules)."""
    c = row["close"]
    rules = {
        "c>MA150/200": c > row["sma150"] and c > row["sma200"],
        "MA150>MA200": row["sma150"] > row["sma200"],
        "MA200 rising": row["sma200"] > row["sma200_prev"],
        "MA50>MA150/200": row["sma50"] > row["sma150"] and row["sma50"] > row["sma200"],
        "c>MA50": c > row["sma50"],
        "+30% off 52wL": c >= 1.30 * row["low52"],
        "within 25% of 52wH": c >= 0.75 * row["high52"],
        "RS>SPY": row["rs_excess"] > 0 and row["rs_line_up"] > 0,
    }
    return sum(bool(v) for v in rules.values()), [k for k, v in rules.items() if not v]


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
    tt_n: int           # Minervini Trend Template rules met (0-8)
    rs_excess: float    # weighted 3/6/9/12m return minus SPY, %
    pivot: float        # buy-stop level (long setups)
    stop: float         # protective stop on the underlying
    risk_pct: float     # pivot -> stop distance, %
    reason: str

    def as_dict(self):
        return asdict(self)


def classify_row(ticker: str, row: pd.Series, cfg: dict) -> Setup:
    c = row["close"]
    tt_on = cfg.get("tt_enabled", True) and ticker not in (cfg.get("tt_exempt") or [])
    tt_n, tt_fail = trend_template(row)
    tt_pass = tt_n >= cfg.get("tt_min_rules", 8)
    tt_req = set(cfg.get("tt_require_for") or []) if tt_on else set()
    tt_bonus = set(cfg.get("tt_bonus_for") or []) if tt_on else set()
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
    elif up_trend and coiled and c >= 0.95 * row["hi_n"] and (tt_pass or "COIL" not in tt_req):
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

    pivot = stop = risk = float("nan")
    if bias == "LONG":
        if setup in tt_bonus and tt_pass:
            score += 1
        if setup == "COIL" and tt_on and tt_pass:
            why.insert(0, "VCP/SEPA")
            if row["vol_dry"] <= cfg.get("tt_vol_dry", 0.8):
                score += 1; why.append(f"vol dry-up {row['vol_dry']:.2f}")
        pivot = float(c) if setup == "BREAKOUT" else float(row["hi_n"])
        max_stop = pivot * (1 - cfg.get("tt_max_stop_pct", 8) / 100)
        stop = max(float(row["low10"]), max_stop)
        if stop >= pivot:
            stop = max_stop
        risk = 100 * (pivot - stop) / pivot
    if tt_on and setup != "NONE" and bias == "LONG":
        why.append(f"TT {tt_n}/8" + ("" if tt_pass else f" (fail: {', '.join(tt_fail[:3])})"))

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
        tt_n=int(tt_n),
        rs_excess=round(float(row["rs_excess"]) * 100, 1),
        pivot=round(pivot, 2),
        stop=round(stop, 2),
        risk_pct=round(risk, 1),
        reason="; ".join(why),
    )


def _bench(cfg: dict) -> pd.Series | None:
    return bench_close(cfg.get("tt_bench", "SPY")) if cfg.get("tt_enabled", True) else None


def latest_setup(ticker: str, bars: pd.DataFrame, cfg: dict) -> Setup:
    d = enrich(bars, cfg, _bench(cfg)).dropna()
    return classify_row(ticker, d.iloc[-1], cfg)


def all_setups(ticker: str, bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Vectorised-ish: classify every row (for backtest)."""
    d = enrich(bars, cfg, _bench(cfg)).dropna()
    rows = [classify_row(ticker, r, cfg).as_dict() for _, r in d.iterrows()]
    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")
