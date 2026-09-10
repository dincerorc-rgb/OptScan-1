"""Market data: daily bars + option chains (yfinance, free, no key)."""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


def daily_bars(ticker: str, years: int = 2) -> pd.DataFrame:
    """OHLCV daily bars, adjusted, ascending. Columns: open high low close volume."""
    df = yf.Ticker(ticker).history(period=f"{years}y", auto_adjust=True)
    if df.empty:
        raise ValueError(f"no price data for {ticker}")
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df.dropna()


def option_chain(ticker: str, dte_min: int, dte_max: int) -> pd.DataFrame:
    """Calls+puts for all expiries inside [dte_min, dte_max]. Adds dte, mid, spread_pct."""
    tk = yf.Ticker(ticker)
    today = dt.date.today()
    frames = []
    for exp in tk.options:
        exp_d = dt.date.fromisoformat(exp)
        dte = (exp_d - today).days
        if dte < dte_min or dte > dte_max:
            continue
        try:
            ch = tk.option_chain(exp)
        except Exception as e:  # yfinance is flaky; skip expiry
            log.warning("%s %s chain failed: %s", ticker, exp, e)
            continue
        for side, df in (("call", ch.calls), ("put", ch.puts)):
            if df is None or df.empty:
                continue
            d = df.copy()
            d["side"] = side
            d["expiry"] = exp
            d["dte"] = dte
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["bid"] = out["bid"].fillna(0.0)
    out["ask"] = out["ask"].fillna(0.0)
    out["mid"] = (out["bid"] + out["ask"]) / 2
    out["spread_pct"] = ((out["ask"] - out["bid"]) / out["mid"].replace(0, float("nan"))).abs()
    out["iv"] = out["impliedVolatility"].astype(float)
    out["oi"] = out["openInterest"].fillna(0).astype(int)
    out["vol"] = out["volume"].fillna(0).astype(int)
    return out


def spot(ticker: str) -> float:
    fi = yf.Ticker(ticker).fast_info
    px = fi.get("last_price") or fi.get("lastPrice")
    if not px:
        px = daily_bars(ticker, 1)["close"].iloc[-1]
    return float(px)
