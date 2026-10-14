"""Tom Nash quality layer (Oct 2026): fundamentals -> grade -> better option contracts.

Implements the quantitative parts of Nash's 12-step process with free yfinance data:
  Step 2  sanity check (2 of 3: revenue growth, profitable, cash > debt)
  Step 3  pre-screen + instant kills + Rule of 40 + scalability
  Step 6  125-point scorecard (7 data categories here; 6 subjective ones from config.yaml)
  Step 8  loan-officer ratios (interest cover, assets/liabilities ex-goodwill, current ratio, D/E)
  Step 12 quick DCF (bull 8% / base 10% / bear 12%, terminal growth from config) + PEG

Grade -> what the scanner does with the option contract (see apply()):
  A  score >= 100, no kills  -> quality earns time: LEAPS allowed outside cheap IV, deep-ITM (stock-replacement) delta
  B  75-99, no kills         -> scanner default
  C  < 75 or 1 kill          -> no LEAPS, debit spreads only, half size
  X  2+ kills / red flags    -> calls: spreads only, quarter size, -1★ (mode 'gate' blocks them); puts: +1★
  ETF                        -> not graded, scanner default
Puts against an A/B name are spreads only, swing horizon, -1★ (you are shorting quality).

Fundamentals change quarterly, so results are cached in state/nash_cache.json (refresh every cache_days).
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import math
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
STATE = Path(__file__).resolve().parent / "state"
CACHE = STATE / "nash_cache.json"
SUBJ = ["moat", "ceo", "pricing_power", "trend", "culture", "talent"]


# ---------- helpers ----------

def _f(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _row(df: pd.DataFrame | None, *names) -> pd.Series | None:
    """First matching row of a yfinance statement, newest column first, NaNs dropped."""
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce").dropna()
            if not s.empty:
                return s.sort_index(ascending=False)
    return None


def _band(x, lo, hi, reverse=False):
    """0/5/10 points: below lo, between, above hi (reverse: lower is better)."""
    if x is None:
        return 0
    if reverse:
        return 10 if x < lo else (5 if x <= hi else 0)
    return 10 if x > hi else (5 if x >= lo else 0)


# ---------- data ----------

def fetch(ticker: str) -> dict:
    """Raw fundamentals from yfinance. Every field may be None."""
    import yfinance as yf
    tk = yf.Ticker(ticker)
    info = tk.info or {}
    out = {"ticker": ticker, "asof": dt.date.today().isoformat(),
           "etf": info.get("quoteType") in ("ETF", "MUTUALFUND", "INDEX")}
    if out["etf"]:
        return out
    g = info.get
    out.update(mcap=_f(g("marketCap")), short_pct=_f(g("shortPercentOfFloat")),
               inst_pct=_f(g("heldPercentInstitutions")), op_margin=_f(g("operatingMargins")),
               rev_growth_q=_f(g("revenueGrowth")), cash=_f(g("totalCash")), debt=_f(g("totalDebt")),
               fcf_ttm=_f(g("freeCashflow")), shares=_f(g("sharesOutstanding")),
               fwd_pe=_f(g("forwardPE")), peg=_f(g("trailingPegRatio") or g("pegRatio")),
               eps_growth=_f(g("earningsGrowth")), price=_f(g("currentPrice") or g("regularMarketPrice")))

    def safe(attr):
        try:
            return getattr(tk, attr)
        except Exception as e:
            log.warning("%s %s: %s", ticker, attr, e)
            return None

    qcf, qbs, qis = safe("quarterly_cashflow"), safe("quarterly_balance_sheet"), safe("quarterly_income_stmt")
    acf, ais = safe("cashflow"), safe("income_stmt")

    fcf_q = _row(qcf, "Free Cash Flow")
    out["fcf_q"] = [float(v) for v in fcf_q.head(6)] if fcf_q is not None else None
    cash_q = _row(qbs, "Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents")
    debt_q = _row(qbs, "Total Debt")
    if cash_q is not None:
        d = debt_q.reindex(cash_q.index).fillna(0) if debt_q is not None else 0 * cash_q
        out["netcash_q"] = [float(v) for v in (cash_q - d).head(6)]
    for k, names in {"assets": ("Total Assets",), "liab": ("Total Liabilities Net Minority Interest",),
                     "goodwill": ("Goodwill And Other Intangible Assets", "Goodwill"),
                     "cur_assets": ("Current Assets",), "cur_liab": ("Current Liabilities",),
                     "equity": ("Stockholders Equity", "Common Stock Equity")}.items():
        s = _row(qbs, *names)
        out[k] = float(s.iloc[0]) if s is not None else None
    for k, names in {"op_inc_q": ("Operating Income",), "int_exp_q": ("Interest Expense",)}.items():
        s = _row(qis, *names)
        out[k] = float(s.head(4).sum()) if s is not None and len(s) >= 4 else None

    rev_a, exp_a = _row(ais, "Total Revenue"), _row(ais, "Total Expenses")
    fcf_a = _row(acf, "Free Cash Flow")
    out["rev_a"] = [float(v) for v in rev_a.head(4)] if rev_a is not None else None
    out["exp_a"] = [float(v) for v in exp_a.head(4)] if exp_a is not None else None
    out["fcf_a"] = [float(v) for v in fcf_a.head(4)] if fcf_a is not None else None

    try:  # EPS beats/misses, last 4 reported quarters
        ed = tk.get_earnings_dates(limit=12)
        ed = ed.dropna(subset=["Reported EPS", "EPS Estimate"]).head(4)
        out["misses_4q"] = int((ed["Reported EPS"] < ed["EPS Estimate"]).sum()) if len(ed) else None
    except Exception:
        out["misses_4q"] = None
    try:  # 5-year total return vs SPY
        px = yf.download([ticker, "SPY"], period="5y", auto_adjust=True, progress=False)["Close"].dropna()
        if len(px) > 1000:
            r = px.iloc[-1] / px.iloc[0] - 1
            out["ret5_vs_spy_pp"] = round(100 * float(r[ticker] - r["SPY"]), 1)
    except Exception:
        pass
    return out


def load(ticker: str, cfg: dict, refresh: bool = False) -> dict:
    """Cached fundamentals; refetch after cache_days."""
    days = (cfg.get("nash") or {}).get("cache_days", 7)
    cache = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text())
        except Exception:
            cache = {}
    c = cache.get(ticker)
    if c and not refresh and (dt.date.today() - dt.date.fromisoformat(c["asof"])).days < days:
        return c
    try:
        c = fetch(ticker)
    except Exception as e:
        log.warning("%s nash fetch failed: %s", ticker, e)
        if c:
            return c  # stale beats nothing
        return {"ticker": ticker, "asof": dt.date.today().isoformat(), "error": str(e)}
    cache[ticker] = c
    STATE.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1, default=str))
    return c


# ---------- scoring ----------

def dcf(f: dict, ncfg: dict) -> dict | None:
    """Nash's 10-20 min DCF: 3y averages, trimmed growth, 5y FCF, terminal g, +cash -debt / shares."""
    rev, fcf, sh = f.get("rev_a"), f.get("fcf_a"), f.get("shares")
    if not rev or len(rev) < 2 or not fcf or not sh:
        return None
    gr = [rev[i] / rev[i + 1] - 1 for i in range(min(3, len(rev) - 1)) if rev[i + 1] > 0]
    mg = [fcf[i] / rev[i] for i in range(min(3, len(fcf), len(rev))) if rev[i] > 0]
    if not gr or not mg:
        return None
    g = min(sum(gr) / len(gr) * ncfg.get("dcf_trim", 0.8), ncfg.get("dcf_max_growth", 0.30))
    m = sum(mg) / len(mg)
    if m < 0.02:
        return {"growth": round(g, 3), "fcf_margin": round(m, 3), "note": "FCF margin < 2% — DCF n/a"}
    tg = ncfg.get("terminal_growth", 0.04)
    out = {"growth": round(g, 3), "fcf_margin": round(m, 3)}
    for case, r in (("bull", 0.08), ("base", 0.10), ("bear", 0.12)):
        pv, r_ = 0.0, rev[0]
        for y in range(1, 6):
            r_ *= 1 + g
            pv += r_ * m / (1 + r) ** y
        term = r_ * m * (1 + tg) / (r - tg) / (1 + r) ** 5
        eq = pv + term + (f.get("cash") or 0) - (f.get("debt") or 0)
        out[case] = round(eq / sh, 2)
    return out


def score(f: dict, cfg: dict) -> dict:
    ncfg = cfg.get("nash") or {}
    t = f["ticker"]
    if f.get("etf") or t in (ncfg.get("exempt") or []):
        return {"ticker": t, "grade": "ETF", "score": None, "kills": [], "flags": []}
    if f.get("error"):
        return {"ticker": t, "grade": "?", "score": None, "kills": [], "flags": [f"no data: {f['error'][:60]}"]}

    cash, debt = f.get("cash"), f.get("debt") or 0
    rev_a, exp_a = f.get("rev_a") or [], f.get("exp_a") or []
    rg = f.get("rev_growth_q")
    rg_a = rev_a[0] / rev_a[1] - 1 if len(rev_a) >= 2 and rev_a[1] > 0 else None
    rg_prev = rev_a[1] / rev_a[2] - 1 if len(rev_a) >= 3 and rev_a[2] > 0 else None
    eg_a = exp_a[0] / exp_a[1] - 1 if len(exp_a) >= 2 and exp_a[1] > 0 else None
    growth = rg if rg is not None else rg_a
    om = f.get("op_margin")
    fcf_margin = (f["fcf_ttm"] / rev_a[0]) if f.get("fcf_ttm") is not None and rev_a and rev_a[0] > 0 else None
    ro40 = round(100 * (growth + fcf_margin)) if growth is not None and fcf_margin is not None else None
    scal = (rg_a > eg_a) if rg_a is not None and eg_a is not None else None

    # ----- Step 6 scorecard: data categories -----
    ratio = (cash / debt) if cash is not None and debt > 0 else (float("inf") if cash else None)
    pts = {
        "cash_debt": 0 if ratio is None or ratio < 1 else (10 if ratio >= 2 else 5),
        "rev_growth": _band(None if growth is None else 100 * growth, 10, 20),
        "op_margin": _band(None if om is None else 100 * om, 10, 25),
        "short_int": _band(None if f.get("short_pct") is None else 100 * f["short_pct"], 5, 10, reverse=True),
        "inst_own": _band(None if f.get("inst_pct") is None else 100 * f["inst_pct"], 30, 60),
        "scalable": 10 if scal else 0,
        "ret5_vs_spy": _band(f.get("ret5_vs_spy_pp"), 10, 20),
    }
    subj = (ncfg.get("subjective") or {}).get(t) or {}
    est = [k for k in SUBJ if k not in subj]
    for k in SUBJ:
        pts[k] = int(subj.get(k, ncfg.get("subjective_default", 5)))
    total = sum(pts.values())

    # ----- Step 3 instant kills (only on real data) -----
    kills = []
    if f.get("mcap") is not None and f["mcap"] < 1e9:
        kills.append("mcap<$1B")
    if f.get("short_pct") is not None and f["short_pct"] > 0.10:
        kills.append(f"SI {100*f['short_pct']:.0f}%")
    if cash is not None and debt > cash:
        kills.append("debt>cash")
    if om is not None and om < ncfg.get("min_op_margin", 0.20):
        kills.append(f"opm {100*om:.0f}%")
    if growth is not None and growth < 0.10 and rg_prev is not None and growth < rg_prev:
        kills.append(f"rev {100*growth:+.0f}% & falling")
    if (f.get("misses_4q") or 0) >= 2:
        kills.append(f"{f['misses_4q']} misses/4Q")
    founded = (ncfg.get("founded") or {}).get(t)
    if founded and dt.date.today().year - int(founded) < 10:
        kills.append("<10y old")

    # ----- Step 3 pre-screen + Step 2 sanity + Step 8 loan officer: flags -----
    flags = []
    fq = f.get("fcf_q") or []
    if fq and min(fq[:4]) <= 0:
        flags.append("FCF−")
    nq = f.get("netcash_q") or []
    if nq and min(nq[:4]) <= 0:
        flags.append("netcash−")
    if ro40 is not None and ro40 < 40:
        flags.append(f"Ro40 {ro40}")
    if scal is False:
        flags.append("exp>rev")
    sanity = sum([bool(rg_a and rg_a > 0), bool(om and om > 0), bool(cash is not None and cash >= debt)])
    if sanity < 2:
        flags.append(f"sanity {sanity}/3")
    lo = {}
    if f.get("op_inc_q") is not None and f.get("int_exp_q"):
        lo["icov"] = round(f["op_inc_q"] / abs(f["int_exp_q"]), 1)
        if lo["icov"] < 4:
            flags.append(f"icov {lo['icov']}x")
    if f.get("assets") and f.get("liab"):
        lo["a_l"] = round((f["assets"] - (f.get("goodwill") or 0)) / f["liab"], 2)
        if lo["a_l"] < 2:
            flags.append(f"A/L {lo['a_l']}")
    if f.get("cur_assets") and f.get("cur_liab"):
        lo["cur"] = round(f["cur_assets"] / f["cur_liab"], 2)
        if lo["cur"] < 1:
            flags.append(f"cur {lo['cur']}")
    if f.get("liab") and f.get("equity") and f["equity"] > 0:
        lo["de"] = round(f["liab"] / f["equity"], 2)
        if lo["de"] >= 4:
            flags.append(f"D/E {lo['de']}")

    # ----- Step 12 valuation -----
    val = dcf(f, ncfg)
    px = f.get("price")
    if val and px and "bear" in val:
        val["vs_px"] = round(100 * (val["base"] / px - 1))
        if val["bear"] < px:
            flags.append("bear DCF < px")
    peg = f.get("peg")
    if peg is not None and peg > 2:
        flags.append(f"PEG {peg:.1f}")

    hard = len(kills) + (1 if ("FCF−" in flags and "debt>cash" in kills) else 0)
    if hard >= 2:
        grade = "X"
    elif hard == 1 or total < ncfg.get("grade_b", 75):
        grade = "C"
    elif total >= ncfg.get("grade_a", 100):
        grade = "A"
    else:
        grade = "B"
    return {"ticker": t, "grade": grade, "score": total, "est": est, "pts": pts, "kills": kills,
            "flags": flags, "ro40": ro40, "loan": lo, "dcf": val, "peg": peg, "asof": f.get("asof")}


def grade(ticker: str, cfg: dict) -> dict:
    return score(load(ticker, cfg), cfg)


# ---------- effect on the option contract ----------

def apply(rec: dict, ns: dict, horizon: str, ocfg: dict, cfg: dict) -> dict:
    """Decide how quality changes the contract. Returns kwargs for options.suggest + size/score tweaks."""
    ncfg = cfg.get("nash") or {}
    mode = ncfg.get("mode", "tilt")
    g = ns.get("grade")
    out = {"horizon": horizon, "force_spread": False, "delta": None, "size_mult": 1.0,
           "score_adj": 0, "block": None, "why": ""}
    if mode == "off" or g in (None, "ETF", "?"):
        return out
    long = rec.get("bias") == "LONG"
    val = ns.get("dcf") or {}
    # Nash wants even the bear case attractive; for contract choice we act on the base case (shown either way)
    px = rec.get("close") or 0
    rich = (px and "base" in val and val["base"] < px) or (ns.get("peg") or 0) > 2

    if long:
        if g == "A":
            if horizon == "swing" and rec.get("setup") in ("PULLBACK", "COIL", "BREAKOUT") \
                    and not (rec.get("iv_rank", 0) >= ocfg["iv_rich_rank"]):
                out["horizon"] = "leaps"
            if out["horizon"] == "leaps":
                out["delta"] = ncfg.get("leaps_delta_a", [0.70, 0.80])
                out["why"] = "Nash A: quality earns time → LEAPS, deep ITM"
                if rich:
                    out["force_spread"] = True
                    out["why"] += "; valuation rich → spread"
        elif g == "C":
            if out["horizon"] == "leaps":
                out["horizon"] = "swing"
            out.update(force_spread=True, size_mult=0.5, why="Nash C: no LEAPS, spread, ½ size")
            if mode == "gate":
                out["block"] = "Nash C — fails Nash screen (gate mode)"
        elif g == "X":
            out.update(horizon="swing", force_spread=True, size_mult=0.25, score_adj=-1,
                       why="Nash X: weak business → spread only, ¼ size, −1★")
            if mode == "gate":
                out["block"] = "Nash X — off the island (gate mode)"
    else:
        if g in ("A", "B"):
            out.update(horizon="swing", force_spread=True, score_adj=-1,
                       why=f"Nash {g}: shorting quality → spread, swing, −1★")
        elif g == "X":
            out.update(score_adj=1, why="Nash X: weak fundamentals confirm the put, +1★")
    return out


# ---------- formatting ----------

def fmt_line(ns: dict) -> str:
    g = ns.get("grade")
    if g in ("ETF", None):
        return ""
    if g == "?":
        return f"Nash ? ({'; '.join(ns.get('flags', []))})"
    s = f"Nash <b>{g}</b> {ns['score']}{'~' if ns.get('est') else ''}/130"
    s += " · kills: " + (", ".join(ns["kills"]) if ns["kills"] else "none")
    if ns.get("ro40") is not None:
        s += f" · Ro40 {ns['ro40']}"
    v = ns.get("dcf") or {}
    if "base" in v:
        s += f" · DCF {v['bear']:.0f}/{v['base']:.0f}/{v['bull']:.0f} ({v['vs_px']:+d}%)"
    if ns.get("peg") is not None:
        s += f" · PEG {ns['peg']:.1f}"
    fl = [x for x in ns.get("flags", []) if not x.startswith("bear") and not x.startswith("PEG")]
    if fl:
        s += "\n   flags: " + ", ".join(fl)
    return s


def board(tickers: list[str], cfg: dict, refresh: bool = False) -> str:
    """Telegram board: every watchlist name graded, best first."""
    rows = []
    for t in tickers:
        try:
            rows.append(score(load(t, cfg, refresh), cfg))
        except Exception as e:
            rows.append({"ticker": t, "grade": "?", "score": None, "kills": [], "flags": [str(e)[:60]]})
    order = {"A": 0, "B": 1, "C": 2, "X": 3, "?": 4, "ETF": 5}
    rows.sort(key=lambda r: (order.get(r["grade"], 9), -(r.get("score") or 0)))
    out = [f"<b>Nash board {dt.date.today()}</b>  (~ = subjective categories still at default 5)"]
    for r in rows:
        if r["grade"] == "ETF":
            continue
        out.append(f"{r['ticker']}: " + fmt_line(r))
    etfs = [r["ticker"] for r in rows if r["grade"] == "ETF"]
    if etfs:
        out.append("<i>not graded (ETF):</i> " + ", ".join(etfs))
    return "\n".join(out)
