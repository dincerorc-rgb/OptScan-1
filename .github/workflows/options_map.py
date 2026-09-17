#!/usr/bin/env python3
"""
options_map.py — options-market map for a list of tickers.

Per ticker:
  * pulls the full chain (yfinance) for expiries inside MAX_DTE
  * OI heatmap (calls / puts / net) by strike x expiry
  * GEX profile (OI-based dealer-gamma proxy) by strike
  * nodes: call wall, put wall, max pain, gamma flip, fresh positioning (vol/OI)
  * P/C by expiry, ATM IV term structure
  * one PNG + one text ladder -> Telegram (and ./out/)

Env:  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID   (optional — prints only if missing)
Args: python options_map.py [TICKERS,COMMA,SEPARATED]
"""
import os, sys, math, json, time, io, datetime as dt
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

DEFAULT = "AAPL,MSFT,GOOGL,AMZN,META,NVDA,TSLA,SPY,QQQ,PLTR,IREN,SPCX,AMD,MU,DRAM,UUUU"
MAX_DTE = 130          # expiries considered
MAX_EXP = 9            # cap on expiries per ticker (nearest weeklies + monthlies)
STRIKE_BAND = 0.25     # +/- % around spot shown on heatmap
R = 0.045              # risk-free for BS gamma
OUT = "out"
os.makedirs(OUT, exist_ok=True)

TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT = os.getenv("TELEGRAM_CHAT_ID")


# ---------- data ----------
def pull(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError("no price history")
    spot = float(hist["Close"].iloc[-1])
    today = dt.date.today()
    exps = []
    for e in t.options:
        d = dt.date.fromisoformat(e)
        dte = (d - today).days
        if 0 < dte <= MAX_DTE:
            exps.append((e, dte))
    # keep nearest 4 + monthlies (3rd Friday) up to MAX_EXP
    keep = [x for x in exps[:4]]
    for e, dte in exps[4:]:
        d = dt.date.fromisoformat(e)
        if 15 <= d.day <= 21 and d.weekday() == 4 and len(keep) < MAX_EXP:
            keep.append((e, dte))
    if len(keep) < MAX_EXP:
        for x in exps:
            if x not in keep and len(keep) < MAX_EXP:
                keep.append(x)
    keep.sort(key=lambda x: x[1])
    frames = []
    for e, dte in keep:
        for attempt in range(3):
            try:
                ch = t.option_chain(e)
                break
            except Exception as ex:
                if attempt == 2:
                    raise
                time.sleep(2)
        for side, df in (("C", ch.calls), ("P", ch.puts)):
            d = df[["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]].copy()
            d.columns = ["strike", "last", "bid", "ask", "vol", "oi", "iv"]
            d["side"] = side
            d["exp"] = e
            d["dte"] = dte
            frames.append(d)
    chain = pd.concat(frames, ignore_index=True)
    chain[["vol", "oi"]] = chain[["vol", "oi"]].fillna(0)
    chain["iv"] = chain["iv"].replace(0, np.nan)
    chain["iv"] = chain["iv"].fillna(chain.groupby("exp")["iv"].transform("median")).fillna(0.5)
    return spot, chain


def bs_gamma(S, K, T, sigma, r=R):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    return pdf / (S * sigma * math.sqrt(T))


# ---------- analytics ----------
def analyse(ticker, spot, chain):
    c = chain.copy()
    c["T"] = c["dte"] / 365.0
    c["gamma"] = [bs_gamma(spot, k, T, iv) for k, T, iv in zip(c.strike, c["T"], c.iv)]
    # GEX proxy ($ per 1% move): dealers long calls sold to them (+), short puts (-)
    sign = np.where(c.side == "C", 1, -1)
    c["gex"] = sign * c.gamma * c.oi * 100 * spot * spot * 0.01

    lo, hi = spot * (1 - STRIKE_BAND), spot * (1 + STRIKE_BAND)
    band = c[(c.strike >= lo) & (c.strike <= hi)]

    # per-strike aggregates (all expiries)
    ps = band.groupby(["strike", "side"]).agg(oi=("oi", "sum"), vol=("vol", "sum")).unstack(fill_value=0)
    oi_c = ps["oi"].get("C", pd.Series(dtype=float))
    oi_p = ps["oi"].get("P", pd.Series(dtype=float))
    vol_c = ps["vol"].get("C", pd.Series(dtype=float))
    vol_p = ps["vol"].get("P", pd.Series(dtype=float))
    gex = band.groupby("strike")["gex"].sum()

    call_wall = float(oi_c.idxmax()) if len(oi_c) else None
    put_wall = float(oi_p.idxmax()) if len(oi_p) else None
    call_wall2 = float(oi_c.drop(call_wall).idxmax()) if len(oi_c) > 1 else None
    put_wall2 = float(oi_p.drop(put_wall).idxmax()) if len(oi_p) > 1 else None

    # max pain — nearest expiry
    near = c[c.exp == c.exp.min()]
    strikes = np.sort(near.strike.unique())
    pain = []
    for s in strikes:
        cc = near[near.side == "C"]; pp = near[near.side == "P"]
        pain.append(((np.maximum(s - cc.strike, 0) * cc.oi).sum() + (np.maximum(pp.strike - s, 0) * pp.oi).sum()))
    max_pain = float(strikes[int(np.argmin(pain))]) if len(pain) else None

    # gamma flip — zero crossing of cumulative net GEX closest to spot
    g = gex.sort_index()
    cum = g.cumsum()
    flip = None
    idx = list(cum.index)
    for i in range(1, len(idx)):
        if (cum.iloc[i - 1] < 0) != (cum.iloc[i] < 0):
            cand = idx[i - 1] if abs(idx[i - 1] - spot) < abs(idx[i] - spot) else idx[i]
            if flip is None or abs(cand - spot) < abs(flip - spot):
                flip = float(cand)
    net_gex = float(g.sum())
    gex_pos = float(g.idxmax()) if len(g) else None
    gex_neg = float(g.idxmin()) if len(g) else None

    # fresh positioning: high vol vs OI (today's flow), min vol 500
    band2 = band[(band.vol >= 500)].copy()
    band2["vo"] = band2.vol / band2.oi.clip(lower=1)
    fresh = band2.sort_values("vo", ascending=False).head(5)[["exp", "side", "strike", "vol", "oi", "vo", "iv"]]

    # P/C by expiry (OI) + ATM IV
    pc = []
    for e, grp in c.groupby("exp"):
        co = grp[grp.side == "C"].oi.sum(); po = grp[grp.side == "P"].oi.sum()
        cv = grp[grp.side == "C"].vol.sum(); pv = grp[grp.side == "P"].vol.sum()
        atm = grp.iloc[(grp.strike - spot).abs().argsort()[:2]]
        pc.append(dict(exp=e, dte=int(grp.dte.iloc[0]), pc_oi=po / max(co, 1), pc_vol=pv / max(cv, 1),
                       atm_iv=float(atm.iv.mean()), oi=int(co + po), vol=int(cv + pv)))
    pc = pd.DataFrame(pc)

    tot_c = c[c.side == "C"].oi.sum(); tot_p = c[c.side == "P"].oi.sum()
    tv_c = c[c.side == "C"].vol.sum(); tv_p = c[c.side == "P"].vol.sum()

    return dict(ticker=ticker, spot=spot, call_wall=call_wall, call_wall2=call_wall2, put_wall=put_wall,
                put_wall2=put_wall2, max_pain=max_pain, near_exp=str(c.exp.min()), flip=flip,
                net_gex=net_gex, gex_pos=gex_pos, gex_neg=gex_neg, pc_oi=tot_p / max(tot_c, 1),
                pc_vol=tv_p / max(tv_c, 1), fresh=fresh, pc=pc, band=band, oi_c=oi_c, oi_p=oi_p,
                vol_c=vol_c, vol_p=vol_p, gex=g)


# ---------- render ----------
def render(a):
    band, spot = a["band"], a["spot"]
    exps = sorted(band.exp.unique())
    strikes = np.sort(band.strike.unique())
    # thin strikes if too many rows
    if len(strikes) > 40:
        step = int(math.ceil(len(strikes) / 40))
        keep = strikes[np.argsort(np.abs(strikes - spot))][:40]
        strikes = np.sort(keep)
    b = band[band.strike.isin(strikes)]
    piv = lambda side, col: b[b.side == side].pivot_table(index="strike", columns="exp", values=col, aggfunc="sum").reindex(index=strikes, columns=exps).fillna(0)
    oi_c, oi_p = piv("C", "oi"), piv("P", "oi")
    vol_c, vol_p = piv("C", "vol"), piv("P", "vol")
    net_oi = oi_c - oi_p
    net_vol = vol_c - vol_p

    fig = plt.figure(figsize=(16, 9), facecolor="#0e1117")
    gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1.1, 1.1, 1.0], wspace=0.35)
    axs = [fig.add_subplot(gs[0, i]) for i in range(4)]
    for ax in axs:
        ax.set_facecolor("#0e1117"); ax.tick_params(colors="#c9d1d9", labelsize=8)
        for s in ax.spines.values(): s.set_color("#30363d")

    def heat(ax, m, title, cmap, diverging):
        v = m.values
        vmax = np.nanmax(np.abs(v)) if v.size else 1
        norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax) if diverging else None
        im = ax.imshow(v, aspect="auto", cmap=cmap, norm=norm, origin="lower")
        ax.set_xticks(range(len(exps))); ax.set_xticklabels([e[5:] for e in exps], rotation=60, color="#c9d1d9")
        ax.set_yticks(range(len(strikes))); ax.set_yticklabels([f"{s:g}" for s in strikes], color="#c9d1d9")
        # spot line
        si = np.searchsorted(strikes, spot) - 0.5
        ax.axhline(si, color="#ffd700", lw=1.5, ls="--")
        ax.set_title(title, color="#e6edf3", fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.yaxis.set_tick_params(color="#c9d1d9"); plt.setp(cb.ax.get_yticklabels(), color="#c9d1d9", fontsize=7)

    heat(axs[0], net_oi, "Net OI  (calls − puts)", "RdYlGn", True)
    heat(axs[1], oi_c + oi_p, "Total OI", "magma", False)
    heat(axs[2], net_vol, "Net volume today (calls − puts)", "RdYlGn", True)

    # GEX profile
    g = a["gex"].reindex(strikes).fillna(0)
    cols = np.where(g.values >= 0, "#2ea043", "#f85149")
    axs[3].barh(range(len(strikes)), g.values / 1e6, color=cols)
    axs[3].set_yticks(range(len(strikes))); axs[3].set_yticklabels([f"{s:g}" for s in strikes], color="#c9d1d9")
    axs[3].axhline(np.searchsorted(strikes, spot) - 0.5, color="#ffd700", lw=1.5, ls="--")
    axs[3].axvline(0, color="#8b949e", lw=0.8)
    axs[3].set_title("GEX proxy  ($M / 1% move)", color="#e6edf3", fontsize=11)
    axs[3].set_xlabel("green = +gamma (pins)   red = −gamma (accelerates)", color="#8b949e", fontsize=8)
    axs[3].set_ylim(-0.5, len(strikes) - 0.5)
    labels = {}
    for lvl, lab in ((a["call_wall"], "CW"), (a["put_wall"], "PW"), (a["flip"], "FLIP"), (a["max_pain"], "MP")):
        if lvl is not None and lvl in strikes:
            labels.setdefault(int(np.where(strikes == lvl)[0][0]), []).append(lab)
    for y, labs in labels.items():
        axs[3].text(axs[3].get_xlim()[1], y, " " + "/".join(labs), color="#e6edf3", fontsize=8, va="center", fontweight="bold")

    fig.suptitle(f"{a['ticker']}   spot {spot:.2f}   |   CW {a['call_wall']:g}  PW {a['put_wall']:g}  MP({a['near_exp'][5:]}) {a['max_pain']:g}  FLIP {a['flip'] if a['flip'] else '—'}   |   P/C OI {a['pc_oi']:.2f}  vol {a['pc_vol']:.2f}   |   net GEX {a['net_gex']/1e6:+.0f}M",
                 color="#e6edf3", fontsize=12, y=0.98)
    fig.text(0.5, 0.005, f"OI-based GEX proxy (not dealer positioning) · strikes ±{int(STRIKE_BAND*100)}% of spot · {dt.datetime.utcnow():%Y-%m-%d %H:%M} UTC · yfinance delayed", ha="center", color="#8b949e", fontsize=8)
    path = f"{OUT}/{a['ticker']}_map.png"
    fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def ladder(a):
    spot = a["spot"]
    L = [f"*{a['ticker']}*  spot {spot:.2f}"]
    L.append(f"CW {a['call_wall']:g} / {a['call_wall2']:g}   PW {a['put_wall']:g} / {a['put_wall2']:g}")
    L.append(f"MP {a['max_pain']:g} ({a['near_exp'][5:]})   FLIP {a['flip'] if a['flip'] else '—'}   GEX {a['net_gex']/1e6:+.0f}M  (+{a['gex_pos']:g} / −{a['gex_neg']:g})")
    L.append(f"P/C  OI {a['pc_oi']:.2f}  vol {a['pc_vol']:.2f}")
    regime = "pinning (+γ)" if a["net_gex"] > 0 else "trending (−γ)"
    pos = "above flip" if a["flip"] and spot > a["flip"] else ("below flip" if a["flip"] else "no flip in band")
    L.append(f"regime: {regime}, {pos}")
    L.append("`exp     dte  P/C-oi P/C-v  ATM-IV`")
    for _, r in a["pc"].iterrows():
        L.append(f"`{r.exp[2:]:8s}{r.dte:>3d}  {r.pc_oi:5.2f}  {r.pc_vol:5.2f}  {r.atm_iv*100:5.1f}%`")
    if len(a["fresh"]):
        L.append("fresh (vol/OI):")
        for _, r in a["fresh"].iterrows():
            L.append(f"`{r.exp[5:]} {r.side}{r.strike:g}  v{int(r.vol)} oi{int(r.oi)} x{r.vo:.1f} iv{r.iv*100:.0f}`")
    return "\n".join(L)


# ---------- telegram ----------
def tg_photo(path, caption):
    if not (TOKEN and CHAT):
        return
    import requests
    with open(path, "rb") as f:
        r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto",
                          data={"chat_id": CHAT, "caption": caption[:1024], "parse_mode": "Markdown"},
                          files={"photo": f}, timeout=60)
    if not r.ok:
        # fall back to plain caption if markdown rejected
        with open(path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{TOKEN}/sendPhoto", data={"chat_id": CHAT, "caption": caption[:1024]}, files={"photo": f}, timeout=60)


def tg_text(text):
    if not (TOKEN and CHAT):
        print(text); return
    import requests
    r = requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT, "text": text[:4096], "parse_mode": "Markdown"}, timeout=30)
    if not r.ok:
        requests.post(f"https://api.telegram.org/bot{TOKEN}/sendMessage", data={"chat_id": CHAT, "text": text[:4096]}, timeout=30)


# ---------- main ----------
def main():
    tickers = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] else os.getenv("TICKERS", DEFAULT)).replace(" ", "").split(",")
    summary, failed, rows = [], [], []
    tg_text(f"options map · {len(tickers)} tickers · {dt.datetime.utcnow():%d %b %H:%M} UTC")
    for tk in tickers:
        try:
            spot, chain = pull(tk)
            a = analyse(tk, spot, chain)
            png = render(a)
            txt = ladder(a)
            tg_photo(png, txt)
            with open(f"{OUT}/{tk}_ladder.txt", "w") as f:
                f.write(txt)
            a["pc"].to_csv(f"{OUT}/{tk}_pc.csv", index=False)
            chain.to_csv(f"{OUT}/{tk}_chain.csv", index=False)
            rows.append(dict(ticker=tk, spot=round(spot, 2), CW=a["call_wall"], PW=a["put_wall"], MP=a["max_pain"], FLIP=a["flip"],
                             netGEX_M=round(a["net_gex"] / 1e6), PC_oi=round(a["pc_oi"], 2), PC_vol=round(a["pc_vol"], 2)))
            summary.append(f"`{tk:5s} {spot:8.2f}  CW {a['call_wall']:g}  PW {a['put_wall']:g}  F {a['flip'] if a['flip'] else '—'}  PC {a['pc_oi']:.2f}`")
            print(txt, "\n")
            time.sleep(1.5)
        except Exception as ex:
            failed.append(f"{tk}: {ex}")
            print("FAIL", tk, ex)
    pd.DataFrame(rows).to_csv(f"{OUT}/summary.csv", index=False)
    tail = "*summary* (spot / call wall / put wall / flip / P-C OI)\n" + "\n".join(summary)
    if failed:
        tail += "\n\nfailed: " + "; ".join(failed)
    tg_text(tail)


if __name__ == "__main__":
    main()
