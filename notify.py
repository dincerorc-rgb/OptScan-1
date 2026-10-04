"""Telegram alerts. Env: TELEGRAM_TOKEN, TELEGRAM_CHAT_ID."""
from __future__ import annotations

import html
import logging
import os

import requests

log = logging.getLogger(__name__)


def send(text: str) -> bool:
    tok, chat = os.getenv("TELEGRAM_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not tok or not chat:
        log.warning("Telegram not configured; printing instead\n%s", text)
        print(text)
        return False
    text = html.escape(text).replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    ok = True
    for i in range(0, len(text), 3900):
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          json={"chat_id": chat, "text": text[i:i + 3900], "parse_mode": "HTML",
                                "disable_web_page_preview": True}, timeout=20)
        ok &= r.ok
        if not r.ok:
            log.error("telegram %s %s", r.status_code, r.text[:200])
    return ok


def fmt_alert(a: dict) -> str:
    """One compact block per hit. a = merged dict from scan."""
    t = a["trade"]
    lines = [
        f"<b>{a['ticker']}</b> {a['close']:.2f}  <b>{a['setup']}</b> {a['bias']} ★{a['score']}",
        f"RSI {a['rsi']:.0f} ADX {a['adx']:.0f} vol {a['vol_ratio']:.1f}x 20d {a['ret20']:+.1f}%",
        f"IV {a['atm_iv']:.0f}% IVR {a['iv_rank']:.0f}({a['iv_rank_src']}) IV/RV {a['iv_rv']:.2f} PCR {a['pcr']}",
    ]
    if a.get("insider_score", 0) >= 1:
        lines.append(f"Form4 cluster {a['insider_score']} ({a['insider_n']} insiders ${a['insider_usd']/1e6:.1f}M)")
    if a.get("bias") == "LONG" and a.get("pivot") == a.get("pivot") and not a.get("tt_exempt"):
        tag = "SEPA ✅" if a.get("horizon") == "sepa" else f"TT {a.get('tt_n', 0)}/8"
        lines.append(f"{tag} RS {a.get('rs_excess', 0):+.0f}% vs SPY · pivot {a['pivot']:.2f} "
                     f"stop {a['stop']:.2f} (-{a['risk_pct']:.1f}%)")
    lines.append(f"BT {a['bt']}  {a['bt_detail']}")
    if a.get("bayes"):
        import bayes
        lines.append(bayes.fmt(a["bayes"]))
    pl = a.get("plan")
    if pl:
        lines.append(f"R plan: entry {pl['entry']:.2f} stop {pl['stop']:.2f} (1R {pl['R']:.2f} = {pl['risk_pct']:.1f}%)"
                     f" · BE-stop @ {pl['be_at']:.2f} (+2R) · tgt {pl['t1']:.2f} (+3R, sell ⅓) · trail EMA50")
        sz = f"   risk ${pl['risk_usd']:,.0f}/trade → shares {pl['shares']}"
        if pl.get("pyr") and pl["shares"] >= 2:
            a, b2, c2 = (pl["pyr"] + [0, 0, 0])[:3]
            sz += f" (buy {a} at pivot, +{b2} at +2–3%, +{c2} at +4–5%; ≤5% above pivot)"
        if "contracts_risk" in pl:
            sz += f" · options {pl['contracts_risk']}x (≈${pl['opt_loss_1x']:,.0f} lost/contract at stop)"
        lines.append(sz)
    if t["structure"] in ("WATCH", "NONE"):
        lines.append(f"→ {t['structure']}: {t['note']}")
    else:
        lines.append(f"→ <b>{t['structure']}</b> {t['expiry']} ({t['dte']}d) {t['legs']} @ {t['debit']:.2f} "
                     f"BE {t['breakeven']:.2f} Δ{t['delta']:.2f}")
        lines.append(f"   size {t['contracts']}x ≈ ${t['premium_usd']:,.0f} · {t['note']}")
    lines.append(f"   {a['reason']}")
    return "\n".join(lines)
