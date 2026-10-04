# optscan on GitHub Actions (phone-only setup)

1. github.com → sign up / log in → **New repository** → name `optscan`, **Private**, Create.
2. **Add file → Upload files** → select every file from this zip (all .py, config.yaml, requirements.txt, README.md) → Commit.
3. **Actions** tab → "set up a workflow yourself" → name it `scan.yml` → paste contents of scan.yml → Commit.
4. **Settings → Secrets and variables → Actions → New repository secret**
   - `TELEGRAM_TOKEN` = your BotFather token
   - `TELEGRAM_CHAT_ID` = your Telegram id
5. **Actions → optscan → Run workflow** → cmd `test-telegram` → Run. Phone buzzes.
6. Run again with `scan` to see the first real pass. After that it runs itself Mon–Fri 15:30 + 20:15 UTC (01:30 + 06:15 AEST Tue–Sat).

Edit `config.yaml` in the browser to change watchlist / positions / net_liq.
Note: GitHub pauses schedules after 60 days with no commits; the workflow commits IV history so it stays alive.

## Minervini SEPA layer (Oct 2026)
- Trend Template (8 rules: SMA50/150/200 stack, SMA200 rising, +30% off 52w low, within 25% of 52w high, RS vs SPY) scored on every name.
- **COIL only fires when TT = 8/8** → that's the VCP/SEPA setup. BREAKOUT/PULLBACK get +1★ on a TT pass.
- Long alerts show `pivot` and `stop` (10-bar low, max 8% under pivot). Stop is on the **stock**, not the option.
- TT-pass COIL/BREAKOUT pick 60–120 DTE, 0.65–0.75Δ calls (spread if IV rank ≥70).
- Tune under `momentum.tt_*` / `options.sepa_*` in config.yaml. Set `tt_enabled: false` to switch off.

## Bayes layer (Oct 2026) — `bayes.py`
Every BUY alert gets a line like `Bayes 55%→68% (×1.74: TT8 RS+ vol≥2x ADX30 tape+)` and `P(BE) 41% vs mkt 32% → edge +9pp`.
- **Prior** = this ticker's backtest win-rate for the setup (+10 sessions), shrunk toward 50% when the sample is small.
- **Signals** = likelihood ratios (TT 8/8, RS vs SPY, volume surge, ADX, hot RSI, Form 4 cluster, P/C, SPY vs EMA50). Combined in log-odds, damped 25% for overlap, capped 0.25×–4×.
- **Posterior** = `revise_probability(prior, LR)` (odds × LR).
- **Edge** = model P(stock past breakeven at expiry) − option-implied N(d2), then **usable = edge − cost (half-spreads + $0.65/contract/leg moving your breakeven) − 3pp margin**. ✅ = usable > 0. `alert_min_usable_pp` can filter on it.
- Sunday `calib-review` scores the log (`state/bayes_log.csv`), shows Brier prior vs posterior, and after 30 resolved alerts learns the LRs into `state/bayes_lr.json`.
- Tune in `config.yaml → bayes:`. `alert_min_post: 0.55` drops weak BUYs.
- The calibration log now records the posterior as `p` (and scanner BUY lines are parsed — they weren't before).

## Prediction-market bot (Oct 2026) — `pm_bot.py`
Polymarket + Kalshi, alerts only. Per market in `pm_markets.yaml`:
`usable edge = fair − average entry (walked through the order book for your stake) − fees − safety margin`.
Alerts only when usable edge > 0, sized at 0.2×Kelly on the cost-adjusted probability; says "edge gone" when it closes. Sunday review compares your Brier to the market's.
1. Actions → **pm-bot** → Run workflow → cmd `discover bitcoin` → slugs arrive on Telegram.
2. Edit `pm_markets.yaml`: add the slug (or Kalshi ticker), prior, signals, `enabled: true`.
3. Run with `scan --dry` to check, then it runs itself every 3h.

**Positions:** add `position: {yes: 260, no: 235, avg_yes: 0.48, avg_no: 0.49}` to a market. The bot splits it into matched pairs (235 — pay $1 whatever happens, locked P&L = 235 × (1 − pair cost)) and directional exposure (+25 YES — the only money at risk). Alerts then say **add / hold / flip** against the Kelly target instead of sizing blind. `positions` command (and the Sunday review) sends the book: locked, at risk, MTM.

## One book — `book.csv` (Oct 2026)
`book.csv` is the only place positions live: `netliq` sets option sizing, every row is checked for SELL/TRIM, and the risk gate uses the same rows. `config.yaml positions:` is retired. Columns: `ticker,bias,premium_usd,label` (ticker = underlying to watch, label = what you hold).

## Minervini expectancy layer (Oct 2026)
Win rate isn't the goal — **expectancy (avg R per trade)** is. 1R = entry − stop.
- **Every long alert** carries an R plan: entry (pivot), stop (≤8%), BE-stop at +2R, target +3R (sell ⅓), trail EMA50.
- **Sizing from the stop**: `risk.risk_pct` (1%) of net liq lost if the stop hits → shares and option contracts (option loss ≈ Δ × R × 100). Option size = min(premium cap, risk cap).
- **Bayes** adds `E[R] = post × avg_win_r − (1−post) × avg_loss_r`.
- **Gate**: R:R < 2 → ⛔, < 3 → ⚠️; stop > 8% away → ⛔.
- **Sunday review headline** = expectancy: win %, avg win R / avg loss R, payoff, E per trade, ΣR — split SEPA vs QM and by setup; break-even win rate shown.
- **Positions in R**: add `entry,stop` (stock prices) to a `book.csv` row → stop hit = EXIT, +2R = MOVE STOP to breakeven, +3R = sell ⅓.
- Alerts are tagged `/SEPA` (Trend Template 8/8) or `/QM` (momentum breakout without TT) in the signal log.

## William O'Neil big-winner rules (Oct 2026) — `config.yaml → oneil:`
- **Pyramid**: R plan splits shares 50% at pivot / 30% at +2–3% / 20% at +4–5%. Never add below entry, never above +5% (buy zone). Open positions in the zone get an **ADD** alert.
- **8-week hold**: position up **+20% within 3 weeks** of `entry_date` → **HOLD** until 8 weeks after entry; TRIM/RSI signals suppressed; exits only if it falls back to cost or on a climax.
- **Cut losses 7–8%**: stops capped at 8% (gate blocks wider).
- **Climax run → SELL INTO STRENGTH**: +25% in 15 sessions plus 2 of: ≥70% above 200d SMA, volume ≥2x, widest range in 3 months, exhaustion gap. Climaxing names get −2★ and "no new entry" on scanner alerts.
- **Options**: SEPA expiries moved to 75–150 DTE so an 8-week hold doesn't run into theta decay.
- Needs `entry,stop,entry_date` on `book.csv` rows to manage open positions.

## Build list (agreed Oct 2026, not built yet)
1. **SEPA / QM split** — tag alerts `[SEPA]` / `[QM]`, separate exits + stats.
2. **Tradier paper bot** — SEPA via options (needs secret `TRADIER_SANDBOX_TOKEN`).
3. **Trading 212 paper bot** — SEPA via shares, stop held at broker (needs `T212_DEMO_KEY`).
4. **TradingView webhook → OptScan** — your hand-drawn pivots/alerts trigger the paper bot.
   TradingView can't call GitHub directly (no auth header), so: TradingView alert → free Cloudflare Worker relay
   (checks a secret phrase in the message) → GitHub `repository_dispatch` → `tv_signal.py` → gate → paper order + Telegram.
   Needs: TradingView paid plan (webhooks), 2FA on TradingView, a Cloudflare account, a GitHub fine-grained token.
5. Minervini rules enrichment from uploaded material (sell rules, VCP specifics, sizing, market regime).
