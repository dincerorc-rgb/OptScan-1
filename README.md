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
