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
