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
