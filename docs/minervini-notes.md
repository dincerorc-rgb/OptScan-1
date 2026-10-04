# Minervini / O'Neil reference notes

Short summaries (own words) of material Dincer shared, plus where each idea lives in the code.
New items: add a dated entry here; rule changes go to RULES.md → Candidate changes, decided at the quarterly review.

| # | Date | Source | Key idea | Status in OptScan |
|---|---|---|---|---|
| 1 | 2026-10-04 | Minervini clip (X) — risk/reward & expectancy | Win rate isn't the goal; ~50% hit rate is normal for top traders. Cut losses at 5–8%, let winners run 15–20%+. Expectancy = win% × avg win − loss% × avg loss. | ✅ Built: R plan, stop-based sizing, R:R gate, expectancy headline in Sunday review |
| 2 | 2026-10-04 | O'Neil clip (via @algridtrader) — capturing big moves | Pyramid up, never down (≈50/30/20, adds only +2–3% above pivot). +20% in <3 weeks → hold 8 weeks. Cut every loss at 7–8%. Sell into climax runs (fast 25–50% surge on huge volume late in a move). | ✅ Built: `oneil:` config, ADD / HOLD / SELL INTO STRENGTH signals, climax no-entry flag, SEPA DTE 75–150 |
| 3 | 2026-10-04 | Minervini speaking (TraderLion clip) — mindset | Expect years of being bad at it before being good. Pick one coach / one style and commit; don't hop. Persistence is the edge. When it doesn't work, assume it's your execution, not the strategy — take responsibility. | ✅ Operating principle: RULES.md v1 frozen, quarterly changes only with data, rule-following checked before blaming rules |

## Open questions (waiting for material)
- VCP specifics: number of contractions, depth of each, volume dry-up thresholds
- Market direction: when to stop buying / go to cash (distribution days, index below 50d, etc.)
- Minervini's exact sell rules beyond the 8% stop (sell into strength vs weakness)
- Position sizing at different account sizes; max positions
