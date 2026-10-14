# OptScan rulebook — v1 (frozen 4 Oct 2026)

> Minervini: commit to one style, stay with it for years, and assume a bad stretch is execution, not the strategy.
> So: **one primary style, written rules, no mid-stream changes.**

## Primary style: Minervini SEPA (+ O'Neil position management)
| Area | Rule |
|---|---|
| Universe | config.yaml watchlist only |
| Setup | Trend Template 8/8 + VCP coil / pivot breakout on volume |
| Entry | Pivot, never > 5% above it; pyramid 50 / 30 / 20 (adds at +2–3%, +4–5%) |
| Stop | ≤ 8% below entry (5–8%), honoured without exception |
| Risk | 1% of account per trade at the stop; max 4 open per account |
| Reward:risk | ≥ 3:1 planned (gate: < 2 blocks, < 3 warns) |
| Manage | +2R → stop to breakeven · +3R → sell ⅓, trail EMA50 |
| Big winners | +20% in < 3 weeks → hold 8 weeks (exit only below cost or climax) |
| Sell | Climax run → sell into strength; close < EMA50 → exit |
| Never | Average down · widen a stop · hold options < 5 DTE · enter on a climax · trade through earnings |

## Secondary (control only): Qullamaggie (QM) — paper only, tagged `/QM`, compared against SEPA. Never real money in v1.

## Change policy
1. **Rules are frozen.** New material (books, clips, posts) goes to the *Candidate changes* list below, not into the code.
2. **Review dates:** quarterly (next: **4 Jan 2027**), and only with ≥ 30 resolved SEPA trades.
3. A change is accepted only if the data supports it (expectancy, avg loss R, rule-break count), then the version goes to v2 with a dated changelog.
4. Bug fixes and safety fixes are allowed anytime; they don't change the rules.
5. A losing streak is **not** a reason to change rules. First check: were the rules followed? (Sunday report tracks rule breaks.)

## Candidate changes (parked until the review)
- VCP specifics (contraction count / depth) — awaiting material
- Market-direction filter (when to go to cash) — awaiting material
- Nash `gate` mode (fail one step = off the island) — decide at review with ≥30 graded trades

## Changelog
- v1.1 · 15 Oct 2026 · Tom Nash quality layer (`nash.py`, `config.yaml → nash:`), added at Dincer's request. SEPA entry/stop/exit rules unchanged; the layer changes only **which option contract** (expiry, delta, spread vs outright, size) and ±1★. Mode `tilt`; `gate` (block C/X calls) parked until the 4 Jan review compares graded vs ungraded expectancy.
- v1 · 4 Oct 2026 · SEPA + R plan + O'Neil pyramid / 8-week hold / climax sell; Bayes = tiebreaker only.
