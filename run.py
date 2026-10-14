#!/usr/bin/env python3
"""optscan CLI.

  python run.py scan                 # full watchlist -> Telegram
  python run.py scan NVDA PLTR --dry # print only
  python run.py backtest NVDA        # setup stats on one name
  python run.py chain PLTR           # IV / structure suggestion only
  python run.py nash [TICKERS] [--dry] [--refresh]   # Nash quality board -> Telegram
  python run.py test-telegram
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(ROOT / "state" / "optscan.log")])

import backtest, data, notify, options, scan, signals  # noqa: E402

cfg = yaml.safe_load((ROOT / "config.yaml").read_text())

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("scan"); s.add_argument("tickers", nargs="*"); s.add_argument("--dry", action="store_true")
s.add_argument("--min-score", type=int, default=2)
b = sub.add_parser("backtest"); b.add_argument("tickers", nargs="+")
c = sub.add_parser("chain"); c.add_argument("ticker"); c.add_argument("--bias", default="LONG")
n = sub.add_parser("nash"); n.add_argument("tickers", nargs="*"); n.add_argument("--dry", action="store_true")
n.add_argument("--refresh", action="store_true")
sub.add_parser("test-telegram")
a = ap.parse_args()

if a.cmd == "scan":
    scan.run(cfg, a.tickers or None, send=not a.dry, min_score=a.min_score)
elif a.cmd == "backtest":
    import pandas as pd
    pd.set_option("display.width", 200)
    for t in a.tickers:
        print(backtest.run(t, data.daily_bars(t, 3), cfg["momentum"]).to_string(index=False))
elif a.cmd == "chain":
    bars = data.daily_bars(a.ticker, 2)
    spot = float(bars.close.iloc[-1])
    oa = options.analyse(a.ticker, spot, signals.realised_vol(bars.close), cfg["options"])
    print({k: v for k, v in oa.items() if k != "chain"})
    print(options.suggest(a.ticker, a.bias, "BREAKOUT", spot, oa["iv_rank"], oa["chain"],
                          cfg["options"], cfg["account"]).as_dict())
elif a.cmd == "nash":
    import nash
    msg = nash.board(a.tickers or cfg["watchlist"], cfg, refresh=a.refresh)
    print(msg) if a.dry else notify.send(msg)
elif a.cmd == "test-telegram":
    print("sent" if notify.send("optscan online ✅") else "not sent - check .env")
