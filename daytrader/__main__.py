"""Command line: python -m daytrader {fetch,backtest,paper,live,flatten}"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from .config import Settings


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="daytrader", description="Claude-reviewed crypto day-trading agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="download historical candles to data/candles/")
    f.add_argument("--days", type=int, default=30)

    b = sub.add_parser("backtest", help="replay history through the strategy")
    b.add_argument("--days", type=int, default=30, help="download this much history if no CSV is cached")
    b.add_argument("--synthetic", action="store_true", help="use random-walk data (offline plumbing test)")
    b.add_argument("--with-llm", action="store_true", help="have Claude review every signal (costs API credits)")
    b.add_argument("--max-llm-calls", type=int, default=50, help="safety cap on Claude calls in one backtest")

    sub.add_parser("paper", help="trade live market data with fake money")

    lv = sub.add_parser("live", help="trade REAL money")
    lv.add_argument("--i-understand-this-uses-real-money", dest="confirm", action="store_true")

    fl = sub.add_parser("flatten", help="close every open position now")
    fl.add_argument("--live", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    settings = Settings.from_env()

    if args.cmd == "fetch":
        _fetch(settings, args.days)
    elif args.cmd == "backtest":
        _backtest(settings, args)
    elif args.cmd == "paper":
        from .agent import run
        run(settings, live=False)
    elif args.cmd == "live":
        if not args.confirm:
            sys.exit("Refusing to trade real money without --i-understand-this-uses-real-money")
        from .agent import run
        run(settings, live=True)
    elif args.cmd == "flatten":
        from .agent import flatten
        flatten(settings, live=args.live)


def _fetch(settings: Settings, days: int) -> dict:
    from .data import csv_path, fetch_history, make_exchange, save_csv

    ex = make_exchange(settings.exchange)
    out = {}
    for sym in settings.symbols:
        df = fetch_history(ex, sym, settings.timeframe, days)
        path = csv_path(settings.data_dir, settings.exchange, sym, settings.timeframe)
        save_csv(df, path)
        print(f"{sym}: {len(df)} bars -> {path}")
        out[sym] = df
    return out


def _backtest(settings: Settings, args) -> None:
    from .backtest import run_backtest
    from .data import csv_path, load_csv, synthetic
    from .engine import Journal

    if args.synthetic:
        import ccxt
        minutes = ccxt.Exchange.parse_timeframe(settings.timeframe) // 60
        candles = {
            sym: synthetic(bars=2000, seed=i, start_price=60_000 / (i * 15 + 1), timeframe_minutes=minutes)
            for i, sym in enumerate(settings.symbols)
        }
    else:
        paths = {s: csv_path(settings.data_dir, settings.exchange, s, settings.timeframe) for s in settings.symbols}
        if all(os.path.exists(p) for p in paths.values()):
            candles = {s: load_csv(p) for s, p in paths.items()}
        else:
            candles = _fetch(settings, args.days)

    reviewer = None
    if args.with_llm:
        from .llm import ClaudeReviewer
        reviewer = _CappedReviewer(
            ClaudeReviewer(settings.claude_model, settings.claude_effort, settings.llm_min_confidence, settings.timeframe),
            args.max_llm_calls,
        )

    journal = Journal(os.path.join(settings.data_dir, "journal_backtest.jsonl"))
    if journal.path and os.path.exists(journal.path):
        os.remove(journal.path)
    report, _ = run_backtest(settings, candles, reviewer, journal)
    span = " / ".join(f"{s}: {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}" for s, df in candles.items())
    print(f"\nBacktest {settings.timeframe}  {span}\nClaude review: {'on' if reviewer else 'off'}\n")
    print(report)
    print(f"\nEvery decision is logged in {journal.path}")


class _CappedReviewer:
    """Stops calling Claude after N reviews so a long backtest can't run up a surprise bill."""

    def __init__(self, inner, cap: int):
        self.inner, self.cap, self.calls = inner, cap, 0

    def review(self, signal, df, account, history):
        from .llm import Review
        if self.calls >= self.cap:
            return Review(False, 0.0, 1.0, 0.0, 0.0, "LLM call cap reached")
        self.calls += 1
        return self.inner.review(signal, df, account, history)


if __name__ == "__main__":
    main()
