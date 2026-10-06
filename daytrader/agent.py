"""The always-on loop for paper and live trading."""

from __future__ import annotations

import logging
import os
import signal
import time

import ccxt
import pandas as pd

from .broker import CCXTBroker, PaperBroker
from .config import Settings
from .data import fetch_recent, make_exchange
from .engine import Engine, Journal
from .indicators import add_indicators
from .llm import ClaudeReviewer

log = logging.getLogger(__name__)


def build_engine(settings: Settings, live: bool) -> tuple[Engine, object, str]:
    mode = "live" if live else "paper"
    if live:
        if not (settings.exchange_api_key and settings.exchange_api_secret):
            raise SystemExit("Live trading needs EXCHANGE_API_KEY and EXCHANGE_API_SECRET in .env")
        if settings.risk.max_capital is None:
            raise SystemExit("Set LIVE_MAX_CAPITAL in .env to cap how much the bot may trade with")
        exchange = make_exchange(
            settings.exchange, settings.exchange_api_key, settings.exchange_api_secret, settings.exchange_api_password
        )
        broker = CCXTBroker(exchange, settings.quote_currency)
    else:
        exchange = make_exchange(settings.exchange)  # public market data only, no keys
        broker = PaperBroker(settings.starting_cash, settings.fee_bps, settings.slippage_bps)

    reviewer = None
    if settings.use_llm:
        reviewer = ClaudeReviewer(
            settings.claude_model, settings.claude_effort, settings.llm_min_confidence, settings.timeframe
        )
    journal = Journal(os.path.join(settings.data_dir, f"journal_{mode}.jsonl"))
    engine = Engine(settings, broker, reviewer, journal)
    return engine, exchange, os.path.join(settings.data_dir, f"state_{mode}.json")


def run(settings: Settings, live: bool = False) -> None:
    engine, exchange, state_path = build_engine(settings, live)
    if engine.load(state_path):
        log.info("Resumed state: %d open position(s), cash %.2f", len(engine.positions), engine.broker.cash())

    stopping = False

    def _stop(*_):
        nonlocal stopping
        stopping = True
        log.info("Stopping after this cycle (positions stay open; run `flatten` to close them)")

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    last_bar: dict[str, pd.Timestamp] = {}
    log.info("%s trading %s on %s %s; Claude review %s",
             "LIVE" if live else "Paper", ", ".join(settings.symbols), settings.exchange,
             settings.timeframe, "on" if settings.use_llm else "off")

    while not stopping:
        for sym in settings.symbols:
            try:
                now = pd.Timestamp.now(tz="UTC")
                price = float(exchange.fetch_ticker(sym)["last"])
                engine.on_price(sym, price, price, price, now)

                df = fetch_recent(exchange, sym, settings.timeframe)
                if df.empty or df.index[-1] == last_bar.get(sym):
                    continue
                last_bar[sym] = df.index[-1]
                engine.on_bar_close(sym, add_indicators(df, settings.strategy), df.index[-1])
            except ccxt.NetworkError as e:
                log.warning("Network error on %s, will retry: %s", sym, e)
            except ccxt.ExchangeError as e:
                log.error("Exchange error on %s: %s", sym, e)
            except Exception:
                # Keep running: a crashed bot leaves open positions with no one watching their stops.
                log.exception("Unexpected error on %s", sym)
        engine.save(state_path)
        _status(engine)
        for _ in range(settings.poll_seconds):
            if stopping:
                break
            time.sleep(1)
    engine.save(state_path)


def flatten(settings: Settings, live: bool = False) -> None:
    engine, exchange, state_path = build_engine(settings, live)
    if not engine.load(state_path) or not engine.positions:
        print("No open positions.")
        return
    now = pd.Timestamp.now(tz="UTC")
    for sym in list(engine.positions):
        engine.last_price[sym] = float(exchange.fetch_ticker(sym)["last"])
    engine.flatten(now, "manual_flatten")
    engine.save(state_path)
    print(f"Closed all positions. Cash: {engine.broker.cash():.2f}")


def _status(engine: Engine) -> None:
    acct = engine.account_snapshot()
    pos = ", ".join(
        f"{p.symbol} {p.qty:.6g}@{p.entry:.6g} (stop {p.stop:.6g}, tgt {p.target:.6g})"
        for p in engine.positions.values()
    ) or "flat"
    halted = "  [HALTED: daily loss limit]" if engine.risk.state.halted else ""
    log.info("equity %.2f  day %+.2f%%  trades %d  | %s%s",
             acct["equity"], acct["day_pnl_pct"], acct["trades_today"], pos, halted)
