"""The trading engine shared by backtests, paper trading and live trading.

Two entry points drive it:
  * on_price(...)      - protective exits (stop / target). The backtest feeds each
                         bar's open/high/low; the live loop feeds the latest price
                         every few seconds so stops don't wait for a bar to close.
  * on_bar_close(...)  - indicator exits, time stop, daily-loss flatten and new
                         entries, evaluated once per closed bar.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Protocol

import pandas as pd

from .config import Settings
from .llm import Review
from .risk import RiskManager
from .strategy import Signal, entry_signal, exit_signal

log = logging.getLogger(__name__)


class Reviewer(Protocol):
    def review(self, signal: Signal, df: pd.DataFrame, account: dict, history: list[dict]) -> Review: ...


@dataclass
class Position:
    symbol: str
    qty: float
    entry: float
    stop: float
    target: float
    initial_stop: float
    opened_at: str
    entry_fee: float
    bars_held: int = 0
    note: str = ""


@dataclass
class ClosedTrade:
    symbol: str
    opened: str
    closed: str
    entry: float
    exit: float
    qty: float
    pnl: float
    r_multiple: float
    reason: str
    fees: float = 0.0


@dataclass
class Journal:
    path: str | None = None
    events: list[dict] = field(default_factory=list)

    def log(self, kind: str, now: pd.Timestamp, **data) -> None:
        event = {"time": str(now), "event": kind, **data}
        self.events.append(event)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")


class Engine:
    def __init__(self, settings: Settings, broker, reviewer: Reviewer | None = None, journal: Journal | None = None):
        self.s = settings
        self.broker = broker
        self.reviewer = reviewer
        self.journal = journal or Journal()
        self.risk = RiskManager(settings.risk)
        self.positions: dict[str, Position] = {}
        self.last_price: dict[str, float] = {}
        self.trades: list[ClosedTrade] = []

    # ---- account -------------------------------------------------------
    def equity(self) -> float:
        held = sum(p.qty * self.last_price.get(sym, p.entry) for sym, p in self.positions.items())
        return self.broker.cash() + held

    def account_snapshot(self) -> dict:
        eq = self.equity()
        start = self.risk.state.start_equity or eq
        return {
            "equity": eq,
            "quote": self.s.quote_currency,
            "day_pnl_pct": (eq - start) / start * 100 if start else 0.0,
            "trades_today": self.risk.state.trades,
            "open_positions": len(self.positions),
            "round_trip_cost_pct": self.s.round_trip_cost_pct * 100,
        }

    # ---- protective exits ---------------------------------------------
    def on_price(self, symbol: str, open_: float, high: float, low: float, now: pd.Timestamp) -> None:
        self.last_price[symbol] = (high + low) / 2 if high != low else high
        pos = self.positions.get(symbol)
        if pos is None:
            return
        # If both levels are inside the range we can't know which came first: assume the stop (conservative).
        if low <= pos.stop:
            self._close(symbol, min(open_, pos.stop), now, "stop")
        elif high >= pos.target:
            self._close(symbol, max(open_, pos.target), now, "target")

    # ---- bar close -----------------------------------------------------
    def on_bar_close(self, symbol: str, df: pd.DataFrame, now: pd.Timestamp) -> None:
        close = float(df["close"].iloc[-1])
        self.last_price[symbol] = close
        self.risk.roll_day(now, self.equity())

        if self.risk.daily_loss_breached(self.equity()):
            if self.positions:
                log.warning("Daily loss limit hit - flattening all positions")
            for sym in list(self.positions):
                self._close(sym, self.last_price.get(sym, close), now, "daily_loss_limit")
            return

        pos = self.positions.get(symbol)
        if pos is not None:
            pos.bars_held += 1
            reason = exit_signal(df)
            if reason is None and pos.bars_held >= self.s.strategy.max_hold_bars:
                reason = "time_stop"
            if reason:
                self._close(symbol, close, now, reason)
            return

        signal = entry_signal(symbol, df, self.s.strategy)
        if signal is None:
            return
        if (signal.target - signal.entry) / signal.entry < self.s.strategy.min_reward_to_cost * self.s.round_trip_cost_pct:
            # The move we're aiming for wouldn't cover fees + slippage with room to spare.
            self.journal.log("signal_blocked", now, symbol=symbol, reason="target_below_costs")
            return
        ok, why = self.risk.can_open(self.equity(), len(self.positions))
        if not ok:
            self.journal.log("signal_blocked", now, symbol=symbol, reason=why)
            return

        if self.reviewer is not None:
            history = [asdict(t) for t in self.trades if t.symbol == symbol][-5:]
            review = self.reviewer.review(signal, df, self.account_snapshot(), history)
        else:
            review = Review(True, 1.0, 1.0, signal.stop, signal.target, "rules only")
        self.journal.log(
            "review", now, symbol=symbol, take=review.take, confidence=review.confidence,
            size_mult=review.size_mult, entry=signal.entry, stop=review.stop or signal.stop,
            target=review.target or signal.target, reasoning=review.reasoning, features=signal.features,
        )
        if review.take:
            self._open(signal, review, now)

    # ---- execution -----------------------------------------------------
    def _open(self, signal: Signal, review: Review, now: pd.Timestamp) -> None:
        equity, cash = self.equity(), self.broker.cash()
        qty = self.risk.size(equity, cash, signal.entry, review.stop, review.size_mult)
        if qty <= 0:
            self.journal.log("signal_blocked", now, symbol=signal.symbol, reason="size_below_minimum")
            return
        fill = self.broker.buy(signal.symbol, qty, signal.entry, now)
        if fill is None:
            return
        # Keep the stop/target the same distance from the actual fill price.
        shift = fill.price - signal.entry
        self.positions[signal.symbol] = Position(
            symbol=signal.symbol, qty=fill.qty, entry=fill.price, stop=review.stop + shift,
            target=review.target + shift, initial_stop=review.stop + shift, opened_at=str(now),
            entry_fee=fill.fee, note=review.reasoning[:200],
        )
        self.risk.record_open()
        self.journal.log("open", now, symbol=signal.symbol, qty=fill.qty, price=fill.price, fee=fill.fee,
                         stop=review.stop + shift, target=review.target + shift)

    def _close(self, symbol: str, ref_price: float, now: pd.Timestamp, reason: str) -> None:
        pos = self.positions.get(symbol)
        if pos is None:
            return
        fill = self.broker.sell(symbol, pos.qty, ref_price, now)
        if fill is None:
            log.error("Could not sell %s - dropping position from the book", symbol)
            self.positions.pop(symbol)
            return
        pnl = (fill.price - pos.entry) * fill.qty - pos.entry_fee - fill.fee
        risk_per_unit = pos.entry - pos.initial_stop
        r_multiple = pnl / (risk_per_unit * fill.qty) if risk_per_unit > 0 else 0.0
        self.positions.pop(symbol)
        self.risk.record_close(pnl)
        trade = ClosedTrade(
            symbol, pos.opened_at, str(now), pos.entry, fill.price, fill.qty, pnl, r_multiple, reason,
            pos.entry_fee + fill.fee,
        )
        self.trades.append(trade)
        self.journal.log("close", now, **asdict(trade))

    def flatten(self, now: pd.Timestamp, reason: str = "manual") -> None:
        for sym in list(self.positions):
            self._close(sym, self.last_price.get(sym, self.positions[sym].entry), now, reason)

    # ---- persistence ---------------------------------------------------
    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "risk": self.risk.to_dict(),
            "broker": self.broker.to_dict(),
            "trades": [asdict(t) for t in self.trades[-200:]],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, path)

    def load(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.positions = {k: Position(**v) for k, v in state["positions"].items()}
        self.risk.load(state["risk"])
        self.broker.load(state["broker"])
        self.trades = [ClosedTrade(**t) for t in state["trades"]]
        return True
