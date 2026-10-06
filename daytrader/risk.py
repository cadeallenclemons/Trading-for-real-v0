"""Hard risk limits. Nothing here can be overridden by the LLM."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from .config import RiskLimits


@dataclass
class DayState:
    day: str = ""
    start_equity: float = 0.0
    trades: int = 0
    realized_pnl: float = 0.0
    halted: bool = False


class RiskManager:
    def __init__(self, limits: RiskLimits):
        self.limits = limits
        self.state = DayState()

    def roll_day(self, now: pd.Timestamp, equity: float) -> None:
        day = now.strftime("%Y-%m-%d")
        if day != self.state.day:
            self.state = DayState(day=day, start_equity=equity)

    def daily_loss_breached(self, equity: float) -> bool:
        if self.state.start_equity <= 0:
            return False
        drawdown_pct = (self.state.start_equity - equity) / self.state.start_equity * 100
        if drawdown_pct >= self.limits.max_daily_loss_pct:
            self.state.halted = True
        return self.state.halted

    def can_open(self, equity: float, open_positions: int) -> tuple[bool, str]:
        if self.state.halted or self.daily_loss_breached(equity):
            return False, "daily_loss_limit"
        if self.state.trades >= self.limits.max_trades_per_day:
            return False, "max_trades_per_day"
        if open_positions >= self.limits.max_open_positions:
            return False, "max_open_positions"
        return True, "ok"

    def size(self, equity: float, cash: float, entry: float, stop: float, size_mult: float = 1.0) -> float:
        """Quantity such that hitting the stop loses risk_per_trade_pct of equity."""
        if entry <= 0 or stop >= entry:
            return 0.0
        capital = equity if self.limits.max_capital is None else min(equity, self.limits.max_capital)
        risk_dollars = capital * self.limits.risk_per_trade_pct / 100 * max(0.0, min(size_mult, 1.0))
        qty = risk_dollars / (entry - stop)
        max_notional = min(capital * self.limits.max_position_pct / 100, cash * 0.98)
        qty = min(qty, max_notional / entry)
        if qty * entry < self.limits.min_notional:
            return 0.0
        return qty

    def record_open(self) -> None:
        self.state.trades += 1

    def record_close(self, pnl: float) -> None:
        self.state.realized_pnl += pnl

    def to_dict(self) -> dict:
        return asdict(self.state)

    def load(self, data: dict) -> None:
        self.state = DayState(**data)
