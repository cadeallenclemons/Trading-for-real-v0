"""Replay historical candles through the same Engine used for live trading."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .config import Settings
from .engine import Engine, Journal, Reviewer
from .indicators import add_indicators


@dataclass
class Report:
    start_equity: float
    end_equity: float
    trades: int
    win_rate: float
    avg_r: float
    profit_factor: float
    max_drawdown_pct: float
    fees_paid: float
    buy_and_hold_pct: dict[str, float]
    exit_reasons: dict[str, int]
    blocked: dict[str, int]

    @property
    def return_pct(self) -> float:
        return (self.end_equity / self.start_equity - 1) * 100

    def __str__(self) -> str:
        bh = ", ".join(f"{k} {v:+.2f}%" for k, v in self.buy_and_hold_pct.items())
        reasons = ", ".join(f"{k}: {v}" for k, v in sorted(self.exit_reasons.items()))
        blocked = ", ".join(f"{k}: {v}" for k, v in sorted(self.blocked.items()))
        pf = "inf" if self.profit_factor == float("inf") else f"{self.profit_factor:.2f}"
        return (
            f"Equity        {self.start_equity:,.2f} -> {self.end_equity:,.2f}  ({self.return_pct:+.2f}%)\n"
            f"Trades        {self.trades}   win rate {self.win_rate:.1%}   avg {self.avg_r:+.2f}R   profit factor {pf}\n"
            f"Max drawdown  {self.max_drawdown_pct:.2f}%\n"
            f"Fees paid     {self.fees_paid:,.2f}\n"
            f"Exits         {reasons or '-'}\n"
            f"Setups skipped {blocked or '-'}\n"
            f"Buy & hold    {bh}"
        )


def run_backtest(
    settings: Settings, candles: dict[str, pd.DataFrame], reviewer: Reviewer | None = None,
    journal: Journal | None = None,
) -> tuple[Report, Engine]:
    broker = PaperBroker(settings.starting_cash, settings.fee_bps, settings.slippage_bps)
    engine = Engine(settings, broker, reviewer, journal)
    frames = {sym: add_indicators(df, settings.strategy) for sym, df in candles.items()}
    positions = {sym: {ts: i for i, ts in enumerate(df.index)} for sym, df in frames.items()}
    timeline = sorted(set().union(*[df.index for df in frames.values()]))

    equity_curve = []
    for ts in timeline:
        for sym, df in frames.items():
            i = positions[sym].get(ts)
            if i is None:
                continue
            bar = df.iloc[i]
            engine.on_price(sym, float(bar["open"]), float(bar["high"]), float(bar["low"]), ts)
            engine.on_bar_close(sym, df.iloc[: i + 1], ts)
        equity_curve.append(engine.equity())
    if timeline:
        engine.flatten(timeline[-1], "end_of_test")
        equity_curve.append(engine.equity())

    curve = np.array(equity_curve) if equity_curve else np.array([settings.starting_cash])
    peak = np.maximum.accumulate(curve)
    max_dd = float(((peak - curve) / peak).max() * 100)
    wins = [t.pnl for t in engine.trades if t.pnl > 0]
    losses = [-t.pnl for t in engine.trades if t.pnl <= 0]
    reasons: dict[str, int] = {}
    for t in engine.trades:
        reasons[t.reason] = reasons.get(t.reason, 0) + 1

    blocked: dict[str, int] = {}
    for e in engine.journal.events:
        if e["event"] == "signal_blocked":
            blocked[e["reason"]] = blocked.get(e["reason"], 0) + 1
        elif e["event"] == "review" and not e["take"]:
            blocked["claude_skipped"] = blocked.get("claude_skipped", 0) + 1

    report = Report(
        start_equity=settings.starting_cash,
        end_equity=float(curve[-1]),
        trades=len(engine.trades),
        win_rate=len(wins) / len(engine.trades) if engine.trades else 0.0,
        avg_r=float(np.mean([t.r_multiple for t in engine.trades])) if engine.trades else 0.0,
        profit_factor=sum(wins) / sum(losses) if losses and sum(losses) > 0 else float("inf") if wins else 0.0,
        max_drawdown_pct=max_dd,
        fees_paid=sum(t.fees for t in engine.trades),
        buy_and_hold_pct={sym: (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100 for sym, df in frames.items()},
        exit_reasons=reasons,
        blocked=blocked,
    )
    return report, engine

