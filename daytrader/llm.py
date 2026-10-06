"""Claude as a trade reviewer.

The rules propose a setup with a stop and target; Claude sees the recent price
action, indicators, account state and its own recent results on the symbol, and
decides whether to take it. Claude may only make a trade *safer* (skip it,
shrink it, tighten the stop); the guardrails in `apply_guardrails` enforce that
no matter what the model returns. Any error means the trade is skipped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import anthropic
import pandas as pd
from pydantic import BaseModel, Field

from .strategy import Signal

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You review intraday long setups on crypto spot markets for an automated day-trading system.

A rule-based scanner has found a "trend pullback reclaim" setup: the trend is up (20 EMA above 50 EMA, price above session VWAP), price dipped to the 9 EMA and the bar that just closed reclaimed it. The scanner proposes an entry at the last close, a stop 1.5 ATR below and a target 2.5 ATR above. Fees and slippage are already included in the costs you are shown.

Your job is to filter. The scanner fires on many setups that look fine mechanically but are poor trades, and each losing trade costs fees on both sides. Weigh:
- Trend quality: are the higher lows and EMA slopes clean, or is price chopping around flat averages?
- Where the entry sits: right under an obvious recent high or after an extended run, the reward is smaller than the target implies.
- Volume: does the reclaim bar have participation, or is it drifting on thin volume?
- Volatility: a recent spike bar or a range that is collapsing both make the ATR-based stop and target less reliable.
- Account context: after losses today, or with a position already open, require a clearly better setup.
- Your recent results on this symbol: if similar setups have been stopped out, say what is different this time or skip.

Take the setups where the evidence lines up; skip when it is mixed. You may also shrink the size (size_multiplier below 1.0), raise the stop (it must stay below entry), or set a nearer target. You cannot widen the stop or increase size. Base your decision only on the data provided; you have no news or order book.

Keep the reasoning to two to four sentences that name the specific evidence."""


class TradeDecision(BaseModel):
    decision: Literal["take", "skip"]
    confidence: float = Field(description="How confident you are that taking this trade has positive expectancy, 0.0 to 1.0.")
    size_multiplier: float = Field(description="Fraction of the standard position size to use, 0.25 to 1.0. Use 1.0 when skipping.")
    stop_price: float | None = Field(description="Tighter stop price (above the proposed stop, below entry), or null to keep the proposed stop.")
    target_price: float | None = Field(description="Nearer target price (above entry), or null to keep the proposed target.")
    reasoning: str


@dataclass
class Review:
    take: bool
    confidence: float
    size_mult: float
    stop: float
    target: float
    reasoning: str


def apply_guardrails(signal: Signal, d: TradeDecision, min_confidence: float) -> Review:
    """Clamp the model's output so it can only reduce risk."""
    confidence = max(0.0, min(1.0, d.confidence))
    size_mult = max(0.25, min(1.0, d.size_multiplier))
    stop = signal.stop
    if d.stop_price is not None and signal.stop < d.stop_price < signal.entry:
        stop = d.stop_price
    target = signal.target
    if d.target_price is not None and signal.entry < d.target_price < signal.target:
        target = d.target_price
    take = d.decision == "take" and confidence >= min_confidence
    reasoning = d.reasoning
    if d.decision == "take" and not take:
        reasoning = f"[confidence {confidence:.2f} below {min_confidence}] {reasoning}"
    return Review(take, confidence, size_mult, stop, target, reasoning)


def _bars_table(df: pd.DataFrame, n: int = 40) -> str:
    tail = df.tail(n)
    lines = ["time_utc,open,high,low,close,volume,ema9,ema20,ema50,vwap,rsi"]
    for ts, r in tail.iterrows():
        lines.append(
            f"{ts:%m-%d %H:%M},{r.open:.6g},{r.high:.6g},{r.low:.6g},{r.close:.6g},{r.volume:.4g},"
            f"{r.ema_fast:.6g},{r.ema_mid:.6g},{r.ema_slow:.6g},{r.vwap:.6g},{r.rsi:.1f}"
        )
    return "\n".join(lines)


def build_prompt(signal: Signal, df: pd.DataFrame, timeframe: str, account: dict, history: list[dict]) -> str:
    risk_pct = (signal.entry - signal.stop) / signal.entry * 100
    reward_pct = (signal.target - signal.entry) / signal.entry * 100
    hist = "\n".join(
        f"- {h['opened']} entry {h['entry']:.6g} exit {h['exit']:.6g} ({h['reason']}): {h['r_multiple']:+.2f}R"
        for h in history
    ) or "- none yet"
    return f"""Symbol: {signal.symbol}   Timeframe: {timeframe}   Signal bar: {signal.time:%Y-%m-%d %H:%M} UTC

Proposed long:
- entry {signal.entry:.6g}
- stop {signal.stop:.6g} (-{risk_pct:.3f}%)
- target {signal.target:.6g} (+{reward_pct:.3f}%), reward/risk {signal.reward_risk:.2f}
- round-trip costs (fees + slippage) {account['round_trip_cost_pct']:.3f}%
- indicators on the signal bar: {signal.features}

Account:
- equity {account['equity']:.2f} {account['quote']}, P&L today {account['day_pnl_pct']:+.2f}%
- trades today {account['trades_today']}, open positions {account['open_positions']}

Your recent closed trades on {signal.symbol}:
{hist}

Last {min(40, len(df))} bars:
{_bars_table(df)}
"""


class ClaudeReviewer:
    def __init__(self, model: str, effort: str, min_confidence: float, timeframe: str, client=None):
        self.client = client or anthropic.Anthropic(timeout=90.0, max_retries=2)
        self.model = model
        self.effort = effort
        self.min_confidence = min_confidence
        self.timeframe = timeframe

    def review(self, signal: Signal, df: pd.DataFrame, account: dict, history: list[dict]) -> Review:
        prompt = build_prompt(signal, df, self.timeframe, account, history)
        try:
            response = self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                output_config={"effort": self.effort},
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=TradeDecision,
                # If a safety classifier declines, let the API retry on its recommended fallback model.
                extra_headers={"anthropic-beta": "server-side-fallback-2026-07-01"},
                extra_body={"fallbacks": "default"},
            )
        except anthropic.APIConnectionError as e:
            return self._skip(f"Claude unreachable: {e}")
        except anthropic.RateLimitError as e:
            return self._skip(f"Claude rate limited: {e}")
        except anthropic.APIStatusError as e:
            return self._skip(f"Claude API error {e.status_code}: {e.message}")
        except Exception as e:  # e.g. output failed schema validation; never trade on a broken review
            return self._skip(f"Could not parse Claude's decision: {type(e).__name__}: {e}")

        if response.stop_reason == "refusal":
            return self._skip("Claude declined to review this request")
        if response.stop_reason == "max_tokens" or response.parsed_output is None:
            return self._skip(f"No usable decision (stop_reason={response.stop_reason})")
        return apply_guardrails(signal, response.parsed_output, self.min_confidence)

    def _skip(self, why: str) -> Review:
        log.warning("LLM review failed, skipping trade: %s", why)
        return Review(False, 0.0, 1.0, 0.0, 0.0, why)
