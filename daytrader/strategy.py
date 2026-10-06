"""Rule-based setup detection.

The rules find candidate trades quickly and deterministically; Claude only
reviews candidates the rules have already found. Long-only, since spot crypto
can't be shorted without margin.

Setup ("trend pullback reclaim"):
  * uptrend: mid EMA above slow EMA, and price above session VWAP
  * pullback: previous bar closed at/below the fast EMA
  * trigger: current bar closes back above the fast EMA
  * momentum: RSI inside [rsi_min, rsi_max] (not weak, not overbought)
  * volatility: ATR not negligible (the engine separately rejects setups whose
    target doesn't clear fees + slippage by a margin)
Stop = entry - stop_atr * ATR, target = entry + target_atr * ATR.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .config import StrategyParams


@dataclass
class Signal:
    symbol: str
    side: str
    entry: float
    stop: float
    target: float
    time: pd.Timestamp
    reason: str
    features: dict = field(default_factory=dict)

    @property
    def reward_risk(self) -> float:
        return (self.target - self.entry) / (self.entry - self.stop)


def entry_signal(symbol: str, df: pd.DataFrame, p: StrategyParams) -> Signal | None:
    """Check the last (just closed) bar of an indicator frame for a long setup."""
    if len(df) < max(p.min_bars, p.ema_slow):
        return None
    r, q = df.iloc[-1], df.iloc[-2]
    if pd.isna(r["atr"]) or pd.isna(r["rsi"]) or r["atr"] <= 0:
        return None

    uptrend = r["ema_mid"] > r["ema_slow"] and r["close"] > r["vwap"]
    reclaim = q["close"] <= q["ema_fast"] and r["close"] > r["ema_fast"]
    momentum = p.rsi_min <= r["rsi"] <= p.rsi_max
    lively = r["atr"] / r["close"] >= p.min_atr_pct
    if not (uptrend and reclaim and momentum and lively):
        return None

    entry = float(r["close"])
    stop = entry - p.stop_atr * float(r["atr"])
    target = entry + p.target_atr * float(r["atr"])
    return Signal(
        symbol=symbol,
        side="buy",
        entry=entry,
        stop=stop,
        target=target,
        time=df.index[-1],
        reason="trend_pullback_reclaim",
        features={
            "rsi": round(float(r["rsi"]), 2),
            "atr": float(r["atr"]),
            "atr_pct": round(float(r["atr"] / r["close"]) * 100, 4),
            "ema_fast": float(r["ema_fast"]),
            "ema_mid": float(r["ema_mid"]),
            "ema_slow": float(r["ema_slow"]),
            "vwap": float(r["vwap"]),
            "vol_ratio": round(float(r["vol_ratio"]), 2),
        },
    )


def exit_signal(df: pd.DataFrame) -> str | None:
    """Indicator-based exit for an open long, evaluated on bar close."""
    r = df.iloc[-1]
    if r["close"] < r["ema_mid"]:
        return "trend_break"
    return None
