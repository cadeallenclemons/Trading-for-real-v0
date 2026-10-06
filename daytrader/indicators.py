"""Technical indicators. All are causal: row i only uses data up to row i."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyParams


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / length, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(loss.notna(), np.nan)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    true_range = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / length, adjust=False).mean()


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets at each UTC midnight (crypto has no session open)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    day = df.index.floor("D")
    pv = (typical * df["volume"]).groupby(day).cumsum()
    vol = df["volume"].groupby(day).cumsum()
    return (pv / vol.replace(0, np.nan)).fillna(typical)


def add_indicators(df: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    out = df.copy()
    out["ema_fast"] = ema(out["close"], p.ema_fast)
    out["ema_mid"] = ema(out["close"], p.ema_mid)
    out["ema_slow"] = ema(out["close"], p.ema_slow)
    out["rsi"] = rsi(out["close"], p.rsi_len)
    out["atr"] = atr(out, p.atr_len)
    out["vwap"] = session_vwap(out)
    out["vol_ratio"] = out["volume"] / out["volume"].rolling(20, min_periods=1).mean()
    return out
