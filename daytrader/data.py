"""Market data: exchange candles via ccxt, CSV cache, and a synthetic generator for offline testing."""

from __future__ import annotations

import os
import time

import ccxt
import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def make_exchange(exchange_id: str, api_key: str = "", secret: str = "", password: str = ""):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unknown exchange {exchange_id!r}; see ccxt.exchanges for the list")
    config = {"enableRateLimit": True}
    if api_key:
        config.update(apiKey=api_key, secret=secret)
        if password:
            config["password"] = password
    return getattr(ccxt, exchange_id)(config)


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["ts", *COLUMNS])
    df.index = pd.to_datetime(df.pop("ts"), unit="ms", utc=True)
    df.index.name = "time"
    return df[~df.index.duplicated(keep="last")].sort_index().astype(float)


def drop_unclosed(df: pd.DataFrame, timeframe: str, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Exchanges return the still-forming candle last; trade only on closed bars."""
    now = now or pd.Timestamp.now(tz="UTC")
    bar = pd.Timedelta(seconds=ccxt.Exchange.parse_timeframe(timeframe))
    return df[df.index + bar <= now]


def fetch_recent(exchange, symbol: str, timeframe: str, limit: int = 300) -> pd.DataFrame:
    return drop_unclosed(to_frame(exchange.fetch_ohlcv(symbol, timeframe, limit=limit)), timeframe)


def fetch_history(exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Page backwards-in-time-safe through history from `days` ago until now."""
    step_ms = ccxt.Exchange.parse_timeframe(timeframe) * 1000
    since = exchange.milliseconds() - days * 86_400_000
    rows: list[list] = []
    while since < exchange.milliseconds() - step_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=300)
        if not batch:
            break
        rows.extend(batch)
        nxt = batch[-1][0] + step_ms
        if nxt <= since:
            break
        since = nxt
        time.sleep(exchange.rateLimit / 1000)
    return drop_unclosed(to_frame(rows), timeframe)


def csv_path(data_dir: str, exchange_id: str, symbol: str, timeframe: str) -> str:
    return os.path.join(data_dir, "candles", f"{exchange_id}_{symbol.replace('/', '-')}_{timeframe}.csv")


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    return df[COLUMNS].astype(float)


def synthetic(bars: int = 5000, start_price: float = 60_000.0, timeframe_minutes: int = 5, seed: int = 0) -> pd.DataFrame:
    """Random walk with drifting trend regimes and volatility clustering.

    Only for testing the plumbing - a strategy that wins on random data proves nothing.
    """
    rng = np.random.default_rng(seed)
    scale = np.sqrt(timeframe_minutes / 5)  # volatility grows with the square root of bar length
    drift = np.repeat(rng.normal(0, 0.0003 * scale, bars // 200 + 1), 200)[:bars]
    vol = 0.0015 * scale * np.exp(np.cumsum(rng.normal(0, 0.05, bars)).clip(-1, 1))
    rets = drift + vol * rng.standard_normal(bars)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, vol * 0.6)) * close
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - np.abs(rng.normal(0, vol * 0.6)) * close
    volume = rng.lognormal(3, 0.5, bars) * (1 + 50 * np.abs(rets))
    index = pd.date_range("2026-01-01", periods=bars, freq=f"{timeframe_minutes}min", tz="UTC", name="time")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=index)
