"""Settings loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


def _get(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value if value else default


def _bool(name: str, default: bool) -> bool:
    return _get(name, str(default)).lower() in {"1", "true", "yes", "on"}


@dataclass
class StrategyParams:
    ema_fast: int = 9
    ema_mid: int = 20
    ema_slow: int = 50
    rsi_len: int = 14
    rsi_min: float = 45.0
    rsi_max: float = 70.0
    atr_len: int = 14
    stop_atr: float = 1.5
    target_atr: float = 2.5
    max_hold_bars: int = 16  # 4 hours on 15m bars
    min_atr_pct: float = 0.0005  # skip dead markets: ATR must be >= 0.05% of price
    min_reward_to_cost: float = 2.0  # target move must be >= 2x round-trip costs
    min_bars: int = 60


@dataclass
class RiskLimits:
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 25.0
    max_daily_loss_pct: float = 2.0
    max_trades_per_day: int = 8
    max_open_positions: int = 2
    min_notional: float = 10.0
    max_capital: float | None = None  # live: never use more than this many quote dollars


@dataclass
class Settings:
    exchange: str = "coinbase"
    symbols: list[str] = field(default_factory=lambda: ["BTC/USD", "ETH/USD"])
    timeframe: str = "15m"
    poll_seconds: int = 15
    fee_bps: float = 25.0
    slippage_bps: float = 5.0
    starting_cash: float = 1000.0

    use_llm: bool = True
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "medium"
    llm_min_confidence: float = 0.6

    exchange_api_key: str = ""
    exchange_api_secret: str = ""
    exchange_api_password: str = ""

    data_dir: str = "data"
    strategy: StrategyParams = field(default_factory=StrategyParams)
    risk: RiskLimits = field(default_factory=RiskLimits)

    @property
    def round_trip_cost_pct(self) -> float:
        return 2 * (self.fee_bps + self.slippage_bps) / 10_000

    @classmethod
    def from_env(cls, env_file: str | None = ".env") -> "Settings":
        if env_file:
            load_dotenv(env_file, override=False)
        live_cap = _get("LIVE_MAX_CAPITAL", "")
        s = cls(
            exchange=_get("EXCHANGE", "coinbase"),
            symbols=[x.strip() for x in _get("SYMBOLS", "BTC/USD,ETH/USD").split(",") if x.strip()],
            timeframe=_get("TIMEFRAME", "15m"),
            poll_seconds=int(_get("POLL_SECONDS", "15")),
            fee_bps=float(_get("FEE_BPS", "25")),
            slippage_bps=float(_get("SLIPPAGE_BPS", "5")),
            starting_cash=float(_get("STARTING_CASH", "1000")),
            use_llm=_bool("USE_LLM", True),
            claude_model=_get("CLAUDE_MODEL", "claude-opus-5-5"),
            claude_effort=_get("CLAUDE_EFFORT", "medium"),
            llm_min_confidence=float(_get("LLM_MIN_CONFIDENCE", "0.6")),
            exchange_api_key=_get("EXCHANGE_API_KEY", ""),
            exchange_api_secret=_get("EXCHANGE_API_SECRET", ""),
            exchange_api_password=_get("EXCHANGE_API_PASSWORD", ""),
            data_dir=_get("DATA_DIR", "data"),
            risk=RiskLimits(
                risk_per_trade_pct=float(_get("RISK_PER_TRADE_PCT", "0.5")),
                max_position_pct=float(_get("MAX_POSITION_PCT", "25")),
                max_daily_loss_pct=float(_get("MAX_DAILY_LOSS_PCT", "2")),
                max_trades_per_day=int(_get("MAX_TRADES_PER_DAY", "8")),
                max_open_positions=int(_get("MAX_OPEN_POSITIONS", "2")),
                max_capital=float(live_cap) if live_cap else None,
            ),
        )
        s.validate()
        return s

    def validate(self) -> None:
        if not self.symbols:
            raise ValueError("SYMBOLS is empty")
        quotes = {sym.split("/")[1] for sym in self.symbols if "/" in sym}
        if len(quotes) != 1 or any("/" not in sym for sym in self.symbols):
            raise ValueError(f"All SYMBOLS must be BASE/QUOTE with one shared quote currency, got {self.symbols}")
        if self.claude_effort not in {"low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"CLAUDE_EFFORT must be low|medium|high|xhigh|max, got {self.claude_effort!r}")
        if not 0 < self.risk.risk_per_trade_pct <= 5:
            raise ValueError("RISK_PER_TRADE_PCT should be between 0 and 5")

    @property
    def quote_currency(self) -> str:
        return self.symbols[0].split("/")[1]
