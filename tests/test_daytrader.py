from types import SimpleNamespace

import anthropic
import httpx2
import pandas as pd
import pytest

from daytrader.backtest import run_backtest
from daytrader.broker import PaperBroker
from daytrader.config import RiskLimits, Settings, StrategyParams
from daytrader.data import drop_unclosed, synthetic
from daytrader.engine import Engine, Journal
from daytrader.indicators import add_indicators, rsi, session_vwap
from daytrader.llm import ClaudeReviewer, Review, TradeDecision, apply_guardrails
from daytrader.risk import RiskManager
from daytrader.strategy import Signal

T0 = pd.Timestamp("2026-01-01 12:00", tz="UTC")


def make_signal(entry=100.0, stop=98.0, target=104.0) -> Signal:
    return Signal("BTC/USD", "buy", entry, stop, target, T0, "test")


# ---- indicators ---------------------------------------------------------

def test_rsi_bounded_and_extremes():
    up = pd.Series(range(1, 60), dtype=float)
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    r = rsi(pd.Series(synthetic(500)["close"].values))
    assert r.dropna().between(0, 100).all()


def test_vwap_resets_each_utc_day():
    idx = pd.to_datetime(["2026-01-01 23:00", "2026-01-01 23:30", "2026-01-02 00:00"], utc=True)
    df = pd.DataFrame({"high": [10, 20, 100], "low": [10, 20, 100], "close": [10, 20, 100], "volume": [1, 1, 1]}, index=idx)
    v = session_vwap(df)
    assert v.iloc[1] == pytest.approx(15)
    assert v.iloc[2] == pytest.approx(100)  # new day, fresh VWAP


def test_drop_unclosed_removes_forming_candle():
    idx = pd.date_range("2026-01-01 00:00", periods=3, freq="15min", tz="UTC")
    df = pd.DataFrame({"close": [1, 2, 3]}, index=idx)
    out = drop_unclosed(df, "15m", now=pd.Timestamp("2026-01-01 00:40", tz="UTC"))
    assert list(out["close"]) == [1, 2]


# ---- risk ---------------------------------------------------------------

def test_position_size_risks_fixed_fraction():
    rm = RiskManager(RiskLimits(risk_per_trade_pct=1.0, max_position_pct=100))
    qty = rm.size(equity=10_000, cash=10_000, entry=100, stop=98)
    assert qty * (100 - 98) == pytest.approx(100)  # 1% of 10k


def test_position_size_capped_by_notional_and_live_capital():
    rm = RiskManager(RiskLimits(risk_per_trade_pct=1.0, max_position_pct=25, max_capital=200))
    qty = rm.size(equity=10_000, cash=10_000, entry=100, stop=99.9)
    assert qty * 100 <= 200 * 0.25 + 1e-9


def test_size_multiplier_cannot_increase_size():
    rm = RiskManager(RiskLimits(risk_per_trade_pct=1.0, max_position_pct=100))
    assert rm.size(10_000, 10_000, 100, 98, size_mult=5.0) == rm.size(10_000, 10_000, 100, 98, 1.0)


def test_daily_loss_limit_halts_and_resets_next_day():
    rm = RiskManager(RiskLimits(max_daily_loss_pct=2))
    rm.roll_day(T0, 1000)
    assert rm.can_open(985, 0)[0]
    assert rm.can_open(979, 0) == (False, "daily_loss_limit")
    assert rm.can_open(1000, 0) == (False, "daily_loss_limit")  # stays halted for the day
    rm.roll_day(T0 + pd.Timedelta(days=1), 979)
    assert rm.can_open(979, 0)[0]


def test_trade_and_position_count_limits():
    rm = RiskManager(RiskLimits(max_trades_per_day=1, max_open_positions=1))
    rm.roll_day(T0, 1000)
    assert rm.can_open(1000, 1) == (False, "max_open_positions")
    rm.record_open()
    assert rm.can_open(1000, 0) == (False, "max_trades_per_day")


# ---- paper broker -------------------------------------------------------

def test_paper_broker_round_trip_costs():
    b = PaperBroker(cash=1000, fee_bps=10, slippage_bps=10)
    buy = b.buy("BTC/USD", 1.0, 100.0, T0)
    assert buy.price == pytest.approx(100.1)
    sell = b.sell("BTC/USD", 1.0, 100.0, T0)
    assert sell.price == pytest.approx(99.9)
    assert b.cash() == pytest.approx(1000 - 100.1 - 0.1001 + 99.9 - 0.0999)
    assert "BTC/USD" not in b.holdings


def test_paper_broker_never_overspends():
    b = PaperBroker(cash=50, fee_bps=10, slippage_bps=0)
    fill = b.buy("BTC/USD", 10.0, 100.0, T0)
    assert b.cash() >= -1e-9 and fill.qty < 0.5


# ---- engine exits -------------------------------------------------------

def engine_with_position(stop=98.0, target=104.0):
    s = Settings(fee_bps=0, slippage_bps=0)
    e = Engine(s, PaperBroker(1000, 0, 0))
    e._open(make_signal(stop=stop, target=target), Review(True, 1, 1, stop, target, ""), T0)
    return e


def test_stop_fills_at_stop_or_worse_on_gap():
    e = engine_with_position()
    e.on_price("BTC/USD", open_=99, high=99.5, low=97, now=T0)
    assert e.trades[-1].exit == pytest.approx(98) and e.trades[-1].reason == "stop"
    e = engine_with_position()
    e.on_price("BTC/USD", open_=95, high=96, low=94, now=T0)  # gapped through the stop
    assert e.trades[-1].exit == pytest.approx(95)


def test_stop_wins_when_bar_hits_both_levels():
    e = engine_with_position()
    e.on_price("BTC/USD", open_=100, high=105, low=97, now=T0)
    assert e.trades[-1].reason == "stop"


def test_target_exit_and_r_multiple():
    e = engine_with_position()
    e.on_price("BTC/USD", open_=101, high=104.5, low=100.5, now=T0)
    t = e.trades[-1]
    assert t.reason == "target" and t.r_multiple == pytest.approx(2.0)


def test_state_round_trip(tmp_path):
    e = engine_with_position()
    path = str(tmp_path / "state.json")
    e.save(path)
    e2 = Engine(e.s, PaperBroker(0, 0, 0))
    assert e2.load(path)
    assert e2.positions["BTC/USD"].stop == 98 and e2.broker.cash() == pytest.approx(e.broker.cash())


# ---- LLM guardrails and reviewer ---------------------------------------

def decision(**kw) -> TradeDecision:
    base = dict(decision="take", confidence=0.8, size_multiplier=1.0, stop_price=None, target_price=None, reasoning="ok")
    return TradeDecision(**{**base, **kw})


def test_guardrails_allow_only_risk_reduction():
    sig = make_signal()
    r = apply_guardrails(sig, decision(stop_price=90, target_price=200, size_multiplier=3), 0.6)
    assert (r.stop, r.target, r.size_mult) == (98, 104, 1.0)  # wider stop / farther target / bigger size ignored
    r = apply_guardrails(sig, decision(stop_price=99, target_price=103, size_multiplier=0.5), 0.6)
    assert (r.stop, r.target, r.size_mult) == (99, 103, 0.5)
    r = apply_guardrails(sig, decision(stop_price=100.5), 0.6)
    assert r.stop == 98  # stop above entry is nonsense, keep the original


def test_guardrails_confidence_threshold():
    assert not apply_guardrails(make_signal(), decision(confidence=0.5), 0.6).take
    assert not apply_guardrails(make_signal(), decision(decision="skip", confidence=0.9), 0.6).take


class FakeMessages:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.kwargs = result, error, None

    def parse(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def reviewer_with(result=None, error=None):
    msgs = FakeMessages(result, error)
    return ClaudeReviewer("claude-opus-5-5", "medium", 0.6, "15m", client=SimpleNamespace(messages=msgs)), msgs


def review_once(reviewer):
    df = add_indicators(synthetic(100), StrategyParams())
    acct = {"equity": 1000, "quote": "USD", "day_pnl_pct": 0, "trades_today": 0, "open_positions": 0, "round_trip_cost_pct": 0.6}
    return reviewer.review(make_signal(), df, acct, [])


def test_reviewer_takes_and_sends_expected_request():
    rv, msgs = reviewer_with(SimpleNamespace(stop_reason="end_turn", parsed_output=decision(stop_price=99)))
    r = review_once(rv)
    assert r.take and r.stop == 99
    assert msgs.kwargs["model"] == "claude-opus-5-5"
    assert msgs.kwargs["output_format"] is TradeDecision
    assert msgs.kwargs["extra_body"] == {"fallbacks": "default"}
    assert "Proposed long" in msgs.kwargs["messages"][0]["content"]


def test_reviewer_fails_closed():
    rv, _ = reviewer_with(SimpleNamespace(stop_reason="refusal", parsed_output=None))
    assert not review_once(rv).take
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    rv, _ = reviewer_with(error=anthropic.APIConnectionError(request=req))
    assert not review_once(rv).take
    rv, _ = reviewer_with(error=ValueError("bad json"))
    assert not review_once(rv).take


# ---- backtest -----------------------------------------------------------

def test_backtest_accounting_is_consistent():
    s = Settings(fee_bps=5, slippage_bps=2, symbols=["BTC/USD", "ETH/USD"])
    candles = {"BTC/USD": synthetic(1500, timeframe_minutes=15, seed=1),
               "ETH/USD": synthetic(1500, 3000, timeframe_minutes=15, seed=2)}
    report, engine = run_backtest(s, candles, journal=Journal())
    assert report.trades > 0 and not engine.positions
    assert report.end_equity == pytest.approx(s.starting_cash + sum(t.pnl for t in engine.trades))


def test_backtest_routes_signals_through_reviewer():
    class SkipAll:
        calls = 0

        def review(self, signal, df, account, history):
            SkipAll.calls += 1
            return Review(False, 0, 1, 0, 0, "no")

    s = Settings(fee_bps=5, slippage_bps=2, symbols=["BTC/USD"])
    report, _ = run_backtest(s, {"BTC/USD": synthetic(1500, timeframe_minutes=15, seed=1)}, SkipAll(), Journal())
    assert SkipAll.calls > 0 and report.trades == 0
    assert report.blocked["claude_skipped"] == SkipAll.calls
