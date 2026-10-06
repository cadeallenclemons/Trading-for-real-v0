# Trading-for-real-v0

An AI day-trading agent for crypto. Fast, rule-based code watches the market, places orders and enforces hard risk limits. **Claude** reviews every trade setup before any money moves.

> **Read this first.** Most day traders lose money, and so do most trading bots. This code is a starting framework, not a proven money-maker. Nothing has been tested on real market data yet. Paper trade for weeks before using real money, never trade money you can't afford to lose, and remember that profits are taxable.

## How it works

```
every 15s                        every closed bar (default 15 min)
─────────                        ─────────────────────────────────
latest price ─► stop / target    candles ─► indicators ─► rule scanner finds a setup?
                hit? ─► SELL                                  │ yes
                                         costs check: target ≥ 2× fees+slippage?
                                                              │ yes
                                         risk check: daily loss / trade count / open positions
                                                              │ ok
                                         Claude reviews chart, indicators, account, past trades
                                         ─► take / skip, may shrink size or tighten stop
                                                              │ take
                                         size so a stop-out loses 0.5% of equity ─► BUY
```

* **Strategy** (`daytrader/strategy.py`): long-only "trend pullback reclaim". Uptrend (20 EMA > 50 EMA, price above VWAP), a dip to the 9 EMA, then a bar closes back above it with RSI between 45 and 70. Stop at 1.5 ATR below entry, target at 2.5 ATR above. Exits on stop, target, a close below the 20 EMA, or after 4 hours.
* **Claude** (`daytrader/llm.py`) acts as a skeptical reviewer. It can only make a trade *safer*. The code clamps its output, so it can never widen a stop, enlarge a position or override a risk limit. If Claude errors, times out or declines, the trade is skipped.
* **Risk limits** (`daytrader/risk.py`) are hard-coded checks Claude can't touch: 0.5% of equity risked per trade, at most 25% of equity in one position, trading stops for the day after a 2% loss, at most 8 trades per day and 2 open positions.
* **One engine** (`daytrader/engine.py`) runs the backtest, paper trading and live trading, so you test the same code you trade with.
* Every signal, Claude's reasoning, each order and each exit is logged to `data/journal_<mode>.jsonl`.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env
```

In `.env`, set at least `ANTHROPIC_API_KEY` (from console.anthropic.com), `EXCHANGE` and `SYMBOLS`. **Set `FEE_BPS` to your real exchange fee tier** (100 bps = 1%). Fees decide whether this can make money at all; see below.

## Usage

```bash
# 1. Check that everything runs, offline, on random data (proves the plumbing, not the strategy)
python -m daytrader backtest --synthetic

# 2. Backtest on real history (downloads candles to data/candles/)
python -m daytrader backtest --days 60

# 3. Same backtest, with Claude reviewing each signal (uses API credits; capped at 50 calls by default)
python -m daytrader backtest --days 60 --with-llm --max-llm-calls 50

# 4. Paper trade: live prices, fake money. Leave it running; Ctrl+C to stop.
python -m daytrader paper

# 5. Real money. Only after paper results hold up. Also needs exchange API keys and LIVE_MAX_CAPITAL in .env.
python -m daytrader live --i-understand-this-uses-real-money

# Close every open position now
python -m daytrader flatten            # paper
python -m daytrader flatten --live     # real
```

Run the tests with `pytest`.

## Fees decide everything

A round trip costs twice the fee plus twice the slippage. At 0.25% fee and 0.05% slippage per side, that's **0.6% per trade**. A typical 5-minute BTC bar moves only about 0.1 to 0.2%, so short-timeframe scalping almost always loses to fees at retail fee tiers. That's why:

* the default timeframe is **15m**, and
* the engine rejects any setup whose profit target isn't at least **2× the round-trip cost**. The backtest prints how many setups were skipped for this (`target_below_costs`).

Fees range from about 0.1% to over 1% per side depending on the exchange and your volume tier. Look up your exact rate, and prefer an exchange with low fees.

## Choosing an exchange

Any exchange [ccxt supports](https://github.com/ccxt/ccxt#supported-cryptocurrency-exchanges) works. Set `EXCHANGE` to its ccxt id, e.g. `coinbase`, `kraken`, `binanceus` or `gemini`. Paper trading only needs public market data, so it needs no exchange keys. For live trading, create an API key with **trade permission only, never withdrawal permission**.

Crypto isn't covered by the US pattern-day-trader rule, so accounts under $25k can trade freely.

## Safety notes for live trading

* `LIVE_MAX_CAPITAL` caps how many dollars the bot will ever size against, whatever your balance is. Start tiny.
* Stops are enforced by the bot checking prices every few seconds, not by orders resting on the exchange. **If the bot or your internet stops, open positions have no stop.** Run it on an always-on machine or server, and check on it.
* The bot assumes it owns the coins it bought. Don't trade the same coins by hand on that account while it runs.
* State is saved to `data/state_live.json`, so a restart resumes open positions.

## Suggested path

1. `backtest --synthetic`, to make sure everything runs.
2. `backtest --days 90` on real data, with and without `--with-llm`. Does Claude's filtering improve results, or only reduce trade count? Watch win rate, average R and drawdown, not just total return.
3. Paper trade for at least 2 to 4 weeks and read the journal. Are Claude's skips sensible?
4. Go live with a small `LIVE_MAX_CAPITAL`, and scale up only if live results match paper.

## Ideas for next steps

* A morning "watchlist" agent: Claude picks which coins to trade today from volume and volatility.
* More setups (breakouts, VWAP mean reversion), with Claude choosing among them.
* Resting stop-loss orders on the exchange as a backup to the bot's own stops.
* Notifications on fills and when the daily loss limit hits.

## About `TradingAgents-main.zip`

That's the open-source [TradingAgents](https://github.com/TauricResearch/TradingAgents) research framework: a multi-agent LLM debate that produces one Buy/Hold/Sell rating per stock per day. It's too slow for intraday trading and has no order execution, so this agent is built separately. Its ideas (analyst roles, bull/bear debate) could feed the daily watchlist step above.
