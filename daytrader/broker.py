"""Order execution: a simulated paper broker and a real ccxt exchange broker."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd


@dataclass
class Fill:
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    time: pd.Timestamp


class PaperBroker:
    """Fills market orders instantly at the reference price plus slippage and fees."""

    def __init__(self, cash: float, fee_bps: float, slippage_bps: float):
        self._cash = cash
        self.holdings: dict[str, float] = {}
        self.fee_rate = fee_bps / 10_000
        self.slip_rate = slippage_bps / 10_000

    def cash(self) -> float:
        return self._cash

    def buy(self, symbol: str, qty: float, ref_price: float, now: pd.Timestamp) -> Fill | None:
        price = ref_price * (1 + self.slip_rate)
        qty = min(qty, self._cash / (price * (1 + self.fee_rate)))
        if qty <= 0:
            return None
        fee = qty * price * self.fee_rate
        self._cash -= qty * price + fee
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) + qty
        return Fill(symbol, "buy", qty, price, fee, now)

    def sell(self, symbol: str, qty: float, ref_price: float, now: pd.Timestamp) -> Fill | None:
        qty = min(qty, self.holdings.get(symbol, 0.0))
        if qty <= 0:
            return None
        price = ref_price * (1 - self.slip_rate)
        fee = qty * price * self.fee_rate
        self._cash += qty * price - fee
        self.holdings[symbol] = self.holdings.get(symbol, 0.0) - qty
        if self.holdings[symbol] <= 1e-12:
            self.holdings.pop(symbol)
        return Fill(symbol, "sell", qty, price, fee, now)

    def to_dict(self) -> dict:
        return {"cash": self._cash, "holdings": self.holdings}

    def load(self, data: dict) -> None:
        self._cash = data["cash"]
        self.holdings = dict(data["holdings"])


class CCXTBroker:
    """Places REAL market orders on a crypto exchange via ccxt."""

    def __init__(self, exchange, quote: str):
        self.ex = exchange
        self.quote = quote
        self.ex.load_markets()

    def cash(self) -> float:
        return float(self.ex.fetch_balance().get("free", {}).get(self.quote, 0.0) or 0.0)

    def buy(self, symbol: str, qty: float, ref_price: float, now: pd.Timestamp) -> Fill | None:
        return self._market(symbol, "buy", qty, ref_price, now)

    def sell(self, symbol: str, qty: float, ref_price: float, now: pd.Timestamp) -> Fill | None:
        base = symbol.split("/")[0]
        held = float(self.ex.fetch_balance().get("free", {}).get(base, 0.0) or 0.0)
        return self._market(symbol, "sell", min(qty, held), ref_price, now)

    def _market(self, symbol: str, side: str, qty: float, ref_price: float, now: pd.Timestamp) -> Fill | None:
        amount = float(self.ex.amount_to_precision(symbol, qty))
        if amount <= 0:
            return None
        # Passing a price lets ccxt compute cost for exchanges whose market buys are priced in quote currency.
        order = self.ex.create_order(symbol, "market", side, amount, ref_price)
        for _ in range(10):
            if order.get("status") == "closed" and order.get("filled"):
                break
            time.sleep(1)
            order = self.ex.fetch_order(order["id"], symbol)
        filled = float(order.get("filled") or 0.0)
        if filled <= 0:
            raise RuntimeError(f"{side} order {order.get('id')} for {symbol} did not fill: {order.get('status')}")
        price = float(order.get("average") or order.get("price") or ref_price)
        fee_info = order.get("fee") or {}
        fee = float(fee_info.get("cost") or 0.0) if fee_info.get("currency") == self.quote else 0.0
        return Fill(symbol, side, filled, price, fee, now)

    def to_dict(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        pass
