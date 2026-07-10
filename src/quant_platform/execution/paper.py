"""Paper execution: deterministic simulated fills and account state.

Fee and slippage default to the validation protocol's v1 floors so paper
results are never more optimistic than validation assumptions. There is no
live-execution code in this codebase; ExecutionMode has exactly one member
by design (Phase 10 gate, workspace risk R-4).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.risk.engine import OrderRequest, Side

PROTOCOL_FEE_RATE = 0.001        # 0.10% per side (validation protocol v1 floor)
PROTOCOL_SLIPPAGE_RATE = 0.0005  # 0.05% for majors (v1 floor)


class ExecutionMode(str, Enum):
    PAPER = "paper"  # the only mode that exists; live requires the Phase 10 gate


class PaperFill(BaseModel):
    model_config = ConfigDict(frozen=True)

    fill_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    strategy_id: str
    symbol: str
    side: Side
    requested_notional: float
    fill_price: float
    quantity: float
    fee: float
    slippage_cost: float


class PaperExchange:
    """Simulates fills: slippage against the trader, fee on notional."""

    def __init__(
        self,
        fee_rate: float = PROTOCOL_FEE_RATE,
        slippage_rate: float = PROTOCOL_SLIPPAGE_RATE,
    ) -> None:
        if fee_rate < 0 or slippage_rate < 0:
            raise ValueError("fee/slippage rates must be non-negative")
        self._fee_rate = fee_rate
        self._slippage_rate = slippage_rate

    def execute(
        self, order: OrderRequest, market_price: float, quantity: float | None = None
    ) -> PaperFill:
        """Fill an order. With ``quantity`` set (closing sells), fill EXACTLY that
        many units at the slipped price and derive the cash notional from it -
        deriving quantity from a pre-slippage notional would over-sell by the
        slippage factor and leave a residual position (M9 cycle-test finding)."""
        if market_price <= 0:
            raise ValueError(f"market price must be positive, got {market_price}")
        drift = 1.0 + self._slippage_rate if order.side is Side.BUY else 1.0 - self._slippage_rate
        fill_price = market_price * drift
        if quantity is None:
            notional = order.notional
            quantity = notional / fill_price
        else:
            if quantity <= 0:
                raise ValueError(f"explicit quantity must be positive, got {quantity}")
            notional = quantity * fill_price
        return PaperFill(
            strategy_id=order.strategy_id,
            symbol=order.symbol,
            side=order.side,
            requested_notional=round(notional, 8),
            fill_price=round(fill_price, 8),
            quantity=round(quantity, 10),
            fee=round(notional * self._fee_rate, 8),
            slippage_cost=round(abs(fill_price - market_price) * quantity, 8),
        )


class PaperAccount:
    """Cash + positions in units; equity marked against supplied prices."""

    def __init__(self, starting_cash: float) -> None:
        if starting_cash <= 0:
            raise ValueError("starting cash must be positive")
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, float] = {}  # symbol -> quantity (signed)

    def apply(self, fill: PaperFill) -> None:
        if fill.side is Side.BUY:
            self.cash -= fill.requested_notional + fill.fee
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) + fill.quantity
        else:
            self.cash += fill.requested_notional - fill.fee
            self.positions[fill.symbol] = self.positions.get(fill.symbol, 0.0) - fill.quantity
        if abs(self.positions.get(fill.symbol, 0.0)) < 1e-12:
            self.positions.pop(fill.symbol, None)

    def notional(self, symbol: str, price: float) -> float:
        return self.positions.get(symbol, 0.0) * price

    def equity(self, prices: dict[str, float]) -> float:
        value = self.cash
        for symbol, quantity in self.positions.items():
            if symbol not in prices:
                raise KeyError(f"no mark price supplied for open position {symbol}")
            value += quantity * prices[symbol]
        return value
