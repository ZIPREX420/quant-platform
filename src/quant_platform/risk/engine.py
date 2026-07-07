"""Deterministic risk engine: the only authority between signals and orders.

Design rules (target architecture):
- pure deterministic code - no LLM anywhere in this path;
- strategy risk parameters (ADR-0005 artifacts) are enforced as HARD CAPS,
  never targets;
- fail-closed: any violated check rejects or shrinks the order; the kill
  switch blocks all exposure-increasing orders for the rest of the day;
- every decision is fully explained (one CheckResult per rule) for the audit
  log - approval without a traceable reason does not exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.data.schemas import PriceHistory


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class PortfolioState(BaseModel):
    """Snapshot the engine judges against. Provided by the caller each time."""

    model_config = ConfigDict(frozen=True)

    equity: float = Field(gt=0, description="current total equity in quote currency")
    equity_start_of_day: float = Field(gt=0)
    positions: dict[str, float] = Field(
        default_factory=dict,
        description="symbol -> signed notional (positive = long)",
    )

    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    def daily_pnl_pct(self) -> float:
        return (self.equity / self.equity_start_of_day - 1.0) * 100.0


class OrderRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_id: str
    symbol: str
    side: Side
    notional: float = Field(gt=0, description="requested order size in quote currency")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class RiskDecision:
    approved: bool
    approved_notional: float
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def reasons(self) -> list[str]:
        return [c.detail for c in self.checks if not c.passed]


def check_price_sanity(
    history: PriceHistory,
    max_staleness_days: int = 2,
    max_daily_move_pct: float = 25.0,
) -> list[CheckResult]:
    """Data-anomaly guards (target architecture failure mode: bad feed).

    A violation should freeze the affected strategy pending human review.
    """
    results = []
    stale = history.staleness_days()
    results.append(
        CheckResult(
            "price_staleness",
            stale <= max_staleness_days,
            f"data is {stale}d old (limit {max_staleness_days}d)",
        )
    )
    if len(history.bars) >= 2:
        prev, last = history.bars[-2].close, history.bars[-1].close
        move = abs(last / prev - 1.0) * 100.0
        results.append(
            CheckResult(
                "price_jump",
                move <= max_daily_move_pct,
                f"last daily move {move:.1f}% (limit {max_daily_move_pct}%)",
            )
        )
    return results


class RiskEngine:
    """Evaluates one order against one strategy's declared risk caps."""

    def __init__(self, risk_params: dict) -> None:
        """risk_params: the `risk` object of a loaded ADR-0005 strategy definition."""
        required = {"max_position_pct_equity", "stop_loss_pct", "max_gross_exposure_pct"}
        missing = required - risk_params.keys()
        if missing:
            raise ValueError(f"risk params missing required caps: {sorted(missing)}")
        self._params = dict(risk_params)

    def evaluate(
        self,
        order: OrderRequest,
        portfolio: PortfolioState,
        sanity: list[CheckResult] | None = None,
    ) -> RiskDecision:
        checks: list[CheckResult] = list(sanity or [])
        current = portfolio.positions.get(order.symbol, 0.0)
        reduces_exposure = (order.side is Side.SELL and current > 0) or (
            order.side is Side.BUY and current < 0
        )

        # 1. Kill switch: daily loss cap blocks anything that adds exposure.
        max_daily_loss = self._params.get("max_daily_loss_pct")
        if max_daily_loss is not None:
            pnl = portfolio.daily_pnl_pct()
            tripped = pnl <= -max_daily_loss
            checks.append(
                CheckResult(
                    "daily_loss_kill_switch",
                    not tripped or reduces_exposure,
                    f"daily pnl {pnl:.2f}% vs cap -{max_daily_loss}%"
                    + (" (reducing order allowed)" if tripped and reduces_exposure else ""),
                )
            )

        # 2. Position size cap (per strategy definition, % of current equity).
        max_position = portfolio.equity * self._params["max_position_pct_equity"] / 100.0
        resulting = abs(current + (order.notional if order.side is Side.BUY else -order.notional))
        headroom = max(0.0, max_position - abs(current)) if not reduces_exposure else order.notional
        size_ok = resulting <= max_position or reduces_exposure
        checks.append(
            CheckResult(
                "position_size_cap",
                size_ok or headroom > 0,
                f"resulting position {resulting:.2f} vs cap {max_position:.2f}",
            )
        )

        # 3. Gross exposure cap across all positions.
        max_gross = portfolio.equity * self._params["max_gross_exposure_pct"] / 100.0
        gross_headroom = max(0.0, max_gross - portfolio.gross_exposure())
        gross_ok = reduces_exposure or order.notional <= gross_headroom
        checks.append(
            CheckResult(
                "gross_exposure_cap",
                gross_ok or gross_headroom > 0,
                f"gross {portfolio.gross_exposure():.2f} + order {order.notional:.2f} "
                f"vs cap {max_gross:.2f}",
            )
        )

        # Decision: any hard failure -> reject; otherwise shrink to headroom.
        hard_failures = [c for c in checks if not c.passed]
        if hard_failures:
            return RiskDecision(False, 0.0, checks)
        if reduces_exposure:
            return RiskDecision(True, order.notional, checks)
        approved = min(order.notional, headroom, gross_headroom)
        if approved <= 0:
            checks.append(CheckResult("headroom", False, "no headroom under caps"))
            return RiskDecision(False, 0.0, checks)
        if approved < order.notional:
            checks.append(
                CheckResult(
                    "size_reduced",
                    True,
                    f"order shrunk {order.notional:.2f} -> {approved:.2f} to respect caps",
                )
            )
        return RiskDecision(True, round(approved, 2), checks)

    def stop_loss_price(self, entry_price: float, side: Side) -> float:
        """Mandatory stop distance from the strategy definition."""
        stop_pct = self._params["stop_loss_pct"] / 100.0
        if side is Side.BUY:
            return round(entry_price * (1.0 - stop_pct), 8)
        return round(entry_price * (1.0 + stop_pct), 8)
