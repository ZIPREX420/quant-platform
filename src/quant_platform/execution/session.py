"""Audited paper-trading session: signal -> risk engine -> paper fill -> audit.

The session is the ONLY component allowed to call the exchange adapter, and it
does so exclusively with notionals approved by the deterministic RiskEngine.
Every signal - approved, shrunk, or rejected - produces an append-only audit
record. A session runs either a LoadedStrategy (ADR-0005 validated tier) or a
LoadedCandidate (ADR-0006 candidate tier, paper only); in both cases the
loader has already enforced the artifact contract, and every audit record is
stamped with the tier so the two records can never be conflated.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.execution.paper import ExecutionMode, PaperAccount, PaperExchange, PaperFill
from quant_platform.risk.engine import CheckResult, OrderRequest, PortfolioState, RiskEngine, Side
from quant_platform.strategies.candidates import LoadedCandidate
from quant_platform.strategies.loader import LoadedStrategy


class AuditRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    audit_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    mode: ExecutionMode
    tier: str = "validated"
    strategy_id: str
    symbol: str
    side: Side
    requested_notional: float
    approved: bool
    approved_notional: float
    checks: list[dict]
    fill: dict | None = None
    equity_after: float | None = None


class ExecutionAudit:
    """Append-only JSONL audit trail for execution decisions."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: AuditRecord) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")

    def records(self) -> list[AuditRecord]:
        if not self._path.exists():
            return []
        return [
            AuditRecord.model_validate_json(line)
            for line in self._path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]


class PaperTradingSession:
    """One strategy, one account, one audit trail. Paper mode only."""

    def __init__(
        self,
        strategy: LoadedStrategy | LoadedCandidate,
        account: PaperAccount,
        audit: ExecutionAudit,
        exchange: PaperExchange | None = None,
    ) -> None:
        self.strategy = strategy
        self.account = account
        self.audit = audit
        self.exchange = exchange or PaperExchange()
        self.engine = RiskEngine(strategy.definition["risk"])
        self.mode = ExecutionMode.PAPER
        self.tier = getattr(strategy, "tier", "validated")

    def process_signal(
        self,
        symbol: str,
        side: Side,
        target_notional: float,
        prices: dict[str, float],
        equity_start_of_day: float,
        sanity: list[CheckResult] | None = None,
    ) -> AuditRecord:
        """Run one signal through risk checks; execute on paper iff approved."""
        if symbol not in prices:
            raise KeyError(f"no price supplied for {symbol}")
        order = OrderRequest(
            strategy_id=self.strategy.id, symbol=symbol, side=side, notional=target_notional
        )
        portfolio = PortfolioState(
            equity=self.account.equity(prices),
            equity_start_of_day=equity_start_of_day,
            positions={
                s: self.account.notional(s, prices[s])
                for s in self.account.positions
                if s in prices
            },
        )
        decision = self.engine.evaluate(order, portfolio, sanity=sanity)

        fill: PaperFill | None = None
        if decision.approved and decision.approved_notional > 0:
            approved_order = OrderRequest(
                strategy_id=order.strategy_id,
                symbol=order.symbol,
                side=order.side,
                notional=decision.approved_notional,
            )
            fill = self.exchange.execute(approved_order, prices[symbol])
            self.account.apply(fill)

        record = AuditRecord(
            mode=self.mode,
            tier=self.tier,
            strategy_id=self.strategy.id,
            symbol=symbol,
            side=side,
            requested_notional=target_notional,
            approved=decision.approved,
            approved_notional=decision.approved_notional,
            checks=[{"name": c.name, "passed": c.passed, "detail": c.detail} for c in decision.checks],
            fill=fill.model_dump(mode="json") if fill else None,
            equity_after=round(self.account.equity(prices), 2),
        )
        self.audit.append(record)
        return record
