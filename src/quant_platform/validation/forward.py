"""Forward-record analyzer (protocol v2 addendum): the paper audit as evidence.

Reconstructs completed round trips per candidate from executions.jsonl and
measures them against the pre-registered thresholds F1-F5 of
docs/validation/validation-protocol-v2-forward.md. Evidence is computed
from records, never assembled by hand; a forward-evidence report must embed
this tool's output verbatim.

Round-trip semantics: fills for one candidate are consumed in order; a trip
opens on the first fill from flat (BUY = long trip, SELL = short trip, M12)
and completes when the position returns to flat (partial closes aggregate
into the same trip). Net return per trip is measured on the ENTRY leg's
notional: long = (proceeds - cost) / cost; short = (proceeds - cost) /
proceeds - exactly what the paper account experienced, costs included.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from quant_platform.execution.session import AuditRecord
from quant_platform.risk.engine import Side
from quant_platform.validation.analysis import monte_carlo, trade_metrics
from quant_platform.validation.trades import Trade

# Pre-registered thresholds (protocol v2 addendum, fixed 2026-07-10).
MIN_DAYS = 180          # F1
MIN_ROUND_TRIPS = 100   # F2
MIN_PROFIT_FACTOR = 1.15  # F3
MC_RUNS = 1000          # F5


class ForwardRecordError(ValueError):
    """The audit record cannot be interpreted as a coherent forward record."""


@dataclass(frozen=True)
class RoundTrip:
    candidate_id: str
    symbol: str
    opened_at: datetime
    closed_at: datetime
    cost: float       # buy notional + buy fees
    proceeds: float   # sell notional - sell fees
    direction: str = "long"

    @property
    def return_fraction(self) -> float:
        if self.direction == "short":
            return (self.proceeds - self.cost) / self.proceeds
        return self.proceeds / self.cost - 1.0

    def as_trade(self) -> Trade:
        return Trade(return_fraction=self.return_fraction)


@dataclass(frozen=True)
class ForwardAssessment:
    candidate_id: str
    round_trips: int
    open_position: bool
    first_fill: datetime | None
    last_fill: datetime | None
    evidence_days: int
    total_return_pct: float | None
    profit_factor: float | None
    max_drawdown_pct: float | None
    mc_p05_terminal_return_pct: float | None
    criteria: dict[str, bool | None]  # F1..F5; None = not yet measurable

    def qualifies(self) -> bool:
        return all(v is True for v in self.criteria.values())

    def summary(self) -> str:
        lines = [f"forward record: {self.candidate_id}"]
        lines.append(
            f"  round trips: {self.round_trips}"
            + (" (+1 open position)" if self.open_position else "")
        )
        if self.first_fill and self.last_fill:
            lines.append(
                f"  window: {self.first_fill.date()} -> {self.last_fill.date()}"
                f" ({self.evidence_days} days)"
            )
        if self.total_return_pct is not None:
            pf = f"{self.profit_factor:.2f}" if self.profit_factor is not None else "inf"
            lines.append(
                f"  net return {self.total_return_pct:+.2f}% | profit factor {pf}"
                f" | max drawdown {self.max_drawdown_pct:.2f}%"
            )
        if self.mc_p05_terminal_return_pct is not None:
            lines.append(f"  MC p05 terminal return: {self.mc_p05_terminal_return_pct:+.2f}%")
        for name, value in self.criteria.items():
            verdict = "PASS" if value is True else ("fail" if value is False else "not yet measurable")
            lines.append(f"  {name}: {verdict}")
        lines.append(
            "  QUALIFIES for a forward-evidence report" if self.qualifies()
            else "  does not (yet) qualify - thresholds are pre-registered and fixed"
        )
        return "\n".join(lines)


def round_trips_for(records: list[AuditRecord], candidate_id: str) -> tuple[list[RoundTrip], bool]:
    """(completed round trips, any position still open?) for one candidate.

    Multi-symbol candidates (M12) hold independent positions per symbol; fills
    are reconstructed per symbol so trips can never interleave across books.
    """
    all_fills = [
        r for r in records
        if r.strategy_id == candidate_id and r.tier == "candidate" and r.fill is not None
    ]
    trips: list[RoundTrip] = []
    any_open = False
    for sym in sorted({r.symbol for r in all_fills}):
        sym_trips, sym_open = _round_trips_one_symbol(
            [r for r in all_fills if r.symbol == sym], candidate_id
        )
        trips.extend(sym_trips)
        any_open = any_open or sym_open
    trips.sort(key=lambda t: t.closed_at)
    return trips, any_open


def _round_trips_one_symbol(
    fills: list[AuditRecord], candidate_id: str
) -> tuple[list[RoundTrip], bool]:
    trips: list[RoundTrip] = []
    position = 0.0  # signed: positive long, negative short
    cost = proceeds = 0.0
    opened_at: datetime | None = None
    symbol = ""
    direction = "long"
    for r in fills:
        f = r.fill
        flat = opened_at is None
        if r.side is Side.BUY:
            if flat:
                opened_at, symbol, direction = r.ts, r.symbol, "long"
            elif direction == "long" and position < -1e-12:
                raise ForwardRecordError(
                    f"{candidate_id}: BUY while bookkeeping is inconsistent at {r.ts}"
                )
            position += f["quantity"]
            cost += f["quantity"] * f["fill_price"] + f["fee"]
        else:
            if flat:
                opened_at, symbol, direction = r.ts, r.symbol, "short"
            elif direction == "long" and position <= 1e-12:
                raise ForwardRecordError(f"{candidate_id}: SELL without open position at {r.ts}")
            position -= f["quantity"]
            proceeds += f["quantity"] * f["fill_price"] - f["fee"]
        # a close that flips the sign is bookkeeping corruption, not a trip
        if direction == "long" and position < -1e-10:
            raise ForwardRecordError(f"{candidate_id}: SELL exceeds long position at {r.ts}")
        if direction == "short" and position > 1e-10:
            raise ForwardRecordError(f"{candidate_id}: BUY exceeds short position at {r.ts}")
        if opened_at is not None and abs(position) <= 1e-10 and (cost > 0 and proceeds > 0):
            trips.append(RoundTrip(
                candidate_id=candidate_id, symbol=symbol,
                opened_at=opened_at, closed_at=r.ts, cost=cost, proceeds=proceeds,
                direction=direction,
            ))
            position, cost, proceeds, opened_at = 0.0, 0.0, 0.0, None
    return trips, opened_at is not None


def assess(records: list[AuditRecord], candidate_id: str) -> ForwardAssessment:
    """Measure one candidate's paper record against the F1-F5 thresholds."""
    trips, open_position = round_trips_for(records, candidate_id)
    fills = [r.ts for r in records
             if r.strategy_id == candidate_id and r.tier == "candidate" and r.fill]
    first, last = (min(fills), max(fills)) if fills else (None, None)
    days = (last - first).days if fills else 0

    total = pf = maxdd = p05 = None
    if trips:
        metrics = trade_metrics([t.as_trade() for t in trips])
        total, pf, maxdd = metrics.total_return_pct, metrics.profit_factor, metrics.max_drawdown_pct
    if len(trips) >= 10:
        p05 = monte_carlo(
            [t.as_trade() for t in trips], runs=MC_RUNS
        ).terminal_return_pct_p05

    criteria: dict[str, bool | None] = {
        "F1_duration_180d": days >= MIN_DAYS if fills else None,
        "F2_round_trips_100": len(trips) >= MIN_ROUND_TRIPS if trips else None,
        "F3_net_positive_pf": (
            (total > 0 and (pf is None or pf >= MIN_PROFIT_FACTOR)) if trips else None
        ),
        "F5_mc_p05_positive": (p05 > 0.0) if p05 is not None else None,
    }
    return ForwardAssessment(
        candidate_id=candidate_id,
        round_trips=len(trips),
        open_position=open_position,
        first_fill=first,
        last_fill=last,
        evidence_days=days,
        total_return_pct=round(total, 2) if total is not None else None,
        profit_factor=round(pf, 3) if pf is not None else None,
        max_drawdown_pct=round(maxdd, 2) if maxdd is not None else None,
        mc_p05_terminal_return_pct=round(p05, 2) if p05 is not None else None,
        criteria=criteria,
    )
