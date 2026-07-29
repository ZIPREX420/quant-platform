"""Forward-record analyzer (protocol v2 addendum): the paper audit as evidence.

Reconstructs completed round trips per candidate from executions.jsonl and
measures them against the pre-registered thresholds of
docs/validation/validation-protocol-v2-forward.md. Evidence is computed from
records, never assembled by hand; a forward-evidence report must embed this
tool's output verbatim.

Coverage of F1-F7 (per the report template's Results table):
- F1 duration, F2 round trips, F3 net+PF, F5 Monte-Carlo p05  -> machine-decided
  from the round-trip series;
- F4 kill-switch breaches (=0)                                -> machine-decided
  from the per-cycle equity curve vs the definition's max_daily_loss_pct;
- F6 window integrity (no unexplained cycle gaps)             -> machine-decided
  from the equity-history cycle sequence.
The remaining pieces are inherently human attestations and are SURFACED, never
auto-passed: F4 "max drawdown within declared caps" (the schema declares no
drawdown cap), F6 "state file never manually repaired", and F7 prediction
review. `qualifies()` therefore means "every machine-decidable criterion
passes"; a forward-evidence report additionally records the three attestations.

Round-trip semantics: fills for one candidate are consumed in order; a trip
opens on the first fill from flat (BUY = long trip, SELL = short trip, M12)
and completes when the position returns to flat (partial closes aggregate into
the same trip). Net return per trip is measured on the ENTRY leg's notional.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
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
    criteria: dict[str, bool | None]  # machine-decidable; None = not yet measurable
    kill_switch_breach_days: int | None = None
    integrity_notes: list[str] = field(default_factory=list)
    prediction: str | None = None

    def qualifies(self) -> bool:
        """All MACHINE-decidable criteria pass. A forward-evidence report also
        needs the surfaced human attestations (F4 caps, F6 no-repair, F7)."""
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
        if self.kill_switch_breach_days is not None:
            lines.append(f"  kill-switch breach days: {self.kill_switch_breach_days}")
        for name, value in self.criteria.items():
            verdict = "PASS" if value is True else ("fail" if value is False else "not yet measurable")
            lines.append(f"  {name}: {verdict}")
        for note in self.integrity_notes:
            lines.append(f"  F6 integrity note: {note}")
        lines.append(
            "  QUALIFIES on machine criteria - human attestations still required "
            "(F4 drawdown-vs-caps, F6 'state never manually repaired', F7 prediction)"
            if self.qualifies()
            else "  does not (yet) qualify - thresholds are pre-registered and fixed"
        )
        lines.append(
            "  F4 max drawdown vs declared caps: HUMAN REVIEW (no drawdown cap in the "
            f"risk schema; measured max drawdown {self.max_drawdown_pct if self.max_drawdown_pct is not None else 'n/a'}%)"
        )
        if self.prediction:
            lines.append(f"  F7 prediction review (HUMAN, compare to outcome): {self.prediction}")
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


_DUST = 1e-6  # |position| below this is flat (accumulated fill-rounding tolerance)


def _round_trips_one_symbol(
    fills: list[AuditRecord], candidate_id: str
) -> tuple[list[RoundTrip], bool]:
    """Reconstruct flat-to-flat round trips for one candidate+symbol.

    Handles a genuine direction flip: a fill that carries the position THROUGH
    zero closes the current trip (with the units that reach flat) and opens the
    opposite-direction trip with the excess - so a long sold through flat becomes
    a short, exactly as a funding-carry candidate legitimately trades. A fill that
    lands within _DUST of flat closes cleanly (accumulated fill rounding, not a
    new trip). BUY units accrue to ``cost`` (notional + fee), SELL units to
    ``proceeds`` (notional - fee); ``direction`` selects the return formula. The
    unused ``candidate_id`` stamps each RoundTrip.
    """
    trips: list[RoundTrip] = []
    position = 0.0            # signed units of the open trip (0 = flat)
    cost = proceeds = 0.0     # accumulators for the open trip
    opened_at: datetime | None = None
    symbol = ""
    direction = "long"

    def accrue(units: float, price: float, fee_per_unit: float, is_buy: bool) -> None:
        nonlocal cost, proceeds
        if is_buy:
            cost += units * price + units * fee_per_unit
        else:
            proceeds += units * price - units * fee_per_unit

    def close(closed_at: datetime) -> None:
        nonlocal position, cost, proceeds, opened_at
        if opened_at is not None and cost > 0 and proceeds > 0:
            trips.append(RoundTrip(
                candidate_id=candidate_id, symbol=symbol, opened_at=opened_at,
                closed_at=closed_at, cost=cost, proceeds=proceeds, direction=direction,
            ))
        position, cost, proceeds, opened_at = 0.0, 0.0, 0.0, None

    for r in fills:
        f = r.fill
        qty, price, fee = f["quantity"], f["fill_price"], f["fee"]
        if qty <= 0:
            continue
        is_buy = r.side is Side.BUY
        fee_per_unit = fee / qty
        signed = qty if is_buy else -qty
        new_position = position + signed

        # Flip: the fill carries the position through zero (beyond dust on both
        # sides). Split it - close the current trip with the units that reach
        # flat, open the opposite-direction trip with the remainder.
        if (
            opened_at is not None
            and abs(position) > _DUST
            and position * new_position < 0
            and abs(new_position) > _DUST
        ):
            closing = abs(position)
            accrue(closing, price, fee_per_unit, is_buy)
            close(r.ts)
            opening = qty - closing
            symbol, direction, opened_at = r.symbol, ("long" if is_buy else "short"), r.ts
            accrue(opening, price, fee_per_unit, is_buy)
            position = opening if is_buy else -opening
            continue

        if opened_at is None:
            symbol, direction, opened_at = r.symbol, ("long" if is_buy else "short"), r.ts
        accrue(qty, price, fee_per_unit, is_buy)
        position = new_position
        if abs(position) <= _DUST:
            close(r.ts)

    return trips, opened_at is not None


def _parse_ts(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _window_rows(
    equity_history: list[dict], first: datetime, last: datetime
) -> list[dict]:
    """Per-cycle equity marks whose timestamp lies within [first, last]."""
    rows = []
    for r in equity_history:
        ts = _parse_ts(r.get("ts", ""))
        if ts is not None and first <= ts <= last:
            rows.append(r)
    return rows


def _kill_switch_breach_days(rows: list[dict], max_daily_loss_pct: float) -> int:
    """UTC days in which account equity fell at least max_daily_loss_pct below
    that day's opening mark - a faithful replay of the live (account-level)
    daily-loss kill switch. Rows must be chronological (append-only)."""
    by_day: dict[str, list[float]] = {}
    for r in rows:
        by_day.setdefault(str(r.get("ts", ""))[:10], []).append(float(r.get("equity", 0.0)))
    breaches = 0
    for equities in by_day.values():
        anchor = equities[0]
        if anchor <= 0:
            continue
        worst_pct = min((e - anchor) / anchor for e in equities) * 100.0
        if worst_pct <= -max_daily_loss_pct:
            breaches += 1
    return breaches


def _integrity(rows: list[dict]) -> tuple[bool, list[str]]:
    """F6 core: the per-cycle equity marks in the window must be a contiguous,
    duplicate-free run of cycle numbers. A gap = a cycle that never recorded a
    mark (a silent stop or a hand-removed record); a duplicate = corruption.
    This would have flagged the 2026-07-24/25 stuck-cycle window automatically."""
    cycles = [int(r["cycle"]) for r in rows if "cycle" in r]
    if not cycles:
        return False, ["no per-cycle equity marks in the evidence window"]
    counts = Counter(cycles)
    notes: list[str] = []
    dupes = sorted(c for c, n in counts.items() if n > 1)
    if dupes:
        notes.append(f"duplicate cycle marks: {dupes}")
    uniq = sorted(counts)
    missing = (uniq[-1] - uniq[0] + 1) - len(uniq)
    if missing > 0:
        notes.append(
            f"{missing} missing cycle mark(s) in window (cycles {uniq[0]}..{uniq[-1]}) "
            "- unexplained gap; explain or disclose before an F6 pass"
        )
    return (not notes), notes


def assess(
    records: list[AuditRecord],
    candidate_id: str,
    equity_history: list[dict] | None = None,
    definition: dict | None = None,
) -> ForwardAssessment:
    """Measure one candidate's paper record against F1-F6.

    equity_history (per-cycle marks) and the candidate definition are optional;
    without them F4 (kill switch) and F6 (integrity) are 'not yet measurable'
    rather than passing - the tool never asserts a criterion it cannot check.
    """
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
        p05 = monte_carlo([t.as_trade() for t in trips], runs=MC_RUNS).terminal_return_pct_p05

    # F4 (kill switch) + F6 (integrity) require the per-cycle equity marks.
    kill_breaches: int | None = None
    f4_kill: bool | None = None
    f6_integrity: bool | None = None
    integrity_notes: list[str] = []
    max_daily_loss = (definition or {}).get("risk", {}).get("max_daily_loss_pct")
    if equity_history and first and last:
        rows = _window_rows(equity_history, first, last)
        if rows:
            f6_integrity, integrity_notes = _integrity(rows)
            if max_daily_loss is not None:
                kill_breaches = _kill_switch_breach_days(rows, max_daily_loss)
                f4_kill = kill_breaches == 0
            else:
                f4_kill = True  # no daily-loss cap declared -> no kill switch to breach

    prediction = (definition or {}).get("tracking", {}).get("prediction")

    criteria: dict[str, bool | None] = {
        "F1_duration_180d": days >= MIN_DAYS if fills else None,
        "F2_round_trips_100": len(trips) >= MIN_ROUND_TRIPS if trips else None,
        "F3_net_positive_pf": (
            (total > 0 and (pf is None or pf >= MIN_PROFIT_FACTOR)) if trips else None
        ),
        "F4_kill_switch_clean": f4_kill,
        "F5_mc_p05_positive": (p05 > 0.0) if p05 is not None else None,
        "F6_integrity_no_gaps": f6_integrity,
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
        kill_switch_breach_days=kill_breaches,
        integrity_notes=integrity_notes,
        prediction=prediction,
    )
