"""Rebuild paper state from the durable ledgers (audit-vs-state integrity).

``executions.jsonl`` (append-only fills) and ``funding-accruals.jsonl``
(append-only funding WAL) are the DURABLE record of what the paper engine did.
``paper-state.json`` is a materialised view of them. A lost end-of-cycle save
(a race between overlapping cycles, or a crash between a fill's audit append and
the save) leaves the view stale: a fill lives in the audit but never reached the
state. This module makes the divergence detectable and repairable.

Two entry points:
  - ``book_divergences`` : cheap read-only check used by the cycle on load - the
    account book (net units per symbol) must equal the audit's net per symbol.
  - ``rebuild_state``    : event-sourced reconstruction (cash, positions, and
    per-candidate open positions) from the ledgers, used by the one-shot repair
    CLI (``quant-reconcile-state``). Never wired into the trading path: repairing
    a book is an explicit, reviewed operation, never a silent auto-heal.
"""
from __future__ import annotations

from datetime import datetime

from quant_platform.execution.paper import PaperAccount, PaperFill
from quant_platform.execution.session import AuditRecord
from quant_platform.execution.state import OpenPosition, PaperState
from quant_platform.risk.engine import Side

DUST = 1e-6  # |net units| below this is treated as flat (accumulated fill rounding)


def _signed(fill: dict) -> float:
    q = float(fill.get("quantity") or 0.0)
    side = fill.get("side")
    is_buy = (side.value if isinstance(side, Side) else str(side)).lower().endswith("buy")
    return q if is_buy else -q


def audit_net_by_symbol(records: list[AuditRecord]) -> dict[str, float]:
    """Net signed units per symbol implied by every filled audit record."""
    net: dict[str, float] = {}
    for r in records:
        if r.fill is None:
            continue
        net[r.symbol] = net.get(r.symbol, 0.0) + _signed(r.fill)
    return {s: q for s, q in net.items() if abs(q) > DUST}


def book_divergences(
    positions: dict[str, float], records: list[AuditRecord], tol: float = DUST
) -> list[str]:
    """Symbols where the account book disagrees with the audit net. Empty = OK."""
    audit = audit_net_by_symbol(records)
    out: list[str] = []
    for symbol in sorted(set(positions) | set(audit)):
        held = positions.get(symbol, 0.0)
        expected = audit.get(symbol, 0.0)
        if abs(held - expected) > tol:
            out.append(f"{symbol}: book {held:.10f} vs audit {expected:.10f} (Δ {held - expected:+.10f})")
    return out


def rebuild_state(
    prior: PaperState,
    records: list[AuditRecord],
    funding_rows: list[dict],
    stop_loss_pct_by_candidate: dict[str, float],
    now: datetime,
) -> PaperState:
    """Reconstruct the full state from the durable ledgers.

    cash = starting_cash + every execution fill's cashflow + every funding delta.
    positions = replay of every execution fill. Per-candidate open positions are
    walked flip-aware: a fill that crosses zero closes the current direction and
    opens the opposite one, so a long that is sold through flat becomes a short
    with its entry at the crossing fill. Funding cursor is set to ``now`` and
    ``funding_net`` reset to 0 for each rebuilt position (past funding is already
    in cash) so the next cycle cannot double-accrue it.
    """
    account = PaperAccount(starting_cash=prior.starting_cash)
    for r in records:
        if r.fill is not None:
            account.apply(PaperFill.model_validate(r.fill))
    for row in funding_rows:
        account.cash += float(row.get("cash_delta", 0.0))
    # Snap near-flat holdings to zero at the same dust threshold the open-position
    # reconstruction uses, so the book and the per-candidate positions agree under
    # PaperState.from_account's net check.
    account.positions = {s: q for s, q in account.positions.items() if abs(q) > DUST}

    # per-(candidate, symbol) flip-aware open-position reconstruction
    by_key: dict[tuple[str, str], list[AuditRecord]] = {}
    for r in records:
        if r.fill is not None:
            by_key.setdefault((r.strategy_id, r.symbol), []).append(r)

    open_positions: list[OpenPosition] = []
    for (candidate_id, symbol), recs in by_key.items():
        recs.sort(key=lambda r: r.ts)
        pos = 0.0
        entry: AuditRecord | None = None
        for r in recs:
            step = _signed(r.fill)
            new = pos + step
            if abs(pos) <= DUST or (pos > 0) != (new > 0):
                entry = r  # opened from flat, or crossed zero into a new direction
            pos = new
        if abs(pos) <= DUST or entry is None:
            continue  # net flat -> no open position
        direction = "long" if pos > 0 else "short"
        stop_frac = stop_loss_pct_by_candidate.get(candidate_id, 50.0) / 100.0
        entry_price = float(entry.fill["fill_price"])
        stop = entry_price * ((1.0 - stop_frac) if direction == "long" else (1.0 + stop_frac))
        open_positions.append(OpenPosition(
            candidate_id=candidate_id, symbol=symbol, direction=direction,
            quantity=round(abs(pos), 10), entry_price=entry_price, entry_ts=entry.ts,
            stop_price=round(stop, 8), entry_fill_id=entry.fill["fill_id"],
            last_funding_ts=now, funding_net=0.0,
        ))

    return PaperState.from_account(
        account,
        tuple(sorted(open_positions, key=lambda p: (p.candidate_id, p.symbol))),
        cycle_count=prior.cycle_count,
        day_anchor_date=prior.day_anchor_date,
        day_anchor_equity=prior.day_anchor_equity,
        last_equity=prior.last_equity,
    )
