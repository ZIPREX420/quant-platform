"""Audit-vs-state reconcile: divergence detection + event-sourced rebuild."""
from datetime import datetime, timezone

import pytest

from quant_platform.execution.reconcile import book_divergences, rebuild_state
from quant_platform.execution.session import AuditRecord
from quant_platform.execution.state import PaperState
from quant_platform.risk.engine import Side

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _fill(cand, side, price, qty, ts, symbol="SOLUSDT"):
    notional = price * qty
    return AuditRecord(
        mode="paper", tier="candidate", strategy_id=cand, symbol=symbol,
        side=side, requested_notional=notional, approved=True, approved_notional=notional,
        checks=[], ts=ts,
        fill={"fill_id": "f" * 12, "ts": ts.isoformat(), "strategy_id": cand, "symbol": symbol,
              "side": side.value, "requested_notional": notional, "fill_price": price,
              "quantity": qty, "fee": round(notional * 0.001, 8), "slippage_cost": 0.0},
        equity_after=10_000.0,
    )


def test_book_divergence_detected_and_clean():
    recs = [_fill("c1", Side.BUY, 100.0, 1.0, T0),
            _fill("c1", Side.SELL, 100.0, 0.4, T0.replace(hour=8))]  # audit net +0.6
    assert book_divergences({"SOLUSDT": 0.6}, recs) == []   # agrees -> clean
    assert book_divergences({"SOLUSDT": 0.5}, recs)         # 0.1 off -> flagged
    assert book_divergences({}, recs)                       # book empty -> flagged


def test_rebuild_heals_lost_fill_that_flipped_direction():
    # opened long 1.0, then sold 1.1 -> genuine flip to short 0.1. A lost save
    # could leave the book long/flat; the rebuild must land net short 0.1.
    recs = [_fill("c1", Side.BUY, 100.0, 1.0, T0),
            _fill("c1", Side.SELL, 100.0, 1.1, T0.replace(hour=8))]
    rebuilt = rebuild_state(PaperState.fresh(10_000.0), recs, [], {"c1": 50.0}, T0.replace(hour=9))
    assert rebuilt.positions["SOLUSDT"] == pytest.approx(-0.1, abs=1e-9)
    assert len(rebuilt.open_positions) == 1
    op = rebuilt.open_positions[0]
    assert op.direction == "short" and op.quantity == pytest.approx(0.1, abs=1e-9)
    assert op.entry_price == 100.0  # entry is the crossing (flip) fill


def test_rebuild_flat_candidate_has_no_open_position():
    recs = [_fill("c1", Side.BUY, 100.0, 1.0, T0),
            _fill("c1", Side.SELL, 100.0, 1.0, T0.replace(hour=8))]
    rebuilt = rebuild_state(PaperState.fresh(10_000.0), recs, [], {"c1": 50.0}, T0.replace(hour=9))
    assert rebuilt.open_positions == ()
    assert "SOLUSDT" not in rebuilt.positions


def test_rebuild_applies_funding_cash_deltas():
    recs = [_fill("c1", Side.BUY, 100.0, 1.0, T0)]  # cash -= 100 + 0.1 fee
    rebuilt = rebuild_state(
        PaperState.fresh(10_000.0), recs, [{"cash_delta": -1.25}], {"c1": 50.0}, T0.replace(hour=9)
    )
    assert rebuilt.cash == pytest.approx(10_000.0 - 100.1 - 1.25, abs=1e-6)


def test_rebuild_two_candidates_one_symbol_net_consistent():
    # short book on SOL held by two candidates; rebuild must satisfy from_account's
    # per-symbol net check (the PR #5 invariant) without raising.
    recs = [
        _fill("flow", Side.SELL, 100.0, 5.0, T0),               # flow short 5
        _fill("carry", Side.BUY, 100.0, 2.0, T0.replace(hour=1)),  # carry long 2
    ]
    rebuilt = rebuild_state(PaperState.fresh(10_000.0), recs,
                            [], {"flow": 50.0, "carry": 50.0}, T0.replace(hour=9))
    assert rebuilt.positions["SOLUSDT"] == pytest.approx(-3.0, abs=1e-9)
    dirs = {p.candidate_id: p.direction for p in rebuilt.open_positions}
    assert dirs == {"flow": "short", "carry": "long"}
