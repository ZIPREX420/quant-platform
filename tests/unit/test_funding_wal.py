"""Funding-accrual durability (write-ahead ledger + reconcile-on-load).

Guards the crash window PR #2's after-save write left open: a failure between
the state save and the ledger append. The ledger is now the write-ahead log
(durably appended BEFORE save) and run_cycle reconciles any rows the last save
did not absorb - so the forward-evidence funding record can neither be lost nor
duplicated. Also covers the funding-ledger integrity probe.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.cycle import run_cycle
from quant_platform.execution.funding import reconcile_from_ledger
from quant_platform.execution.state import OpenPosition, StateStore
from quant_platform.monitoring.status import check_funding_ledger
from tests.unit.test_cycle import T0, flat, paths, write_candidate
from tests.unit.test_funding_accrual_atomicity import (
    _accrual_rows,
    _AccrualClient,
    _seed_open_btc_position,
)

UTC = timezone.utc
_T2 = "2026-07-10T02:00:00Z"  # T0 + 2h, the _AccrualClient's due funding event


def _write_ledger_row(audit_path, candidate_id, cash_delta=-0.0475):
    ledger = audit_path.parent / "funding-accruals.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(json.dumps({
        "ts": "2026-07-10T09:00:00Z", "candidate_id": candidate_id, "symbol": "BTCUSDT",
        "direction": "long", "funding_time": _T2, "rate": 0.001, "quantity": 0.5,
        "settle_price": 95.0, "cash_delta": cash_delta,
    }) + "\n", encoding="utf-8")


def test_reconcile_replays_unsaved_ledger_row_no_duplicate(tmp_path):
    """Crash between WAL append and save: row on disk, cursor unadvanced -> the
    next cycle replays it (cash/cursor/funding_net once) and does NOT re-accrue."""
    cands, state_path, audit_path = paths(tmp_path)
    held = write_candidate(cands)
    _seed_open_btc_position(state_path, candidate_id=held)   # cursor None, funding_net 0
    _write_ledger_row(audit_path, held)                      # simulate the pre-save crash

    feed = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT")  # feed still serves the event
    run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())

    assert len(_accrual_rows(audit_path)) == 1                # replayed, NOT duplicated
    pos = StateStore(state_path).load().open_positions[0]
    assert pos.last_funding_ts == datetime(2026, 7, 10, 2, 0, tzinfo=UTC)
    assert pos.funding_net == pytest.approx(-0.0475)
    assert StateStore(state_path).load().cash == pytest.approx(10_000.0 - 0.0475)


def test_happy_path_writes_ledger_and_holds_invariant(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    held = write_candidate(cands)
    _seed_open_btc_position(state_path, candidate_id=held)
    feed = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT")
    run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())

    rows = _accrual_rows(audit_path)
    assert len(rows) == 1
    net = StateStore(state_path).load().open_positions[0].funding_net
    assert sum(r["cash_delta"] for r in rows) == pytest.approx(net)   # ledger == state
    # re-run same bar: cursor advanced -> no new accrual, ledger stable
    feed2 = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT")
    run_cycle(cands, state_path, audit_path, client=feed2, now=feed2.now_after_bars())
    assert len(_accrual_rows(audit_path)) == 1


def test_reconcile_from_ledger_is_idempotent():
    pos = OpenPosition(candidate_id="c", symbol="BTCUSDT", direction="long", quantity=1.0,
                       entry_price=100.0, entry_ts=T0, stop_price=90.0, entry_fill_id="f" * 12)
    rows = [{"candidate_id": "c", "symbol": "BTCUSDT", "funding_time": _T2, "cash_delta": -0.2}]
    now = T0 + timedelta(hours=5)
    updated, delta = reconcile_from_ledger({("c", "BTCUSDT"): pos}, rows, now)
    assert delta == pytest.approx(-0.2)
    p2 = updated[("c", "BTCUSDT")]
    assert p2.funding_net == pytest.approx(-0.2)
    updated2, delta2 = reconcile_from_ledger({("c", "BTCUSDT"): p2}, rows, now)   # again
    assert not updated2 and delta2 == 0.0


def test_ledger_probe_healthy_after_normal_cycle(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    held = write_candidate(cands)
    _seed_open_btc_position(state_path, candidate_id=held)
    feed = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT")
    run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
    check = check_funding_ledger(state_path, audit_path.parent / "funding-accruals.jsonl")
    assert check.healthy


def test_ledger_probe_degraded_when_row_lost(tmp_path):
    state_path = tmp_path / "paper-state.json"
    state_path.write_text(json.dumps({
        "version": 2, "updated_at": "2026-07-10T00:00:00Z", "starting_cash": 10000.0,
        "cash": 9999.9525, "positions": {"BTCUSDT": 0.5},
        "open_positions": [{
            "candidate_id": "c", "symbol": "BTCUSDT", "direction": "long", "quantity": 0.5,
            "entry_price": 95.0, "entry_ts": "2026-07-10T00:00:00Z", "stop_price": 90.0,
            "entry_fill_id": "f" * 12, "last_funding_ts": _T2, "funding_net": -0.0475,
        }],
        "cycle_count": 5,
    }), encoding="utf-8")
    # ledger missing the -0.0475 row the state already absorbed
    check = check_funding_ledger(state_path, tmp_path / "funding-accruals.jsonl")
    assert not check.healthy and "diverges" in check.detail
