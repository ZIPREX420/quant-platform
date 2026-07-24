"""Regression: funding-accrual rows must not be committed when a cycle refuses.

The accrual cursor (`last_funding_ts`) lives in paper-state.json, so the
funding-accruals.jsonl sidecar must be written only AFTER the state is saved.
Otherwise a refusal after accrual (e.g. a funding-feed failure on a later
candidate) strands rows on disk that the unsaved cursor re-accrues next cycle,
duplicating the forward-evidence funding record.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

from quant_platform.cycle import run_cycle
from quant_platform.data.binance_client import BinanceClientError, FundingEvent
from quant_platform.execution.state import OpenPosition, PaperState, StateStore
from tests.unit.test_cycle import T0, FakeClient, flat, paths, write_candidate
from tests.unit.test_cycle_funding_refusal import _write_funding_gated_candidate


class _AccrualClient(FakeClient):
    """Held-symbol funding returns one due event; another symbol's feed is down."""

    def __init__(self, ohlc, due_symbol="BTCUSDT", down_symbol=None):
        super().__init__(ohlc)
        self._due_symbol = due_symbol
        self._down_symbol = down_symbol

    def funding_rates(self, symbol, limit=100):
        if self._down_symbol is not None and symbol == self._down_symbol:
            raise BinanceClientError(f"funding feed unreachable for {symbol}")
        if symbol == self._due_symbol:
            return [FundingEvent(funding_time=T0 + timedelta(hours=2), rate=0.001)]
        return []


def _seed_open_btc_position(state_path, candidate_id, quantity=0.5):
    """Persist a fresh account already holding a long BTCUSDT position."""
    account = PaperState.fresh(10_000.0).restore_account()
    account.positions["BTCUSDT"] = quantity
    pos = OpenPosition(
        candidate_id=candidate_id, symbol="BTCUSDT", direction="long",
        quantity=quantity, entry_price=95.0, entry_ts=T0,
        stop_price=90.0, entry_fill_id="seed00000000",
    )
    StateStore(state_path).save(PaperState.from_account(account, (pos,), cycle_count=3))


def _accrual_rows(audit_path):
    p = audit_path.parent / "funding-accruals.jsonl"
    if not p.exists():
        return []
    return [json.loads(ln) for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def test_accrual_sidecar_written_on_successful_cycle(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    held = write_candidate(cands)  # plain BTCUSDT candidate, holds the seeded position
    _seed_open_btc_position(state_path, candidate_id=held)

    feed = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT")
    run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())

    rows = _accrual_rows(audit_path)
    assert len(rows) == 1 and rows[0]["symbol"] == "BTCUSDT"
    # cursor advanced in saved state -> a second run accrues nothing new (no dup)
    feed2 = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT")
    run_cycle(cands, state_path, audit_path, client=feed2, now=feed2.now_after_bars())
    assert len(_accrual_rows(audit_path)) == 1


def test_accrual_sidecar_not_written_when_cycle_refuses(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    held = write_candidate(cands)                            # holds the BTCUSDT position
    _write_funding_gated_candidate(cands, symbol="ETHUSDT")  # funding-gated, feed down
    _seed_open_btc_position(state_path, candidate_id=held)
    before = StateStore(state_path).load()

    # BTC funding has a due event (accrual produces a row); ETH funding feed is down
    feed = _AccrualClient(flat(95, 10), due_symbol="BTCUSDT", down_symbol="ETHUSDT")
    with pytest.raises(BinanceClientError):
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())

    # the refusal must leave NO funding record and NO state change
    assert _accrual_rows(audit_path) == []
    after = StateStore(state_path).load()
    assert after.cycle_count == before.cycle_count
    assert after.cash == before.cash
    assert after.open_positions[0].last_funding_ts == before.open_positions[0].last_funding_ts
