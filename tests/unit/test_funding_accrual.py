"""Funding accrual on held paper positions (ADR-0007 cost tier) + the
paper-cycle staleness alarm (heartbeat SLO for the forward-evidence clock)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from quant_platform.data.binance_client import FundingEvent
from quant_platform.dashboard import render_dashboard
from quant_platform.execution.funding import accrue_open_positions
from quant_platform.execution.state import OpenPosition, PaperState, StateStore
from quant_platform.journal import DecisionJournal
from quant_platform.monitoring.status import check_paper_state
from quant_platform.signals.rules import Bar

UTC = timezone.utc
T0 = datetime(2026, 7, 20, 0, 0, tzinfo=UTC)


def _pos(direction="long", quantity=2.0, entry=T0, last_funding=None, funding_net=0.0):
    return OpenPosition(
        candidate_id="cand", symbol="BTCUSDT", direction=direction,
        quantity=quantity, entry_price=100.0, entry_ts=entry,
        stop_price=50.0 if direction == "long" else 200.0,
        entry_fill_id="f" * 12,
        last_funding_ts=last_funding, funding_net=funding_net,
    )


def _event(hours_after_t0: int, rate: float) -> FundingEvent:
    return FundingEvent(funding_time=T0 + timedelta(hours=hours_after_t0), rate=rate)


def _bars(hours: int, close: float = 100.0) -> list[Bar]:
    return [
        Bar(date=(T0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M"),
            open=close, high=close, low=close, close=close)
        for h in range(hours)
    ]


def _market(hours: int = 24, close: float = 100.0):
    return {"BTCUSDT": (_bars(hours, close), "1h")}


class TestAccrual:
    def test_long_pays_positive_rate(self):
        positions = {("cand", "BTCUSDT"): _pos("long", quantity=2.0)}
        events = {"BTCUSDT": [_event(8, 0.001)]}
        now = T0 + timedelta(hours=10)
        updated, rows, delta = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, now)
        # long pays: -0.001 * 2 * 100 = -0.2
        assert delta == -0.2
        assert rows[0]["cash_delta"] == -0.2
        pos = updated[("cand", "BTCUSDT")]
        assert pos.funding_net == -0.2
        assert pos.last_funding_ts == T0 + timedelta(hours=8)

    def test_short_receives_positive_rate(self):
        positions = {("cand", "BTCUSDT"): _pos("short", quantity=2.0)}
        events = {"BTCUSDT": [_event(8, 0.001)]}
        _, _, delta = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=10))
        assert delta == 0.2

    def test_long_receives_negative_rate(self):
        positions = {("cand", "BTCUSDT"): _pos("long", quantity=2.0)}
        events = {"BTCUSDT": [_event(8, -0.0005)]}
        _, _, delta = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=10))
        assert delta == 0.1

    def test_cursor_prevents_double_accrual(self):
        positions = {("cand", "BTCUSDT"): _pos(last_funding=T0 + timedelta(hours=8))}
        events = {"BTCUSDT": [_event(8, 0.001)]}  # already applied
        updated, rows, delta = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=10))
        assert not updated and not rows and delta == 0.0

    def test_multi_event_catch_up_accumulates(self):
        positions = {("cand", "BTCUSDT"): _pos("long", quantity=1.0)}
        events = {"BTCUSDT": [_event(8, 0.001), _event(16, 0.002)]}
        updated, rows, delta = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=20))
        assert len(rows) == 2
        assert delta == -0.3  # 0.001*100 + 0.002*100, rounded to 8dp by the accruer
        assert updated[("cand", "BTCUSDT")].last_funding_ts == T0 + timedelta(hours=16)

    def test_events_before_entry_are_ignored(self):
        positions = {("cand", "BTCUSDT"): _pos(entry=T0 + timedelta(hours=9))}
        events = {"BTCUSDT": [_event(8, 0.01)]}  # settled before entry
        updated, rows, _ = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=12))
        assert not updated and not rows

    def test_future_events_are_ignored(self):
        positions = {("cand", "BTCUSDT"): _pos()}
        events = {"BTCUSDT": [_event(16, 0.01)]}
        _, rows, _ = accrue_open_positions(
            positions, events, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=10))
        assert not rows

    def test_settle_price_uses_matching_bar_close(self):
        positions = {("cand", "BTCUSDT"): _pos("long", quantity=1.0)}
        events = {"BTCUSDT": [_event(8, 0.001)]}
        _, rows, _ = accrue_open_positions(
            positions, events, _market(close=250.0), {"BTCUSDT": 999.0},
            T0 + timedelta(hours=10))
        assert rows[0]["settle_price"] == 250.0

    def test_settle_price_falls_back_to_mark(self):
        positions = {("cand", "BTCUSDT"): _pos("long", quantity=1.0)}
        events = {"BTCUSDT": [_event(8, 0.001)]}
        _, rows, _ = accrue_open_positions(
            positions, events, {}, {"BTCUSDT": 123.0}, T0 + timedelta(hours=10))
        assert rows[0]["settle_price"] == 123.0

    def test_no_events_is_a_no_op(self):
        positions = {("cand", "BTCUSDT"): _pos()}
        updated, rows, delta = accrue_open_positions(
            positions, {}, _market(), {"BTCUSDT": 100.0}, T0 + timedelta(hours=10))
        assert not updated and not rows and delta == 0.0


class TestStateV2Migration:
    def test_v1_state_file_upgrades_on_load(self, tmp_path):
        v1 = {
            "version": 1,
            "updated_at": "2026-07-20T00:00:00Z",
            "starting_cash": 10000.0,
            "cash": 9500.0,
            "positions": {"BTCUSDT": 1.0},
            "open_positions": [{
                "candidate_id": "cand", "symbol": "BTCUSDT", "direction": "long",
                "quantity": 1.0, "entry_price": 100.0,
                "entry_ts": "2026-07-20T00:00:00Z", "stop_price": 50.0,
                "entry_fill_id": "abcabcabcabc",
            }],
            "cycle_count": 69,
        }
        path = tmp_path / "paper-state.json"
        path.write_text(json.dumps(v1), encoding="utf-8")
        state = StateStore(path).load()
        assert state.version == 2
        pos = state.open_positions[0]
        assert pos.last_funding_ts is None
        assert pos.funding_net == 0.0

    def test_v2_round_trips_funding_fields(self, tmp_path):
        pos = _pos(last_funding=T0 + timedelta(hours=8), funding_net=-0.2)
        account = PaperState.fresh(1000.0).restore_account()
        account.cash = 900.0
        account.positions = {"BTCUSDT": pos.quantity}
        state = PaperState.from_account(account, (pos,), cycle_count=1)
        store = StateStore(tmp_path / "s.json")
        store.save(state)
        loaded = store.load()
        assert loaded.open_positions[0].funding_net == -0.2
        assert loaded.open_positions[0].last_funding_ts == T0 + timedelta(hours=8)


class TestPaperCycleStaleness:
    def _write_state(self, path: Path, age_hours: float, cycles: int = 5) -> None:
        updated = datetime.now(UTC) - timedelta(hours=age_hours)
        path.write_text(json.dumps({
            "updated_at": updated.isoformat().replace("+00:00", "Z"),
            "cycle_count": cycles,
        }), encoding="utf-8")

    def test_missing_file_is_healthy(self, tmp_path):
        check = check_paper_state(tmp_path / "nope.json")
        assert check.healthy

    def test_fresh_state_is_healthy(self, tmp_path):
        path = tmp_path / "paper-state.json"
        self._write_state(path, age_hours=1.0)
        check = check_paper_state(path, max_age_hours=3.0)
        assert check.healthy
        assert check.metrics["cycles"] == 5

    def test_stale_state_is_degraded(self, tmp_path):
        path = tmp_path / "paper-state.json"
        self._write_state(path, age_hours=7.5)
        check = check_paper_state(path, max_age_hours=3.0)
        assert not check.healthy
        assert "NOT ticking" in check.detail

    def test_corrupt_state_is_degraded(self, tmp_path):
        path = tmp_path / "paper-state.json"
        path.write_text("{not json", encoding="utf-8")
        assert not check_paper_state(path).healthy


class TestDashboardStaleBanner:
    def _render(self, age_hours: float, tmp_path: Path) -> str:
        state = PaperState.fresh(1000.0)
        state = state.model_copy(
            update={"updated_at": datetime.now(UTC) - timedelta(hours=age_hours)})
        journal = DecisionJournal(tmp_path / "journal.jsonl")
        return render_dashboard(state, [], [], journal, [],
                                generated_at=datetime.now(UTC))

    def test_stale_state_shows_banner(self, tmp_path):
        assert "CYCLE STALE" in self._render(8.0, tmp_path)

    def test_fresh_state_has_no_banner(self, tmp_path):
        assert "CYCLE STALE" not in self._render(0.5, tmp_path)
