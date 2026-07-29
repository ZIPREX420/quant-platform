"""Forward-record analyzer: round-trip reconstruction + pre-registered thresholds."""
from datetime import datetime, timedelta, timezone

import pytest

from quant_platform.execution.session import AuditRecord
from quant_platform.risk.engine import Side
from quant_platform.validation.forward import (
    assess,
    round_trips_for,
)

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def fill_record(candidate, side, price, qty, ts, fee=None):
    notional = price * qty
    fee = fee if fee is not None else round(notional * 0.001, 8)
    return AuditRecord(
        mode="paper", tier="candidate", strategy_id=candidate, symbol="SOLUSDT",
        side=side, requested_notional=notional, approved=True, approved_notional=notional,
        checks=[], ts=ts,
        fill={"fill_id": "f" * 12, "ts": ts.isoformat(), "strategy_id": candidate,
              "symbol": "SOLUSDT", "side": side.value, "requested_notional": notional,
              "fill_price": price, "quantity": qty, "fee": fee, "slippage_cost": 0.0},
        equity_after=10_000.0,
    )


def trip(candidate, day, entry, exit_, qty=1.0):
    return [
        fill_record(candidate, Side.BUY, entry, qty, T0 + timedelta(days=day)),
        fill_record(candidate, Side.SELL, exit_, qty, T0 + timedelta(days=day, hours=8)),
    ]


def test_round_trip_reconstruction_with_costs():
    records = trip("c1", 0, 100.0, 110.0)
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and not open_pos
    # cost 100+0.1 fee, proceeds 110-0.11 fee -> ~ +9.78%
    assert trips[0].return_fraction == pytest.approx(
        (110 - 0.11) / (100 + 0.1) - 1.0
    )


def test_partial_closes_aggregate_into_one_trip():
    records = [
        fill_record("c1", Side.BUY, 100.0, 2.0, T0),
        fill_record("c1", Side.SELL, 105.0, 1.2, T0 + timedelta(hours=8)),
        fill_record("c1", Side.SELL, 108.0, 0.8, T0 + timedelta(hours=16)),
    ]
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and not open_pos
    assert trips[0].proceeds == pytest.approx(
        1.2 * 105 - 0.126 + 0.8 * 108 - 0.0864
    )


def test_open_position_not_counted_as_trip():
    records = trip("c1", 0, 100.0, 110.0) + [
        fill_record("c1", Side.BUY, 100.0, 1.0, T0 + timedelta(days=1))
    ]
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and open_pos


def test_other_candidates_and_validated_tier_excluded():
    records = trip("c1", 0, 100.0, 110.0) + trip("c2", 0, 100.0, 50.0)
    validated = fill_record("c1", Side.BUY, 100.0, 1.0, T0 + timedelta(days=2))
    records.append(validated.model_copy(update={"tier": "validated"}))
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and not open_pos  # c2 and validated-tier ignored


def test_sell_from_flat_opens_short_trip():
    # M12: a SELL from flat is a short open, closed by a BUY
    records = [
        fill_record("c1", Side.SELL, 100.0, 1.0, T0),
        fill_record("c1", Side.BUY, 90.0, 1.0, T0 + timedelta(hours=8)),
    ]
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and not open_pos and trips[0].direction == "short"
    # proceeds = 100 - 0.1 fee; cost = 90 + 0.09 fee; return on entry notional
    expected = ((100 - 0.1) - (90 + 0.09)) / (100 - 0.1)
    assert trips[0].return_fraction == pytest.approx(expected)


def test_sell_through_flat_flips_long_to_short():
    # A SELL larger than the open long is a legitimate flip (funding-carry
    # candidates do this): it closes the long trip at flat and opens a short with
    # the remainder. It must reconstruct, not raise.
    records = [
        fill_record("c1", Side.BUY, 100.0, 1.0, T0),
        fill_record("c1", Side.SELL, 100.0, 2.0, T0 + timedelta(hours=8)),
    ]
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and trips[0].direction == "long" and open_pos


def test_full_flip_long_to_short_to_flat_is_two_trips():
    records = [
        fill_record("c1", Side.BUY, 100.0, 1.0, T0),                        # open long
        fill_record("c1", Side.SELL, 110.0, 2.0, T0 + timedelta(hours=8)),  # close long +10%, open short 1.0
        fill_record("c1", Side.BUY, 105.0, 1.0, T0 + timedelta(hours=16)),  # close short (sold 110, bought 105)
    ]
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 2 and not open_pos
    assert trips[0].direction == "long" and trips[1].direction == "short"


def test_dust_overshoot_closes_cleanly_without_spurious_short():
    # selling a hair more than held (accumulated fill rounding) closes flat -
    # no spurious micro-short trip, no open position.
    records = [
        fill_record("c1", Side.BUY, 100.0, 1.0, T0),
        fill_record("c1", Side.SELL, 100.0, 1.0 + 1e-9, T0 + timedelta(hours=8)),
    ]
    trips, open_pos = round_trips_for(records, "c1")
    assert len(trips) == 1 and not open_pos and trips[0].direction == "long"


class TestAssessment:
    def test_no_fills_all_criteria_unmeasurable(self):
        a = assess([], "c1")
        assert a.round_trips == 0 and not a.qualifies()
        assert all(v is None for v in a.criteria.values())

    def test_qualifying_record(self):
        # 120 profitable trips over 200 days, slight noise for realistic MC
        records = []
        for i in range(120):
            exit_price = 103.0 if i % 5 else 96.0  # mostly wins, some losses
            records += trip("c1", day=i * 200 // 120, entry=100.0, exit_=exit_price)
        eq = [{"ts": (T0 + timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "cycle": d + 1, "equity": 10_000.0 + d} for d in range(201)]
        defn = {"risk": {"max_daily_loss_pct": 5}, "tracking": {"prediction": "wins net"}}
        a = assess(records, "c1", equity_history=eq, definition=defn)
        assert a.round_trips == 120 and a.evidence_days >= 180
        assert a.criteria["F1_duration_180d"] and a.criteria["F2_round_trips_100"]
        assert a.criteria["F3_net_positive_pf"] and a.criteria["F5_mc_p05_positive"]
        assert a.criteria["F4_kill_switch_clean"] and a.criteria["F6_integrity_no_gaps"]
        assert a.kill_switch_breach_days == 0 and a.qualifies()
        assert "QUALIFIES" in a.summary()

    def test_losing_record_fails_f3(self):
        records = []
        for i in range(110):
            records += trip("c1", day=i * 2, entry=100.0, exit_=99.0)
        a = assess(records, "c1")
        assert a.criteria["F3_net_positive_pf"] is False
        assert not a.qualifies()
        assert "does not (yet) qualify" in a.summary()

    def test_short_record_not_yet_measurable_not_failed(self):
        a = assess(trip("c1", 0, 100.0, 110.0), "c1")
        assert a.criteria["F1_duration_180d"] is False  # 0 days measured
        assert a.criteria["F2_round_trips_100"] is False
        assert a.criteria["F5_mc_p05_positive"] is None  # <10 trips: no MC yet
        assert not a.qualifies()


class TestF4F6F7:
    """kill switch (F4), window integrity (F6), prediction surfacing (F7)."""

    def _profitable_120(self):
        records = []
        for i in range(120):
            exit_price = 103.0 if i % 5 else 96.0
            records += trip("c1", day=i * 200 // 120, entry=100.0, exit_=exit_price)
        return records

    def _gapless_equity(self, n=201):
        return [{"ts": (T0 + timedelta(days=d)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                 "cycle": d + 1, "equity": 10_000.0 + d} for d in range(n)]

    def test_no_equity_history_leaves_f4_f6_unmeasurable(self):
        a = assess(self._profitable_120(), "c1")
        assert a.criteria["F4_kill_switch_clean"] is None
        assert a.criteria["F6_integrity_no_gaps"] is None
        assert not a.qualifies()

    def test_kill_switch_breach_fails_f4(self):
        eq = self._gapless_equity()
        eq.insert(11, {"ts": (T0 + timedelta(days=10, hours=12)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "cycle": 0, "equity": 10_010.0 * 0.90})  # -10% intraday, same day
        for k, r in enumerate(eq):
            r["cycle"] = k + 1  # renumber contiguous so F6 stays clean, isolating F4
        a = assess(self._profitable_120(), "c1", equity_history=eq,
                   definition={"risk": {"max_daily_loss_pct": 5}})
        assert a.criteria["F4_kill_switch_clean"] is False
        assert a.kill_switch_breach_days >= 1 and not a.qualifies()

    def test_cycle_gap_fails_f6_integrity(self):
        eq = self._gapless_equity()
        del eq[50]  # a missing cycle mark inside the window
        a = assess(self._profitable_120(), "c1", equity_history=eq,
                   definition={"risk": {"max_daily_loss_pct": 5}})
        assert a.criteria["F6_integrity_no_gaps"] is False
        assert any("missing cycle" in n for n in a.integrity_notes)
        assert not a.qualifies()

    def test_prediction_surfaced_for_f7(self):
        defn = {"risk": {"max_daily_loss_pct": 5},
                "tracking": {"prediction": "loses per symbol (falsification tracker)"}}
        a = assess(self._profitable_120(), "c1",
                   equity_history=self._gapless_equity(), definition=defn)
        assert a.prediction == "loses per symbol (falsification tracker)"
        assert "F7 prediction review" in a.summary() and "falsification tracker" in a.summary()
