"""M12.2 vol-targeted sizing + M12.3 multi-symbol candidates."""
import json
from datetime import datetime, timedelta, timezone

from quant_platform.cycle import _vol_scale, run_cycle
from quant_platform.execution.session import ExecutionAudit
from quant_platform.execution.state import StateStore
from quant_platform.risk.engine import Side
from quant_platform.signals.rules import Bar
from quant_platform.validation.forward import round_trips_for
from tests.unit.test_cycle import FakeClient, flat, paths, write_candidate
from tests.unit.test_forward import fill_record


def bars_with_vol(step_pct: float, n: int = 40) -> list[Bar]:
    closes, price = [], 100.0
    for i in range(n):
        price *= (1 + step_pct / 100) if i % 2 == 0 else 1 / (1 + step_pct / 100)
        closes.append(price)
    return [Bar(date=f"2026-07-10T{i:02d}:00", open=c, high=c, low=c, close=c)
            for i, c in enumerate(closes)]


class TestVolScale:
    def test_no_target_full_scale(self):
        assert _vol_scale(bars_with_vol(2.0), "1h", None) == 1.0

    def test_high_vol_scales_down(self):
        # ~2%/bar hourly -> annualized ~187%; expected scale = 50/187 ~ 0.27
        scale = _vol_scale(bars_with_vol(2.0), "1h", 50.0)
        assert 0.2 < scale < 0.35

    def test_low_vol_capped_at_one(self):
        assert _vol_scale(bars_with_vol(0.001), "1h", 50.0) == 1.0  # never leverages up

    def test_insufficient_data_full_scale(self):
        assert _vol_scale(bars_with_vol(2.0)[:5], "1h", 50.0) == 1.0


class TestVolScaleInCycle:
    def test_entry_scaled_down_under_target(self, tmp_path):
        cands, state_path, audit_path = paths(tmp_path)
        write_candidate(cands)
        f = cands / "cycle-test-cand.json"
        definition = json.loads(f.read_text())
        definition["risk"]["vol_target_annual_pct"] = 20.0
        f.write_text(json.dumps(definition))
        ohlc = [(100 + (i % 2) * 8, 108, 99, 100 + (i % 2) * 8) for i in range(12)]
        ohlc += [(105, 105, 104, 105), (106, 106, 106, 106)]
        feed = FakeClient(ohlc)
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        records = ExecutionAudit(audit_path).records()
        assert records, "entry expected"
        # unscaled target would be 5% of 10k = 500; vol scaling must shrink it hard
        assert records[-1].requested_notional < 100.0


class TestMultiSymbol:
    def test_independent_positions_per_symbol(self, tmp_path):
        cands, state_path, audit_path = paths(tmp_path)
        write_candidate(cands)
        f = cands / "cycle-test-cand.json"
        definition = json.loads(f.read_text())
        definition["universe"]["symbols"] = ["BTCUSDT", "ETHUSDT"]
        f.write_text(json.dumps(definition))
        feed = FakeClient(flat(95, 8) + [(105, 105, 104, 105), (106, 106, 106, 106)])
        r = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        assert [x.action for x in r.results] == ["enter", "enter"]
        assert {x.symbol for x in r.results} == {"BTCUSDT", "ETHUSDT"}
        state = StateStore(state_path).load()
        assert len(state.open_positions) == 2
        assert set(state.positions) == {"BTCUSDT", "ETHUSDT"}

    def test_forward_trips_do_not_interleave_across_symbols(self):
        t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rec = [
            fill_record("c1", Side.BUY, 100.0, 1.0, t0).model_copy(
                update={"symbol": "BTCUSDT"}),
            fill_record("c1", Side.BUY, 10.0, 5.0, t0 + timedelta(hours=1)).model_copy(
                update={"symbol": "ETHUSDT"}),
            fill_record("c1", Side.SELL, 11.0, 5.0, t0 + timedelta(hours=2)).model_copy(
                update={"symbol": "ETHUSDT"}),
            fill_record("c1", Side.SELL, 110.0, 1.0, t0 + timedelta(hours=3)).model_copy(
                update={"symbol": "BTCUSDT"}),
        ]
        trips, open_pos = round_trips_for(rec, "c1")
        assert len(trips) == 2 and not open_pos
        assert {t.symbol for t in trips} == {"BTCUSDT", "ETHUSDT"}
        assert all(t.return_fraction > 0 for t in trips)
