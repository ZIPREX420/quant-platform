"""M12 short-side semantics: every rule the exact mirror of the long path."""
import json
from pathlib import Path

import pytest

from quant_platform.signals.evaluator import evaluate_candidate
from quant_platform.signals.rules import Bar
from quant_platform.validation.backtest import run_backtest

REPO = Path(__file__).resolve().parents[2]


def bar(date, o, h=None, lo=None, c=None, funding=None):
    c = c if c is not None else o
    return Bar(date=date, open=o, high=h or max(o, c), low=lo or min(o, c),
               close=c, funding=funding)


def signal(direction, entry_above=100, exit_below=0.001):
    return {
        "kind": "declarative-rules", "parameters": {}, "direction": direction,
        "entry_rules": [{"indicator": "close", "operator": "greater_than", "operand": entry_above}],
        "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": exit_below}],
    }


class TestShortBacktest:
    def test_short_profits_when_price_falls(self):
        bars = [bar("d1", 100), bar("d2", 100, c=105),   # entry signal at d2 close
                bar("d3", 105), bar("d4", 105, c=0.0001),  # exit signal at d4 close
                bar("d5", 90)]                              # exit fill at d5 open
        trades = run_backtest(signal("short"), bars, fee_rate=0.0, slippage_rate=0.0)
        assert len(trades) == 1
        t = trades[0]
        assert t.entry_price == 105 and t.exit_price == 90
        assert t.return_fraction == pytest.approx(1 - 90 / 105)  # +14.3%

    def test_long_short_symmetry_zero_costs(self):
        # long on a rising path == short on the mirrored falling path
        rising = [bar("d1", 100), bar("d2", 100, c=105), bar("d3", 105),
                  bar("d4", 105, c=0.0001), bar("d5", 120)]
        falling = [bar("d1", 100), bar("d2", 100, c=105), bar("d3", 105),
                   bar("d4", 105, c=0.0001), bar("d5", 105 * 105 / 120)]
        lt = run_backtest(signal("long"), rising, fee_rate=0.0, slippage_rate=0.0)
        st = run_backtest(signal("short"), falling, fee_rate=0.0, slippage_rate=0.0)
        # long gains 120/105-1 = 14.29%; short gains 1 - (105*105/120)/105 = 1-105/120 = 12.5%
        assert lt[0].return_fraction == pytest.approx(120 / 105 - 1)
        assert st[0].return_fraction == pytest.approx(1 - 105 / 120)

    def test_short_stop_sits_above_and_triggers_on_high(self):
        bars = [bar("d1", 100), bar("d2", 100, c=105), bar("d3", 100),
                bar("d4", 100, h=112)]  # entry fills d3 open @100; stop=105; d4 high 112 >= 105
        trades = run_backtest(signal("short"), bars, fee_rate=0.0, slippage_rate=0.0,
                              stop_loss_pct=5.0)
        assert len(trades) == 1 and trades[0].stopped_out
        assert trades[0].exit_price == pytest.approx(105.0)  # filled AT the stop
        assert trades[0].return_fraction == pytest.approx(1 - 105 / 100)  # -5%

    def test_short_gap_through_stop_fills_at_open(self):
        bars = [bar("d1", 100), bar("d2", 100, c=105), bar("d3", 100),
                bar("d4", 115, h=118)]  # gaps OPEN above the 105 stop -> fill at 115 (worse)
        trades = run_backtest(signal("short"), bars, fee_rate=0.0, slippage_rate=0.0,
                              stop_loss_pct=5.0)
        assert trades[0].stopped_out and trades[0].exit_price == pytest.approx(115.0)

    def test_short_receives_positive_funding(self):
        bars = [bar("d1", 100), bar("d2", 100, c=105), bar("d3", 100),
                bar("d4", 100, funding=0.001), bar("d5", 100, c=0.0001), bar("d6", 100)]
        with_f = run_backtest(signal("short"), bars, fee_rate=0.0, slippage_rate=0.0)
        no_f = run_backtest(
            signal("short"),
            [b if b.funding is None else bar(b.date, b.open) for b in bars],
            fee_rate=0.0, slippage_rate=0.0,
        )
        # flat price, +0.1% funding event -> short RECEIVES it (long would pay)
        assert with_f[0].return_fraction == pytest.approx(no_f[0].return_fraction + 0.001)

    def test_short_slippage_against_trader_both_legs(self):
        bars = [bar("d1", 100), bar("d2", 100, c=105), bar("d3", 100),
                bar("d4", 100, c=0.0001), bar("d5", 100)]
        trades = run_backtest(signal("short"), bars, fee_rate=0.0, slippage_rate=0.001)
        t = trades[0]
        assert t.entry_price == pytest.approx(100 * 0.999)   # sell fills lower
        assert t.exit_price == pytest.approx(100 * 1.001)    # buy fills higher
        assert t.return_fraction < 0                         # flat price -> slippage-only loss

    def test_unknown_direction_refused(self):
        import pytest as _pytest
        from quant_platform.signals.rules import RuleError
        bad = signal("long")
        bad["direction"] = "sideways"
        with _pytest.raises(RuleError, match="unsupported direction"):
            run_backtest(bad, [bar("d1", 100)] * 3)


class TestShortEvaluator:
    BARS = [Bar(date="2026-07-10T10:00", open=100, high=101, low=99, close=100),
            Bar(date="2026-07-10T11:00", open=100, high=108, low=100, close=101)]

    def test_short_stop_breach_on_high(self):
        d = evaluate_candidate(signal("short"), self.BARS, in_position=True, stop_price=107.0)
        assert d.action == "exit" and "bar high 108" in d.reason

    def test_short_stop_not_hit_by_low(self):
        d = evaluate_candidate(signal("short"), self.BARS, in_position=True, stop_price=110.0)
        assert d.action == "hold"


class TestShortCycleEndToEnd:
    def test_short_lifecycle_via_cycle(self, tmp_path):
        from tests.unit.test_cycle import FakeClient, flat, paths, write_candidate
        from quant_platform.cycle import run_cycle
        from quant_platform.execution.session import ExecutionAudit
        from quant_platform.execution.state import StateStore

        cands, state_path, audit_path = paths(tmp_path)
        write_candidate(cands)
        # flip the test candidate to short
        f = cands / "cycle-test-cand.json"
        definition = json.loads(f.read_text())
        definition["signal"]["direction"] = "short"
        f.write_text(json.dumps(definition))

        # cycle 1: entry (close 105 > 100) -> SELL to open at live 106
        feed = FakeClient(flat(95, 8) + [(105, 105, 104, 105), (106, 106, 106, 106)])
        r1 = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        assert [x.action for x in r1.results] == ["enter"]
        state = StateStore(state_path).load()
        pos = state.open_positions[0]
        assert pos.direction == "short"
        assert state.positions["BTCUSDT"] < 0                    # account is short
        assert pos.stop_price == pytest.approx(pos.entry_price * 1.05)  # stop ABOVE
        assert pos.entry_price == pytest.approx(106 * 0.9995)    # sell slipped DOWN

        # cycle 2: bar high breaches the stop -> BUY to close
        breach = pos.stop_price + 1
        feed = FakeClient(flat(95, 7) + [(105, 105, 104, 105), (106, breach, 105, 106),
                                         (112, 112, 112, 112)])
        r2 = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        assert [x.action for x in r2.results] == ["exit"]
        assert r2.results[0].reason.startswith("stop-breach")
        final = StateStore(state_path).load()
        assert final.open_positions == () and final.positions == {}
        assert final.cash < 10_000.0  # stopped short lost money (price rose)
        records = ExecutionAudit(audit_path).records()
        assert [rec.side.value for rec in records] == ["sell", "buy"]

    def test_short_forward_round_trip_from_cycle_audit(self, tmp_path):
        # reuse the audit produced above via a fresh run, then measure it
        self.test_short_lifecycle_via_cycle(tmp_path)
        from quant_platform.execution.session import ExecutionAudit
        from quant_platform.validation.forward import round_trips_for
        records = ExecutionAudit(tmp_path / "executions.jsonl").records()
        trips, open_pos = round_trips_for(records, "cycle-test-cand")
        assert len(trips) == 1 and not open_pos
        assert trips[0].direction == "short" and trips[0].return_fraction < 0
