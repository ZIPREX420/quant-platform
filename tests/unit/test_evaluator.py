"""Live signal evaluator: same rule machinery as the backtester, cycle timing."""
import pytest

from quant_platform.signals.evaluator import attach_funding, evaluate_candidate
from quant_platform.signals.rules import Bar, RuleError


def bar(date, close, low=None, high=None):
    return Bar(date=date, open=close, high=high or close, low=low or close, close=close)


SIGNAL = {
    "kind": "declarative-rules",
    "parameters": {},
    "entry_rules": [{"indicator": "close", "operator": "greater_than", "operand": 100}],
    "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": 90}],
}


def test_enter_when_flat_and_entry_true():
    bars = [bar("2026-07-10T10:00", 99), bar("2026-07-10T11:00", 105)]
    d = evaluate_candidate(SIGNAL, bars, in_position=False)
    assert d.action == "enter" and d.reason == "entry-rules" and d.close == 105


def test_hold_when_flat_and_entry_false():
    bars = [bar("2026-07-10T10:00", 99), bar("2026-07-10T11:00", 98)]
    assert evaluate_candidate(SIGNAL, bars, in_position=False).action == "hold"


def test_no_reentry_signal_while_in_position():
    bars = [bar("2026-07-10T10:00", 105), bar("2026-07-10T11:00", 106)]
    d = evaluate_candidate(SIGNAL, bars, in_position=True, stop_price=50.0)
    assert d.action == "hold"


def test_exit_rules_fire():
    bars = [bar("2026-07-10T10:00", 95), bar("2026-07-10T11:00", 89)]
    d = evaluate_candidate(SIGNAL, bars, in_position=True, stop_price=50.0)
    assert d.action == "exit" and d.reason == "exit-rules"


def test_stop_breach_preempts_exit_rules():
    # bar low breached the stop during the bar; the rule exit is only actionable
    # now - the stop fired first chronologically (backtester chronology)
    bars = [bar("2026-07-10T10:00", 95), bar("2026-07-10T11:00", 89, low=84)]
    d = evaluate_candidate(SIGNAL, bars, in_position=True, stop_price=85.0)
    assert d.action == "exit" and d.reason.startswith("stop-breach")


def test_in_position_requires_stop():
    bars = [bar("2026-07-10T10:00", 95), bar("2026-07-10T11:00", 96)]
    with pytest.raises(RuleError, match="stop_price"):
        evaluate_candidate(SIGNAL, bars, in_position=True)


def test_too_few_bars_refused():
    with pytest.raises(RuleError, match="at least 2"):
        evaluate_candidate(SIGNAL, [bar("2026-07-10T10:00", 95)], in_position=False)


def test_crossing_semantics_match_backtester():
    signal = {
        "kind": "declarative-rules", "parameters": {},
        "entry_rules": [{"indicator": "close", "operator": "crosses_above",
                         "operand": {"indicator": "sma", "window": 3}}],
        "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": 0.001}],
    }
    closes = [100, 100, 100, 95, 96, 103]  # dips below sma3 then crosses above
    bars = [bar(f"2026-07-10T{h:02d}:00", c) for h, c in enumerate(closes)]
    assert evaluate_candidate(signal, bars, in_position=False).action == "enter"
    # one bar earlier there is no crossing yet
    assert evaluate_candidate(signal, bars[:-1], in_position=False).action == "hold"


class TestAttachFunding:
    def test_events_attach_and_future_events_dropped(self):
        bars = [bar("2026-07-10T08:00", 100), bar("2026-07-10T09:00", 101)]
        out = attach_funding(bars, [("2026-07-10T08:00", 0.0001), ("2026-07-10T16:00", 0.0002)])
        assert out[0].funding == 0.0001 and out[1].funding is None

    def test_misaligned_events_refused(self):
        bars = [bar("2026-07-10T08:00", 100), bar("2026-07-10T09:00", 101)]
        with pytest.raises(RuleError, match="misaligned"):
            attach_funding(bars, [("2026-07-10T08:30", 0.0001)])
