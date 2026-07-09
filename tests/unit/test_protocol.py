"""Protocol helpers: window attribution and splits."""
import pytest

from quant_platform.validation.backtest import Bar
from quant_platform.validation.protocol import is_oos_split, walk_forward, with_windows

SIG = {
    "kind": "declarative-rules", "parameters": {"fast": 2, "slow": 4},
    "entry_rules": [{"indicator": "sma", "window": 2, "operator": "crosses_above",
                     "operand": {"indicator": "sma", "window": 4}}],
    "exit_rules": [{"indicator": "sma", "window": 2, "operator": "crosses_below",
                    "operand": {"indicator": "sma", "window": 4}}],
}


def bars(n=400):
    import math
    out = []
    for i in range(n):
        c = 100 + 20 * math.sin(i / 15)  # oscillating -> plenty of crosses
        out.append(Bar(date=f"2025-{1 + i // 28:02d}-{1 + i % 28:02d}",
                       open=c, high=c + 1, low=c - 1, close=c))
    return out


def test_split_fractions():
    isb, oos = is_oos_split(bars(100), 0.3)
    assert len(isb) == 70 and len(oos) == 30
    with pytest.raises(ValueError):
        is_oos_split(bars(100), 0.9)


def test_walk_forward_windows_cover_and_attribute():
    results = walk_forward(SIG, bars(400), n_windows=5, warmup_bars=50)
    assert len(results) == 5
    assert results[0].test_start > bars(400)[49].date  # warmup respected
    # windows are sequential and non-overlapping
    for a, b in zip(results, results[1:]):
        assert a.test_end < b.test_start
    # every attributed trade entered inside its window
    for r in results:
        for t in r.trades:
            assert r.test_start <= t.entry_date <= r.test_end
    assert sum(len(r.trades) for r in results) > 0


def test_with_windows_rewrites_both_sides():
    s = with_windows(SIG, 40, 160)
    assert s["entry_rules"][0]["window"] == 40
    assert s["exit_rules"][0]["operand"]["window"] == 160
    assert SIG["entry_rules"][0]["window"] == 2  # original untouched
