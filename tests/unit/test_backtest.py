"""Reference backtester: crossover detection, next-bar-open fills, cost math."""
import pytest

from quant_platform.validation.backtest import Bar, RuleError, run_backtest

SMA_CROSS = {
    "kind": "declarative-rules",
    "parameters": {"fast": 2, "slow": 4},
    "entry_rules": [{"indicator": "sma", "window": 2, "operator": "crosses_above",
                     "operand": {"indicator": "sma", "window": 4}}],
    "exit_rules": [{"indicator": "sma", "window": 2, "operator": "crosses_below",
                    "operand": {"indicator": "sma", "window": 4}}],
}


def bars_from_closes(closes, start_day=1):
    return [
        Bar(date=f"2026-01-{start_day + i:02d}", open=c, high=c * 1.01, low=c * 0.99, close=c)
        for i, c in enumerate(closes)
    ]


def test_single_round_trip_with_known_prices():
    # closes: fall then rise then fall -> one golden cross, one death cross
    closes = [100, 90, 80, 70, 60, 80, 100, 120, 110, 90, 70, 60]
    bars = bars_from_closes(closes)
    trades = run_backtest(SMA_CROSS, bars, fee_rate=0.0, slippage_rate=0.0)
    assert len(trades) == 1
    t = trades[0]
    assert not t.forced_exit
    # verify fill = next bar OPEN after the signal close (opens == closes here)
    entry_idx = [b.date for b in bars].index(t.entry_date)
    assert bars[entry_idx].open == t.entry_price
    assert t.return_fraction == pytest.approx(t.exit_price / t.entry_price - 1.0)


def test_costs_reduce_return_exactly():
    closes = [100, 90, 80, 70, 60, 80, 100, 120, 110, 90, 70, 60]
    bars = bars_from_closes(closes)
    gross = run_backtest(SMA_CROSS, bars, fee_rate=0.0, slippage_rate=0.0)[0]
    net = run_backtest(SMA_CROSS, bars, fee_rate=0.001, slippage_rate=0.0005)[0]
    expected_ratio = ((1 - 0.0005) / (1 + 0.0005)) * (1 - 0.001) ** 2
    assert (1 + net.return_fraction) == pytest.approx((1 + gross.return_fraction) * expected_ratio, rel=1e-9)


def test_open_position_force_closed_at_end():
    closes = [100, 90, 80, 70, 60, 80, 100, 120, 140, 160]  # cross up, never down
    trades = run_backtest(SMA_CROSS, bars_from_closes(closes), fee_rate=0.0, slippage_rate=0.0)
    assert len(trades) == 1 and trades[0].forced_exit


def test_no_signals_no_trades():
    closes = list(range(100, 116))  # monotone rise: fast starts above slow, no cross
    assert run_backtest(SMA_CROSS, bars_from_closes(closes), 0.0, 0.0) == []


def test_threshold_rule_and_numeric_operand():
    sig = {
        "kind": "declarative-rules", "parameters": {},
        "entry_rules": [{"indicator": "close", "operator": "greater_than", "operand": 105}],
        "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": 95}],
    }
    closes = [100, 106, 110, 94, 90, 100]
    trades = run_backtest(sig, bars_from_closes(closes), 0.0, 0.0)
    assert len(trades) == 1
    assert trades[0].entry_price == 110  # signal at close 106 (day2) -> fill open day3
    assert trades[0].exit_price == 90    # signal at close 94 (day4) -> fill open day5


def test_unknown_kind_rejected():
    with pytest.raises(RuleError, match="unsupported signal kind"):
        run_backtest({"kind": "python"}, bars_from_closes([1, 2, 3]), 0.0, 0.0)


def test_rsi_and_high_n_series_produce_trades_without_error():
    sig = {
        "kind": "declarative-rules", "parameters": {},
        "entry_rules": [{"indicator": "rsi", "window": 3, "operator": "less_than", "operand": 30}],
        "exit_rules": [{"indicator": "rsi", "window": 3, "operator": "greater_than", "operand": 70}],
    }
    closes = [100, 95, 90, 85, 80, 85, 92, 99, 106, 110]
    trades = run_backtest(sig, bars_from_closes(closes), 0.0, 0.0)
    assert len(trades) >= 1
