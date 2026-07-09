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


class TestStopLoss:
    """Stop from the risk block: intra-bar trigger, gap handling, precedence."""

    SIG = {
        "kind": "declarative-rules", "parameters": {},
        "entry_rules": [{"indicator": "close", "operator": "greater_than", "operand": 100}],
        "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": 1}],  # never
    }

    def make_bars(self, specs):
        # specs: list of (open, high, low, close)
        return [Bar(date=f"2026-02-{i + 1:02d}", open=o, high=h, low=lo, close=c)
                for i, (o, h, lo, c) in enumerate(specs)]

    def test_intrabar_stop_fills_at_stop_level(self):
        bars = self.make_bars([
            (100, 101, 99, 101),   # entry signal at close
            (102, 103, 101, 102),  # entry fill at open 102 -> stop at 91.8 (10%)
            (102, 104, 100, 103),  # holds
            (101, 102, 90, 95),    # low 90 touches 91.8 -> stopped at stop level
        ])
        trades = run_backtest(self.SIG, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert len(trades) == 1 and trades[0].stopped_out
        assert trades[0].exit_price == pytest.approx(102 * 0.9)
        assert trades[0].return_fraction == pytest.approx(-0.10)

    def test_gap_through_fills_at_open_worse_than_stop(self):
        bars = self.make_bars([
            (100, 101, 99, 101),
            (102, 103, 101, 102),   # entry at 102, stop 91.8
            (80, 85, 78, 82),       # gaps open at 80 < stop -> fill at 80
        ])
        trades = run_backtest(self.SIG, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert trades[0].stopped_out and trades[0].exit_price == 80
        assert trades[0].return_fraction == pytest.approx(80 / 102 - 1)

    def test_stop_takes_precedence_over_rule_exit(self):
        sig = dict(self.SIG, exit_rules=[{"indicator": "close", "operator": "less_than",
                                          "operand": 95}])
        bars = self.make_bars([
            (100, 101, 99, 101),
            (102, 103, 101, 102),  # entry
            (100, 101, 94, 94),    # close 94 -> rule exit signal AND next bar hits stop
            (92, 93, 90, 91),      # stop (91.8) checked first: open 92 > stop, low 90 <= stop
        ])
        trades = run_backtest(sig, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert trades[0].stopped_out  # not a rule exit

    def test_no_stop_behaviour_unchanged(self):
        bars = self.make_bars([
            (100, 101, 99, 101),
            (102, 103, 101, 102),
            (101, 102, 60, 70),    # deep intra-bar dip, no stop configured
            (70, 75, 65, 72),
        ])
        trades = run_backtest(self.SIG, bars, 0.0, 0.0, stop_loss_pct=None)
        assert len(trades) == 1 and trades[0].forced_exit and not trades[0].stopped_out

    def test_stop_range_validated(self):
        with pytest.raises(ValueError, match="out of range"):
            run_backtest(self.SIG, self.make_bars([(1, 1, 1, 1)] * 4), 0.0, 0.0, stop_loss_pct=150)

    def test_reentry_after_stop_uses_new_stop_level(self):
        bars = self.make_bars([
            (100, 101, 99, 101),   # entry signal
            (102, 103, 101, 102),  # entry @102, stop 91.8
            (91, 92, 90, 101),     # open 91 < stop -> gap fill @91; close 101 -> re-entry signal
            (200, 201, 199, 200),  # re-entry @200, stop 180
            (185, 186, 179, 181),  # low 179 <= 180 -> stopped at 180
        ])
        trades = run_backtest(self.SIG, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert len(trades) == 2
        assert trades[0].exit_price == 91 and trades[1].exit_price == pytest.approx(180.0)
