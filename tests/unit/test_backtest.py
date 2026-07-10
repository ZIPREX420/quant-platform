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

    def test_rule_exit_at_open_preempts_later_intrabar_stop(self):
        # chronology: the rule exit fills AT the open (92), before price falls
        # to the stop (91.8) later in the bar - so this is NOT a stop-out.
        sig = dict(self.SIG, exit_rules=[{"indicator": "close", "operator": "less_than",
                                          "operand": 95}])
        bars = self.make_bars([
            (100, 101, 99, 101),
            (102, 103, 101, 102),  # entry @102, stop 91.8
            (100, 101, 94, 94),    # close 94 -> rule exit signal
            (92, 93, 90, 91),      # exit fills at open 92; intra-bar 90 is moot
        ])
        trades = run_backtest(sig, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert not trades[0].stopped_out and trades[0].exit_price == 92

    def test_gap_through_stop_on_rule_exit_bar_is_a_stop(self):
        sig = dict(self.SIG, exit_rules=[{"indicator": "close", "operator": "less_than",
                                          "operand": 95}])
        bars = self.make_bars([
            (100, 101, 99, 101),
            (102, 103, 101, 102),  # entry @102, stop 91.8
            (100, 101, 94, 94),    # rule exit signal
            (88, 89, 85, 86),      # opens BELOW the stop -> flagged stopped, fill at open
        ])
        trades = run_backtest(sig, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert trades[0].stopped_out and trades[0].exit_price == 88

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


class TestFunding:
    """Funding accrual and funding-series rules (schema v1.1)."""

    SIG = {
        "kind": "declarative-rules", "parameters": {},
        "entry_rules": [{"indicator": "close", "operator": "greater_than", "operand": 100}],
        "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": 90}],
    }

    def bars_with_funding(self):
        # entry signal at bar0 close (101>100); fill bar1 open; two funding
        # events while held (+0.01 and -0.005); exit signal bar3 close (85<90); fill bar4
        return [
            Bar("2026-03-01T00:00", 100, 102, 99, 101),
            Bar("2026-03-01T08:00", 100, 102, 99, 100, funding=0.01),
            Bar("2026-03-01T16:00", 100, 102, 99, 100, funding=-0.005),
            Bar("2026-03-02T00:00", 100, 102, 84, 85),
            Bar("2026-03-02T08:00", 100, 102, 99, 100),
        ]

    def test_long_pays_positive_receives_negative_funding(self):
        trades = run_backtest(self.SIG, self.bars_with_funding(), 0.0, 0.0)
        assert len(trades) == 1
        # entry fills at bar1 open: bar1's own event (+0.01) is NOT paid
        # (position born at that instant); bar2's event (-0.005) IS received.
        # price roundtrip 100 -> 100 => return = +0.005 exactly.
        assert trades[0].return_fraction == pytest.approx(0.005, abs=1e-9)

    def test_rule_exit_at_open_skips_that_bars_event(self):
        bars = [
            Bar("2026-03-01T00:00", 100, 102, 99, 101),                 # entry signal
            Bar("2026-03-01T08:00", 100, 102, 99, 85),                  # fill; exit signal (85<90)
            Bar("2026-03-01T16:00", 100, 102, 99, 100, funding=0.02),   # exit fill AT open: no pay
        ]
        trades = run_backtest(self.SIG, bars, 0.0, 0.0)
        assert trades[0].return_fraction == pytest.approx(0.0, abs=1e-9)

    def test_intrabar_stop_pays_that_bars_open_event(self):
        sig = dict(self.SIG, exit_rules=[{"indicator": "close", "operator": "less_than",
                                          "operand": 1}])  # never rule-exits
        bars = [
            Bar("2026-03-01T00:00", 100, 102, 99, 101),                        # entry signal
            Bar("2026-03-01T08:00", 100, 102, 99, 100),                        # fill @100, stop 90
            Bar("2026-03-01T16:00", 95, 96, 88, 92, funding=0.01),             # survives open, pays, then stopped @90
        ]
        trades = run_backtest(sig, bars, 0.0, 0.0, stop_loss_pct=10.0)
        assert trades[0].stopped_out
        expected = (90.0 / 100.0) * (1 - 0.01) - 1.0
        assert trades[0].return_fraction == pytest.approx(expected, abs=1e-9)

    def test_funding_outside_position_ignored(self):
        bars = [
            Bar("2026-03-01T00:00", 100, 102, 99, 50, funding=0.05),   # not in position
            Bar("2026-03-01T08:00", 100, 102, 99, 101),                # entry signal
            Bar("2026-03-01T16:00", 100, 102, 99, 85),                 # fill here; exit signal
            Bar("2026-03-02T00:00", 100, 102, 99, 100),                # exit fill
        ]
        trades = run_backtest(self.SIG, bars, 0.0, 0.0)
        assert trades[0].return_fraction == pytest.approx(0.0, abs=1e-9)

    def test_funding_series_rule_triggers(self):
        sig = {
            "kind": "declarative-rules", "parameters": {},
            "entry_rules": [{"indicator": "sma", "window": 2, "series": "funding",
                             "operator": "less_than", "operand": -0.0001}],
            "exit_rules": [{"indicator": "close", "series": "funding",
                            "operator": "greater_than", "operand": 0.0}],
        }
        bars = [
            Bar("d1", 100, 101, 99, 100, funding=-0.001),
            Bar("d2", 100, 101, 99, 100, funding=-0.002),  # sma2=-0.0015 < -0.0001 -> entry
            Bar("d3", 100, 101, 99, 100),                  # entry fill
            Bar("d4", 100, 101, 99, 100, funding=0.001),   # last funding > 0 -> exit signal
            Bar("d5", 100, 101, 99, 100),                  # exit fill
        ]
        trades = run_backtest(sig, bars, 0.0, 0.0)
        assert len(trades) == 1
        assert trades[0].entry_date == "d3" and trades[0].exit_date == "d5"

    def test_unsupported_funding_indicator_rejected(self):
        sig = dict(self.SIG, entry_rules=[{"indicator": "rsi", "window": 5, "series": "funding",
                                           "operator": "less_than", "operand": 30}])
        with pytest.raises(RuleError, match="not supported on the funding series"):
            run_backtest(sig, self.bars_with_funding(), 0.0, 0.0)

    def test_merge_funding_alignment_guard(self, tmp_path):
        from quant_platform.validation.backtest import merge_funding
        f = tmp_path / "funding.csv"
        f.write_text("funding_time_utc,funding_rate\n2026-03-01T08:00,0.0001\n"
                     "2099-01-01T00:00,0.0001\n", encoding="utf-8")
        with pytest.raises(ValueError, match="misaligned"):
            merge_funding(self.bars_with_funding(), f)
        f.write_text("funding_time_utc,funding_rate\n2026-03-01T08:00,0.0001\n", encoding="utf-8")
        merged = merge_funding(self.bars_with_funding(), f)
        assert merged[1].funding == 0.0001
