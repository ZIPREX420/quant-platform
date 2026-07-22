"""hour_utc indicator: UTC-hour series for time-of-day entry filters."""
from __future__ import annotations

from quant_platform.signals.rules import Bar, _series
from quant_platform.validation.backtest import run_backtest


def _bars(n=72, start_hour=0):
    out = []
    for i in range(n):
        day, hour = divmod(start_hour + i, 24)
        out.append(Bar(date=f"2026-01-{day+1:02d}T{hour:02d}:00",
                       open=100.0, high=101.0, low=99.0, close=100.0))
    return out


class TestHourSeries:
    def test_series_is_the_utc_open_hour(self):
        vals = _series("hour_utc", None, _bars(30, start_hour=5))
        assert vals[:4] == [5.0, 6.0, 7.0, 8.0]
        assert vals[19] == 0.0  # rolls over midnight

    def test_no_window_required(self):
        assert _series("hour_utc", None, _bars(3)) == [0.0, 1.0, 2.0]


class TestHourGatedBacktest:
    def test_entries_only_inside_the_hour_window(self):
        sig = {"kind": "declarative-rules",
               "entry_rules": [
                   {"indicator": "hour_utc", "operator": "greater_than", "operand": 16.5},
                   {"indicator": "hour_utc", "operator": "less_than", "operand": 19.5}],
               "exit_rules": [
                   {"indicator": "hour_utc", "operator": "greater_than", "operand": 21.5}]}
        trades = run_backtest(sig, _bars(72), stop_loss_pct=50.0)
        # Next-bar-open discipline: a rule true at bar H's close fills at
        # H+1's open. Window 17-19 true -> first fill at 18:00; exit rule
        # first true at 22:00 -> fill at 23:00.
        assert trades
        for t in trades:
            assert int(t.entry_date[11:13]) == 18
            assert int(t.exit_date[11:13]) == 23
