"""Per-cycle equity curve: the equity-history sidecar and its dashboard use."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from quant_platform.cli_dashboard import read_equity_history
from quant_platform.dashboard import render_dashboard
from quant_platform.execution.state import PaperState
from quant_platform.journal import DecisionJournal


def _render(tmp_path, history):
    state = PaperState.fresh(10_000.0)
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    return render_dashboard(
        state, [], [], journal, [],
        generated_at=datetime.now(timezone.utc),
        equity_history=history,
    )


class TestReadEquityHistory:
    def test_missing_file_is_empty(self, tmp_path):
        assert read_equity_history(tmp_path / "nope.jsonl") == []

    def test_reads_rows_and_skips_garbage(self, tmp_path):
        path = tmp_path / "equity-history.jsonl"
        path.write_text(
            json.dumps({"ts": "t1", "cycle": 1, "equity": 10000.0}) + "\n"
            + "{broken\n"
            + "\n"
            + json.dumps({"ts": "t2", "cycle": 2, "equity": 10006.5}) + "\n",
            encoding="utf-8",
        )
        rows = read_equity_history(path)
        assert [r["equity"] for r in rows] == [10000.0, 10006.5]


class TestDashboardEquityCurve:
    def test_history_renders_per_cycle_curve(self, tmp_path):
        history = [{"ts": f"t{i}", "cycle": i, "equity": 10_000.0 + i} for i in range(5)]
        html = _render(tmp_path, history)
        assert "Equity (per cycle, 5 marks)" in html
        assert "<svg" in html  # the curve actually drew

    def test_single_mark_still_shows_placeholder(self, tmp_path):
        html = _render(tmp_path, [{"ts": "t0", "cycle": 1, "equity": 10_000.0}])
        assert "Equity curve appears after" in html

    def test_no_history_falls_back_to_fill_points(self, tmp_path):
        html = _render(tmp_path, None)
        assert "<h2>Equity</h2>" in html  # legacy label, fills-based path

    def test_non_numeric_equity_rows_are_skipped(self, tmp_path):
        history = [
            {"ts": "t0", "cycle": 1, "equity": 10_000.0},
            {"ts": "t1", "cycle": 2, "equity": "oops"},
            {"ts": "t2", "cycle": 3, "equity": 10_001.0},
        ]
        html = _render(tmp_path, history)
        assert "Equity (per cycle, 2 marks)" in html
