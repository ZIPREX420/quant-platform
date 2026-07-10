"""Dashboard rendering: honest empty states, correct data, escaped output."""
import json
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.cli_dashboard import parse_ledger
from quant_platform.dashboard import render_dashboard
from quant_platform.execution.session import AuditRecord
from quant_platform.execution.state import OpenPosition, PaperState
from quant_platform.journal import DecisionJournal, MemoRecord, OutcomeRecord
from quant_platform.risk.engine import Side
from quant_platform.strategies.candidates import load_candidate

REPO = Path(__file__).resolve().parents[2]
NOW = datetime(2026, 7, 10, 12, 0, tzinfo=timezone.utc)


def make_audit(approved=True, equity=10_000.0, tier="candidate"):
    return AuditRecord(
        mode="paper", tier=tier, strategy_id="cand-x", symbol="BTCUSDT", side=Side.BUY,
        requested_notional=500.0, approved=approved, approved_notional=500.0 if approved else 0.0,
        checks=[{"name": "price_jump", "passed": approved, "detail": "d"}],
        fill={"fill_id": "f" * 12, "fill_price": 100.05, "quantity": 5.0} if approved else None,
        equity_after=equity,
    )


def test_empty_workspace_renders_empty_states(tmp_path):
    html = render_dashboard(None, [], [], DecisionJournal(tmp_path / "j.jsonl"), [], NOW)
    assert "run m9-cycle.bat for the first time" in html
    assert "No execution decisions yet" in html
    assert "No candidates registered" in html
    assert "No desk memos yet" in html
    assert "No signed validation reports" in html
    assert "Live trading does not exist in this codebase" in html


def test_full_dashboard_content(tmp_path):
    state = PaperState(
        updated_at=NOW, starting_cash=10_000.0, cash=9_400.0,
        positions={"BTCUSDT": 5.0},
        open_positions=(OpenPosition(
            candidate_id="cand-x", symbol="BTCUSDT", quantity=5.0, entry_price=100.05,
            entry_ts=NOW, stop_price=95.0, entry_fill_id="f" * 12,
        ),),
        cycle_count=42, last_equity=10_150.0,
    )
    records = [make_audit(equity=10_000.0), make_audit(equity=10_150.0),
               make_audit(approved=False, equity=10_150.0)]
    candidate = load_candidate(REPO / "config/candidates/sol-funding-carry-tracker.json")
    journal = DecisionJournal(tmp_path / "j.jsonl")
    memo_id = journal.append_memo(MemoRecord(
        symbol="BTC-USD",
        context=json.loads(Path(REPO / "tests/unit/fixtures/context.json").read_text())
        if (REPO / "tests/unit/fixtures/context.json").exists() else _ctx(),
        memo="Confidence: MEDIUM", model="test", confidence="MEDIUM",
    ))
    journal.append_outcome(OutcomeRecord(
        memo_record_id=memo_id, horizon_days=7, realized_return_pct=-3.21,
    ))
    ledger = [{"report": "sol-funding-carry-v0.1.0.md", "strategy": "s", "result": "fail",
               "date": "Seppe Willemsens, 2026-07-10", "sha256": "ab" * 32}]

    html = render_dashboard(state, records, [candidate], journal, ledger, NOW)
    assert "10150.00" in html and "42" in html            # KPIs
    assert "cand-x" in html and "95.0000" in html         # position + stop
    assert "rejected" in html and "price_jump" in html    # failed check surfaced
    assert "sol-funding-carry-tracker" in html and "H1" in html
    assert "LOSES money net of protocol costs" in html    # prediction verbatim
    assert "-3.21%" in html and "MEDIUM" in html          # outcome loop
    assert "sol-funding-carry-v0.1.0.md" in html          # ledger
    assert "<script" not in html                          # static artifact, no JS


def _ctx():
    return {
        "symbol": "BTC-USD", "as_of": "2026-07-01", "source": "test", "stale_days": 0,
        "last_close": 100.0, "returns_pct": {}, "volatility_annualized_pct": {},
        "trend": {}, "range_365d": {"low": 1, "high": 2}, "avg_volume_30d": None, "bars": 60,
    }


def test_html_escaping(tmp_path):
    candidate_path = tmp_path / "evil.json"
    definition = json.loads(
        (REPO / "config/candidates/sol-funding-carry-tracker.json").read_text()
    )
    definition["id"] = "evil-candidate"
    definition["tracking"]["prediction"] = (
        "<script>alert(1)</script> a prediction long enough to satisfy the schema."
    )
    candidate_path.write_text(json.dumps(definition))
    candidate = load_candidate(candidate_path)
    html = render_dashboard(None, [], [candidate], DecisionJournal(tmp_path / "j.jsonl"), [], NOW)
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_parse_ledger(tmp_path):
    ledger_md = tmp_path / "signed-reports.md"
    ledger_md.write_text(
        "# ledger\n\n| Report | Strategy | Result | Signed off | sha256 |\n|---|---|---|---|---|\n"
        "| a-v0.1.0.md | a v0.1.0 | fail | S, 2026-07-09 | `" + "ab" * 32 + "` |\n",
        encoding="utf-8",
    )
    rows = parse_ledger(ledger_md)
    assert len(rows) == 1 and rows[0]["report"] == "a-v0.1.0.md" and rows[0]["result"] == "fail"
    assert parse_ledger(tmp_path / "missing.md") == []
