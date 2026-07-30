"""DecisionJournal: append-only memo/outcome records, pending queue."""

import pytest

from quant_platform.data.context import MarketContext
from quant_platform.journal import DecisionJournal, MemoRecord, OutcomeRecord, extract_confidence


def ctx() -> MarketContext:
    return MarketContext(
        symbol="BTC-USD", as_of="2026-07-07", source="openbb/yfinance", stale_days=0,
        last_close=64000.0, returns_pct={"7d": 1.0, "30d": 2.0, "90d": -3.0, "365d": -40.0},
        volatility_annualized_pct={"30d": 34.0, "90d": 36.0},
        trend={"sma50": 66000.0, "sma200": 74000.0,
               "price_vs_sma50_pct": -3.0, "price_vs_sma200_pct": -13.5},
        range_365d={"high": 124000.0, "low": 58000.0,
                    "drawdown_from_high_pct": -48.0, "recovery_from_low_pct": 10.0},
        avg_volume_30d=1000.0, bars=400,
    )


def memo(**kw) -> MemoRecord:
    return MemoRecord(symbol="BTC-USD", context=ctx(), memo="## Confidence\n**MEDIUM** - test",
                      model="test/stub", **kw)


def test_append_and_read_roundtrip(tmp_path):
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    rid = journal.append_memo(memo())
    records = journal.memos()
    assert len(records) == 1 and records[0].record_id == rid
    assert records[0].context.last_close == 64000.0
    assert records[0].created_at.tzinfo is not None


def test_outcome_lifecycle_and_pending(tmp_path):
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    rid1 = journal.append_memo(memo())
    rid2 = journal.append_memo(memo())
    assert {m.record_id for m in journal.pending()} == {rid1, rid2}
    journal.append_outcome(OutcomeRecord(memo_record_id=rid1, horizon_days=7,
                                         realized_return_pct=2.5, notes="tracked"))
    assert [m.record_id for m in journal.pending()] == [rid2]
    outcomes = journal.outcomes_for(rid1)
    assert len(outcomes) == 1 and outcomes[0].realized_return_pct == 2.5


def test_outcome_for_unknown_memo_rejected(tmp_path):
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    with pytest.raises(KeyError):
        journal.append_outcome(OutcomeRecord(memo_record_id="nope", horizon_days=7,
                                             realized_return_pct=0.0))


def test_extract_confidence():
    # canonical machine tag (the editor's required final line), tolerant of markdown/case
    assert extract_confidence("...\n\nCONFIDENCE: HIGH") == "HIGH"
    assert extract_confidence("CONFIDENCE: **Medium**") == "MEDIUM"
    assert extract_confidence("blah\nconfidence: low\n") == "LOW"
    # the LAST tag wins - it is the mandated closing line
    assert extract_confidence("early confidence high hedge ... \nCONFIDENCE: LOW") == "LOW"
    # older memo format: a **LEVEL** marker in a Confidence section
    assert extract_confidence("## Confidence\n\n**MEDIUM** - because") == "MEDIUM"
    # genuinely absent -> None (the capture-gap the cli warns on)
    assert extract_confidence("no rating here") is None
    assert extract_confidence("Both cases overstate confidence from thin data") is None


def test_editor_prompt_mandates_confidence_tag():
    """Regression guard: the machine-readable rating instruction must stay in the
    editor prompt, or memos silently lose the field H4 scores (the sonnet-5 drift)."""
    from quant_platform.agents.prompts import EDITOR

    assert "CONFIDENCE: LOW" in EDITOR
    assert "CONFIDENCE: MEDIUM" in EDITOR
    assert "CONFIDENCE: HIGH" in EDITOR
