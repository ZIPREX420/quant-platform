"""Status probe: verdicts and metrics per surface, without live services."""
import json
import os
import time

from quant_platform.data.context import MarketContext
from quant_platform.journal import DecisionJournal, MemoRecord
from quant_platform.monitoring.status import (
    check_audit_trail,
    check_cache,
    check_http_service,
    check_journal,
)


def ctx() -> MarketContext:
    return MarketContext(
        symbol="BTC-USD", as_of="2026-07-09", source="test", stale_days=0, last_close=1.0,
        returns_pct={"7d": 0.0, "30d": 0.0, "90d": 0.0, "365d": 0.0},
        volatility_annualized_pct={"30d": 1.0, "90d": 1.0},
        trend={"sma50": 1.0, "sma200": 1.0, "price_vs_sma50_pct": 0.0, "price_vs_sma200_pct": 0.0},
        range_365d={"high": 1.0, "low": 1.0, "drawdown_from_high_pct": 0.0, "recovery_from_low_pct": 0.0},
        avg_volume_30d=1.0, bars=400,
    )


def test_http_unreachable_is_unhealthy():
    check = check_http_service("svc", "http://127.0.0.1:1/nope", timeout_seconds=0.3)
    assert not check.healthy and "unreachable" in check.detail


def test_journal_missing_is_healthy_cold_start(tmp_path):
    check = check_journal(tmp_path / "none.jsonl")
    assert check.healthy and check.metrics["memos"] == 0


def test_journal_pending_backlog_flags(tmp_path):
    journal = DecisionJournal(tmp_path / "j.jsonl")
    for _ in range(3):
        journal.append_memo(MemoRecord(symbol="BTC-USD", context=ctx(), memo="m", model="t"))
    check = check_journal(tmp_path / "j.jsonl", max_pending=2)
    assert not check.healthy and "pending outcomes" in check.detail
    assert check.metrics == {"memos": 3, "pending": 3}


def test_audit_counts_rejections(tmp_path):
    p = tmp_path / "exec.jsonl"
    p.write_text(json.dumps({"approved": True}) + "\n" + json.dumps({"approved": False}) + "\n")
    check = check_audit_trail(p)
    assert check.healthy and check.metrics == {"records": 2, "rejections": 1}


def test_cache_staleness(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    old = time.time() - 100 * 3600
    os.utime(f, (old, old))
    assert not check_cache(tmp_path, max_age_hours=48).healthy
    os.utime(f, None)
    assert check_cache(tmp_path, max_age_hours=48).healthy


def test_cache_empty_is_healthy(tmp_path):
    assert check_cache(tmp_path / "missing").healthy
