"""record_outcomes: horizon gating, realized-return math, fetch-failure resilience."""
from datetime import date, datetime, timedelta, timezone

import httpx

from quant_platform.cli_outcomes import record_outcomes
from quant_platform.data.context import MarketContext
from quant_platform.data.openbb_client import OpenBBClient
from quant_platform.journal import DecisionJournal, MemoRecord


def ctx(as_of: str, last_close: float) -> MarketContext:
    return MarketContext(
        symbol="BTCUSD", as_of=as_of, source="openbb/yfinance", stale_days=0,
        last_close=last_close, returns_pct={"7d": 0.0, "30d": 0.0, "90d": 0.0, "365d": 0.0},
        volatility_annualized_pct={"30d": 30.0, "90d": 30.0},
        trend={"sma50": 1.0, "sma200": 1.0, "price_vs_sma50_pct": 0.0, "price_vs_sma200_pct": 0.0},
        range_365d={"high": 2.0, "low": 0.5, "drawdown_from_high_pct": 0.0,
                    "recovery_from_low_pct": 0.0},
        avg_volume_30d=1.0, bars=400,
    )


def memo(journal: DecisionJournal, as_of: str, last_close: float) -> str:
    record = MemoRecord(symbol="BTCUSD", context=ctx(as_of, last_close),
                        memo="**MEDIUM**", model="test/stub")
    return journal.append_memo(record)


def price_payload(as_of: str, closes: list[float]):
    base = date.fromisoformat(as_of)
    rows = [{"date": (base + timedelta(days=i)).isoformat(),
             "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1}
            for i, c in enumerate(closes)]
    return {"results": rows}


NOW = datetime(2026, 7, 20, tzinfo=timezone.utc)


def test_realized_return_recorded(tmp_path):
    journal = DecisionJournal(tmp_path / "j.jsonl")
    rid = memo(journal, "2026-07-07", 100.0)

    def handler(request):
        return httpx.Response(200, json=price_payload("2026-07-07", [100.0, 105.0, 110.0]))

    with OpenBBClient(transport=httpx.MockTransport(handler)) as client:
        recorded = record_outcomes(journal, client, horizon_days=7, now=NOW)
    assert len(recorded) == 1
    assert recorded[0].memo_record_id == rid
    assert recorded[0].realized_return_pct == 10.0  # 100 -> 110
    assert journal.pending() == []


def test_horizon_gating(tmp_path):
    journal = DecisionJournal(tmp_path / "j.jsonl")
    memo(journal, "2026-07-18", 100.0)  # only 2 days old at NOW

    with OpenBBClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=price_payload("2026-07-18", [100.0])))) as client:
        recorded = record_outcomes(journal, client, horizon_days=7, now=NOW)
    assert recorded == [] and len(journal.pending()) == 1


def test_fetch_failure_keeps_memo_pending(tmp_path):
    journal = DecisionJournal(tmp_path / "j.jsonl")
    memo(journal, "2026-07-07", 100.0)

    with OpenBBClient(transport=httpx.MockTransport(
            lambda r: httpx.Response(500, text="down"))) as client:
        recorded = record_outcomes(journal, client, horizon_days=7, now=NOW)
    assert recorded == [] and len(journal.pending()) == 1  # retried next run
