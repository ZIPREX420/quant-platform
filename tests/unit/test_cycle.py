"""Multi-cycle simulation: the M9 engine room proven end-to-end on a fake feed."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from quant_platform.cycle import CycleError, run_cycle
from quant_platform.data.binance_client import KlineBar
from quant_platform.execution.session import ExecutionAudit
from quant_platform.execution.state import OpenPosition, PaperState, StateStore

REPO = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 7, 10, 0, 0, tzinfo=timezone.utc)


def write_candidate(directory: Path, **risk_overrides) -> str:
    definition = json.loads(
        (REPO / "config/strategies/example-btc-trend.json").read_text(encoding="utf-8")
    )
    del definition["validation_report"]
    definition["id"] = "cycle-test-cand"
    definition["universe"]["symbols"] = ["BTCUSDT"]
    definition["signal"] = {
        "kind": "declarative-rules",
        "parameters": {},
        "entry_rules": [{"indicator": "close", "operator": "greater_than", "operand": 100}],
        "exit_rules": [{"indicator": "close", "operator": "less_than", "operand": 90}],
    }
    definition["risk"] = {
        "max_position_pct_equity": 5, "stop_loss_pct": 5, "max_gross_exposure_pct": 50,
        "max_daily_loss_pct": 2, **risk_overrides,
    }
    definition["data_dependencies"] = [
        {"series": "ohlcv", "frequency": "1h", "lookback_bars": 10}
    ]
    definition["tracking"] = {
        "prediction": (
            "Pre-registered pytest candidate: exercises entry, hold, stop and "
            "exit paths across simulated cycles."
        ),
        "registered_by": "pytest",
        "registered_date": "2026-07-10",
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "cycle-test-cand.json").write_text(json.dumps(definition), encoding="utf-8")
    return definition["id"]


class FakeClient:
    """Serves a scripted sequence of (open, high, low, close) 1h bars; the final
    tuple is the FORMING bar (its close = current live price)."""

    def __init__(self, ohlc: list[tuple]):
        self.ohlc = ohlc

    def klines(self, symbol, interval, limit=500, include_unclosed=False, now=None):
        bars = []
        for i, (o, h, lo, c) in enumerate(self.ohlc):
            open_time = T0 + timedelta(hours=i)
            bars.append(KlineBar(
                open_time=open_time, close_time=open_time + timedelta(hours=1),
                open=o, high=h, low=lo, close=c, volume=1.0,
            ))
        # emulate the real client: the last bar is forming (close_time > now)
        object.__setattr__(bars[-1], "__dict__", bars[-1].__dict__)
        if not include_unclosed:
            bars = bars[:-1]
        return bars[-limit:]

    def funding_rates(self, symbol, limit=100):
        return []

    def close(self):
        pass

    def now_after_bars(self) -> datetime:
        """A 'now' that falls just inside the final (forming) bar."""
        return T0 + timedelta(hours=len(self.ohlc) - 1, minutes=5)


def flat(price, n):
    return [(price, price, price, price)] * n


def paths(tmp_path):
    return (tmp_path / "candidates", tmp_path / "paper-state.json", tmp_path / "executions.jsonl")


def test_full_lifecycle_enter_hold_stop(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    write_candidate(cands)

    # --- cycle 1: last closed bar 105 > 100 -> ENTER at live price 106 ---
    feed = FakeClient(flat(95, 8) + [(105, 105, 104, 105), (106, 106, 106, 106)])
    r1 = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
    assert [x.action for x in r1.results] == ["enter"]
    assert r1.results[0].approved and r1.results[0].fill_price == pytest.approx(106 * 1.0005)
    state = StateStore(state_path).load()
    assert state.cycle_count == 1 and len(state.open_positions) == 1
    stop = state.open_positions[0].stop_price
    assert stop == pytest.approx(106 * 1.0005 * 0.95)

    # --- cycle 2: price drifts, stop untouched -> HOLD; state persists ---
    feed = FakeClient(flat(95, 7) + [(105, 105, 104, 105), (106, 107, 104, 106), (105, 105, 105, 105)])
    r2 = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
    assert [x.action for x in r2.results] == ["hold"]
    assert StateStore(state_path).load().cycle_count == 2

    # --- cycle 3: last closed bar low 98 <= stop -> STOP EXIT at live 99 ---
    feed = FakeClient(flat(95, 6) + [(105, 105, 104, 105), (106, 107, 104, 106),
                                     (105, 105, 98, 99), (99, 99, 99, 99)])
    r3 = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
    assert [x.action for x in r3.results] == ["exit"]
    assert r3.results[0].reason.startswith("stop-breach")
    final = StateStore(state_path).load()
    assert final.cycle_count == 3 and final.open_positions == ()
    # a stopped round trip must have lost money, and the account must be flat
    assert final.cash < 10_000.0
    assert final.positions == {}

    # --- audit completeness: entry + exit, both candidate tier ---
    records = ExecutionAudit(audit_path).records()
    assert len(records) == 2
    assert all(rec.tier == "candidate" for rec in records)
    assert [rec.side.value for rec in records] == ["buy", "sell"]


def test_idempotent_within_same_bar(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    write_candidate(cands)
    feed = FakeClient(flat(95, 8) + [(105, 105, 104, 105), (106, 106, 106, 106)])
    now = feed.now_after_bars()
    run_cycle(cands, state_path, audit_path, client=feed, now=now)
    r2 = run_cycle(cands, state_path, audit_path, client=feed, now=now)  # same bar, again
    assert [x.action for x in r2.results] == ["hold"]  # position exists -> no re-entry
    assert len(ExecutionAudit(audit_path).records()) == 1  # exactly one fill, not two


def test_orphaned_position_refused(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    write_candidate(cands)
    state = PaperState.fresh(10_000.0)
    orphan = OpenPosition(
        candidate_id="ghost-cand", symbol="ETHUSDT", quantity=1.0, entry_price=100.0,
        entry_ts=T0, stop_price=95.0, entry_fill_id="deadbeef0000",
    )
    account = state.restore_account()
    account.positions["ETHUSDT"] = 1.0
    StateStore(state_path).save(
        PaperState.from_account(account, (orphan,), cycle_count=5)
    )
    feed = FakeClient(flat(95, 10))
    with pytest.raises(CycleError, match="ghost-cand"):
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())


def test_bad_feed_jump_rejected_by_risk_engine(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    write_candidate(cands)
    # entry condition true, but the last closed bar jumped +90% -> sanity fails
    feed = FakeClient(flat(60, 8) + [(114, 114, 60, 114), (114, 114, 114, 114)])
    r = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
    assert r.results[0].action == "enter" and r.results[0].approved is False
    records = ExecutionAudit(audit_path).records()
    assert records[-1].approved is False
    failed = [c for c in records[-1].checks if not c["passed"]]
    assert any(c["name"] == "price_jump" for c in failed)


def test_stale_feed_rejected(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    write_candidate(cands)
    feed = FakeClient(flat(95, 8) + [(105, 105, 104, 105), (106, 106, 106, 106)])
    stale_now = feed.now_after_bars() + timedelta(hours=6)
    r = run_cycle(cands, state_path, audit_path, client=feed, now=stale_now)
    assert r.results[0].approved is False


def test_no_candidates_is_a_valid_cycle(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    cands.mkdir()
    feed = FakeClient(flat(95, 3))
    r = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
    assert r.results == () and r.equity == 10_000.0
    assert StateStore(state_path).load().cycle_count == 1


class TestOutcomeLoopIntegration:
    """M10: the cycle resolves due memo outcomes automatically."""

    def _journal_with_due_memo(self, ws: Path):
        from quant_platform.journal import DecisionJournal, MemoRecord
        journal = DecisionJournal(ws / "reports" / "research" / "journal.jsonl")
        journal.append_memo(MemoRecord(
            symbol="BTCUSD",
            context={
                "symbol": "BTCUSD", "as_of": "2026-06-20", "source": "t", "stale_days": 0,
                "last_close": 100.0, "returns_pct": {}, "volatility_annualized_pct": {},
                "trend": {}, "range_365d": {}, "avg_volume_30d": None, "bars": 60,
            },
            memo="Confidence: LOW", model="test", confidence="LOW",
        ))
        return journal

    def test_due_outcome_recorded_via_mock_service(self, tmp_path):
        import httpx
        from quant_platform.cli_cycle import resolve_due_outcomes
        from quant_platform.data.openbb_client import OpenBBClient

        journal = self._journal_with_due_memo(tmp_path)

        def handler(request: httpx.Request) -> httpx.Response:
            if "system/version" in str(request.url):
                return httpx.Response(200, json={"version": "test"})
            return httpx.Response(200, json={"results": [
                {"date": "2026-06-20", "open": 100, "high": 100, "low": 100,
                 "close": 100.0, "volume": 1},
                {"date": "2026-07-09", "open": 110, "high": 110, "low": 110,
                 "close": 110.0, "volume": 1},
            ]})

        client = OpenBBClient(transport=httpx.MockTransport(handler))
        assert resolve_due_outcomes(tmp_path, client=client) == 1
        outcomes = journal.outcomes_for(journal.memos()[0].record_id)
        assert len(outcomes) == 1
        assert outcomes[0].realized_return_pct == 10.0
        # idempotent: resolved memos are not re-resolved
        assert resolve_due_outcomes(tmp_path, client=client) == 0

    def test_offline_service_returns_none(self, tmp_path):
        import httpx
        from quant_platform.cli_cycle import resolve_due_outcomes
        from quant_platform.data.openbb_client import OpenBBClient

        self._journal_with_due_memo(tmp_path)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("service down")

        client = OpenBBClient(transport=httpx.MockTransport(handler))
        assert resolve_due_outcomes(tmp_path, client=client) is None

    def test_empty_journal_short_circuits(self, tmp_path):
        from quant_platform.cli_cycle import resolve_due_outcomes
        assert resolve_due_outcomes(tmp_path, client=object()) == 0  # never touched


class TestMarketStructureRecorder:
    @staticmethod
    def rich_fake(ohlc, oi_hours=3):
        from datetime import timedelta
        from quant_platform.data.binance_client import OpenInterestPoint, PremiumBar

        class RichFake(FakeClient):
            def open_interest_hist(self, symbol, period="1h", limit=30):
                return [OpenInterestPoint(
                    ts=T0 + timedelta(hours=i), open_interest=100.0 + i,
                    open_interest_value=6_000_000.0,
                ) for i in range(oi_hours)][-limit:]

            def premium_index_klines(self, symbol, interval, limit=500,
                                     start_time_ms=None, include_unclosed=False, now=None):
                return [PremiumBar(open_time=T0, close_time=T0 + timedelta(hours=1),
                                   open=-0.0004, high=-0.0002, low=-0.0008, close=-0.0006)]

        return RichFake(ohlc)

    def test_oi_window_and_basis_appended(self, tmp_path):
        import json
        cands, state_path, audit_path = paths(tmp_path)
        write_candidate(cands)
        feed = self.rich_fake(flat(95, 10), oi_hours=3)
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        rows = [json.loads(x) for x in
                (tmp_path / "market-structure.jsonl").read_text().splitlines()]
        oi_rows = [r for r in rows if "oi" in r]
        basis_rows = [r for r in rows if "basis_close" in r]
        assert len(oi_rows) == 3 and oi_rows[-1]["oi"] == 102.0  # full window captured
        assert len(basis_rows) == 1 and basis_rows[0]["basis_close"] == -0.0006

    def test_missed_cycles_caught_up_without_duplicates(self, tmp_path):
        import json
        cands, state_path, audit_path = paths(tmp_path)
        write_candidate(cands)
        feed = self.rich_fake(flat(95, 10), oi_hours=2)
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        # "PC was off": next cycle's window contains 4 new + 2 already-recorded points
        feed = self.rich_fake(flat(95, 10), oi_hours=6)
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        rows = [json.loads(x) for x in
                (tmp_path / "market-structure.jsonl").read_text().splitlines()]
        oi_ts = [r["oi_ts"] for r in rows if "oi" in r]
        assert len(oi_ts) == 6 and len(set(oi_ts)) == 6 and oi_ts == sorted(oi_ts)
        # re-run with nothing new: no further OI rows
        feed = self.rich_fake(flat(95, 10), oi_hours=6)
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        rows2 = [json.loads(x) for x in
                 (tmp_path / "market-structure.jsonl").read_text().splitlines()]
        assert len([r for r in rows2 if "oi" in r]) == 6

    def test_plain_fake_client_skips_gracefully(self, tmp_path):
        cands, state_path, audit_path = paths(tmp_path)
        write_candidate(cands)
        feed = FakeClient(flat(95, 10))
        r = run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())
        assert r.cycle_count == 1  # cycle unaffected
        assert not (tmp_path / "market-structure.jsonl").exists()
