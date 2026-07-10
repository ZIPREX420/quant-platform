"""Paper execution: fill math, account integrity, audited session with risk gating."""
import hashlib
import json
from pathlib import Path

import pytest

from quant_platform.execution import (
    ExecutionAudit,
    PaperAccount,
    PaperExchange,
    PaperTradingSession,
)
from quant_platform.risk.engine import OrderRequest, Side
from quant_platform.strategies.loader import load_strategy

REPO = Path(__file__).resolve().parents[2]


def loaded_strategy(tmp_path):
    """A contract-complete strategy (reuses the schema example + real signed report)."""
    ws = tmp_path / "ws"
    (ws / "docs" / "validation").mkdir(parents=True)
    report = ws / "docs" / "validation" / "r.md"
    report.write_text("result: pass\n", encoding="utf-8")
    definition = json.loads((REPO / "config/strategies/example-btc-trend.json").read_text())
    definition["id"] = "exec-test"
    definition["validation_report"].update(
        path="docs/validation/r.md",
        sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        signed_off_by="pytest",
    )
    p = tmp_path / "s.json"
    p.write_text(json.dumps(definition), encoding="utf-8")
    return load_strategy(p, ws)
    # risk caps in the example: max_position 5%, gross 50%, daily loss 2%, stop 10%


def order(notional, side=Side.BUY):
    return OrderRequest(strategy_id="exec-test", symbol="BTC-USD", side=side, notional=notional)


class TestPaperExchange:
    def test_buy_fill_slips_up_and_charges_fee(self):
        fill = PaperExchange().execute(order(1000.0), market_price=100.0)
        assert fill.fill_price == 100.05          # +0.05% slippage
        assert fill.quantity == pytest.approx(1000.0 / 100.05)
        assert fill.fee == 1.0                    # 0.10% of notional
        assert fill.slippage_cost == pytest.approx(0.05 * fill.quantity, rel=1e-6)

    def test_sell_fill_slips_down(self):
        fill = PaperExchange().execute(order(1000.0, Side.SELL), market_price=100.0)
        assert fill.fill_price == 99.95

    def test_bad_price_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            PaperExchange().execute(order(100.0), market_price=0.0)


class TestPaperAccount:
    def test_buy_then_sell_roundtrip_costs_only_fees_and_slippage(self):
        account = PaperAccount(10_000.0)
        exchange = PaperExchange()
        buy = exchange.execute(order(1000.0), 100.0)
        account.apply(buy)
        assert account.cash == pytest.approx(10_000.0 - 1000.0 - 1.0)
        # sell the exact quantity back at the same market price
        sell_notional = buy.quantity * 99.95  # what a full exit yields at fill price
        sell = exchange.execute(order(sell_notional, Side.SELL), 100.0)
        # force identical quantity for a clean roundtrip check
        assert sell.quantity == pytest.approx(buy.quantity, rel=1e-9)
        account.apply(sell)
        assert account.positions == {}
        total_cost = 10_000.0 - account.cash
        assert 0 < total_cost < 4.0  # fees + slippage only, ~0.3% of 1000

    def test_equity_marking_requires_prices(self):
        account = PaperAccount(10_000.0)
        account.apply(PaperExchange().execute(order(1000.0), 100.0))
        with pytest.raises(KeyError, match="mark price"):
            account.equity({})
        assert account.equity({"BTC-USD": 100.0}) == pytest.approx(9_998.5, abs=0.2)


class TestPaperTradingSession:
    def session(self, tmp_path):
        return PaperTradingSession(
            strategy=loaded_strategy(tmp_path),
            account=PaperAccount(10_000.0),
            audit=ExecutionAudit(tmp_path / "audit.jsonl"),
        )

    def test_approved_signal_fills_and_audits(self, tmp_path):
        s = self.session(tmp_path)
        rec = s.process_signal("BTC-USD", Side.BUY, 400.0, {"BTC-USD": 100.0}, 10_000.0)
        assert rec.approved and rec.fill is not None
        assert rec.fill["requested_notional"] == 400.0
        assert s.account.positions["BTC-USD"] > 0
        stored = s.audit.records()
        assert len(stored) == 1 and stored[0].audit_id == rec.audit_id
        assert {c["name"] for c in stored[0].checks} >= {"position_size_cap", "gross_exposure_cap"}

    def test_oversized_signal_shrunk_to_cap(self, tmp_path):
        s = self.session(tmp_path)
        rec = s.process_signal("BTC-USD", Side.BUY, 2_000.0, {"BTC-USD": 100.0}, 10_000.0)
        assert rec.approved and rec.approved_notional == 500.0  # 5% cap of 10k

    def test_kill_switch_day_blocks_and_audits_rejection(self, tmp_path):
        s = self.session(tmp_path)
        # equity 10k, but day started at 10.5k -> -4.76% day vs 2% cap
        rec = s.process_signal("BTC-USD", Side.BUY, 100.0, {"BTC-USD": 100.0}, 10_500.0)
        assert not rec.approved and rec.fill is None
        assert any(c["name"] == "daily_loss_kill_switch" and not c["passed"] for c in rec.checks)
        assert s.account.positions == {}  # nothing executed
        assert len(s.audit.records()) == 1  # rejection still audited

    def test_missing_price_raises(self, tmp_path):
        with pytest.raises(KeyError, match="no price"):
            self.session(tmp_path).process_signal("ETH-USD", Side.BUY, 100.0, {}, 10_000.0)

    def test_unvalidated_strategy_cannot_even_enter(self, tmp_path):
        # the session type only accepts LoadedStrategy; the loader is the gate -
        # here we confirm the example (unsigned report) is refused upstream.
        from quant_platform.strategies.loader import StrategyLoadError
        with pytest.raises(StrategyLoadError):
            load_strategy(REPO / "config/strategies/example-btc-trend.json", tmp_path)


class TestCandidateTierInSession:
    """ADR-0006: candidates run in the SAME session with the SAME governance,
    and every audit record is stamped with their tier."""

    def loaded_candidate(self, tmp_path):
        definition = json.loads(
            (REPO / "config/strategies/example-btc-trend.json").read_text()
        )
        del definition["validation_report"]
        definition["id"] = "cand-exec-test"
        definition["tracking"] = {
            "prediction": (
                "Pre-registered: expected to lose net of costs; forward record "
                "exists to test the 2024 regime-inversion finding."
            ),
            "registered_by": "pytest",
            "registered_date": "2026-07-10",
        }
        p = tmp_path / "c.json"
        p.write_text(json.dumps(definition), encoding="utf-8")
        from quant_platform.strategies.candidates import load_candidate
        return load_candidate(p)

    def test_candidate_fill_audited_with_tier(self, tmp_path):
        session = PaperTradingSession(
            strategy=self.loaded_candidate(tmp_path),
            account=PaperAccount(starting_cash=10_000.0),
            audit=ExecutionAudit(tmp_path / "executions.jsonl"),
        )
        record = session.process_signal(
            "BTC-USD", Side.BUY, 400.0, {"BTC-USD": 100.0}, 10_000.0
        )
        assert record.tier == "candidate"
        assert record.approved and record.fill is not None
        on_disk = session.audit.records()
        assert on_disk[-1].tier == "candidate"

    def test_candidate_risk_caps_enforced_identically(self, tmp_path):
        # example caps: max_position 5% of equity -> 10k equity caps at 500
        session = PaperTradingSession(
            strategy=self.loaded_candidate(tmp_path),
            account=PaperAccount(starting_cash=10_000.0),
            audit=ExecutionAudit(tmp_path / "executions.jsonl"),
        )
        record = session.process_signal(
            "BTC-USD", Side.BUY, 5_000.0, {"BTC-USD": 100.0}, 10_000.0
        )
        assert record.approved_notional <= 500.0 + 1e-9

    def test_validated_strategy_records_stamped_validated(self, tmp_path):
        session = PaperTradingSession(
            strategy=loaded_strategy(tmp_path),
            account=PaperAccount(starting_cash=10_000.0),
            audit=ExecutionAudit(tmp_path / "executions.jsonl"),
        )
        record = session.process_signal(
            "BTC-USD", Side.BUY, 400.0, {"BTC-USD": 100.0}, 10_000.0
        )
        assert record.tier == "validated"
