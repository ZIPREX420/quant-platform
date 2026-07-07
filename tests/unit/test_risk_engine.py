"""RiskEngine: caps are hard, kill-switch fail-closed, decisions fully explained."""
from datetime import date, datetime, timedelta, timezone

import pytest

from quant_platform.data.schemas import OHLCVBar, PriceHistory
from quant_platform.risk import (
    OrderRequest,
    PortfolioState,
    RiskEngine,
    Side,
    check_price_sanity,
)

RISK = {
    "max_position_pct_equity": 5.0,   # 500 on 10k equity
    "stop_loss_pct": 10.0,
    "max_gross_exposure_pct": 50.0,   # 5000 on 10k equity
    "max_daily_loss_pct": 2.0,
}


def engine() -> RiskEngine:
    return RiskEngine(RISK)


def portfolio(equity=10_000.0, sod=10_000.0, positions=None) -> PortfolioState:
    return PortfolioState(equity=equity, equity_start_of_day=sod, positions=positions or {})


def order(notional, side=Side.BUY, symbol="BTC-USD") -> OrderRequest:
    return OrderRequest(strategy_id="s1", symbol=symbol, side=side, notional=notional)


class TestPositionSizing:
    def test_within_cap_approved_in_full(self):
        d = engine().evaluate(order(400), portfolio())
        assert d.approved and d.approved_notional == 400

    def test_oversized_order_shrunk_to_cap(self):
        d = engine().evaluate(order(2_000), portfolio())
        assert d.approved and d.approved_notional == 500  # 5% of 10k
        assert any(c.name == "size_reduced" for c in d.checks)

    def test_existing_position_reduces_headroom(self):
        d = engine().evaluate(order(400), portfolio(positions={"BTC-USD": 300}))
        assert d.approved and d.approved_notional == 200

    def test_no_headroom_rejected(self):
        d = engine().evaluate(order(100), portfolio(positions={"BTC-USD": 500}))
        assert not d.approved and d.approved_notional == 0


class TestGrossExposure:
    def test_gross_cap_shrinks_across_symbols(self):
        pf = portfolio(positions={"ETH-USD": 4_800})
        d = engine().evaluate(order(400), pf)
        assert d.approved and d.approved_notional == 200  # gross headroom 5000-4800

    def test_gross_cap_full_rejected(self):
        pf = portfolio(positions={"ETH-USD": 5_000})
        d = engine().evaluate(order(100), pf)
        assert not d.approved


class TestKillSwitch:
    def test_daily_loss_blocks_new_exposure(self):
        pf = portfolio(equity=9_700, sod=10_000)  # -3% day vs 2% cap
        d = engine().evaluate(order(100), pf)
        assert not d.approved
        assert any("daily_loss_kill_switch" == c.name and not c.passed for c in d.checks)

    def test_daily_loss_allows_reducing_orders(self):
        pf = portfolio(equity=9_700, sod=10_000, positions={"BTC-USD": 400})
        d = engine().evaluate(order(400, side=Side.SELL), pf)
        assert d.approved and d.approved_notional == 400

    def test_boundary_not_tripped(self):
        pf = portfolio(equity=9_810, sod=10_000)  # -1.9%
        assert engine().evaluate(order(100), pf).approved


class TestExplainability:
    def test_every_decision_carries_checks(self):
        d = engine().evaluate(order(100), portfolio())
        assert {c.name for c in d.checks} >= {
            "daily_loss_kill_switch", "position_size_cap", "gross_exposure_cap",
        }

    def test_rejection_reasons_populated(self):
        pf = portfolio(equity=9_000, sod=10_000)
        d = engine().evaluate(order(100), pf)
        assert d.reasons and "daily pnl" in d.reasons[0]


class TestStopLoss:
    def test_long_and_short_stops(self):
        e = engine()
        assert e.stop_loss_price(100.0, Side.BUY) == 90.0
        assert e.stop_loss_price(100.0, Side.SELL) == 110.0


class TestConstruction:
    def test_missing_caps_rejected(self):
        with pytest.raises(ValueError, match="missing required caps"):
            RiskEngine({"stop_loss_pct": 5})


class TestPriceSanity:
    def history(self, last_move_pct=1.0, days_old=0):
        end = date.today() - timedelta(days=days_old)
        prev_close = 100.0
        last_close = prev_close * (1 + last_move_pct / 100.0)
        bars = (
            OHLCVBar(date=end - timedelta(days=1), open=prev_close, high=prev_close * 1.3,
                     low=prev_close * 0.7, close=prev_close, volume=1),
            OHLCVBar(date=end, open=prev_close, high=max(prev_close, last_close) * 1.01,
                     low=min(prev_close, last_close) * 0.99, close=last_close, volume=1),
        )
        return PriceHistory(symbol="BTC-USD", source="test",
                            fetched_at=datetime.now(timezone.utc), bars=bars)

    def test_clean_data_passes(self):
        assert all(c.passed for c in check_price_sanity(self.history()))

    def test_stale_data_flagged(self):
        results = check_price_sanity(self.history(days_old=5))
        assert any(c.name == "price_staleness" and not c.passed for c in results)

    def test_price_jump_flagged(self):
        results = check_price_sanity(self.history(last_move_pct=28.0))
        assert any(c.name == "price_jump" and not c.passed for c in results)

    def test_sanity_failure_rejects_order(self):
        sanity = check_price_sanity(self.history(last_move_pct=28.0))
        d = engine().evaluate(order(100), portfolio(), sanity=sanity)
        assert not d.approved
