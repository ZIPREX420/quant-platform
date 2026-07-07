"""build_market_context: deterministic math, staleness, guardrails."""
from datetime import date, datetime, timedelta, timezone

import pytest

from quant_platform.data.context import build_market_context
from quant_platform.data.schemas import OHLCVBar, PriceHistory


def linear_history(n=400, start_price=100.0, step=0.5) -> PriceHistory:
    base = date(2025, 1, 1)
    bars = []
    price = start_price
    for i in range(n):
        price += step
        bars.append(OHLCVBar(
            date=base + timedelta(days=i),
            open=price - 0.2, high=price + 1, low=price - 1, close=price, volume=1000 + i,
        ))
    return PriceHistory(
        symbol="TEST-USD", source="openbb/yfinance",
        fetched_at=datetime.now(timezone.utc), bars=tuple(bars),
    )


def test_context_math_on_linear_series():
    history = linear_history()
    last = history.bars[-1].close
    ctx = build_market_context(history)
    assert ctx.symbol == "TEST-USD" and ctx.bars == 400
    assert ctx.last_close == round(last, 2)
    # rising series: positive returns, price above both SMAs, at 365d high
    assert ctx.returns_pct["30d"] > 0 and ctx.returns_pct["365d"] > 0
    assert ctx.trend["price_vs_sma50_pct"] > 0 and ctx.trend["price_vs_sma200_pct"] > 0
    assert ctx.range_365d["drawdown_from_high_pct"] == 0.0
    expected_vol = round(sum(b.volume for b in history.bars[-30:]) / 30, 0)
    assert ctx.avg_volume_30d == expected_vol
    # exact sma50 check: closes are arithmetic sequence
    closes = [b.close for b in history.bars]
    assert ctx.trend["sma50"] == round(sum(closes[-50:]) / 50, 2)


def test_staleness_flag():
    history = linear_history()
    now = datetime.now(timezone.utc) + timedelta(days=500)
    ctx = build_market_context(history, now=now)
    assert ctx.stale_days >= 500 - 400


def test_short_history_rejected():
    with pytest.raises(ValueError, match="insufficient history"):
        build_market_context(linear_history(n=59))


def test_no_sma200_below_200_bars():
    ctx = build_market_context(linear_history(n=150))
    assert ctx.trend["sma200"] is None and ctx.trend["price_vs_sma200_pct"] is None
