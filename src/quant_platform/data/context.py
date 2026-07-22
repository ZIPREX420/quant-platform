"""Deterministic market-context builder: PriceHistory -> MarketContext.

This is the data the research desk reasons from. Everything here is pure
arithmetic over bars - no opinions, no indicators zoo. Ported from the
validated ADR-0004 spike (experiments/prototypes/research-desk-spike).
"""
from __future__ import annotations

from datetime import datetime
from statistics import mean, pstdev

from pydantic import BaseModel, ConfigDict

from quant_platform.data.schemas import PriceHistory

TRADING_DAYS_PER_YEAR_CRYPTO = 365  # crypto trades continuously


def _pct(a: float, b: float) -> float:
    return round((a / b - 1.0) * 100.0, 2)


class MarketContext(BaseModel):
    """Snapshot statistics the desk receives. All percentages, no raw series."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: str
    source: str
    stale_days: int
    last_close: float
    returns_pct: dict[str, float | None]
    volatility_annualized_pct: dict[str, float]
    trend: dict[str, float | None]
    range_365d: dict[str, float]
    avg_volume_30d: float | None
    bars: int
    # M10 enrichment - optional so pre-M10 journal records still validate
    funding: dict[str, float | None] | None = None
    macro: dict[str, float | None] | None = None


def funding_snapshot(events: list[tuple[datetime, float]]) -> dict[str, float | None]:
    """Positioning summary from perp funding events (rate per settlement, e.g. 8h).

    Percentages per settlement period; sign convention: positive = longs pay.
    """
    if not events:
        return {"last_pct": None, "avg_3d_pct": None, "avg_30d_pct": None}
    rates = [r for _, r in sorted(events, key=lambda e: e[0])]

    def avg_pct(n: int) -> float | None:
        return round(mean(rates[-n:]) * 100, 4) if len(rates) >= n else None

    return {
        "last_pct": round(rates[-1] * 100, 4),
        "avg_3d_pct": avg_pct(9),     # 9 settlements = 3 days at 8h
        "avg_30d_pct": avg_pct(90),
    }


def build_market_context(
    history: PriceHistory,
    now: datetime | None = None,
    funding: dict[str, float | None] | None = None,
    macro: dict[str, float | None] | None = None,
) -> MarketContext:
    """Compute the desk's context from a daily-bar history (>= 60 bars).

    funding/macro are optional pre-computed enrichment blocks (M10); they are
    attached verbatim so the memo's journal record preserves exactly what the
    desk saw.
    """
    if len(history.bars) < 60:
        raise ValueError(
            f"insufficient history for {history.symbol}: {len(history.bars)} bars (need >= 60)"
        )
    closes = [b.close for b in history.bars]
    vols = [b.volume for b in history.bars if b.volume is not None]
    last = closes[-1]

    def ret_over(days: int) -> float | None:
        return _pct(last, closes[-days - 1]) if len(closes) > days else None

    daily_rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    ann = TRADING_DAYS_PER_YEAR_CRYPTO ** 0.5
    sma50 = mean(closes[-50:])
    sma200 = mean(closes[-200:]) if len(closes) >= 200 else None
    high = max(closes)
    low = min(closes)

    return MarketContext(
        symbol=history.symbol,
        as_of=history.bars[-1].date.isoformat(),
        source=history.source,
        stale_days=history.staleness_days(now),
        last_close=round(last, 2),
        returns_pct={
            "7d": ret_over(7), "30d": ret_over(30),
            "90d": ret_over(90), "365d": ret_over(364),
        },
        volatility_annualized_pct={
            "30d": round(pstdev(daily_rets[-30:]) * ann * 100, 1),
            "90d": round(pstdev(daily_rets[-90:]) * ann * 100, 1),
        },
        trend={
            "sma50": round(sma50, 2),
            "sma200": round(sma200, 2) if sma200 else None,
            "price_vs_sma50_pct": _pct(last, sma50),
            "price_vs_sma200_pct": _pct(last, sma200) if sma200 else None,
        },
        range_365d={
            "high": round(high, 2),
            "low": round(low, 2),
            "drawdown_from_high_pct": _pct(last, high),
            "recovery_from_low_pct": _pct(last, low),
        },
        avg_volume_30d=round(mean(vols[-30:]), 0) if len(vols) >= 30 else None,
        bars=len(history.bars),
        funding=funding,
        macro=macro,
    )
