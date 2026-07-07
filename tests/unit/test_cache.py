"""Unit tests for PriceHistoryCache."""
from datetime import date, datetime, timedelta, timezone

from quant_platform.data.cache import PriceHistoryCache
from quant_platform.data.schemas import OHLCVBar, PriceHistory

START, END = date(2026, 1, 1), date(2026, 1, 2)


def history(fetched_at=None) -> PriceHistory:
    return PriceHistory(
        symbol="BTC-USD", source="openbb/yfinance",
        fetched_at=fetched_at or datetime.now(timezone.utc),
        bars=(OHLCVBar(date=START, open=10, high=12, low=9, close=11, volume=1),
              OHLCVBar(date=END, open=11, high=13, low=10, close=12, volume=2)),
    )


def test_roundtrip(tmp_path):
    cache = PriceHistoryCache(tmp_path)
    cache.put(history(), START, END)
    got = cache.get("BTC-USD", "openbb/yfinance", START, END)
    assert got is not None and got.last_close == 12 and len(got.bars) == 2


def test_miss(tmp_path):
    assert PriceHistoryCache(tmp_path).get("ETH-USD", "openbb/yfinance", START, END) is None


def test_expiry(tmp_path):
    cache = PriceHistoryCache(tmp_path)
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    cache.put(history(fetched_at=old), START, END)
    assert cache.get("BTC-USD", "openbb/yfinance", START, END, max_age=timedelta(hours=24)) is None
    assert cache.get("BTC-USD", "openbb/yfinance", START, END) is not None  # no TTL -> hit


def test_corrupt_entry_is_miss_and_removed(tmp_path):
    cache = PriceHistoryCache(tmp_path)
    path = cache.put(history(), START, END)
    path.write_text("{not json", encoding="utf-8")
    assert cache.get("BTC-USD", "openbb/yfinance", START, END) is None
    assert not path.exists()


def test_source_slug_separates_entries(tmp_path):
    cache = PriceHistoryCache(tmp_path)
    cache.put(history(), START, END)
    assert cache.get("BTC-USD", "openbb/fmp", START, END) is None
