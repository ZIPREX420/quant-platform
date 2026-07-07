"""Unit tests for quant_platform.data.schemas."""
from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from quant_platform.data.schemas import OHLCVBar, PriceHistory


def bar(d="2026-01-02", o=10.0, h=12.0, lo=9.0, c=11.0, v=100.0) -> OHLCVBar:
    return OHLCVBar(date=date.fromisoformat(d), open=o, high=h, low=lo, close=c, volume=v)


class TestOHLCVBar:
    def test_valid_bar(self):
        b = bar()
        assert b.close == 11.0 and b.volume == 100.0

    def test_volume_optional(self):
        assert bar(v=None).volume is None

    @pytest.mark.parametrize("kwargs", [
        dict(h=8.0),            # high < low
        dict(o=13.0),           # open above high
        dict(c=8.5),            # close below low
        dict(o=-1.0),           # non-positive price
    ])
    def test_invariants_rejected(self, kwargs):
        with pytest.raises((ValidationError, ValueError)):
            bar(**kwargs)

    def test_frozen(self):
        with pytest.raises(ValidationError):
            bar().close = 1.0


class TestPriceHistory:
    def new(self, dates=("2026-01-01", "2026-01-02")):
        return PriceHistory(
            symbol="BTCUSD",
            source="openbb/yfinance",
            fetched_at=datetime.now(timezone.utc),
            bars=tuple(bar(d=d) for d in dates),
        )

    def test_valid(self):
        h = self.new()
        assert h.last_close == 11.0 and len(h.bars) == 2

    def test_unsorted_rejected(self):
        with pytest.raises(ValidationError):
            self.new(dates=("2026-01-02", "2026-01-01"))

    def test_duplicates_rejected(self):
        with pytest.raises(ValidationError):
            self.new(dates=("2026-01-01", "2026-01-01"))

    def test_empty_rejected(self):
        with pytest.raises(ValidationError):
            PriceHistory(symbol="X", source="s",
                         fetched_at=datetime.now(timezone.utc), bars=())

    def test_naive_fetched_at_rejected(self):
        with pytest.raises(ValidationError):
            PriceHistory(symbol="X", source="s",
                         fetched_at=datetime(2026, 1, 1), bars=(bar(),))

    def test_staleness(self):
        h = self.new()
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)
        assert h.staleness_days(now) == 3
