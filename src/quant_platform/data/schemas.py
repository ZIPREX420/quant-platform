"""Internal market-data contracts.

Every external source (OpenBB REST, exchange-native paths) is normalized into
these schemas at the data-service boundary; nothing downstream ever sees a
provider-specific payload.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OHLCVBar(BaseModel):
    """One OHLCV bar. Volume may be absent for some providers/instruments."""

    model_config = ConfigDict(frozen=True)

    date: date
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float | None = Field(default=None, ge=0)

    def model_post_init(self, __context) -> None:
        if self.high < self.low:
            raise ValueError(f"high {self.high} < low {self.low}")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open {self.open} outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close {self.close} outside [low, high]")


class PriceHistory(BaseModel):
    """A series of OHLCV bars with provenance and staleness metadata."""

    model_config = ConfigDict(frozen=True)

    symbol: str = Field(min_length=1)
    source: str = Field(min_length=1, description="e.g. 'openbb/yfinance'")
    fetched_at: datetime
    bars: tuple[OHLCVBar, ...] = Field(min_length=1)

    @field_validator("fetched_at")
    @classmethod
    def fetched_at_must_be_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("fetched_at must be timezone-aware")
        return v

    @field_validator("bars")
    @classmethod
    def bars_sorted_unique(cls, v: tuple[OHLCVBar, ...]) -> tuple[OHLCVBar, ...]:
        dates = [b.date for b in v]
        if dates != sorted(dates):
            raise ValueError("bars must be sorted by date ascending")
        if len(set(dates)) != len(dates):
            raise ValueError("bars contain duplicate dates")
        return v

    @property
    def last_close(self) -> float:
        return self.bars[-1].close

    def staleness_days(self, now: datetime | None = None) -> int:
        now = now or datetime.now(timezone.utc)
        return (now.date() - self.bars[-1].date).days
