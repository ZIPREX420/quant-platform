"""HTTP client for Binance PUBLIC market-data endpoints (no API keys, ever).

Feeds the M9 paper-trading cycle with current market data:
  - spot klines   (api.binance.com/api/v3/klines)
  - perp funding  (fapi.binance.com/fapi/v1/fundingRate)

Correctness rule: the final kline Binance returns is the still-forming bar.
``klines()`` drops it by default (``include_unclosed=False``) so downstream
signal evaluation only ever sees CLOSED bars - the same no-lookahead
discipline the validation backtester enforces.

Read-only market data; this module can never place an order (risk R-4).
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

SPOT_BASE_URL = "https://api.binance.com"
PERP_BASE_URL = "https://fapi.binance.com"

SUPPORTED_INTERVALS = ("15m", "1h", "4h", "1d")
MAX_LIMIT = 1000


class BinanceClientError(RuntimeError):
    """Transport failure, non-2xx response, or malformed payload."""


class KlineBar(BaseModel):
    """One CLOSED spot kline, normalized."""

    model_config = ConfigDict(frozen=True)

    open_time: datetime
    close_time: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)

    @field_validator("open_time", "close_time")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v


class PremiumBar(BaseModel):
    """One CLOSED premium-index kline (perp basis vs index). Values are RATES
    (e.g. -0.0005 = perp trades 5 bps under index) and are routinely negative,
    so no positivity invariant applies."""

    model_config = ConfigDict(frozen=True)

    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float

    @field_validator("open_time", "close_time")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("timestamps must be timezone-aware")
        return v


class OpenInterestPoint(BaseModel):
    """One open-interest history point. NOTE: Binance retains only ~30 days of
    OI history - this series must be accumulated forward (M13)."""

    model_config = ConfigDict(frozen=True)

    ts: datetime
    open_interest: float = Field(ge=0)          # contracts (base units)
    open_interest_value: float = Field(ge=0)    # quote value

    @field_validator("ts")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("ts must be timezone-aware")
        return v


class FundingEvent(BaseModel):
    """One perp funding-rate settlement event."""

    model_config = ConfigDict(frozen=True)

    funding_time: datetime
    rate: float

    @field_validator("funding_time")
    @classmethod
    def _aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("funding_time must be timezone-aware")
        return v


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


class BinanceClient:
    """Minimal, typed client for the two public endpoints the cycle needs."""

    def __init__(
        self,
        spot_base_url: str = SPOT_BASE_URL,
        perp_base_url: str = PERP_BASE_URL,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._spot = httpx.Client(base_url=spot_base_url, timeout=timeout_seconds, transport=transport)
        self._perp = httpx.Client(base_url=perp_base_url, timeout=timeout_seconds, transport=transport)

    def close(self) -> None:
        self._spot.close()
        self._perp.close()

    def __enter__(self) -> "BinanceClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _get(self, client: httpx.Client, path: str, params: dict) -> list:
        try:
            response = client.get(path, params=params)
        except httpx.HTTPError as exc:
            raise BinanceClientError(f"transport failure for {path}: {exc}") from exc
        if response.status_code != 200:
            raise BinanceClientError(
                f"{path} returned {response.status_code}: {response.text[:300]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise BinanceClientError(f"{path}: response is not JSON") from exc
        if not isinstance(payload, list):
            raise BinanceClientError(f"{path}: expected a JSON array, got {type(payload).__name__}")
        return payload

    def klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        include_unclosed: bool = False,
        now: datetime | None = None,
    ) -> list[KlineBar]:
        """Latest ``limit`` spot klines, oldest first, unclosed final bar dropped."""
        if interval not in SUPPORTED_INTERVALS:
            raise BinanceClientError(
                f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"
            )
        if not 1 <= limit <= MAX_LIMIT:
            raise BinanceClientError(f"limit must be in [1, {MAX_LIMIT}], got {limit}")
        rows = self._get(
            self._spot, "/api/v3/klines",
            {"symbol": symbol, "interval": interval, "limit": limit},
        )
        bars: list[KlineBar] = []
        for row in rows:
            try:
                bars.append(KlineBar(
                    open_time=_ms_to_dt(int(row[0])),
                    open=float(row[1]), high=float(row[2]),
                    low=float(row[3]), close=float(row[4]),
                    volume=float(row[5]),
                    close_time=_ms_to_dt(int(row[6])),
                ))
            except (IndexError, TypeError, ValueError) as exc:
                raise BinanceClientError(f"malformed kline row: {row!r}: {exc}") from exc
        if not include_unclosed:
            cutoff = now or datetime.now(timezone.utc)
            bars = [b for b in bars if b.close_time <= cutoff]
        if bars != sorted(bars, key=lambda b: b.open_time):
            raise BinanceClientError("klines not ascending by open_time")
        return bars

    def premium_index_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 500,
        start_time_ms: int | None = None,
        include_unclosed: bool = False,
        now: datetime | None = None,
    ) -> list[PremiumBar]:
        """Premium-index (basis) klines, oldest first, unclosed final bar dropped.

        History reaches back to ~2019-12 for BTCUSDT; paginate with
        start_time_ms for full-depth ingestion.
        """
        if interval not in SUPPORTED_INTERVALS:
            raise BinanceClientError(
                f"unsupported interval {interval!r}; supported: {SUPPORTED_INTERVALS}"
            )
        if not 1 <= limit <= MAX_LIMIT:
            raise BinanceClientError(f"limit must be in [1, {MAX_LIMIT}], got {limit}")
        params: dict = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time_ms is not None:
            params["startTime"] = start_time_ms
        rows = self._get(self._perp, "/fapi/v1/premiumIndexKlines", params)
        bars: list[PremiumBar] = []
        for row in rows:
            try:
                bars.append(PremiumBar(
                    open_time=_ms_to_dt(int(row[0])),
                    open=float(row[1]), high=float(row[2]),
                    low=float(row[3]), close=float(row[4]),
                    close_time=_ms_to_dt(int(row[6])),
                ))
            except (IndexError, TypeError, ValueError) as exc:
                raise BinanceClientError(f"malformed premium kline row: {row!r}: {exc}") from exc
        if not include_unclosed:
            cutoff = now or datetime.now(timezone.utc)
            bars = [b for b in bars if b.close_time <= cutoff]
        if bars != sorted(bars, key=lambda b: b.open_time):
            raise BinanceClientError("premium klines not ascending by open_time")
        return bars

    def open_interest_hist(
        self, symbol: str, period: str = "1h", limit: int = 30
    ) -> list[OpenInterestPoint]:
        """Open-interest history, oldest first. Venue retains ~30 DAYS only -
        callers must persist points forward; history cannot be backfilled."""
        if period not in ("5m", "15m", "30m", "1h", "4h", "1d"):
            raise BinanceClientError(f"unsupported OI period {period!r}")
        if not 1 <= limit <= 500:
            raise BinanceClientError(f"limit must be in [1, 500], got {limit}")
        rows = self._get(
            self._perp, "/futures/data/openInterestHist",
            {"symbol": symbol, "period": period, "limit": limit},
        )
        points: list[OpenInterestPoint] = []
        for row in rows:
            try:
                points.append(OpenInterestPoint(
                    ts=_ms_to_dt(int(row["timestamp"])),
                    open_interest=float(row["sumOpenInterest"]),
                    open_interest_value=float(row["sumOpenInterestValue"]),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise BinanceClientError(f"malformed OI row: {row!r}: {exc}") from exc
        points.sort(key=lambda p: p.ts)
        return points

    def funding_rates(self, symbol: str, limit: int = 100) -> list[FundingEvent]:
        """Latest ``limit`` settled funding events for a perp symbol, oldest first."""
        if not 1 <= limit <= MAX_LIMIT:
            raise BinanceClientError(f"limit must be in [1, {MAX_LIMIT}], got {limit}")
        rows = self._get(
            self._perp, "/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit}
        )
        events: list[FundingEvent] = []
        for row in rows:
            try:
                events.append(FundingEvent(
                    funding_time=_ms_to_dt(int(row["fundingTime"])),
                    rate=float(row["fundingRate"]),
                ))
            except (KeyError, TypeError, ValueError) as exc:
                raise BinanceClientError(f"malformed funding row: {row!r}: {exc}") from exc
        events.sort(key=lambda e: e.funding_time)
        return events
