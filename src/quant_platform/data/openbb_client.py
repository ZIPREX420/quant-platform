"""HTTP client for a locally running OpenBB Platform REST API.

Process-boundary consumption per ADR-0003: this module never imports openbb.
Start the service with automation/bootstrap/m4-openbb-api.bat (workspace).
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import httpx

from quant_platform.data.schemas import OHLCVBar, PriceHistory

DEFAULT_BASE_URL = "http://127.0.0.1:6900"


class OpenBBClientError(RuntimeError):
    """Raised for transport failures, non-2xx responses, or malformed payloads."""


class OpenBBClient:
    """Minimal, typed client for the OpenBB REST endpoints the platform uses."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._client = httpx.Client(
            base_url=base_url,
            timeout=timeout_seconds,
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "OpenBBClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def health(self) -> bool:
        """True iff the OpenBB REST service answers its OpenAPI probe."""
        try:
            response = self._client.get("/api/v1/system/version")
            if response.status_code == 200:
                return True
            # older/newer builds may not expose system/version; fall back to docs
            return self._client.get("/openapi.json").status_code == 200
        except httpx.HTTPError:
            return False

    def _historical(
        self,
        endpoint: str,
        symbol: str,
        start_date: date,
        end_date: date,
        provider: str,
    ) -> PriceHistory:
        """Shared fetch-and-normalize path for every OHLCV-shaped endpoint."""
        try:
            response = self._client.get(
                endpoint,
                params={
                    "symbol": symbol,
                    "provider": provider,
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                },
            )
        except httpx.HTTPError as exc:
            raise OpenBBClientError(f"transport failure: {exc}") from exc
        if response.status_code != 200:
            raise OpenBBClientError(
                f"OpenBB REST returned {response.status_code}: {response.text[:300]}"
            )
        try:
            results = response.json()["results"]
        except (KeyError, ValueError) as exc:
            raise OpenBBClientError(f"malformed payload: {exc}") from exc
        if not results:
            raise OpenBBClientError(f"no data returned for {symbol}")

        bars = []
        try:
            for row in results:
                bars.append(
                    OHLCVBar(
                        date=date.fromisoformat(str(row["date"])[:10]),
                        open=row["open"],
                        high=row["high"],
                        low=row["low"],
                        close=row["close"],
                        volume=row.get("volume"),
                    )
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise OpenBBClientError(f"row failed normalization: {exc}") from exc
        bars.sort(key=lambda b: b.date)

        return PriceHistory(
            symbol=symbol,
            source=f"openbb/{provider}",
            fetched_at=datetime.now(timezone.utc),
            bars=tuple(bars),
        )

    def crypto_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        provider: str = "yfinance",
    ) -> PriceHistory:
        """Daily OHLCV history for a crypto pair, normalized to PriceHistory."""
        return self._historical(
            "/api/v1/crypto/price/historical", symbol, start_date, end_date, provider
        )

    def index_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        provider: str = "yfinance",
    ) -> PriceHistory:
        """Daily index history (e.g. ^GSPC, ^VIX) - macro context for the desk."""
        return self._historical(
            "/api/v1/index/price/historical", symbol, start_date, end_date, provider
        )

    def currency_historical(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        provider: str = "yfinance",
    ) -> PriceHistory:
        """Daily FX pair history (e.g. EURUSD) - macro context for the desk."""
        return self._historical(
            "/api/v1/currency/price/historical", symbol, start_date, end_date, provider
        )
