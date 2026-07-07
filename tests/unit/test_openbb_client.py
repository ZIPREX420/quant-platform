"""Unit tests for OpenBBClient against a mock transport (no network, no openbb)."""
from datetime import date

import httpx
import pytest

from quant_platform.data.openbb_client import OpenBBClient, OpenBBClientError

GOOD_PAYLOAD = {
    "results": [
        {"date": "2026-01-02", "open": 10, "high": 12, "low": 9, "close": 11, "volume": 100},
        {"date": "2026-01-01", "open": 9, "high": 11, "low": 8, "close": 10, "volume": 90},
    ],
    "provider": "yfinance",
}


def client_with(handler) -> OpenBBClient:
    return OpenBBClient(transport=httpx.MockTransport(handler))


def test_crypto_historical_normalizes_and_sorts():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/crypto/price/historical"
        assert request.url.params["symbol"] == "BTCUSD"
        return httpx.Response(200, json=GOOD_PAYLOAD)

    with client_with(handler) as c:
        history = c.crypto_historical("BTCUSD", date(2026, 1, 1), date(2026, 1, 2))
    assert [b.date.isoformat() for b in history.bars] == ["2026-01-01", "2026-01-02"]
    assert history.last_close == 11
    assert history.source == "openbb/yfinance"
    assert history.fetched_at.tzinfo is not None


def test_non_200_raises():
    with client_with(lambda r: httpx.Response(500, text="boom")) as c:
        with pytest.raises(OpenBBClientError, match="500"):
            c.crypto_historical("BTCUSD", date(2026, 1, 1), date(2026, 1, 2))


def test_empty_results_raises():
    with client_with(lambda r: httpx.Response(200, json={"results": []})) as c:
        with pytest.raises(OpenBBClientError, match="no data"):
            c.crypto_historical("BTCUSD", date(2026, 1, 1), date(2026, 1, 2))


def test_malformed_payload_raises():
    with client_with(lambda r: httpx.Response(200, json={"nope": 1})) as c:
        with pytest.raises(OpenBBClientError, match="malformed"):
            c.crypto_historical("BTCUSD", date(2026, 1, 1), date(2026, 1, 2))


def test_bad_row_raises():
    payload = {"results": [{"date": "2026-01-01", "open": 1}]}
    with client_with(lambda r: httpx.Response(200, json=payload)) as c:
        with pytest.raises(OpenBBClientError, match="normalization"):
            c.crypto_historical("BTCUSD", date(2026, 1, 1), date(2026, 1, 2))


def test_transport_failure_raises():
    def handler(request):
        raise httpx.ConnectError("refused")
    with client_with(handler) as c:
        with pytest.raises(OpenBBClientError, match="transport"):
            c.crypto_historical("BTCUSD", date(2026, 1, 1), date(2026, 1, 2))


def test_health_true_on_200():
    with client_with(lambda r: httpx.Response(200, json={"version": "x"})) as c:
        assert c.health() is True


def test_health_false_on_connect_error():
    def handler(request):
        raise httpx.ConnectError("refused")
    with client_with(handler) as c:
        assert c.health() is False


def test_index_and_currency_share_normalization():
    seen_paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        return httpx.Response(200, json=GOOD_PAYLOAD)

    with client_with(handler) as c:
        idx = c.index_historical("^GSPC", date(2026, 1, 1), date(2026, 1, 2))
        fx = c.currency_historical("EURUSD", date(2026, 1, 1), date(2026, 1, 2))
    assert seen_paths == [
        "/api/v1/index/price/historical",
        "/api/v1/currency/price/historical",
    ]
    assert idx.last_close == 11 and fx.last_close == 11
    assert idx.symbol == "^GSPC" and fx.symbol == "EURUSD"


def test_index_error_paths_match_crypto():
    with client_with(lambda r: httpx.Response(200, json={"results": []})) as c:
        with pytest.raises(OpenBBClientError, match="no data"):
            c.index_historical("^GSPC", date(2026, 1, 1), date(2026, 1, 2))
