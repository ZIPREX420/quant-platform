"""BinanceClient against a mock transport (no network, no keys)."""
from datetime import datetime, timezone

import httpx
import pytest

from quant_platform.data.binance_client import BinanceClient, BinanceClientError

H = 3_600_000  # one hour in ms
T0 = 1_750_000_000_000  # fixed epoch-ms anchor for deterministic tests


def kline_row(i: int, close: float = 100.0) -> list:
    open_ms = T0 + i * H
    return [open_ms, "99.0", "101.0", "98.0", str(close), "12.5", open_ms + H - 1,
            "0", 0, "0", "0", "0"]


def make_client(handler) -> BinanceClient:
    return BinanceClient(transport=httpx.MockTransport(handler))


def spot_handler(rows):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v3/klines"
        return httpx.Response(200, json=rows)
    return handler


def test_klines_normalized_and_ascending():
    rows = [kline_row(i, close=100.0 + i) for i in range(5)]
    now = datetime.fromtimestamp((T0 + 10 * H) / 1000, tz=timezone.utc)
    with make_client(spot_handler(rows)) as client:
        bars = client.klines("BTCUSDT", "1h", limit=5, now=now)
    assert len(bars) == 5
    assert bars[0].close == 100.0 and bars[-1].close == 104.0
    assert bars[0].open_time.tzinfo is not None


def test_unclosed_final_bar_dropped():
    rows = [kline_row(i) for i in range(3)]
    # 'now' falls INSIDE the third bar -> it is still forming and must be dropped
    now = datetime.fromtimestamp((T0 + 2 * H + H // 2) / 1000, tz=timezone.utc)
    with make_client(spot_handler(rows)) as client:
        bars = client.klines("BTCUSDT", "1h", limit=3, now=now)
        assert len(bars) == 2
        all_bars = client.klines("BTCUSDT", "1h", limit=3, now=now, include_unclosed=True)
        assert len(all_bars) == 3


def test_unsupported_interval_refused():
    with make_client(spot_handler([])) as client:
        with pytest.raises(BinanceClientError, match="unsupported interval"):
            client.klines("BTCUSDT", "3m")


def test_malformed_kline_row_refused():
    rows = [kline_row(0), [123, "not-enough-fields"]]
    with make_client(spot_handler(rows)) as client:
        with pytest.raises(BinanceClientError, match="malformed kline row"):
            client.klines("BTCUSDT", "1h")


def test_http_error_translated():
    def handler(request):
        return httpx.Response(451, text="unavailable for legal reasons")
    with make_client(handler) as client:
        with pytest.raises(BinanceClientError, match="returned 451"):
            client.klines("BTCUSDT", "1h")


def test_transport_failure_translated():
    def handler(request):
        raise httpx.ConnectError("boom")
    with make_client(handler) as client:
        with pytest.raises(BinanceClientError, match="transport failure"):
            client.klines("BTCUSDT", "1h")


def test_funding_rates_sorted_and_typed():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/fapi/v1/fundingRate"
        return httpx.Response(200, json=[
            {"symbol": "BTCUSDT", "fundingTime": T0 + 8 * H, "fundingRate": "0.0002"},
            {"symbol": "BTCUSDT", "fundingTime": T0, "fundingRate": "-0.0001"},
        ])
    with make_client(handler) as client:
        events = client.funding_rates("BTCUSDT", limit=2)
    assert [e.rate for e in events] == [-0.0001, 0.0002]  # oldest first
    assert events[0].funding_time < events[1].funding_time


def test_non_array_payload_refused():
    def handler(request):
        return httpx.Response(200, json={"code": -1121, "msg": "Invalid symbol."})
    with make_client(handler) as client:
        with pytest.raises(BinanceClientError, match="expected a JSON array"):
            client.klines("BTCUSDT", "1h")
