"""Regression: a funding-feed failure must refuse the cycle as a clean no-op.

Guards the invariant that funding for every funding-gated candidate is fetched
BEFORE any order is processed. A lazy fetch inside the trading loop could raise
AFTER an earlier candidate's fill was already appended to the append-only
execution audit, leaving the audit ahead of the (never-saved) paper state.
This reproduces exactly that ordering: a first candidate that enters and would
fill, and a second, funding-gated candidate whose funding feed is down.
"""
from __future__ import annotations

import json

import pytest

from quant_platform.cycle import run_cycle
from quant_platform.data.binance_client import BinanceClientError
from quant_platform.execution.session import ExecutionAudit
from quant_platform.execution.state import StateStore
from tests.unit.test_cycle import REPO, FakeClient, flat, paths, write_candidate


def _write_funding_gated_candidate(directory, symbol="ETHUSDT"):
    """A schema-valid, funding-gated candidate on its own symbol (funding leg
    shape mirrors config/candidates/flow-gated-trend-short.json)."""
    definition = json.loads(
        (REPO / "config/strategies/example-btc-trend.json").read_text(encoding="utf-8")
    )
    del definition["validation_report"]
    definition["id"] = "funding-gated-cand"
    definition["universe"]["symbols"] = [symbol]
    definition["signal"] = {
        "kind": "declarative-rules",
        "direction": "long",
        "parameters": {},
        "entry_rules": [
            {"indicator": "sma", "window": 9, "series": "funding",
             "operator": "less_than", "operand": 0},
        ],
        "exit_rules": [
            {"indicator": "close", "operator": "less_than", "operand": 90},
        ],
    }
    definition["risk"] = {
        "max_position_pct_equity": 5, "stop_loss_pct": 5,
        "max_gross_exposure_pct": 50, "max_daily_loss_pct": 2,
    }
    definition["data_dependencies"] = [
        {"series": "ohlcv", "frequency": "1h", "lookback_bars": 10},
        {"series": "funding", "frequency": "1h", "lookback_bars": 10},
    ]
    definition["tracking"] = {
        "prediction": (
            "Pre-registered pytest candidate: its only purpose is to exercise "
            "the funding-feed failure path during a paper cycle."
        ),
        "registered_by": "pytest",
        "registered_date": "2026-07-10",
    }
    (directory / "zz-funding-gated-cand.json").write_text(
        json.dumps(definition), encoding="utf-8"
    )
    return definition["id"]


class _FundingDownClient(FakeClient):
    """Klines behave normally; the funding feed is down for one symbol."""

    def __init__(self, ohlc, down_symbol="ETHUSDT"):
        super().__init__(ohlc)
        self._down_symbol = down_symbol

    def funding_rates(self, symbol, limit=100):
        if symbol == self._down_symbol:
            raise BinanceClientError(f"funding feed unreachable for {symbol}")
        return []


def test_funding_feed_failure_refuses_without_audit_leak(tmp_path):
    cands, state_path, audit_path = paths(tmp_path)
    # candidate A (sorts first): a plain long that WOULD enter and fill this cycle
    write_candidate(cands)  # 'cycle-test-cand.json', BTCUSDT, non-funding
    # candidate B (sorts later): funding-gated on ETHUSDT, whose funding feed is down
    _write_funding_gated_candidate(cands, symbol="ETHUSDT")

    # last closed bar 105 > 100 -> candidate A's entry condition is satisfied
    feed = _FundingDownClient(
        flat(95, 8) + [(105, 105, 104, 105), (106, 106, 106, 106)],
        down_symbol="ETHUSDT",
    )

    with pytest.raises(BinanceClientError):
        run_cycle(cands, state_path, audit_path, client=feed, now=feed.now_after_bars())

    # a refusal must be a clean no-op: no fill leaked into the append-only audit,
    # and no paper state was written (first run -> load() stays None).
    assert ExecutionAudit(audit_path).records() == []
    assert StateStore(state_path).load() is None
