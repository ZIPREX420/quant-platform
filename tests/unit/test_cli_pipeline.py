"""Full desk pipeline with mock HTTP transport and stub LLM - no network, no keys."""
from datetime import date, timedelta

import httpx

from quant_platform.cli import run_desk
from quant_platform.data.cache import PriceHistoryCache
from quant_platform.data.openbb_client import OpenBBClient
from quant_platform.journal import DecisionJournal


def payload(n=400):
    base = date(2025, 1, 1)
    rows = []
    price = 100.0
    for i in range(n):
        price += 0.5
        rows.append({"date": (base + timedelta(days=i)).isoformat(),
                     "open": price - 0.2, "high": price + 1, "low": price - 1,
                     "close": price, "volume": 1000})
    return {"results": rows}


class FakeResp:
    def __init__(self, text):
        self.content = text
        self.usage_metadata = {"input_tokens": 200, "output_tokens": 100}


class StubLLM:
    def __init__(self):
        self.n = 0

    def invoke(self, messages):
        self.n += 1
        if self.n == 4:
            return FakeResp("## Regime\n...\n## Confidence\n**HIGH** - stub\n")
        return FakeResp(f"[view {self.n}]")


def test_run_desk_end_to_end(tmp_path):
    calls = {"http": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["http"] += 1
        return httpx.Response(200, json=payload())

    client = OpenBBClient(transport=httpx.MockTransport(handler))
    journal = DecisionJournal(tmp_path / "journal.jsonl")
    cache = PriceHistoryCache(tmp_path / "cache")

    memo_path, record_id = run_desk(
        symbol="BTCUSD", client=client, llm=StubLLM(), model_id="test/stub",
        journal=journal, out_dir=tmp_path / "out", cache=cache, enrich=False,
    )
    assert memo_path.is_file()
    text = memo_path.read_text(encoding="utf-8")
    assert "Confidence" in text and record_id in memo_path.name
    records = journal.memos()
    assert len(records) == 1
    assert records[0].confidence == "HIGH"
    assert records[0].model == "test/stub"
    assert len(records[0].usage) == 4
    assert records[0].context.symbol == "BTCUSD"

    # second run: cache hit - no additional HTTP call
    memo_path2, _ = run_desk(
        symbol="BTCUSD", client=client, llm=StubLLM(), model_id="test/stub",
        journal=journal, out_dir=tmp_path / "out", cache=cache, enrich=False,
    )
    assert calls["http"] == 1
    assert len(journal.memos()) == 2
