"""ResearchGraph: role sequence, state accumulation, staleness injection, budget guard."""
import pytest

from quant_platform.orchestration.research_graph import BudgetExceeded, ResearchGraph

CONTEXT_JSON = '{"symbol": "BTC-USD", "last_close": 64000.0}'


class FakeResp:
    def __init__(self, content, tokens=100):
        self.content = content
        self.usage_metadata = {"input_tokens": tokens, "output_tokens": tokens // 2}


class StubLLM:
    def __init__(self, tokens=100, block_content=False):
        self.calls = []
        self.tokens = tokens
        self.block_content = block_content

    def invoke(self, messages):
        self.calls.append(messages)
        n = len(self.calls)
        content = [{"type": "text", "text": f"[role {n}]"}] if self.block_content else f"[role {n}]"
        return FakeResp(content, self.tokens)


def test_four_roles_and_state_accumulation():
    llm = StubLLM()
    graph = ResearchGraph(llm)
    memo = graph.run(CONTEXT_JSON)
    assert memo == "[role 4]"
    assert len(llm.calls) == 4 and len(graph.usage) == 4
    final_user = llm.calls[3][1][1]
    assert "64000.0" in final_user
    assert "Macro analyst view" in final_user and "Risk review" in final_user


def test_block_content_normalized():
    graph = ResearchGraph(StubLLM(block_content=True))
    assert graph.run(CONTEXT_JSON) == "[role 4]"


def test_staleness_warning_injected():
    llm = StubLLM()
    ResearchGraph(llm).run(CONTEXT_JSON, stale_days=3)
    assert "STALENESS WARNING" in llm.calls[0][0][1]
    llm2 = StubLLM()
    ResearchGraph(llm2).run(CONTEXT_JSON, stale_days=0)
    assert "STALENESS WARNING" not in llm2.calls[0][0][1]


def test_budget_guard_aborts():
    llm = StubLLM(tokens=1000)  # 1500 tokens per role
    graph = ResearchGraph(llm, token_budget=2000)
    with pytest.raises(BudgetExceeded, match="budget 2000 exhausted"):
        graph.run(CONTEXT_JSON)
    assert len(llm.calls) < 4  # aborted mid-run


def test_provider_error_wrapped_with_diagnosis():
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("Error code: 429 - rate limited")

    with pytest.raises(RuntimeError, match="DESK-ERROR at role 'macro_view'"):
        ResearchGraph(FailingLLM()).run(CONTEXT_JSON)
