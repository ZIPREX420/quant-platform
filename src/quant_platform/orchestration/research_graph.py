"""Research-desk orchestration: the 4-role LangGraph DAG with a budget guard.

    macro_analyst -> asset_analyst -> risk_reviewer -> editor

Governance properties (Paperclip-informed, per ADR-0004):
- hard token budget per run: the graph aborts before a node that would run
  after the cumulative input+output tokens exceed the budget;
- every role call is metered (tokens, seconds) into the run's usage log;
- agents produce text only - no tool access, no execution authority.
"""
from __future__ import annotations

import time

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from quant_platform.agents import prompts
from quant_platform.agents.providers import content_to_text, explain_provider_error

DEFAULT_TOKEN_BUDGET = 40_000


class BudgetExceeded(RuntimeError):
    """The run's token budget was exhausted before all roles completed."""


class DeskState(TypedDict):
    context_json: str
    stale_days: int
    macro_view: str
    asset_view: str
    risk_review: str
    memo: str


class ResearchGraph:
    """Compiled 4-role desk graph bound to one LLM and one usage log."""

    def __init__(self, llm, token_budget: int = DEFAULT_TOKEN_BUDGET) -> None:
        self._llm = llm
        self._budget = token_budget
        self.usage: list[dict] = []
        self._graph = self._build()

    def _spent(self) -> int:
        return sum((u.get("input_tokens") or 0) + (u.get("output_tokens") or 0) for u in self.usage)

    def _node(self, role_prompt: str, state_key: str):
        def node(state: DeskState) -> dict:
            if self._spent() >= self._budget:
                raise BudgetExceeded(
                    f"token budget {self._budget} exhausted before role '{state_key}' "
                    f"(spent {self._spent()})"
                )
            parts = [f"Market context (JSON):\n{state['context_json']}"]
            if state.get("macro_view"):
                parts.append(f"Macro analyst view:\n{state['macro_view']}")
            if state.get("asset_view"):
                parts.append(f"Asset analyst view:\n{state['asset_view']}")
            if state.get("risk_review"):
                parts.append(f"Risk review:\n{state['risk_review']}")
            system = role_prompt + "\n\n" + prompts.GROUND_RULES
            if state.get("stale_days", 0) > 1:
                system += "\n\n" + prompts.STALENESS_WARNING.format(days=state["stale_days"])
            t0 = time.time()
            try:
                resp = self._llm.invoke([("system", system), ("user", "\n\n---\n\n".join(parts))])
            except Exception as exc:
                raise RuntimeError(explain_provider_error(exc, state_key)) from exc
            meta = getattr(resp, "usage_metadata", None) or {}
            self.usage.append(
                {
                    "role": state_key,
                    "seconds": round(time.time() - t0, 1),
                    "input_tokens": meta.get("input_tokens"),
                    "output_tokens": meta.get("output_tokens"),
                }
            )
            return {state_key: content_to_text(resp.content)}

        return node

    def _build(self):
        g = StateGraph(DeskState)
        g.add_node("macro_analyst", self._node(prompts.MACRO_ANALYST, "macro_view"))
        g.add_node("asset_analyst", self._node(prompts.ASSET_ANALYST, "asset_view"))
        g.add_node("risk_reviewer", self._node(prompts.RISK_REVIEWER, "risk_review"))
        g.add_node("editor", self._node(prompts.EDITOR, "memo"))
        g.add_edge(START, "macro_analyst")
        g.add_edge("macro_analyst", "asset_analyst")
        g.add_edge("asset_analyst", "risk_reviewer")
        g.add_edge("risk_reviewer", "editor")
        g.add_edge("editor", END)
        return g.compile()

    def run(self, context_json: str, stale_days: int = 0) -> str:
        """Execute the desk once; returns the memo text."""
        final = self._graph.invoke(
            {"context_json": context_json, "stale_days": stale_days}
        )
        return final["memo"]
