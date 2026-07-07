"""LLM provider factory and response handling for the research desk.

Operational lessons baked in (learned during the ADR-0004 spike):
- current reasoning models reject `temperature` (400) - only send it when
  explicitly configured;
- message content may arrive as a list of content blocks - normalize at this
  boundary;
- provider errors are translated into one actionable line before the raw
  detail, so operators are never left with a bare traceback.
"""
from __future__ import annotations

import os

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5.4-mini",
}


class DeskProviderError(RuntimeError):
    """LLM call failed; message contains an actionable diagnosis."""


def make_llm(max_tokens: int = 1500):
    """Build a langchain chat model from environment configuration.

    Returns (llm, model_id). Env: ANTHROPIC_API_KEY / OPENAI_API_KEY,
    optional DESK_PROVIDER, DESK_MODEL, DESK_TEMPERATURE.
    """
    provider = os.environ.get("DESK_PROVIDER")
    if not provider:
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.environ.get("OPENAI_API_KEY"):
            provider = "openai"
        else:
            raise DeskProviderError(
                "No LLM key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "(optionally DESK_PROVIDER / DESK_MODEL)."
            )
    model = os.environ.get("DESK_MODEL", DEFAULT_MODELS[provider])
    kwargs = {}
    if os.environ.get("DESK_TEMPERATURE"):
        kwargs["temperature"] = float(os.environ["DESK_TEMPERATURE"])
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, max_tokens=max_tokens, **kwargs), f"anthropic/{model}"
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, max_completion_tokens=max_tokens, **kwargs), f"openai/{model}"


def content_to_text(content) -> str:
    """Normalize message content (str, or list of blocks) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(p for p in parts if p)
    return str(content)


def explain_provider_error(exc: Exception, role: str) -> str:
    """One actionable line per known failure mode; full detail follows."""
    msg = str(exc)
    if "insufficient_quota" in msg or "exceeded your current quota" in msg:
        hint = (
            "No API credits on this account. Add prepaid credit "
            "(platform.openai.com/billing or console.anthropic.com), or switch provider."
        )
    elif "model_not_found" in msg or "does not exist or you do not have access" in msg:
        hint = "Model unavailable to this key. Override with DESK_MODEL=<model>."
    elif "401" in msg or "invalid_api_key" in msg or "authentication" in msg.lower():
        hint = "The API key was rejected (401). Check for copy/paste truncation."
    elif "deprecated" in msg and "400" in msg:
        hint = (
            "The provider rejected a request parameter as deprecated for this model. "
            "Re-run without overrides, or set DESK_MODEL to a different model."
        )
    elif "429" in msg:
        hint = "Rate limited (429). Wait a minute and re-run."
    else:
        hint = "Unrecognized provider error; full detail below."
    return (
        f"DESK-ERROR at role '{role}': {hint}\n"
        f"--- provider detail ---\n{type(exc).__name__}: {msg[:800]}"
    )
