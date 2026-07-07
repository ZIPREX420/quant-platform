# quant-platform

Primary development repository of PROJECT GENESIS: an institutional-grade, AI-powered
cryptocurrency trading platform. All original integration code, custom agents,
orchestration, execution engine, risk engine, APIs, monitoring, and deployment glue
live here. Upstream projects (OpenBB, TradingAgents, Paperclip, QuantDinger) are
consumed as external systems — see ../../docs/architecture/repository-map.md.

## Layout (src-layout package: quant_platform)
- src/quant_platform/agents         — custom AI research agents (reasoning only; no execution authority)
- src/quant_platform/orchestration  — workflow coordination between agents
- src/quant_platform/data           — data service boundary to OpenBB (REST only; AGPL isolation, R-1)
- src/quant_platform/risk           — deterministic risk engine (never LLM-driven)
- src/quant_platform/execution      — execution adapters (paper-first; live gated by Phase 10 criteria)
- src/quant_platform/api            — platform APIs
- src/quant_platform/monitoring     — observability
- tests/                            — unit + integration (run: PYTHONPATH=src pytest)
- config/strategies/                — ADR-0005 strategy artifacts + JSON Schema
- deploy/                           — deployment definitions

## Principles
Correctness, reliability, security, statistical rigor. No profitability claims.
Every strategy is untrusted until validated (workspace docs/validation/).
