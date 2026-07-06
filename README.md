# quant-platform

Primary development repository of PROJECT GENESIS: an institutional-grade, AI-powered
cryptocurrency trading platform. All original integration code, custom agents,
orchestration, execution engine, risk engine, APIs, monitoring, and deployment glue
live here. Upstream projects (OpenBB, TradingAgents, Paperclip, QuantDinger) are
consumed as external systems — see ../../docs/architecture/repository-map.md.

## Layout
- src/agents         — custom AI research agents (reasoning only; no execution authority)
- src/orchestration  — workflow coordination between agents
- src/data           — data service boundary to OpenBB (API boundary; AGPL isolation, see risk R-1)
- src/risk           — deterministic risk engine (never LLM-driven)
- src/execution      — execution adapters (paper-first; live gated by Phase 10 criteria)
- src/api            — platform APIs
- src/monitoring     — observability
- tests/             — unit + integration
- config/            — configuration (secrets never committed)
- deploy/            — deployment definitions

## Principles
Correctness, reliability, security, statistical rigor. No profitability claims.
Every strategy is untrusted until validated (workspace docs/validation/).
