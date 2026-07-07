"""quant-desk: generate one research memo end to end.

Pipeline: OpenBB REST (via cache) -> MarketContext -> ResearchGraph -> memo
file + DecisionJournal entry. Requires the OpenBB REST service (see workspace
automation/bootstrap/m4-openbb-api.bat) and an LLM key in the environment.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from quant_platform.data.cache import PriceHistoryCache
from quant_platform.data.context import build_market_context
from quant_platform.data.openbb_client import OpenBBClient
from quant_platform.journal import DecisionJournal, MemoRecord, extract_confidence
from quant_platform.observability import configure_json_logging

log = logging.getLogger("quant_platform.desk")

CACHE_MAX_AGE_HOURS = 6


def run_desk(
    symbol: str,
    client: OpenBBClient,
    llm,
    model_id: str,
    journal: DecisionJournal,
    out_dir: Path,
    cache: PriceHistoryCache | None = None,
    lookback_days: int = 400,
    token_budget: int | None = None,
) -> tuple[Path, str]:
    """Run one desk cycle. Returns (memo_path, journal_record_id)."""
    from quant_platform.orchestration.research_graph import DEFAULT_TOKEN_BUDGET, ResearchGraph

    end = date.today()
    start = end - timedelta(days=lookback_days)

    history = None
    if cache is not None:
        history = cache.get(
            symbol, "openbb/yfinance", start, end, max_age=timedelta(hours=CACHE_MAX_AGE_HOURS)
        )
        if history is not None:
            log.info("cache hit", extra={"symbol": symbol})
    if history is None:
        history = client.crypto_historical(symbol, start, end)
        if cache is not None:
            cache.put(history, start, end)

    context = build_market_context(history)
    if context.stale_days > 1:
        log.warning("stale data", extra={"symbol": symbol, "stale_days": context.stale_days})

    graph = ResearchGraph(llm, token_budget=token_budget or DEFAULT_TOKEN_BUDGET)
    memo = graph.run(context.model_dump_json(indent=2), stale_days=context.stale_days)

    record = MemoRecord(
        symbol=symbol,
        context=context,
        memo=memo,
        model=model_id,
        confidence=extract_confidence(memo),
        usage=graph.usage,
    )
    record_id = journal.append_memo(record)

    out_dir.mkdir(parents=True, exist_ok=True)
    memo_path = out_dir / f"memo-{symbol}-{context.as_of}-{record_id}.md"
    header = (
        f"# Research memo - {symbol} - {context.as_of}\n\n"
        f"*Model {model_id} | journal record {record_id} | data staleness "
        f"{context.stale_days}d | tokens {sum((u.get('input_tokens') or 0) + (u.get('output_tokens') or 0) for u in graph.usage)}*\n\n"
    )
    memo_path.write_text(header + memo + "\n", encoding="utf-8")
    log.info(
        "memo written",
        extra={"symbol": symbol, "record_id": record_id, "confidence": record.confidence},
    )
    return memo_path, record_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", nargs="?", default="BTCUSD")
    parser.add_argument("--openbb-url", default="http://127.0.0.1:6900")
    parser.add_argument("--out-dir", default="reports/research")
    parser.add_argument("--journal", default="reports/research/journal.jsonl")
    parser.add_argument("--cache-dir", default="datasets/cache")
    args = parser.parse_args()

    configure_json_logging()
    from quant_platform.agents.providers import DeskProviderError, make_llm

    try:
        llm, model_id = make_llm()
    except DeskProviderError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    with OpenBBClient(base_url=args.openbb_url) as client:
        if not client.health():
            print(
                f"OpenBB REST not reachable at {args.openbb_url} - start it with "
                "automation/bootstrap/m4-openbb-api.bat",
                file=sys.stderr,
            )
            raise SystemExit(1)
        memo_path, record_id = run_desk(
            symbol=args.symbol,
            client=client,
            llm=llm,
            model_id=model_id,
            journal=DecisionJournal(args.journal),
            out_dir=Path(args.out_dir),
            cache=PriceHistoryCache(args.cache_dir),
        )
    print(f"memo: {memo_path}\njournal record: {record_id}")


if __name__ == "__main__":
    main()
