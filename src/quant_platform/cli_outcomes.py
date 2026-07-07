"""quant-desk-outcomes: resolve pending journal memos against realized returns.

For each pending memo at least --horizon days old, fetches history and records
an OutcomeRecord with the realized return from the memo's as_of close to the
latest close. Closes the memo-vs-outcome loop (M6) without any LLM involvement.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from quant_platform.data.openbb_client import OpenBBClient, OpenBBClientError
from quant_platform.journal import DecisionJournal, OutcomeRecord
from quant_platform.observability import configure_json_logging

log = logging.getLogger("quant_platform.outcomes")


def record_outcomes(
    journal: DecisionJournal,
    client: OpenBBClient,
    horizon_days: int = 7,
    now: datetime | None = None,
) -> list[OutcomeRecord]:
    """Resolve every pending memo older than horizon_days. Returns new records."""
    now = now or datetime.now(timezone.utc)
    recorded: list[OutcomeRecord] = []
    for memo in journal.pending():
        as_of = date.fromisoformat(memo.context.as_of)
        age_days = (now.date() - as_of).days
        if age_days < horizon_days:
            continue
        try:
            history = client.crypto_historical(
                memo.symbol, as_of, now.date() + timedelta(days=1)
            )
        except OpenBBClientError as exc:
            log.warning(
                "outcome fetch failed", extra={"record_id": memo.record_id, "error": str(exc)}
            )
            continue
        realized = round((history.last_close / memo.context.last_close - 1.0) * 100.0, 2)
        outcome = OutcomeRecord(
            memo_record_id=memo.record_id,
            horizon_days=age_days,
            realized_return_pct=realized,
            notes=f"auto: {memo.context.last_close} -> {history.last_close} "
                  f"({memo.context.as_of} -> {history.bars[-1].date.isoformat()})",
        )
        journal.append_outcome(outcome)
        recorded.append(outcome)
        log.info(
            "outcome recorded",
            extra={
                "record_id": memo.record_id,
                "symbol": memo.symbol,
                "realized_return_pct": realized,
                "confidence_was": memo.confidence,
            },
        )
    return recorded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", default="reports/research/journal.jsonl")
    parser.add_argument("--openbb-url", default="http://127.0.0.1:6900")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()

    configure_json_logging()
    journal = DecisionJournal(Path(args.journal))
    with OpenBBClient(base_url=args.openbb_url) as client:
        if not client.health():
            print(f"OpenBB REST not reachable at {args.openbb_url}", file=sys.stderr)
            raise SystemExit(1)
        recorded = record_outcomes(journal, client, horizon_days=args.horizon)
    print(f"outcomes recorded: {len(recorded)}; still pending: {len(journal.pending())}")


if __name__ == "__main__":
    main()
