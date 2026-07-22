"""Perp funding accrual for held paper positions (ADR-0007 cost tier).

The validation backtester already models funding for rule-driven strategies;
until now the LIVE paper cycle did not, so a position held across funding
settlements paid and received nothing. That flatters every carry-exposed
candidate - including the H1 falsification tracker, whose entire hypothesis
IS the funding flow. This module closes that gap.

Sign convention (Binance): a POSITIVE funding rate means longs pay shorts.
    long  position: cash_delta = -rate * quantity * settle_price
    short position: cash_delta = +rate * quantity * settle_price

Accrual is cursor-based per position (``last_funding_ts``), exactly like the
M13 open-interest capture: a missed cycle - or an offline weekend - is caught
up on the next run with no double-accrual, because only events strictly newer
than the cursor (or the position's ``entry_ts`` before any accrual) are
applied. The settle price is the close of the bar whose open matches the
funding event's timestamp when that bar is in the fetched window, else the
current mark price. Every accrual is returned as an audit row for the
``funding-accruals.jsonl`` sidecar; the executions.jsonl contract is untouched
so the forward-evidence analyzer keeps reading exactly what it always has.
"""
from __future__ import annotations

from datetime import datetime

from quant_platform.data.binance_client import FundingEvent
from quant_platform.execution.state import OpenPosition
from quant_platform.signals.rules import Bar


def accrue_open_positions(
    open_positions: dict[tuple[str, str], OpenPosition],
    funding_cache: dict[str, list[FundingEvent]],
    market: dict[str, tuple[list[Bar], str]],
    latest_price: dict[str, float],
    now: datetime,
) -> tuple[dict[tuple[str, str], OpenPosition], list[dict], float]:
    """Accrue unapplied funding events on every held position.

    Returns (updated positions by key, audit rows, total cash delta). Pure
    function: no I/O, no clock reads - fully deterministic and unit-testable.
    Positions without new events are omitted from the returned dict.
    """
    updated: dict[tuple[str, str], OpenPosition] = {}
    rows: list[dict] = []
    total_delta = 0.0
    for key, pos in open_positions.items():
        events = funding_cache.get(pos.symbol, [])
        if not events:
            continue
        cursor = pos.last_funding_ts or pos.entry_ts
        due = [e for e in events if cursor < e.funding_time <= now]
        if not due:
            continue
        bar_close_by_open = _bar_closes(market.get(pos.symbol))
        pos_delta = 0.0
        last_ts = cursor
        for event in due:
            settle_price = bar_close_by_open.get(
                event.funding_time.strftime("%Y-%m-%dT%H:%M"),
                latest_price.get(pos.symbol),
            )
            if settle_price is None:
                continue  # no price reference at all - leave for next cycle
            sign = -1.0 if pos.direction == "long" else 1.0
            delta = sign * event.rate * pos.quantity * settle_price
            pos_delta += delta
            last_ts = event.funding_time
            rows.append({
                "ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "candidate_id": pos.candidate_id,
                "symbol": pos.symbol,
                "direction": pos.direction,
                "funding_time": event.funding_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rate": event.rate,
                "quantity": pos.quantity,
                "settle_price": settle_price,
                "cash_delta": round(delta, 8),
            })
        if last_ts == cursor:
            continue
        total_delta += pos_delta
        updated[key] = pos.model_copy(update={
            "last_funding_ts": last_ts,
            "funding_net": round(pos.funding_net + pos_delta, 8),
        })
    return updated, rows, round(total_delta, 8)


def _bar_closes(entry: tuple[list[Bar], str] | None) -> dict[str, float]:
    if entry is None:
        return {}
    bars, _interval = entry
    return {bar.date: bar.close for bar in bars}
