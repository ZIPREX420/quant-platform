"""The paper-trading cycle (M9): live data -> signals -> risk -> paper fills.

One cycle = for every registered candidate: fetch closed bars, evaluate its
rules with the shared machinery, route any signal through the RiskEngine and
PaperExchange inside an audited PaperTradingSession, then persist state
atomically. Idempotent: running twice inside the same bar re-evaluates the
same closed bar and (holding) produces no duplicate entries, because entry
signals are crossings/conditions on the same bar and positions are tracked.

Safety: candidates only (ADR-0006), paper only (ExecutionMode has one member),
public data only (BinanceClient has no keys), audit before state save so a
crash between the two is detectable (audit longer than state.cycle_count).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.data.binance_client import BinanceClient, KlineBar
from quant_platform.execution.paper import PaperAccount
from quant_platform.execution.session import ExecutionAudit, PaperTradingSession
from quant_platform.execution.state import OpenPosition, PaperState, StateStore
from quant_platform.risk.engine import CheckResult, Side
from quant_platform.signals.evaluator import attach_funding, evaluate_candidate
from quant_platform.signals.rules import Bar
from quant_platform.strategies.candidates import LoadedCandidate, load_candidate_dir

INTERVAL_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
DEFAULT_LOOKBACK = 500
MAX_JUMP_PCT = 25.0  # bar-to-bar move beyond this freezes the candidate (bad feed guard)


class CycleError(Exception):
    """The cycle refused to run or to continue. The message states why."""


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    symbol: str
    action: str          # enter | exit | hold | frozen
    reason: str
    approved: bool | None = None   # None when no order was attempted
    fill_price: float | None = None


@dataclass(frozen=True)
class CycleReport:
    cycle_count: int
    ran_at: str
    equity: float
    cash: float
    results: tuple[CandidateResult, ...]

    def summary_line(self) -> str:
        actions = ", ".join(f"{r.candidate_id}:{r.action}" for r in self.results) or "no candidates"
        return (
            f"cycle {self.cycle_count} @ {self.ran_at} | equity {self.equity:.2f} "
            f"| cash {self.cash:.2f} | {actions}"
        )


def _rules_use_funding(signal: dict) -> bool:
    for rule in list(signal["entry_rules"]) + list(signal["exit_rules"]):
        if rule.get("series") == "funding":
            return True
        operand = rule.get("operand")
        if isinstance(operand, dict) and operand.get("series") == "funding":
            return True
    return False


def _to_rule_bars(klines: list[KlineBar]) -> list[Bar]:
    return [
        Bar(
            date=k.open_time.strftime("%Y-%m-%dT%H:%M"),
            open=k.open, high=k.high, low=k.low, close=k.close,
        )
        for k in klines
    ]


def _sanity_checks(bars: list[Bar], interval: str, now: datetime) -> list[CheckResult]:
    """Live-feed guards: fresh last bar, no absurd bar-to-bar jump."""
    checks = []
    last_open = datetime.strptime(bars[-1].date, "%Y-%m-%dT%H:%M").replace(tzinfo=timezone.utc)
    age = (now - last_open).total_seconds()
    limit = 3 * INTERVAL_SECONDS[interval]
    checks.append(CheckResult(
        "bar_freshness", age <= limit,
        f"last closed bar opened {age:.0f}s ago (limit {limit}s for {interval})",
    ))
    move = abs(bars[-1].close / bars[-2].close - 1.0) * 100.0
    checks.append(CheckResult(
        "price_jump", move <= MAX_JUMP_PCT,
        f"last bar-to-bar move {move:.2f}% (limit {MAX_JUMP_PCT}%)",
    ))
    return checks


def _candidate_symbol_and_dep(candidate: LoadedCandidate) -> tuple[str, str, int]:
    symbols = candidate.definition["universe"]["symbols"]
    if len(symbols) != 1:
        raise CycleError(
            f"{candidate.id}: cycle v1 requires exactly one symbol per candidate, got {symbols}"
        )
    ohlcv = [d for d in candidate.definition["data_dependencies"] if d["series"] == "ohlcv"]
    if len(ohlcv) != 1:
        raise CycleError(f"{candidate.id}: exactly one ohlcv data dependency required")
    dep = ohlcv[0]
    lookback = int(dep.get("lookback_bars", DEFAULT_LOOKBACK))
    return symbols[0], dep["frequency"], min(lookback + 2, 1000)


def run_cycle(
    candidates_dir: Path | str,
    state_path: Path | str,
    audit_path: Path | str,
    client: BinanceClient | None = None,
    starting_cash: float = 10_000.0,
    now: datetime | None = None,
) -> CycleReport:
    now = now or datetime.now(timezone.utc)
    candidates = load_candidate_dir(candidates_dir)

    store = StateStore(state_path)
    state = store.load()
    if state is None:
        state = PaperState.fresh(starting_cash)
    account: PaperAccount = state.restore_account()
    open_by_candidate: dict[str, OpenPosition] = {
        p.candidate_id: p for p in state.open_positions
    }
    registered = {c.id for c in candidates}
    orphans = sorted(cid for cid in open_by_candidate if cid not in registered)
    if orphans:
        raise CycleError(
            f"open paper positions belong to unregistered candidates {orphans} - "
            f"re-register them or close the positions explicitly before cycling "
            f"(refusing to orphan positions silently)."
        )

    owns_client = client is None
    client = client or BinanceClient()
    audit = ExecutionAudit(audit_path)
    results: list[CandidateResult] = []
    today = now.strftime("%Y-%m-%d")

    try:
        # mark prices for every symbol we hold or watch (single fetch per symbol)
        latest_price: dict[str, float] = {}
        market: dict[str, tuple[list[Bar], str]] = {}
        for candidate in candidates:
            symbol, interval, limit = _candidate_symbol_and_dep(candidate)
            if symbol not in market:
                klines = client.klines(symbol, interval, limit=limit, include_unclosed=True, now=now)
                closed = [k for k in klines if k.close_time <= now]
                if len(closed) < 2:
                    raise CycleError(f"{symbol}: fewer than 2 closed bars returned")
                latest_price[symbol] = klines[-1].close  # forming bar close = live price
                market[symbol] = (_to_rule_bars(closed), interval)

        # daily kill-switch anchor: reset at UTC day rollover
        anchor_date, anchor_equity = state.day_anchor_date, state.day_anchor_equity
        if anchor_date != today or anchor_equity is None:
            anchor_date, anchor_equity = today, account.equity(latest_price) if latest_price else account.cash

        for candidate in candidates:
            symbol, interval, _ = _candidate_symbol_and_dep(candidate)
            bars, _ = market[symbol]
            if _rules_use_funding(candidate.definition["signal"]):
                events = client.funding_rates(symbol, limit=1000)
                bars = attach_funding(
                    bars,
                    [(e.funding_time.strftime("%Y-%m-%dT%H:%M"), e.rate) for e in events],
                )

            position = open_by_candidate.get(candidate.id)
            decision = evaluate_candidate(
                candidate.definition["signal"],
                bars,
                in_position=position is not None,
                stop_price=position.stop_price if position else None,
            )
            sanity = _sanity_checks(bars, interval, now)

            if decision.action == "hold":
                results.append(CandidateResult(candidate.id, symbol, "hold", decision.reason))
                continue

            price = latest_price[symbol]
            direction = candidate.definition["signal"].get("direction", "long")
            if decision.action == "enter":
                equity = account.equity(latest_price)
                target = equity * candidate.definition["risk"]["max_position_pct_equity"] / 100.0
                session = PaperTradingSession(candidate, account, audit)
                open_side = Side.BUY if direction == "long" else Side.SELL
                record = session.process_signal(
                    symbol, open_side, target, latest_price, anchor_equity, sanity=sanity
                )
                if record.approved and record.fill:
                    stop_frac = candidate.definition["risk"]["stop_loss_pct"] / 100.0
                    stop = record.fill["fill_price"] * (
                        (1.0 - stop_frac) if direction == "long" else (1.0 + stop_frac)
                    )
                    open_by_candidate[candidate.id] = OpenPosition(
                        candidate_id=candidate.id,
                        symbol=symbol,
                        direction=direction,
                        quantity=record.fill["quantity"],
                        entry_price=record.fill["fill_price"],
                        entry_ts=now,
                        stop_price=round(stop, 8),
                        entry_fill_id=record.fill["fill_id"],
                    )
                results.append(CandidateResult(
                    candidate.id, symbol, "enter", decision.reason,
                    approved=record.approved,
                    fill_price=record.fill["fill_price"] if record.fill else None,
                ))
            else:  # exit
                session = PaperTradingSession(candidate, account, audit)
                close_side = Side.SELL if position.direction == "long" else Side.BUY
                record = session.process_signal(
                    symbol, close_side, position.quantity * price, latest_price,
                    anchor_equity, sanity=sanity, close_quantity=position.quantity,
                )
                if record.approved and record.fill:
                    remaining = position.quantity - record.fill["quantity"]
                    if remaining <= 1e-10:
                        del open_by_candidate[candidate.id]
                    else:  # risk engine shrank the close - keep the remainder tracked
                        open_by_candidate[candidate.id] = position.model_copy(
                            update={"quantity": remaining}
                        )
                results.append(CandidateResult(
                    candidate.id, symbol, "exit", decision.reason,
                    approved=record.approved,
                    fill_price=record.fill["fill_price"] if record.fill else None,
                ))
    finally:
        if owns_client:
            client.close()

    equity = account.equity(latest_price) if latest_price else account.cash
    new_state = PaperState.from_account(
        account,
        tuple(sorted(open_by_candidate.values(), key=lambda p: p.candidate_id)),
        cycle_count=state.cycle_count + 1,
        day_anchor_date=anchor_date,
        day_anchor_equity=anchor_equity,
        last_equity=round(equity, 8),
    )
    store.save(new_state)

    return CycleReport(
        cycle_count=new_state.cycle_count,
        ran_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        equity=round(equity, 2),
        cash=round(account.cash, 2),
        results=tuple(results),
    )
