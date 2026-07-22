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
from quant_platform.execution.funding import accrue_open_positions
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


def _vol_scale(bars: list[Bar], interval: str, vol_target_annual_pct: float | None) -> float:
    """Risk overlay (M12): scale exposure by min(1, target/realized vol).

    Never scales ABOVE 1 (no leveraging up); insufficient data -> full scale
    (the hard caps still bound everything).
    """
    if vol_target_annual_pct is None:
        return 1.0
    closes = [b.close for b in bars[-31:]]
    if len(closes) < 11:
        return 1.0
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    from statistics import pstdev  # noqa: PLC0415
    per_year = (365 * 86400) / INTERVAL_SECONDS[interval]
    realized = pstdev(rets) * (per_year ** 0.5) * 100.0
    if realized <= 0:
        return 1.0
    return min(1.0, vol_target_annual_pct / realized)


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


def _candidate_symbols_and_dep(candidate: LoadedCandidate) -> tuple[list[str], str, int]:
    """All universe symbols (M12: multi-symbol supported), shared ohlcv dependency."""
    symbols = candidate.definition["universe"]["symbols"]
    ohlcv = [d for d in candidate.definition["data_dependencies"] if d["series"] == "ohlcv"]
    if len(ohlcv) != 1:
        raise CycleError(f"{candidate.id}: exactly one ohlcv data dependency required")
    dep = ohlcv[0]
    lookback = int(dep.get("lookback_bars", DEFAULT_LOOKBACK))
    return list(symbols), dep["frequency"], min(lookback + 2, 1000)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to a JSONL sidecar (mkdir on first write)."""
    import json as _json  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(_json.dumps(row) + "\n")


OI_CATCHUP_POINTS = 48  # 1h points fetched per cycle: PC downtime <= 2 days loses nothing


def _record_market_structure(client, symbols, path: Path, now: datetime) -> int:
    """Append OI + basis points (M13 forward accumulation), gap-tolerant.

    Open interest CANNOT be backfilled beyond ~30 days, so each cycle fetches
    a 48-point window and writes only points NEWER than a per-symbol cursor
    (sidecar .cursor.json): a missed cycle - or a whole offline weekend - is
    caught up on the next run, with no duplicate rows. Basis is backfillable
    at the venue, so only its latest closed value is snapshotted. Best-effort:
    never affects the cycle result. Returns rows written.
    """
    import json as _json  # noqa: PLC0415

    cursor_path = path.with_suffix(".cursor.json")
    try:
        cursors: dict = _json.loads(cursor_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        cursors = {}

    rows = []
    for symbol in sorted(symbols):
        last_seen = cursors.get(symbol, "")
        try:
            for point in client.open_interest_hist(symbol, period="1h", limit=OI_CATCHUP_POINTS):
                oi_ts = point.ts.strftime("%Y-%m-%dT%H:%M:%SZ")
                if oi_ts <= last_seen:
                    continue
                rows.append({
                    "cycle_ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "symbol": symbol,
                    "oi": point.open_interest, "oi_value": point.open_interest_value,
                    "oi_ts": oi_ts,
                })
                last_seen = oi_ts
            cursors[symbol] = last_seen
        except Exception:  # noqa: BLE001 - forward recording is best-effort
            pass
        try:
            pk = client.premium_index_klines(symbol, "1h", limit=2, now=now)
            if pk:
                rows.append({
                    "cycle_ts": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "symbol": symbol,
                    "basis_close": pk[-1].close,
                    "basis_ts": pk[-1].close_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                })
        except Exception:  # noqa: BLE001
            pass

    if rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(_json.dumps(row) + "\n")
        tmp = cursor_path.with_suffix(".json.tmp")
        tmp.write_text(_json.dumps(cursors, indent=2), encoding="utf-8")
        tmp.replace(cursor_path)
    return len(rows)


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
    open_positions: dict[tuple[str, str], OpenPosition] = {
        (p.candidate_id, p.symbol): p for p in state.open_positions
    }
    registered = {c.id for c in candidates}
    orphans = sorted({cid for cid, _ in open_positions if cid not in registered})
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
            symbols, interval, limit = _candidate_symbols_and_dep(candidate)
            for symbol in symbols:
                if symbol not in market:
                    klines = client.klines(
                        symbol, interval, limit=limit, include_unclosed=True, now=now
                    )
                    closed = [k for k in klines if k.close_time <= now]
                    if len(closed) < 2:
                        raise CycleError(f"{symbol}: fewer than 2 closed bars returned")
                    latest_price[symbol] = klines[-1].close  # forming bar close = live price
                    market[symbol] = (_to_rule_bars(closed), interval)

        funding_cache: dict[str, list] = {}

        # Accrue perp funding on every held position (ADR-0007 cost tier).
        # Cursor-based per position, so downtime is caught up like M13's OI
        # capture; failures leave the cursor unmoved and retry next cycle.
        if open_positions:
            for symbol in sorted({p.symbol for p in open_positions.values()}):
                if symbol not in funding_cache:
                    try:
                        funding_cache[symbol] = client.funding_rates(symbol, limit=1000)
                    except Exception:  # noqa: BLE001 - accrual retries next cycle
                        funding_cache[symbol] = []
            accrued, accrual_rows, cash_delta = accrue_open_positions(
                open_positions, funding_cache, market, latest_price, now
            )
            if accrual_rows:
                account.cash += cash_delta
                open_positions.update(accrued)
                _append_jsonl(
                    Path(audit_path).parent / "funding-accruals.jsonl", accrual_rows
                )

        # daily kill-switch anchor: reset at UTC day rollover
        anchor_date, anchor_equity = state.day_anchor_date, state.day_anchor_equity
        if anchor_date != today or anchor_equity is None:
            anchor_date, anchor_equity = today, account.equity(latest_price) if latest_price else account.cash
        for candidate in candidates:
          symbols, interval, _ = _candidate_symbols_and_dep(candidate)
          for symbol in symbols:
            bars, _ = market[symbol]
            if _rules_use_funding(candidate.definition["signal"]):
                if symbol not in funding_cache:
                    funding_cache[symbol] = client.funding_rates(symbol, limit=1000)
                bars = attach_funding(
                    bars,
                    [(e.funding_time.strftime("%Y-%m-%dT%H:%M"), e.rate)
                     for e in funding_cache[symbol]],
                )

            position = open_positions.get((candidate.id, symbol))
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
                scale = _vol_scale(
                    bars, interval, candidate.definition["risk"].get("vol_target_annual_pct")
                )
                record = session.process_signal(
                    symbol, open_side, target * scale, latest_price, anchor_equity, sanity=sanity
                )
                if record.approved and record.fill:
                    stop_frac = candidate.definition["risk"]["stop_loss_pct"] / 100.0
                    stop = record.fill["fill_price"] * (
                        (1.0 - stop_frac) if direction == "long" else (1.0 + stop_frac)
                    )
                    open_positions[(candidate.id, symbol)] = OpenPosition(
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
                        del open_positions[(candidate.id, symbol)]
                    else:  # risk engine shrank the close - keep the remainder tracked
                        open_positions[(candidate.id, symbol)] = position.model_copy(
                            update={"quantity": remaining}
                        )
                results.append(CandidateResult(
                    candidate.id, symbol, "exit", decision.reason,
                    approved=record.approved,
                    fill_price=record.fill["fill_price"] if record.fill else None,
                ))
        if market:  # M13: forward-accumulate what cannot be backfilled
            _record_market_structure(
                client, list(market), Path(audit_path).parent / "market-structure.jsonl", now
            )
    finally:
        if owns_client:
            client.close()

    equity = account.equity(latest_price) if latest_price else account.cash
    new_state = PaperState.from_account(
        account,
        tuple(sorted(open_positions.values(), key=lambda p: (p.candidate_id, p.symbol))),
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
