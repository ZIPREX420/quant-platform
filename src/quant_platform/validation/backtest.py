"""Reference backtester for ADR-0005 declarative-rules strategies.

Purpose: reproducible protocol validation of the declarative signal contract
(strategy.schema.json). Long-only, one position, full-equity-fraction sizing.

Execution model (protocol v1 assumptions, deliberately pessimistic):
- signals evaluated on bar close; fills at NEXT bar open (no look-ahead);
- slippage against the trader on every fill; fee per side on notional;
- last open position is force-closed at the final bar's open (reported).

This is validation tooling: it shares the rule vocabulary with the schema and
is never imported by the execution path.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_platform.validation.trades import Trade

INDICATORS = ("close", "sma", "ema", "rsi", "atr", "high_n", "low_n")


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    funding: float | None = None  # perp funding-rate event occurring in this bar, if any


@dataclass(frozen=True)
class BacktestTrade:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_fraction: float  # net of fees and slippage
    forced_exit: bool = False
    stopped_out: bool = False

    def as_trade(self) -> Trade:
        return Trade(return_fraction=self.return_fraction)


class RuleError(ValueError):
    """The rule set references something this reference implementation lacks."""


def _funding_series(kind: str, window: int | None, bars: list[Bar]) -> list[float | None]:
    """Funding-series indicators. Windows count funding EVENTS (e.g. 9 = 3 days
    at 8h funding), not bars. 'close' = last observed event rate."""
    n = len(bars)
    out: list[float | None] = [None] * n
    events: list[float] = []
    for i, bar in enumerate(bars):
        if bar.funding is not None:
            events.append(bar.funding)
        if kind == "close":
            out[i] = events[-1] if events else None
        elif kind == "sma":
            if window is None:
                raise RuleError("funding sma requires a window (event count)")
            if len(events) >= window:
                out[i] = sum(events[-window:]) / window
        else:
            raise RuleError(f"indicator '{kind}' not supported on the funding series")
    return out


def _series(
    kind: str, window: int | None, bars: list[Bar], series: str = "price"
) -> list[float | None]:
    if series == "funding":
        return _funding_series(kind, window, bars)
    closes = [b.close for b in bars]
    n = len(bars)
    if kind == "close":
        return list(closes)
    if window is None:
        raise RuleError(f"indicator '{kind}' requires a window")
    out: list[float | None] = [None] * n
    if kind == "sma":
        acc = 0.0
        for i, c in enumerate(closes):
            acc += c
            if i >= window:
                acc -= closes[i - window]
            if i >= window - 1:
                out[i] = acc / window
        return out
    if kind == "ema":
        k = 2.0 / (window + 1)
        ema = None
        for i, c in enumerate(closes):
            ema = c if ema is None else c * k + ema * (1 - k)
            if i >= window - 1:
                out[i] = ema
        return out
    if kind == "rsi":
        gains = losses = 0.0
        avg_gain = avg_loss = None
        for i in range(1, n):
            change = closes[i] - closes[i - 1]
            gain, loss = max(change, 0.0), max(-change, 0.0)
            if i <= window:
                gains += gain
                losses += loss
                if i == window:
                    avg_gain, avg_loss = gains / window, losses / window
            else:
                avg_gain = (avg_gain * (window - 1) + gain) / window
                avg_loss = (avg_loss * (window - 1) + loss) / window
            if i >= window and avg_loss is not None:
                out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
        return out
    if kind == "atr":
        trs = []
        for i in range(1, n):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - closes[i - 1]),
                abs(bars[i].low - closes[i - 1]),
            )
            trs.append(tr)
            if len(trs) >= window:
                out[i] = sum(trs[-window:]) / window
        return out
    if kind == "high_n":
        for i in range(window - 1, n):
            out[i] = max(b.high for b in bars[i - window + 1 : i + 1])
        return out
    if kind == "low_n":
        for i in range(window - 1, n):
            out[i] = min(b.low for b in bars[i - window + 1 : i + 1])
        return out
    raise RuleError(f"unknown indicator '{kind}'")


def _operand_series(operand, bars: list[Bar]) -> list[float | None]:
    if isinstance(operand, (int, float)):
        return [float(operand)] * len(bars)
    return _series(
        operand["indicator"], operand.get("window"), bars, operand.get("series", "price")
    )


def _rule_signal(rule: dict, bars: list[Bar]) -> list[bool]:
    left = _series(rule["indicator"], rule.get("window"), bars, rule.get("series", "price"))
    right = _operand_series(rule["operand"], bars)
    op = rule["operator"]
    n = len(bars)
    out = [False] * n
    for i in range(n):
        lv, rv = left[i], right[i]
        if lv is None or rv is None:
            continue
        if op == "greater_than":
            out[i] = lv > rv
        elif op == "less_than":
            out[i] = lv < rv
        elif op in ("crosses_above", "crosses_below"):
            if i == 0 or left[i - 1] is None or right[i - 1] is None:
                continue
            if op == "crosses_above":
                out[i] = left[i - 1] <= right[i - 1] and lv > rv
            else:
                out[i] = left[i - 1] >= right[i - 1] and lv < rv
        else:
            raise RuleError(f"unknown operator '{op}'")
    return out


def _all_true(rules: list[dict], bars: list[Bar]) -> list[bool]:
    signals = [_rule_signal(rule, bars) for rule in rules]
    return [all(s[i] for s in signals) for i in range(len(bars))]


def run_backtest(
    signal: dict,
    bars: list[Bar],
    fee_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    stop_loss_pct: float | None = None,
) -> list[BacktestTrade]:
    """Execute the strategy definition's `signal` object over bars.

    stop_loss_pct comes from the definition's `risk` block (single source of
    truth - the M6 v0.1.0 failure was validating rules without the declared
    stop). Stop handling is intra-bar and chronological:
    - triggered when bar.low touches the stop level;
    - gap-through: if the bar OPENS at/below the stop, fill at the open (worse);
    - a rule exit fills AT the open and therefore preempts an intra-bar stop
      that would only have triggered later in the same bar (true event order).
    """
    if signal.get("kind") != "declarative-rules":
        raise RuleError(f"unsupported signal kind: {signal.get('kind')}")
    if stop_loss_pct is not None and not 0 < stop_loss_pct < 100:
        raise ValueError(f"stop_loss_pct out of range: {stop_loss_pct}")
    if len(bars) < 3:
        return []
    entries = _all_true(signal["entry_rules"], bars)
    exits = _all_true(signal["exit_rules"], bars)

    trades: list[BacktestTrade] = []
    in_position = False
    entry_price = 0.0
    entry_date = ""
    stop_level = 0.0
    funding_factor = 1.0  # multiplicative accrual: long pays positive funding

    def close_position(exit_price: float, exit_date: str, stopped: bool, forced: bool = False):
        net = (exit_price / entry_price) * (1 - fee_rate) * (1 - fee_rate) * funding_factor
        trades.append(
            BacktestTrade(
                entry_date=entry_date,
                exit_date=exit_date,
                entry_price=round(entry_price, 8),
                exit_price=round(exit_price, 8),
                return_fraction=round(net - 1.0, 8),
                forced_exit=forced,
                stopped_out=stopped,
            )
        )

    # Funding boundary semantics (documented, tested): an event accrues only if
    # the position exists strictly across it. The entry-fill bar's own event is
    # not paid (position born at that instant); an exit AT the open (rule exit
    # or gap-stop) does not pay that open's event; an intra-bar stop DOES pay
    # it, because the position survived the open.
    for i in range(len(bars) - 1):  # signal at close i -> fill at open i+1
        nxt = bars[i + 1]
        if in_position:
            exits_at_open = exits[i] or (
                stop_loss_pct is not None and nxt.open <= stop_level
            )
            if exits_at_open:
                stopped = stop_loss_pct is not None and nxt.open <= stop_level
                close_position(nxt.open * (1 - slippage_rate), nxt.date, stopped=stopped)
                in_position = False
                continue
            if nxt.funding is not None:
                funding_factor *= 1.0 - nxt.funding  # long pays positive, receives negative
            if stop_loss_pct is not None and nxt.low <= stop_level:  # touched intra-bar
                close_position(stop_level * (1 - slippage_rate), nxt.date, stopped=True)
                in_position = False
                continue
        elif entries[i]:
            entry_price = nxt.open * (1 + slippage_rate)
            entry_date = nxt.date
            in_position = True
            funding_factor = 1.0
            if stop_loss_pct is not None:
                stop_level = entry_price * (1 - stop_loss_pct / 100.0)
    if in_position:
        last = bars[-1]
        close_position(last.open * (1 - slippage_rate), last.date, stopped=False, forced=True)
    return trades


def load_bars_csv(path) -> list[Bar]:
    import csv
    from pathlib import Path

    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        bars = [
            Bar(
                date=row["date"],
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )
            for row in reader
        ]
    if bars != sorted(bars, key=lambda b: b.date):
        raise ValueError("bars must be date-ascending")
    return bars


def merge_funding(bars: list[Bar], funding_csv_path) -> list[Bar]:
    """Attach funding events to the bar covering each event timestamp.

    Funding times are exact bar boundaries (00/08/16 UTC on 1h bars); an event
    is attached to the bar whose timestamp matches its own, i.e. it is known by
    that bar's close (no look-ahead: signals fill next-bar-open as always).
    """
    import csv
    from dataclasses import replace
    from pathlib import Path

    events: dict[str, float] = {}
    with Path(funding_csv_path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            events[row["funding_time_utc"][:16]] = float(row["funding_rate"])
    matched = 0
    out = []
    for bar in bars:
        rate = events.get(bar.date[:16])
        if rate is not None:
            matched += 1
            out.append(replace(bar, funding=rate))
        else:
            out.append(bar)
    if matched < len(events) * 0.9:
        raise ValueError(
            f"only {matched}/{len(events)} funding events matched bar timestamps - "
            "bar/funding series appear misaligned"
        )
    return out
