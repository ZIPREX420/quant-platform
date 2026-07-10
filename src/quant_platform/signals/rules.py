"""Shared declarative-rule machinery (single source of truth).

This module is THE implementation of the strategy schema's rule vocabulary.
Both the validation backtester and the live paper-cycle signal evaluator
import it - by construction there is no second implementation whose
semantics could drift (M9 requirement).

Semantics (enforced by tests):
- indicators are computed over CLOSED bars only; callers must never pass a
  still-forming bar (BinanceClient drops it by default);
- high_n/low_n = highest high / lowest low of the PRIOR n bars, current bar
  excluded (Donchian breakout semantics, commit 82eb8e9);
- funding-series windows count funding EVENTS, not bars;
- crosses_above/crosses_below require both series defined on the previous
  bar (no signal on the first defined bar).
"""
from __future__ import annotations

from dataclasses import dataclass

INDICATORS = ("close", "sma", "ema", "rsi", "atr", "high_n", "low_n")


@dataclass(frozen=True)
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    funding: float | None = None  # perp funding-rate event occurring in this bar, if any


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
        # highest high of the PRIOR n bars (current bar excluded) - Donchian
        # semantics; including the current bar would make breakout rules like
        # `close crosses_above high_n` unsatisfiable by construction.
        for i in range(window, n):
            out[i] = max(b.high for b in bars[i - window : i])
        return out
    if kind == "low_n":
        for i in range(window, n):
            out[i] = min(b.low for b in bars[i - window : i])
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


