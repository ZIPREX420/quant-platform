"""Live signal evaluation for the paper-trading cycle (M9).

Answers exactly one question per candidate per cycle: enter, exit, or hold -
using the SAME rule machinery the validation backtester uses
(quant_platform.signals.rules), over closed bars only.

Timing model (mirrors the backtester's next-bar-open discipline):
- the backtester evaluates rules at bar i's close and fills at bar i+1's open;
- the live cycle runs shortly after a bar closes, evaluates rules on that
  just-closed bar, and fills at the current market price - which IS the next
  bar's early trading. Same information set, same fill timing.

Stop handling: the cycle observes a stop breach on the last CLOSED bar
(low <= stop) and exits at current market price. This differs from the
backtester's intra-bar fill at the stop level: the live exit can be worse
(price kept falling) or better (it bounced). It is never based on
information the backtest would not have had. Chronology matches the
backtester: a stop breached during the last bar preempts a rule exit that
could only be acted on now.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_platform.signals.rules import Bar, RuleError, _all_true

MIN_BARS = 2  # crossings need a previous bar; real rule sets need far more


@dataclass(frozen=True)
class SignalDecision:
    """One evaluation outcome. action is 'enter' | 'exit' | 'hold'."""

    action: str
    reason: str
    bar_date: str
    close: float


def evaluate_candidate(
    signal: dict,
    bars: list[Bar],
    in_position: bool,
    stop_price: float | None = None,
) -> SignalDecision:
    """Evaluate a candidate's declarative rules on the last closed bar."""
    if signal.get("kind") != "declarative-rules":
        raise RuleError(f"unsupported signal kind {signal.get('kind')!r}")
    if len(bars) < MIN_BARS:
        raise RuleError(f"need at least {MIN_BARS} closed bars, got {len(bars)}")
    if in_position and stop_price is None:
        raise RuleError("in_position requires the position's stop_price")

    last = bars[-1]

    if in_position:
        if last.low <= stop_price:
            return SignalDecision(
                action="exit",
                reason=f"stop-breach: bar low {last.low} <= stop {stop_price}",
                bar_date=last.date,
                close=last.close,
            )
        if _all_true(signal["exit_rules"], bars)[-1]:
            return SignalDecision(
                action="exit", reason="exit-rules", bar_date=last.date, close=last.close
            )
        return SignalDecision(
            action="hold", reason="position held; no exit condition", bar_date=last.date,
            close=last.close,
        )

    if _all_true(signal["entry_rules"], bars)[-1]:
        return SignalDecision(
            action="enter", reason="entry-rules", bar_date=last.date, close=last.close
        )
    return SignalDecision(
        action="hold", reason="flat; no entry condition", bar_date=last.date, close=last.close
    )


def attach_funding(bars: list[Bar], events: list[tuple[str, float]]) -> list[Bar]:
    """Attach funding events (minute-precision 'YYYY-MM-DDTHH:MM' keys) to bars.

    Same alignment contract as validation's merge_funding: an event belongs to
    the bar whose timestamp prefix matches, i.e. it is known by that bar's
    close. Events outside the bar window are ignored (older history, or newer
    than the last closed bar); unmatched events INSIDE the window indicate
    misalignment and raise.
    """
    from dataclasses import replace

    lookup = {key[:16]: rate for key, rate in events}
    out, matched = [], 0
    for bar in bars:
        rate = lookup.get(bar.date[:16])
        if rate is not None:
            matched += 1
            out.append(replace(bar, funding=rate))
        else:
            out.append(bar)
    first_key, last_key = bars[0].date[:16], bars[-1].date[:16]
    in_range = [k for k in lookup if first_key <= k <= last_key]
    if matched < len(in_range):
        raise RuleError(
            f"only {matched}/{len(in_range)} in-range funding events matched bar "
            f"timestamps - bar/funding series appear misaligned"
        )
    return out
