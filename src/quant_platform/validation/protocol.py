"""Protocol v1 orchestration helpers: IS/OOS split and walk-forward windows.

For fixed-parameter strategies (no optimization), the train segment serves as
indicator warmup only; this must be declared in the report.
"""
from __future__ import annotations

from dataclasses import dataclass

from quant_platform.validation.backtest import Bar, BacktestTrade, run_backtest


@dataclass(frozen=True)
class WindowResult:
    label: str
    test_start: str
    test_end: str
    trades: list[BacktestTrade]


def trades_entered_within(trades: list[BacktestTrade], start: str, end: str) -> list[BacktestTrade]:
    return [t for t in trades if start <= t.entry_date <= end]


def is_oos_split(bars: list[Bar], oos_fraction: float = 0.3) -> tuple[list[Bar], list[Bar]]:
    if not 0.05 <= oos_fraction <= 0.5:
        raise ValueError("oos_fraction must be in [0.05, 0.5]")
    cut = int(len(bars) * (1 - oos_fraction))
    return bars[:cut], bars[cut:]


def run_oos(signal: dict, bars: list[Bar], oos_fraction: float = 0.3,
            fee_rate: float = 0.001, slippage_rate: float = 0.0005) -> WindowResult:
    """Backtest the full series, report only trades entered in the OOS segment
    (full-series run gives the OOS segment proper indicator warmup)."""
    _, oos = is_oos_split(bars, oos_fraction)
    all_trades = run_backtest(signal, bars, fee_rate, slippage_rate)
    return WindowResult(
        label="OOS",
        test_start=oos[0].date,
        test_end=oos[-1].date,
        trades=trades_entered_within(all_trades, oos[0].date, oos[-1].date),
    )


def walk_forward(signal: dict, bars: list[Bar], n_windows: int = 5, warmup_bars: int = 250,
                 fee_rate: float = 0.001, slippage_rate: float = 0.0005) -> list[WindowResult]:
    """Split the post-warmup history into n sequential test windows.

    Each window's backtest runs from bar 0 (full warmup + prior history) but
    only trades ENTERED inside the window are attributed to it - equivalent to
    rolling deployment of a fixed-parameter strategy.
    """
    if len(bars) <= warmup_bars + n_windows:
        raise ValueError("not enough bars for the requested windows")
    testable = bars[warmup_bars:]
    size = len(testable) // n_windows
    results = []
    all_trades = run_backtest(signal, bars, fee_rate, slippage_rate)
    for w in range(n_windows):
        seg = testable[w * size : (w + 1) * size] if w < n_windows - 1 else testable[w * size :]
        results.append(
            WindowResult(
                label=f"WF{w + 1}",
                test_start=seg[0].date,
                test_end=seg[-1].date,
                trades=trades_entered_within(all_trades, seg[0].date, seg[-1].date),
            )
        )
    return results


def with_windows(signal: dict, fast: int, slow: int) -> dict:
    """Return a copy of an SMA-cross signal with perturbed windows (sensitivity)."""
    import copy

    out = copy.deepcopy(signal)
    out["parameters"] = {"fast": fast, "slow": slow}
    for rules in (out["entry_rules"], out["exit_rules"]):
        for rule in rules:
            rule["window"] = fast
            rule["operand"]["window"] = slow
    return out
