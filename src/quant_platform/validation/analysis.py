"""Trade-list statistics: protocol metrics + Monte Carlo (protocol v1 SS5).

Pure stdlib; deterministic given a seed. Compounded equity model with full
reinvestment - consistent with per-trade return fractions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, pstdev

from quant_platform.validation.trades import Trade


@dataclass(frozen=True)
class TradeMetrics:
    trades: int
    win_rate_pct: float
    profit_factor: float | None  # None when no losing trades (infinite)
    expectancy_pct: float
    total_return_pct: float
    max_drawdown_pct: float
    return_over_maxdd: float | None


@dataclass(frozen=True)
class MonteCarloResult:
    runs: int
    seed: int
    terminal_return_pct_p05: float
    terminal_return_pct_p50: float
    max_drawdown_pct_p05: float  # 5th percentile WORST drawdown (most negative)
    max_drawdown_pct_p50: float
    prob_negative_terminal_pct: float


def _equity_curve(returns: list[float]) -> list[float]:
    equity = 1.0
    curve = [equity]
    for r in returns:
        equity *= 1.0 + r
        curve.append(equity)
    return curve


def _max_drawdown(curve: list[float]) -> float:
    """Most negative peak-to-trough move, as a percentage (<= 0)."""
    peak = curve[0]
    worst = 0.0
    for value in curve:
        peak = max(peak, value)
        worst = min(worst, value / peak - 1.0)
    return worst * 100.0


def trade_metrics(trades: list[Trade]) -> TradeMetrics:
    if not trades:
        raise ValueError("no trades")
    rets = [t.return_fraction for t in trades]
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    curve = _equity_curve(rets)
    max_dd = _max_drawdown(curve)
    total = (curve[-1] - 1.0) * 100.0
    return TradeMetrics(
        trades=len(rets),
        win_rate_pct=round(100.0 * len(wins) / len(rets), 1),
        profit_factor=round(gross_win / gross_loss, 3) if gross_loss > 0 else None,
        expectancy_pct=round(mean(rets) * 100.0, 3),
        total_return_pct=round(total, 2),
        max_drawdown_pct=round(max_dd, 2),
        return_over_maxdd=round(total / abs(max_dd), 2) if max_dd < 0 else None,
    )


def sharpe_like(trades: list[Trade], periods_per_year: float) -> float | None:
    """Per-trade Sharpe scaled by trade frequency. None if undefined (constant returns)."""
    rets = [t.return_fraction for t in trades]
    if len(rets) < 2:
        return None
    sd = pstdev(rets)
    if sd == 0:
        return None
    return round(mean(rets) / sd * periods_per_year**0.5, 2)


def monte_carlo(trades: list[Trade], runs: int = 1000, seed: int = 42) -> MonteCarloResult:
    """Bootstrap (sample with replacement) the trade sequence `runs` times.

    Resampling with replacement is strictly more conservative than pure
    shuffling for drawdown estimation (loss clusters can repeat).
    """
    if len(trades) < 10:
        raise ValueError(f"Monte Carlo needs >= 10 trades, got {len(trades)}")
    rng = random.Random(seed)
    rets = [t.return_fraction for t in trades]
    n = len(rets)
    terminals: list[float] = []
    drawdowns: list[float] = []
    for _ in range(runs):
        sample = [rets[rng.randrange(n)] for _ in range(n)]
        curve = _equity_curve(sample)
        terminals.append((curve[-1] - 1.0) * 100.0)
        drawdowns.append(_max_drawdown(curve))
    terminals.sort()
    drawdowns.sort()  # ascending: most negative (worst) first

    def pct(sorted_vals: list[float], q: float) -> float:
        idx = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
        return sorted_vals[idx]

    negative = sum(1 for t in terminals if t < 0)
    return MonteCarloResult(
        runs=runs,
        seed=seed,
        terminal_return_pct_p05=round(pct(terminals, 0.05), 2),
        terminal_return_pct_p50=round(pct(terminals, 0.50), 2),
        max_drawdown_pct_p05=round(pct(drawdowns, 0.05), 2),
        max_drawdown_pct_p50=round(pct(drawdowns, 0.50), 2),
        prob_negative_terminal_pct=round(100.0 * negative / runs, 1),
    )
