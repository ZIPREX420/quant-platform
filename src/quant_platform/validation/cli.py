"""quant-validate-trades: analyze a backtest trade-list CSV per protocol v1.

Emits the Monte Carlo and headline-metric sections ready to paste into the
validation report (docs/validation/TEMPLATE-validation-report.md SS5).
"""
from __future__ import annotations

import argparse
import sys

from quant_platform.validation.analysis import monte_carlo, sharpe_like, trade_metrics
from quant_platform.validation.trades import TradeListError, load_trades_csv


def render_report_sections(csv_name: str, metrics, mc, sharpe) -> str:
    pf = metrics.profit_factor if metrics.profit_factor is not None else "inf (no losses)"
    rl = metrics.return_over_maxdd if metrics.return_over_maxdd is not None else "n/a"
    sh = sharpe if sharpe is not None else "n/a (constant or <2 trades)"
    return f"""## Headline metrics (from {csv_name})

- Trades: {metrics.trades} (protocol floor: >=100 across OOS+WF)
- Win rate: {metrics.win_rate_pct}% | Profit factor: {pf} (criterion: >1.15 OOS)
- Expectancy: {metrics.expectancy_pct}%/trade | Sharpe (per-trade, scaled): {sh}
- Total return: {metrics.total_return_pct}% | Max drawdown: {metrics.max_drawdown_pct}%
- Return/MaxDD: {rl}

## 5. Monte Carlo ({mc.runs} bootstrap runs, seed {mc.seed})

- Method: bootstrap resampling with replacement of the trade sequence
- 5th-pct terminal return: {mc.terminal_return_pct_p05}% | median: {mc.terminal_return_pct_p50}%
- 5th-pct (worst) max drawdown: {mc.max_drawdown_pct_p05}% | median: {mc.max_drawdown_pct_p50}%
- Probability of negative terminal equity: {mc.prob_negative_terminal_pct}%
- Criterion check: 5th-pct max drawdown must lie within the strategy's declared risk caps.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="trade-list CSV exported from the backtester")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trades-per-year", type=float, default=12.0,
                        help="approx trade frequency for Sharpe scaling (default 12)")
    args = parser.parse_args()

    try:
        trades = load_trades_csv(args.csv)
        metrics = trade_metrics(trades)
        mc = monte_carlo(trades, runs=args.runs, seed=args.seed)
    except (TradeListError, ValueError) as exc:
        print(f"VALIDATE-ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    sharpe = sharpe_like(trades, periods_per_year=args.trades_per_year)
    print(render_report_sections(args.csv, metrics, mc, sharpe))


if __name__ == "__main__":
    main()
