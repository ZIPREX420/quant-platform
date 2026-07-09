"""quant_platform.validation - statistical analysis for the validation protocol.

Human-invoked promotion tooling (protocol v1, docs/validation/). Never imported
by the execution path.
"""

from quant_platform.validation.analysis import monte_carlo, trade_metrics
from quant_platform.validation.trades import Trade, TradeListError, load_trades_csv

__all__ = ["Trade", "TradeListError", "load_trades_csv", "trade_metrics", "monte_carlo"]
