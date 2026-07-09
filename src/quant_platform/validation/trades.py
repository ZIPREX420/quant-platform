"""Trade-list loading with column auto-detection.

Backtest exports differ by tool (QuantDinger CSV, custom notebooks). We only
need one number per trade: the return (fraction or percent) or absolute pnl
plus an entry notional. Column names are auto-detected; ambiguity is an error,
never a guess.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

RETURN_COLUMNS = ("return_pct", "profit_pct", "pnl_pct", "return", "profit_ratio")
PNL_COLUMNS = ("pnl", "profit", "profit_abs", "pnl_abs")
NOTIONAL_COLUMNS = ("notional", "cost", "entry_value", "stake_amount", "amount")


class TradeListError(ValueError):
    """The CSV could not be interpreted unambiguously."""


@dataclass(frozen=True)
class Trade:
    return_fraction: float  # e.g. 0.02 == +2%


def _pick(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    hits = [c for c in candidates if c in fieldnames]
    if len(hits) > 1:
        raise TradeListError(f"ambiguous columns {hits}; rename to keep exactly one")
    return hits[0] if hits else None


def load_trades_csv(path: Path | str, percent_threshold: float = 1.5) -> list[Trade]:
    """Load trades from CSV.

    Interpretation order:
    1. a return column - values are fractions unless the file's max magnitude
       exceeds percent_threshold, in which case the whole file is treated as
       percent and divided by 100 (decided once per file, never per row);
    2. else pnl + notional columns (return = pnl / notional).
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise TradeListError(f"{path.name}: no header row")
        fields = [f.strip().lower() for f in reader.fieldnames]
        rows = [{k.strip().lower(): v for k, v in row.items() if k} for row in reader]
    if not rows:
        raise TradeListError(f"{path.name}: no trade rows")

    ret_col = _pick(fields, RETURN_COLUMNS)
    if ret_col:
        try:
            values = [float(r[ret_col]) for r in rows]
        except (KeyError, TypeError, ValueError) as exc:
            raise TradeListError(f"{path.name}: bad value in '{ret_col}': {exc}") from exc
        if max(abs(v) for v in values) > percent_threshold:
            values = [v / 100.0 for v in values]
        return [Trade(return_fraction=v) for v in values]

    pnl_col = _pick(fields, PNL_COLUMNS)
    notional_col = _pick(fields, NOTIONAL_COLUMNS)
    if pnl_col and notional_col:
        trades = []
        for r in rows:
            try:
                pnl, notional = float(r[pnl_col]), float(r[notional_col])
            except (KeyError, TypeError, ValueError) as exc:
                raise TradeListError(f"{path.name}: bad pnl/notional row: {exc}") from exc
            if notional <= 0:
                raise TradeListError(f"{path.name}: non-positive notional {notional}")
            trades.append(Trade(return_fraction=pnl / notional))
        return trades

    raise TradeListError(
        f"{path.name}: no usable columns. Provide one of {RETURN_COLUMNS} "
        f"or both of {PNL_COLUMNS} + {NOTIONAL_COLUMNS}. Found: {fields}"
    )
