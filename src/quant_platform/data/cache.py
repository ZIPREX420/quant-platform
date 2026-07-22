"""File-backed cache for PriceHistory (workspace datasets/cache/).

Degradation path per the target architecture: when the OpenBB service is
down, the data service serves cached history and the consumer must check
`PriceHistory.staleness_days()` / the `max_age` it passed here.
"""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from quant_platform.data.schemas import PriceHistory

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _slug(value: str) -> str:
    return _SAFE.sub("-", value)


class PriceHistoryCache:
    """One JSON file per (symbol, source, start, end) tuple."""

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, symbol: str, source: str, start: date, end: date) -> Path:
        name = f"{_slug(symbol)}__{_slug(source)}__{start.isoformat()}__{end.isoformat()}.json"
        return self._root / name

    def put(self, history: PriceHistory, start: date, end: date) -> Path:
        path = self._path(history.symbol, history.source, start, end)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(history.model_dump_json(), encoding="utf-8")
        tmp.replace(path)  # atomic on POSIX and NTFS
        return path

    def get(
        self,
        symbol: str,
        source: str,
        start: date,
        end: date,
        max_age: timedelta | None = None,
    ) -> PriceHistory | None:
        """Return the cached history, or None on miss/expiry/corruption.

        Corrupt entries are treated as misses (and removed) rather than raised:
        the caller's fallback is a fresh fetch either way.
        """
        path = self._path(symbol, source, start, end)
        if not path.exists():
            return None
        try:
            history = PriceHistory.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            path.unlink(missing_ok=True)
            return None
        if max_age is not None:
            age = datetime.now(timezone.utc) - history.fetched_at
            if age > max_age:
                return None
        return history
