"""Decision journal: every research memo is recorded with its inputs.

Append-only JSONL, one MemoRecord per line. Outcomes are recorded later as
follow-up lines referencing the memo id (append-only keeps the file auditable;
readers merge). This is the substrate for M6 memo-vs-outcome evaluation,
modeled on TradingAgents' TradingMemoryLog pattern.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from quant_platform.data.context import MarketContext


class MemoRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    record_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    kind: str = "memo"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    symbol: str
    context: MarketContext
    memo: str
    model: str
    confidence: str | None = None
    usage: list[dict] = Field(default_factory=list)


class OutcomeRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: str = "outcome"
    memo_record_id: str
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    horizon_days: int
    realized_return_pct: float
    notes: str = ""


def extract_confidence(memo: str) -> str | None:
    """Pull the LOW/MEDIUM/HIGH rating out of a desk memo, if present."""
    upper = memo.upper()
    for level in ("MEDIUM", "HIGH", "LOW"):  # MEDIUM first: substring-safe order
        if f"**{level}**" in memo or f"CONFIDENCE: {level}" in upper:
            return level
    return None


class DecisionJournal:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append_memo(self, record: MemoRecord) -> str:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json() + "\n")
        return record.record_id

    def append_outcome(self, outcome: OutcomeRecord) -> None:
        known = {r.record_id for r in self.memos()}
        if outcome.memo_record_id not in known:
            raise KeyError(f"no memo record {outcome.memo_record_id} in journal")
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(outcome.model_dump_json() + "\n")

    def _lines(self) -> list[dict]:
        if not self._path.exists():
            return []
        rows = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def memos(self) -> list[MemoRecord]:
        return [MemoRecord.model_validate(r) for r in self._lines() if r.get("kind") == "memo"]

    def outcomes_for(self, record_id: str) -> list[OutcomeRecord]:
        return [
            OutcomeRecord.model_validate(r)
            for r in self._lines()
            if r.get("kind") == "outcome" and r.get("memo_record_id") == record_id
        ]

    def pending(self) -> list[MemoRecord]:
        """Memos that have no recorded outcome yet - the M6 review queue."""
        resolved = {r["memo_record_id"] for r in self._lines() if r.get("kind") == "outcome"}
        return [m for m in self.memos() if m.record_id not in resolved]
