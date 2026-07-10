"""Persistent paper-trading state (M9): the account survives between cycles.

One JSON file (default reports/research/paper-state.json) holds the account
snapshot plus per-position engine metadata (entry price, stop level, owning
candidate). Rules:

  - writes are atomic (tmp file + os.replace, same pattern as the data cache);
  - a MISSING file is a legitimate first run -> ``load()`` returns None and
    the caller decides to initialize;
  - a CORRUPT or invalid file is NEVER silently reset -> StateError. Losing
    track of open paper positions silently would poison the forward-evidence
    record, which is the whole point of the paper loop.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quant_platform.execution.paper import PaperAccount

STATE_VERSION = 1


class StateError(Exception):
    """The state file exists but cannot be trusted. Refuse; never reset."""


class OpenPosition(BaseModel):
    """Engine metadata for one open paper position (owned by one candidate)."""

    model_config = ConfigDict(frozen=True)

    candidate_id: str
    symbol: str
    quantity: float = Field(gt=0)
    entry_price: float = Field(gt=0)
    entry_ts: datetime
    stop_price: float = Field(gt=0)
    entry_fill_id: str


class PaperState(BaseModel):
    """Full persisted state of the paper-trading engine."""

    model_config = ConfigDict(frozen=True)

    version: int = STATE_VERSION
    updated_at: datetime
    starting_cash: float = Field(gt=0)
    cash: float
    positions: dict[str, float] = Field(default_factory=dict)
    open_positions: tuple[OpenPosition, ...] = ()
    cycle_count: int = Field(default=0, ge=0)
    day_anchor_date: str | None = None  # UTC date of the daily kill-switch anchor
    day_anchor_equity: float | None = Field(default=None, gt=0)

    def restore_account(self) -> PaperAccount:
        account = PaperAccount(starting_cash=self.starting_cash)
        account.cash = self.cash
        account.positions = dict(self.positions)
        return account

    @classmethod
    def fresh(cls, starting_cash: float) -> "PaperState":
        return cls(
            updated_at=datetime.now(timezone.utc),
            starting_cash=starting_cash,
            cash=starting_cash,
        )

    @classmethod
    def from_account(
        cls,
        account: PaperAccount,
        open_positions: tuple[OpenPosition, ...],
        cycle_count: int,
        day_anchor_date: str | None = None,
        day_anchor_equity: float | None = None,
    ) -> "PaperState":
        # engine metadata and account book must agree - refuse divergence
        for pos in open_positions:
            held = account.positions.get(pos.symbol, 0.0)
            if abs(held - pos.quantity) > 1e-9:
                raise StateError(
                    f"open-position metadata for {pos.symbol} ({pos.quantity}) does not "
                    f"match account book ({held}) - refusing to persist inconsistent state"
                )
        return cls(
            updated_at=datetime.now(timezone.utc),
            starting_cash=account.starting_cash,
            cash=account.cash,
            positions=dict(account.positions),
            open_positions=open_positions,
            cycle_count=cycle_count,
            day_anchor_date=day_anchor_date,
            day_anchor_equity=day_anchor_equity,
        )


class StateStore:
    """Atomic load/save of PaperState at a fixed path."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> PaperState | None:
        """None iff the file does not exist (first run). Corrupt -> StateError."""
        if not self._path.exists():
            return None
        try:
            state = PaperState.model_validate_json(self._path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError) as exc:
            raise StateError(
                f"paper state at {self._path} is unreadable or invalid: {exc}. "
                f"REFUSING to reset automatically - inspect the file (and the last "
                f"audit records in executions.jsonl) and repair or archive it manually."
            ) from exc
        if state.version != STATE_VERSION:
            raise StateError(
                f"paper state version {state.version} != supported {STATE_VERSION}; "
                f"migrate explicitly before running."
            )
        return state

    def save(self, state: PaperState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp.replace(self._path)  # atomic on POSIX and NTFS
