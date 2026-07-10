"""Persistent paper state: atomic, restorable, and NEVER silently reset."""
from datetime import datetime, timezone

import pytest

from quant_platform.execution.paper import PaperAccount
from quant_platform.execution.state import (
    OpenPosition,
    PaperState,
    StateError,
    StateStore,
)


def make_position(symbol="BTCUSDT", qty=0.5) -> OpenPosition:
    return OpenPosition(
        candidate_id="test-candidate", symbol=symbol, quantity=qty,
        entry_price=100.0, entry_ts=datetime.now(timezone.utc),
        stop_price=95.0, entry_fill_id="abc123def456",
    )


def test_missing_file_is_first_run(tmp_path):
    assert StateStore(tmp_path / "paper-state.json").load() is None


def test_round_trip_restores_account(tmp_path):
    account = PaperAccount(starting_cash=10_000.0)
    account.cash = 9_400.0
    account.positions = {"BTCUSDT": 0.5}
    state = PaperState.from_account(account, (make_position(),), cycle_count=7)
    store = StateStore(tmp_path / "paper-state.json")
    store.save(state)

    loaded = store.load()
    assert loaded is not None and loaded.cycle_count == 7
    restored = loaded.restore_account()
    assert restored.cash == 9_400.0
    assert restored.starting_cash == 10_000.0
    assert restored.positions == {"BTCUSDT": 0.5}
    assert loaded.open_positions[0].stop_price == 95.0


def test_corrupt_file_refused_never_reset(tmp_path):
    path = tmp_path / "paper-state.json"
    path.write_text("{truncated garbage", encoding="utf-8")
    store = StateStore(path)
    with pytest.raises(StateError, match="REFUSING to reset"):
        store.load()
    # the file must be untouched after the refusal
    assert path.read_text(encoding="utf-8") == "{truncated garbage"


def test_unknown_version_refused(tmp_path):
    state = PaperState.fresh(10_000.0).model_copy(update={"version": 99})
    store = StateStore(tmp_path / "paper-state.json")
    store.save(state)
    with pytest.raises(StateError, match="version 99"):
        store.load()


def test_inconsistent_metadata_refused():
    account = PaperAccount(starting_cash=10_000.0)
    account.positions = {"BTCUSDT": 0.5}
    with pytest.raises(StateError, match="does not match account book"):
        PaperState.from_account(account, (make_position(qty=0.7),), cycle_count=1)


def test_save_is_atomic_no_tmp_left(tmp_path):
    store = StateStore(tmp_path / "paper-state.json")
    store.save(PaperState.fresh(10_000.0))
    leftovers = [p.name for p in tmp_path.iterdir()]
    assert leftovers == ["paper-state.json"]
