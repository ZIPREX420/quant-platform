"""Single-instance cycle lock: concurrent runs refuse; stale locks self-heal."""
import json
import time

import pytest

from quant_platform.execution.cyclelock import STALE_SECONDS, CycleLockBusy, cycle_lock


def test_concurrent_acquire_refuses(tmp_path):
    state = tmp_path / "paper-state.json"
    with cycle_lock(state):
        with pytest.raises(CycleLockBusy):
            with cycle_lock(state):
                pass


def test_lock_released_on_normal_exit(tmp_path):
    state = tmp_path / "paper-state.json"
    with cycle_lock(state):
        pass
    with cycle_lock(state):  # re-acquire proves release
        pass
    assert not state.with_suffix(".lock").exists()


def test_lock_released_on_error(tmp_path):
    state = tmp_path / "paper-state.json"
    with pytest.raises(ValueError):
        with cycle_lock(state):
            raise ValueError("boom")
    assert not state.with_suffix(".lock").exists()  # finally removed it


def test_stale_lock_is_stolen(tmp_path):
    state = tmp_path / "paper-state.json"
    lock = state.with_suffix(".lock")
    lock.write_text(json.dumps({"pid": 999999, "ts": time.time() - STALE_SECONDS - 10}))
    with cycle_lock(state):  # stale -> stolen, acquired
        pass
    assert not lock.exists()


def test_fresh_foreign_lock_not_stolen(tmp_path):
    state = tmp_path / "paper-state.json"
    lock = state.with_suffix(".lock")
    lock.write_text(json.dumps({"pid": 999999, "ts": time.time()}))
    with pytest.raises(CycleLockBusy):
        with cycle_lock(state):
            pass
    assert lock.exists()  # a fresh foreign lock is left intact
