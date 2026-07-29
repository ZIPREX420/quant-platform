"""Single-instance guard for the paper cycle.

The cycle is a read-modify-write of ``paper-state.json``: load state, apply the
cycle's fills (each appended immediately to the append-only ``executions.jsonl``),
then save state ONCE at the end. Two overlapping runs - e.g. a manual
``m9-cycle`` firing while the hourly ``m10`` scheduled task runs - both load the
same state, both append their fills to the (append-only) audit, and both save:
last writer wins, silently dropping one run's state update while its fill lives
on in the audit. That is exactly the audit-vs-state divergence observed on
sol-funding-carry-tracker (one fill in ``executions.jsonl`` absent from state).

This module serialises cycles with an exclusive lock file created atomically
(``O_CREAT | O_EXCL``). A second concurrent cycle refuses cleanly rather than
racing. A lock older than ``STALE_SECONDS`` (a crashed run that never released)
is stolen so the cadence self-heals. Cross-platform: relies only on the
atomicity of ``O_EXCL`` file creation, which holds on both NTFS and POSIX.
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

STALE_SECONDS = 600  # a cycle finishing in >10 min has crashed; its lock is stealable


class CycleLockBusy(Exception):
    """Another cycle holds a fresh lock. The caller should refuse this run."""


def _read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def cycle_lock(state_path: Path | str, now: float | None = None) -> Iterator[None]:
    """Hold an exclusive per-state lock for the duration of the block.

    Raises CycleLockBusy if a fresh lock is already held. A stale lock (older
    than STALE_SECONDS) is stolen. The lock file sits next to the state file and
    is always removed on exit, including on error.
    """
    now = time.time() if now is None else now
    lock_path = Path(state_path).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held = _read_lock(lock_path)
        age = now - float(held.get("ts", 0.0))
        if age < STALE_SECONDS:
            raise CycleLockBusy(
                f"another cycle holds {lock_path.name} (pid {held.get('pid', '?')}, "
                f"{age:.0f}s old); refusing to run concurrently"
            ) from None
        # stale: steal it (best-effort remove, then re-create exclusively)
        with contextlib.suppress(OSError):
            lock_path.unlink()
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)

    try:
        os.write(fd, json.dumps({"pid": os.getpid(), "ts": now}).encode("utf-8"))
        os.close(fd)
        yield
    finally:
        with contextlib.suppress(OSError):
            lock_path.unlink()
