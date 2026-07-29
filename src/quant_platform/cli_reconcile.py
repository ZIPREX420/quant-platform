"""quant-reconcile-state: rebuild paper state from the durable ledgers.

The cycle refuses when paper-state.json disagrees with executions.jsonl (a fill
that reached the append-only audit but whose state save was lost). This tool
reconstructs the state from the durable ledgers - executions.jsonl (fills) +
funding-accruals.jsonl - and, with --apply, writes it back (backing up the old
file first). Dry-run by default: it prints exactly what would change, so a repair
is always reviewed - never the silent reset the state contract forbids.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from quant_platform.cli_cycle import find_workspace_root
from quant_platform.execution.reconcile import book_divergences, rebuild_state
from quant_platform.execution.session import ExecutionAudit
from quant_platform.execution.state import StateError, StateStore
from quant_platform.strategies.candidates import CandidateLoadError, load_candidate_dir


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-reconcile-state", description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--apply", action="store_true",
                        help="write the rebuilt state (default: dry-run, show the diff only)")
    args = parser.parse_args(argv)

    root = args.repo_root or find_workspace_root(Path.cwd().resolve())
    ws = root.parents[1] if root.parent.name == "repositories" else root
    reports = ws / "reports" / "research"
    store = StateStore(reports / "paper-state.json")

    try:
        state = store.load()
    except StateError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    if state is None:
        print("no paper-state.json yet (first run) - nothing to reconcile.")
        return 0

    records = ExecutionAudit(reports / "executions.jsonl").records()
    divs = book_divergences(state.restore_account().positions, records)
    if not divs:
        print("state and audit already agree - nothing to reconcile.")
        return 0

    print("DIVERGENCE (account book vs execution audit):")
    for d in divs:
        print(f"  {d}")

    try:
        candidates = load_candidate_dir(root / "config" / "candidates")
    except CandidateLoadError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    stops = {c.id: float(c.definition["risk"]["stop_loss_pct"]) for c in candidates}

    now = datetime.now(timezone.utc)
    rebuilt = rebuild_state(state, records, _load_jsonl(reports / "funding-accruals.jsonl"), stops, now)

    print("\nREBUILD from executions.jsonl + funding-accruals.jsonl:")
    print(f"  cash:      {state.cash:.8f} -> {rebuilt.cash:.8f}")
    print(f"  positions: {dict(state.positions)}")
    print(f"          -> {dict(rebuilt.positions)}")
    print("  open positions ->")
    for p in rebuilt.open_positions:
        print(f"    {p.candidate_id} {p.symbol} {p.direction} qty={p.quantity} entry={p.entry_price}")

    if not args.apply:
        print("\nDRY RUN - nothing written. Re-run with --apply to save "
              "(the old paper-state.json is backed up first).")
        return 0

    backup = reports / f"paper-state.json.bak-{now.strftime('%Y%m%dT%H%M%SZ')}"
    backup.write_text(store.path.read_text(encoding="utf-8"), encoding="utf-8")
    store.save(rebuilt)
    print(f"\nAPPLIED. Old state backed up to {backup.name}. Re-run the cycle to continue.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
