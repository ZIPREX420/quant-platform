"""quant-cycle: run one paper-trading cycle over the registered candidates.

Exit codes: 0 = cycle completed; 1 = refused (bad state, bad feed, orphaned
positions, unloadable candidates) - the message says exactly why. Designed to
be safe at any frequency >= the bar interval; a scheduler may simply retry
next cycle.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_platform.cycle import CycleError, run_cycle
from quant_platform.data.binance_client import BinanceClientError
from quant_platform.execution.state import StateError
from quant_platform.strategies.candidates import CandidateLoadError


def find_workspace_root(start: Path) -> Path:
    """Walk up until the PROJECT GENESIS workspace root (has config/candidates)."""
    for parent in [start, *start.parents]:
        if (parent / "config" / "candidates").is_dir():
            return parent
    raise SystemExit(
        "could not locate a repo root containing config/candidates - "
        "run from within the quant-platform repository or pass --repo-root."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-cycle", description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="quant-platform repo root (default: auto-detect upward from cwd)")
    parser.add_argument("--starting-cash", type=float, default=10_000.0,
                        help="paper account starting cash (first run only; default 10000)")
    args = parser.parse_args(argv)

    root = args.repo_root or find_workspace_root(Path.cwd().resolve())
    candidates_dir = root / "config" / "candidates"
    # workspace root = parent of repositories/quant-platform when embedded, else repo root
    ws = root.parents[1] if root.parent.name == "repositories" else root
    reports = ws / "reports" / "research"

    try:
        report = run_cycle(
            candidates_dir=candidates_dir,
            state_path=reports / "paper-state.json",
            audit_path=reports / "executions.jsonl",
            starting_cash=args.starting_cash,
        )
    except (CycleError, StateError, CandidateLoadError, BinanceClientError) as exc:
        print(f"CYCLE-REFUSED: {exc}", file=sys.stderr)
        return 1

    print(report.summary_line())
    try:  # refresh the operator dashboard; never fail the cycle over rendering
        from quant_platform.cli_dashboard import generate  # noqa: PLC0415
        generate(root, ws, ws / "reports" / "dashboard.html")
        print(f"dashboard: {ws / 'reports' / 'dashboard.html'}")
    except Exception as exc:  # noqa: BLE001 - cycle result must survive dashboard bugs
        print(f"WARNING: dashboard render failed: {exc}", file=sys.stderr)
    for r in report.results:
        marker = "" if r.approved is None else (" [approved]" if r.approved else " [REJECTED]")
        fill = f" fill@{r.fill_price}" if r.fill_price else ""
        print(f"  {r.candidate_id} ({r.symbol}): {r.action} - {r.reason}{marker}{fill}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
