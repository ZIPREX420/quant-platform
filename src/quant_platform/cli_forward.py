"""quant-forward: measure candidates' paper records against protocol v2 thresholds.

Prints one assessment per registered candidate (or one, if given an id).
This output is the ONLY admissible source of numbers for a forward-evidence
validation report (docs/validation/validation-protocol-v2-forward.md).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_platform.cli_cycle import find_workspace_root
from quant_platform.execution.session import ExecutionAudit
from quant_platform.strategies.candidates import CandidateLoadError, load_candidate_dir
from quant_platform.validation.forward import ForwardRecordError, assess


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-forward", description=__doc__)
    parser.add_argument("candidate_id", nargs="?", default=None,
                        help="assess one candidate (default: all registered)")
    parser.add_argument("--repo-root", type=Path, default=None)
    args = parser.parse_args(argv)

    root = args.repo_root or find_workspace_root(Path.cwd().resolve())
    ws = root.parents[1] if root.parent.name == "repositories" else root
    records = ExecutionAudit(ws / "reports" / "research" / "executions.jsonl").records()

    try:
        candidates = load_candidate_dir(root / "config" / "candidates")
    except CandidateLoadError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    ids = [c.id for c in candidates]
    if args.candidate_id:
        if args.candidate_id not in ids:
            print(f"REFUSED: '{args.candidate_id}' is not a registered candidate "
                  f"(registered: {ids or 'none'})", file=sys.stderr)
            return 1
        ids = [args.candidate_id]

    if not ids:
        print("no candidates registered (config/candidates/)")
        return 0
    for cid in ids:
        try:
            print(assess(records, cid).summary())
        except ForwardRecordError as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 1
        candidate = next(c for c in candidates if c.id == cid)
        print(f"  prediction: {candidate.prediction}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
