"""Compute a validation report's sha256 and emit the strategy-definition block.

Usage:  python -m quant_platform.strategies.sign_report <report.md> [--workspace-root DIR]

Prints the JSON `validation_report` block to embed in the strategy definition.
The path is stored workspace-relative so the loader can resolve it (ADR-0005).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from quant_platform.strategies.loader import _sha256


def build_report_reference(report_path: Path, workspace_root: Path) -> dict:
    report_path = report_path.resolve()
    workspace_root = workspace_root.resolve()
    try:
        rel = report_path.relative_to(workspace_root)
    except ValueError as exc:
        raise ValueError(
            f"report {report_path} is not inside workspace root {workspace_root}"
        ) from exc
    return {
        "path": rel.as_posix(),
        "sha256": _sha256(report_path),
        "signed_off_by": "<name>",
        "signed_off_date": date.today().isoformat(),
        "result": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report")
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()
    report = Path(args.report)
    if not report.is_file():
        print(f"no such report file: {report}", file=sys.stderr)
        raise SystemExit(1)
    block = build_report_reference(report, Path(args.workspace_root))
    print(json.dumps({"validation_report": block}, indent=2))
    print(
        "\nReminder: fill signed_off_by with the reviewer's name; result must "
        "reflect the checklist honestly. Report is immutable after sign-off.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
