"""quant-dashboard: render the operator dashboard to reports/dashboard.html.

All file IO happens here; rendering itself is pure (quant_platform.dashboard).
Every input is optional - a missing file renders as its honest empty state,
never as an error, so the dashboard works from the first day of a workspace.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from quant_platform.cli_cycle import find_workspace_root
from quant_platform.dashboard import render_dashboard
from quant_platform.execution.session import ExecutionAudit
from quant_platform.execution.state import StateError, StateStore
from quant_platform.journal import DecisionJournal
from quant_platform.strategies.candidates import CandidateLoadError, load_candidate_dir


def parse_ledger(path: Path) -> list[dict]:
    """Rows of the signed-reports markdown table (immutability ledger)."""
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0].endswith(".md"):
            rows.append({
                "report": cells[0], "strategy": cells[1], "result": cells[2],
                "date": cells[3], "sha256": cells[4].strip("`"),
            })
    return rows


def read_equity_history(path: Path) -> list[dict]:
    """Per-cycle equity marks (equity-history.jsonl). Missing file -> [];
    malformed lines are skipped - the dashboard must render regardless."""
    if not path.is_file():
        return []
    import json
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def generate(repo_root: Path, workspace_root: Path, out_path: Path) -> Path:
    reports = workspace_root / "reports"
    try:
        state = StateStore(reports / "research" / "paper-state.json").load()
    except StateError as exc:  # visible on the dashboard is better than absent
        print(f"WARNING: {exc}", file=sys.stderr)
        state = None
    audit_records = ExecutionAudit(reports / "research" / "executions.jsonl").records()
    try:
        candidates = load_candidate_dir(repo_root / "config" / "candidates")
    except CandidateLoadError as exc:
        print(f"WARNING: candidate registry unreadable: {exc}", file=sys.stderr)
        candidates = []
    journal = DecisionJournal(reports / "research" / "journal.jsonl")
    ledger = parse_ledger(workspace_root / "docs" / "validation" / "signed-reports.md")
    equity_history = read_equity_history(reports / "research" / "equity-history.jsonl")

    html = render_dashboard(
        state, audit_records, candidates, journal, ledger,
        equity_history=equity_history,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".html.tmp")
    tmp.write_text(html, encoding="utf-8")
    tmp.replace(out_path)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quant-dashboard", description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None,
                        help="output path (default <workspace>/reports/dashboard.html)")
    args = parser.parse_args(argv)

    root = args.repo_root or find_workspace_root(Path.cwd().resolve())
    ws = root.parents[1] if root.parent.name == "repositories" else root
    out = args.out or ws / "reports" / "dashboard.html"
    written = generate(root, ws, out)
    print(f"dashboard written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
