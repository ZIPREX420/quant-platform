"""Platform status probe: one command answers "is everything alive and fresh?"

Right-sized observability for a single-operator platform (target architecture
defers Prometheus/k8s deliberately): each surface gets a StatusCheck with a
health verdict and evidence; the CLI exits non-zero when anything is degraded,
so it can back scheduled checks or a pre-research preflight.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

from quant_platform.journal import DecisionJournal


@dataclass(frozen=True)
class StatusCheck:
    name: str
    healthy: bool
    detail: str
    metrics: dict = field(default_factory=dict)


def check_http_service(name: str, url: str, timeout_seconds: float = 5.0) -> StatusCheck:
    t0 = time.time()
    try:
        response = httpx.get(url, timeout=timeout_seconds)
        ms = round((time.time() - t0) * 1000)
        ok = response.status_code == 200
        return StatusCheck(
            name, ok, f"HTTP {response.status_code} in {ms}ms",
            {"status_code": response.status_code, "latency_ms": ms},
        )
    except httpx.HTTPError as exc:
        return StatusCheck(name, False, f"unreachable: {type(exc).__name__}")


def check_journal(path: Path | str, max_pending: int = 25, max_last_memo_days: int | None = None) -> StatusCheck:
    path = Path(path)
    if not path.exists():
        return StatusCheck("journal", True, "no journal yet (no memos generated)",
                           {"memos": 0, "pending": 0})
    journal = DecisionJournal(path)
    memos = journal.memos()
    pending = journal.pending()
    metrics = {"memos": len(memos), "pending": len(pending)}
    issues = []
    if len(pending) > max_pending:
        issues.append(f"{len(pending)} pending outcomes (> {max_pending}) - run quant-desk-outcomes")
    if memos and max_last_memo_days is not None:
        age = (datetime.now(timezone.utc) - memos[-1].created_at).days
        metrics["last_memo_age_days"] = age
        if age > max_last_memo_days:
            issues.append(f"last memo {age}d old (> {max_last_memo_days}d)")
    return StatusCheck("journal", not issues, "; ".join(issues) or f"{len(memos)} memos, {len(pending)} pending", metrics)


def check_audit_trail(path: Path | str) -> StatusCheck:
    path = Path(path)
    if not path.exists():
        return StatusCheck("execution_audit", True, "no executions yet (paper loop idle)", {"records": 0})
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    import json
    rejected = sum(1 for ln in lines if not json.loads(ln).get("approved", False))
    return StatusCheck("execution_audit", True, f"{len(lines)} records ({rejected} rejections)",
                       {"records": len(lines), "rejections": rejected})


def check_cache(directory: Path | str, max_age_hours: float = 48.0) -> StatusCheck:
    directory = Path(directory)
    files = list(directory.glob("*.json")) if directory.exists() else []
    if not files:
        return StatusCheck("cache", True, "empty (cold start is normal)", {"entries": 0})
    newest_age_h = (time.time() - max(f.stat().st_mtime for f in files)) / 3600
    metrics = {"entries": len(files), "newest_age_hours": round(newest_age_h, 1)}
    if newest_age_h > max_age_hours:
        return StatusCheck("cache", False,
                           f"newest entry {newest_age_h:.0f}h old (> {max_age_hours:.0f}h) - data path may be idle/broken",
                           metrics)
    return StatusCheck("cache", True, f"{len(files)} entries, newest {newest_age_h:.1f}h old", metrics)


def run_all(
    openbb_url: str = "http://127.0.0.1:6900",
    quantdinger_url: str = "http://127.0.0.1:5000",
    journal_path: str = "reports/research/journal.jsonl",
    audit_path: str = "reports/research/executions.jsonl",
    cache_dir: str = "datasets/cache",
) -> list[StatusCheck]:
    return [
        check_http_service("openbb_rest", f"{openbb_url}/openapi.json"),
        check_http_service("quantdinger_backend", f"{quantdinger_url}/api/health"),
        check_journal(journal_path),
        check_audit_trail(audit_path),
        check_cache(cache_dir),
    ]


def main() -> None:
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openbb-url", default="http://127.0.0.1:6900")
    parser.add_argument("--quantdinger-url", default="http://127.0.0.1:5000")
    parser.add_argument("--journal", default="reports/research/journal.jsonl")
    parser.add_argument("--audit", default="reports/research/executions.jsonl")
    parser.add_argument("--cache-dir", default="datasets/cache")
    args = parser.parse_args()

    checks = run_all(args.openbb_url, args.quantdinger_url, args.journal, args.audit, args.cache_dir)
    degraded = [c for c in checks if not c.healthy]
    for c in checks:
        print(json.dumps({"check": c.name, "healthy": c.healthy, "detail": c.detail, **c.metrics}))
    print(f"\nSTATUS: {'DEGRADED (' + ', '.join(c.name for c in degraded) + ')' if degraded else 'ALL HEALTHY'}")
    sys.exit(1 if degraded else 0)


if __name__ == "__main__":
    main()
