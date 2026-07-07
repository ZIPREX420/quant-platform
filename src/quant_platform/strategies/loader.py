"""Strategy-artifact loader: the enforcement point of ADR-0005.

A strategy is loadable only if ALL of the following hold:
  1. The definition validates against config/strategies/strategy.schema.json
     (the single source of truth - no duplicated validation logic here).
  2. Its validation report exists at the declared workspace-relative path.
  3. The report's sha256 matches the declared digest (tamper evidence).

Anything else raises StrategyLoadError. Execution code must only ever receive
LoadedStrategy instances produced here.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import jsonschema

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "config" / "strategies" / "strategy.schema.json"


class StrategyLoadError(Exception):
    """A strategy definition was refused. The message states exactly why."""


@dataclass(frozen=True)
class LoadedStrategy:
    """A validated, report-backed strategy definition."""

    id: str
    version: str
    definition: dict
    definition_path: Path
    report_path: Path


def _schema() -> dict:
    try:
        return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StrategyLoadError(f"strategy schema unreadable at {_SCHEMA_PATH}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_strategy(definition_path: Path | str, workspace_root: Path | str) -> LoadedStrategy:
    """Load one strategy definition, enforcing the full ADR-0005 contract."""
    definition_path = Path(definition_path)
    workspace_root = Path(workspace_root)

    try:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise StrategyLoadError(f"{definition_path.name}: unreadable or invalid JSON: {exc}") from exc

    try:
        jsonschema.Draft202012Validator(_schema()).validate(definition)
    except jsonschema.ValidationError as exc:
        raise StrategyLoadError(
            f"{definition_path.name}: schema violation at "
            f"{'/'.join(str(p) for p in exc.absolute_path) or '<root>'}: {exc.message}"
        ) from exc

    report_rel = definition["validation_report"]["path"]
    report_path = (workspace_root / report_rel).resolve()
    if not report_path.is_file():
        raise StrategyLoadError(
            f"{definition_path.name}: validation report missing at {report_rel} "
            f"- a strategy without its report is not loadable (ADR-0005)."
        )

    declared = definition["validation_report"]["sha256"]
    actual = _sha256(report_path)
    if actual != declared:
        raise StrategyLoadError(
            f"{definition_path.name}: validation report digest mismatch "
            f"(declared {declared[:12]}..., actual {actual[:12]}...) - report was "
            f"modified after sign-off or the definition is stale."
        )

    return LoadedStrategy(
        id=definition["id"],
        version=definition["version"],
        definition=definition,
        definition_path=definition_path,
        report_path=report_path,
    )


def load_strategy_dir(directory: Path | str, workspace_root: Path | str) -> list[LoadedStrategy]:
    """Load every *.json strategy in a directory (schema file excluded).

    Fail-closed: any refused strategy aborts the whole load, so execution can
    never start with a partial strategy set by accident.
    """
    directory = Path(directory)
    loaded: list[LoadedStrategy] = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "strategy.schema.json":
            continue
        loaded.append(load_strategy(path, workspace_root))
    return loaded
