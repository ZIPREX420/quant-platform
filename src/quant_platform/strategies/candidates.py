"""Candidate loader: the enforcement point of ADR-0006's candidate tier.

A *candidate* is a schema-valid strategy definition WITHOUT a validation
report, carrying a pre-registered prediction of what its paper record will
show. Candidates may only ever reach PaperTradingSession - they are
hypotheses under forward test, not trading recommendations.

The candidate schema is DERIVED from the strategy schema at load time (one
rule vocabulary, one source of truth): ``validation_report`` is removed and
``tracking`` is required. Because both schemas set additionalProperties to
false, the two artifact kinds are structurally disjoint:

  * a candidate file containing a validation_report is refused here, and
  * a candidate file can never satisfy load_strategy (no validation_report).

Neither loader can substitute for the other.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

import jsonschema

from quant_platform.strategies.loader import StrategyLoadError, _schema

_TRACKING_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["prediction", "registered_by", "registered_date"],
    "properties": {
        "prediction": {
            "type": "string",
            "minLength": 40,
            "maxLength": 2000,
            "description": (
                "Pre-registered, falsifiable statement of what this candidate's "
                "paper record is expected to show and why (ADR-0006)."
            ),
        },
        "registered_by": {"type": "string", "minLength": 1},
        "registered_date": {"type": "string", "format": "date"},
        "hypothesis": {
            "type": "string",
            "maxLength": 200,
            "description": "Research-agenda hypothesis id, e.g. 'H1'.",
        },
        "journal_ref": {
            "type": "string",
            "maxLength": 200,
            "description": "Decision-journal memo id or notebook path backing this registration.",
        },
    },
}


class CandidateLoadError(StrategyLoadError):
    """A candidate definition was refused. The message states exactly why."""


@dataclass(frozen=True)
class LoadedCandidate:
    """A schema-valid, prediction-backed candidate. Paper tier only (ADR-0006)."""

    id: str
    version: str
    definition: dict
    definition_path: Path
    prediction: str
    tier: str = field(default="candidate", init=False)


def candidate_schema() -> dict:
    """The candidate schema, derived from the strategy schema (single vocabulary)."""
    schema = copy.deepcopy(_schema())
    schema["title"] = "Candidate definition (ADR-0006 candidate tier - paper trading only)"
    schema["description"] = (
        "An unvalidated strategy definition registered for forward testing on "
        "paper. Must NOT carry a validation_report; must carry a pre-registered "
        "tracking.prediction. Derived from strategy.schema.json at load time."
    )
    schema["required"] = [r for r in schema["required"] if r != "validation_report"] + ["tracking"]
    del schema["properties"]["validation_report"]
    schema["properties"]["tracking"] = copy.deepcopy(_TRACKING_SCHEMA)
    return schema


def load_candidate(definition_path: Path | str) -> LoadedCandidate:
    """Load one candidate definition, enforcing the ADR-0006 candidate contract."""
    definition_path = Path(definition_path)

    try:
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CandidateLoadError(f"{definition_path.name}: unreadable or invalid JSON: {exc}") from exc

    if isinstance(definition, dict) and "validation_report" in definition:
        raise CandidateLoadError(
            f"{definition_path.name}: carries a validation_report - a validated "
            f"strategy must be loaded via load_strategy(), never as a candidate "
            f"(ADR-0006 tier separation)."
        )

    try:
        jsonschema.Draft202012Validator(candidate_schema()).validate(definition)
    except jsonschema.ValidationError as exc:
        raise CandidateLoadError(
            f"{definition_path.name}: schema violation at "
            f"{'/'.join(str(p) for p in exc.absolute_path) or '<root>'}: {exc.message}"
        ) from exc

    return LoadedCandidate(
        id=definition["id"],
        version=definition["version"],
        definition=definition,
        definition_path=definition_path,
        prediction=definition["tracking"]["prediction"],
    )


def load_candidate_dir(directory: Path | str) -> list[LoadedCandidate]:
    """Load every *.json candidate in a directory.

    Fail-closed: any refused candidate aborts the whole load, so a cycle can
    never silently run a partial candidate set.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise CandidateLoadError(f"candidate directory does not exist: {directory}")
    return [load_candidate(path) for path in sorted(directory.glob("*.json"))]
