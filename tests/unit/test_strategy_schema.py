"""The strategy JSON Schema (ADR-0005) must accept the example and reject violations."""
import json
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "config/strategies/strategy.schema.json").read_text())
EXAMPLE = json.loads((ROOT / "config/strategies/example-btc-trend.json").read_text())


def validate(doc):
    jsonschema.Draft202012Validator(SCHEMA).validate(doc)


def test_example_validates():
    validate(EXAMPLE)


def test_failed_validation_result_rejected():
    doc = json.loads(json.dumps(EXAMPLE))
    doc["validation_report"]["result"] = "fail"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_missing_validation_report_rejected():
    doc = json.loads(json.dumps(EXAMPLE))
    del doc["validation_report"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_oversized_risk_rejected():
    doc = json.loads(json.dumps(EXAMPLE))
    doc["risk"]["max_position_pct_equity"] = 50
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_unknown_field_rejected():
    doc = json.loads(json.dumps(EXAMPLE))
    doc["surprise"] = True
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_bad_semver_rejected():
    doc = json.loads(json.dumps(EXAMPLE))
    doc["version"] = "1.0"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)
