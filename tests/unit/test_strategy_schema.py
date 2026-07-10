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


class TestSchemaV12ForwardEvidence:
    def test_report_type_accepted(self, tmp_path):
        import hashlib
        import json
        from pathlib import Path
        from quant_platform.strategies.loader import load_strategy
        repo = Path(__file__).resolve().parents[2]
        ws = tmp_path / "ws"
        (ws / "docs" / "validation").mkdir(parents=True)
        report = ws / "docs" / "validation" / "r.md"
        report.write_text("result: pass (forward-evidence)\n", encoding="utf-8")
        definition = json.loads(
            (repo / "config/strategies/example-btc-trend.json").read_text()
        )
        definition["id"] = "fwd-test"
        definition["validation_report"].update(
            path="docs/validation/r.md",
            sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
            signed_off_by="pytest",
            type="forward-evidence",
        )
        path = tmp_path / "s.json"
        path.write_text(json.dumps(definition), encoding="utf-8")
        assert load_strategy(path, ws).definition["validation_report"]["type"] == "forward-evidence"

    def test_unknown_report_type_refused(self, tmp_path):
        import json
        import pytest
        from pathlib import Path
        from quant_platform.strategies.loader import StrategyLoadError, load_strategy
        repo = Path(__file__).resolve().parents[2]
        definition = json.loads(
            (repo / "config/strategies/example-btc-trend.json").read_text()
        )
        definition["validation_report"]["type"] = "vibes"
        path = tmp_path / "s.json"
        path.write_text(json.dumps(definition), encoding="utf-8")
        with pytest.raises(StrategyLoadError, match="schema violation"):
            load_strategy(path, tmp_path)
