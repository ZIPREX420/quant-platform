"""sign_report: correct sha256, workspace-relative path, containment check."""
import hashlib
import json
from pathlib import Path

import pytest

from quant_platform.strategies.loader import load_strategy
from quant_platform.strategies.sign_report import build_report_reference

REPO = Path(__file__).resolve().parents[2]


def test_block_matches_loader_expectations(tmp_path):
    ws = tmp_path / "ws"
    (ws / "docs" / "validation").mkdir(parents=True)
    report = ws / "docs" / "validation" / "s-v1.md"
    report.write_text("report body\n", encoding="utf-8")

    block = build_report_reference(report, ws)
    assert block["path"] == "docs/validation/s-v1.md"
    assert block["sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert block["result"] == "pass"

    # end-to-end: a definition using this block loads successfully
    definition = json.loads((REPO / "config/strategies/example-btc-trend.json").read_text())
    definition["validation_report"] = {**block, "signed_off_by": "pytest"}
    def_path = tmp_path / "s.json"
    def_path.write_text(json.dumps(definition), encoding="utf-8")
    loaded = load_strategy(def_path, ws)
    assert loaded.report_path == report.resolve()


def test_report_outside_workspace_rejected(tmp_path):
    ws = tmp_path / "ws"; ws.mkdir()
    outside = tmp_path / "elsewhere.md"
    outside.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not inside workspace root"):
        build_report_reference(outside, ws)
