"""The loader must refuse anything that violates ADR-0005 - these tests are the contract."""
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from quant_platform.strategies import LoadedStrategy, StrategyLoadError, load_strategy, load_strategy_dir

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "config/strategies/example-btc-trend.json"


def make_valid_artifact(tmp_path: Path) -> tuple[Path, Path]:
    """A fabricated-but-contract-complete strategy: definition + real report with matching sha."""
    ws = tmp_path / "ws"
    (ws / "docs" / "validation").mkdir(parents=True)
    report = ws / "docs" / "validation" / "test-strategy-v0.1.0.md"
    report.write_text("# validation report\nresult: pass\n", encoding="utf-8")
    definition = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    definition["id"] = "test-strategy"
    definition["validation_report"]["path"] = "docs/validation/test-strategy-v0.1.0.md"
    definition["validation_report"]["sha256"] = hashlib.sha256(report.read_bytes()).hexdigest()
    definition["validation_report"]["signed_off_by"] = "pytest"
    def_path = tmp_path / "test-strategy.json"
    def_path.write_text(json.dumps(definition), encoding="utf-8")
    return def_path, ws


def test_valid_artifact_loads(tmp_path):
    def_path, ws = make_valid_artifact(tmp_path)
    loaded = load_strategy(def_path, ws)
    assert isinstance(loaded, LoadedStrategy)
    assert loaded.id == "test-strategy" and loaded.version == "0.1.0"
    assert loaded.report_path.is_file()


def test_example_refused_missing_report(tmp_path):
    # the shipped schema example points at a report that does not exist -> must be refused
    with pytest.raises(StrategyLoadError, match="report missing"):
        load_strategy(EXAMPLE, tmp_path)


def test_tampered_report_refused(tmp_path):
    def_path, ws = make_valid_artifact(tmp_path)
    report = ws / "docs" / "validation" / "test-strategy-v0.1.0.md"
    report.write_text("# validation report\nresult: pass\nEDITED AFTER SIGN-OFF\n", encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="digest mismatch"):
        load_strategy(def_path, ws)


def test_schema_violation_refused(tmp_path):
    def_path, ws = make_valid_artifact(tmp_path)
    definition = json.loads(def_path.read_text(encoding="utf-8"))
    definition["risk"]["max_position_pct_equity"] = 99  # beyond hard cap
    def_path.write_text(json.dumps(definition), encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="schema violation"):
        load_strategy(def_path, ws)


def test_invalid_json_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{oops", encoding="utf-8")
    with pytest.raises(StrategyLoadError, match="invalid JSON"):
        load_strategy(bad, tmp_path)


def test_dir_load_fails_closed(tmp_path):
    def_path, ws = make_valid_artifact(tmp_path)
    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    shutil.copy(def_path, strat_dir / "good.json")
    shutil.copy(EXAMPLE, strat_dir / "bad-example.json")  # will be refused
    with pytest.raises(StrategyLoadError):
        load_strategy_dir(strat_dir, ws)


def test_dir_load_skips_schema_and_loads_valid(tmp_path):
    def_path, ws = make_valid_artifact(tmp_path)
    strat_dir = tmp_path / "strategies"
    strat_dir.mkdir()
    shutil.copy(def_path, strat_dir / "good.json")
    shutil.copy(REPO / "config/strategies/strategy.schema.json", strat_dir / "strategy.schema.json")
    loaded = load_strategy_dir(strat_dir, ws)
    assert [s.id for s in loaded] == ["test-strategy"]
