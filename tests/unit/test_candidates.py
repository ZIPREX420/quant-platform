"""ADR-0006 candidate-tier contract: these tests ARE the tier separation."""
import json
from pathlib import Path

import pytest

from quant_platform.strategies import (
    CandidateLoadError,
    LoadedCandidate,
    StrategyLoadError,
    load_candidate,
    load_candidate_dir,
    load_strategy,
)

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "config/strategies/example-btc-trend.json"

PREDICTION = (
    "Pre-registered prediction: this candidate loses money net of costs, "
    "because the regime finding says naive funding carry inverted in 2024."
)


def make_candidate(tmp_path: Path, **overrides) -> Path:
    definition = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    del definition["validation_report"]
    definition["id"] = "test-candidate"
    definition["tracking"] = {
        "prediction": PREDICTION,
        "registered_by": "pytest",
        "registered_date": "2026-07-10",
    }
    definition.update(overrides)
    path = tmp_path / "test-candidate.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    return path


def test_valid_candidate_loads(tmp_path):
    loaded = load_candidate(make_candidate(tmp_path))
    assert isinstance(loaded, LoadedCandidate)
    assert loaded.id == "test-candidate"
    assert loaded.tier == "candidate"
    assert loaded.prediction == PREDICTION


def test_candidate_with_validation_report_refused(tmp_path):
    # a validated strategy must never masquerade as a candidate
    definition = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    definition["tracking"] = {
        "prediction": PREDICTION, "registered_by": "pytest", "registered_date": "2026-07-10",
    }
    path = tmp_path / "masquerade.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    with pytest.raises(CandidateLoadError, match="carries a validation_report"):
        load_candidate(path)


def test_candidate_never_satisfies_validated_loader(tmp_path):
    # the reverse direction: a candidate can never sneak into the validated path
    path = make_candidate(tmp_path)
    with pytest.raises(StrategyLoadError, match="schema violation"):
        load_strategy(path, tmp_path)


def test_missing_prediction_refused(tmp_path):
    path = make_candidate(tmp_path, tracking={"registered_by": "x", "registered_date": "2026-07-10"})
    with pytest.raises(CandidateLoadError, match="schema violation at tracking"):
        load_candidate(path)


def test_trivial_prediction_refused(tmp_path):
    # a prediction too short to be falsifiable is refused (minLength 40)
    path = make_candidate(
        tmp_path,
        tracking={"prediction": "it goes up", "registered_by": "x", "registered_date": "2026-07-10"},
    )
    with pytest.raises(CandidateLoadError, match="schema violation"):
        load_candidate(path)


def test_risk_caps_still_enforced(tmp_path):
    # tier separation relaxes the report requirement, NEVER the risk vocabulary
    definition = json.loads(make_candidate(tmp_path).read_text(encoding="utf-8"))
    definition["risk"]["max_position_pct_equity"] = 99
    path = tmp_path / "over-cap.json"
    path.write_text(json.dumps(definition), encoding="utf-8")
    with pytest.raises(CandidateLoadError, match="schema violation"):
        load_candidate(path)


def test_dir_load_fail_closed(tmp_path):
    make_candidate(tmp_path)
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CandidateLoadError):
        load_candidate_dir(tmp_path)


def test_dir_load_sorted_and_complete(tmp_path):
    make_candidate(tmp_path)
    definition = json.loads((tmp_path / "test-candidate.json").read_text(encoding="utf-8"))
    definition["id"] = "another-candidate"
    (tmp_path / "another.json").write_text(json.dumps(definition), encoding="utf-8")
    loaded = load_candidate_dir(tmp_path)
    assert [c.id for c in loaded] == ["another-candidate", "test-candidate"]


def test_missing_dir_refused(tmp_path):
    with pytest.raises(CandidateLoadError, match="does not exist"):
        load_candidate_dir(tmp_path / "nope")


def test_candidate_error_is_strategy_error():
    # callers guarding with StrategyLoadError catch candidate refusals too
    assert issubclass(CandidateLoadError, StrategyLoadError)
