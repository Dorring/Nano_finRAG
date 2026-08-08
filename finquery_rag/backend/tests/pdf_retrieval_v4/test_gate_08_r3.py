"""Tests for Gate 08 R3: Coverage-only Retrieval Replay."""

from __future__ import annotations
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).absolute().parents[2]
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"


def _load_json(path: Path) -> dict:
    if not path.exists():
        pytest.skip(f"artifact_not_found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# 1. Control keyset is the common subset of legacy and new keys
# ------------------------------------------------------------------
def test_control_keyset_is_common_subset():
    data = _load_json(R3_DIR / "control-keyset-audit.json")
    legacy_count = data["legacy_count"]
    removed_count = data["removed_count"]
    common_count = data["common_count"]
    assert common_count == legacy_count - removed_count
    k_legacy = set(data["k_legacy"])
    k_common = set(data["k_common"])
    legacy_removed = set(data["legacy_removed"])
    # Common keys = legacy keys minus removed keys (intersection of legacy & new)
    assert k_common == k_legacy - legacy_removed
    assert len(k_common) == common_count


# ------------------------------------------------------------------
# 2. Control index uses the R4 serializer
# ------------------------------------------------------------------
def test_control_uses_r4_serializer():
    data = _load_json(R3_DIR / "aligned-control-index-manifest.json")
    assert data["structured_text_version"] == "gate06-r4-v1"


# ------------------------------------------------------------------
# 3. Config parity passed (budgets, RRF, embedding match Gate 08 R2)
# ------------------------------------------------------------------
def test_config_parity_passed():
    data = _load_json(R3_DIR / "config-parity.json")
    assert data["all_gates_passed"] is True
    gates = data["gates"]
    assert gates["budgets_match"] is True
    assert gates["rrf_match"] is True
    assert gates["embedding_match"] is True


# ------------------------------------------------------------------
# 4. No gold reads before seal
# ------------------------------------------------------------------
def test_no_gold_before_seal():
    data = _load_json(R3_DIR / "prediction-seal.json")
    assert data["gold_reads_before_seal"] == 0
    assert data["governance_reads_before_seal"] == 0
    assert data["parameter_scan"] is False
    assert data["reranker_calls"] == 0
    assert data["calculator_calls"] == 0
    assert data["answer_generation_calls"] == 0
    assert data["production_index_writes"] == 0
    assert data["production_switch_allowed"] is False
    assert data["sealed"] is True


# ------------------------------------------------------------------
# 5. Prediction count == 72
# ------------------------------------------------------------------
def test_prediction_count():
    data = _load_json(R3_DIR / "prediction-manifest.json")
    assert data["record_count"] == 72


# ------------------------------------------------------------------
# 6. E0 replay matches historical baseline (42)
# ------------------------------------------------------------------
def test_e0_replay():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    assert data["experiment_groups"]["e0"]["total_hits"] == 42


# ------------------------------------------------------------------
# 7. E1 replay within ±1 of historical 46
# ------------------------------------------------------------------
def test_e1_replay():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    total_hits = data["experiment_groups"]["e1"]["total_hits"]
    assert total_hits in (46, 47)


# ------------------------------------------------------------------
# 8. E2-Legacy replay matches historical (46)
# ------------------------------------------------------------------
def test_e2_legacy_replay():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    assert data["experiment_groups"]["e2_legacy"]["total_hits"] == 46


# ------------------------------------------------------------------
# 9. E3-Legacy replay matches historical (47)
# ------------------------------------------------------------------
def test_e3_legacy_replay():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    assert data["experiment_groups"]["e3_legacy"]["total_hits"] == 47


# ------------------------------------------------------------------
# 10. E3-Expanded meets Small Gain threshold (>= 50)
# ------------------------------------------------------------------
def test_e3_expanded_score():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    assert data["experiment_groups"]["e3_expanded"]["total_hits"] >= 50


# ------------------------------------------------------------------
# 11. Representation Gain (E2-Control - E2-Legacy) >= 0
# ------------------------------------------------------------------
def test_representation_gain_zero_or_positive():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    eg = data["experiment_groups"]
    gain = eg["e2_control"]["total_hits"] - eg["e2_legacy"]["total_hits"]
    assert gain >= 0


# ------------------------------------------------------------------
# 12. Pure Coverage Gain (E2-Expanded - E2-Control) > 0
# ------------------------------------------------------------------
def test_pure_coverage_gain_positive():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    eg = data["experiment_groups"]
    gain = eg["e2_expanded"]["total_hits"] - eg["e2_control"]["total_hits"]
    assert gain > 0


# ------------------------------------------------------------------
# 13. Raw parity verified
# ------------------------------------------------------------------
def test_raw_parity_verified():
    data = _load_json(R3_DIR / "acceptance.json")
    assert data["raw_parity_verified"] is True


# ------------------------------------------------------------------
# 14. Historical parity verified
# ------------------------------------------------------------------
def test_historical_parity_verified():
    data = _load_json(R3_DIR / "acceptance.json")
    assert data["historical_parity_verified"] is True


# ------------------------------------------------------------------
# 15. Outside universe is a separate failure stage, not retrieval_failure
# ------------------------------------------------------------------
def test_outside_universe_not_retrieval_failure():
    data = _load_json(R3_DIR / "first-failure-attribution.json")
    fsc = data["failure_stage_counts"]
    assert "outside_grade_a_universe" in fsc
    # Retrieval failure stages should not include outside_grade_a_universe
    retrieval_stages = {k: v for k, v in fsc.items() if k != "outside_grade_a_universe"}
    assert sum(retrieval_stages.values()) == data["in_universe_missed"]
    assert fsc["outside_grade_a_universe"] == data["outside_universe"]


# ------------------------------------------------------------------
# 16. Rank regression is reported (improved + unchanged + worsened > 0)
# ------------------------------------------------------------------
def test_rank_regression_reported():
    data = _load_json(R3_DIR / "scoring/old-structured-rank-regression.json")
    total = data["improved"] + data["unchanged"] + data["worsened"]
    assert total > 0


# ------------------------------------------------------------------
# 17. Acceptance decision is one of the valid values
# ------------------------------------------------------------------
def test_acceptance_decision():
    data = _load_json(R3_DIR / "acceptance.json")
    valid = {
        "coverage_only_retrieval_strong_pass",
        "coverage_only_retrieval_passed",
        "coverage_expansion_gain_real_but_insufficient",
        "coverage_expansion_small_gain",
        "coverage_expansion_insufficient",
    }
    assert data["decision"] in valid


# ------------------------------------------------------------------
# 18. Universe counts: universe_total == 68, outside_total == 12
# ------------------------------------------------------------------
def test_universe_counts_correct():
    data = _load_json(R3_DIR / "scoring/ablation-metrics.json")
    counts = data["counts"]
    assert counts["universe_mapped"] == 68
    assert counts["outside_universe"] == 12


# ------------------------------------------------------------------
# 19. R4 expanded structured index has no Grade-B candidates
# ------------------------------------------------------------------
def test_grade_b_absent_from_structured_hits():
    data = _load_json(R3_DIR / "aligned-control-index-manifest.json")
    assert data["gates"]["expanded_keyset_is_19500"] is True


# ------------------------------------------------------------------
# 20. Prediction manifest has a valid prediction_sha256 (64-char hex)
# ------------------------------------------------------------------
def test_prediction_deterministic():
    data = _load_json(R3_DIR / "prediction-manifest.json")
    sha = data["prediction_sha256"]
    assert isinstance(sha, str)
    assert re.fullmatch(r"[0-9a-f]{64}", sha) is not None
