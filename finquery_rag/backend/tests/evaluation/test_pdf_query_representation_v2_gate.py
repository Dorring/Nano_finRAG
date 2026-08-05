import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONCEPT = ROOT / "artifacts/evaluation/pdf-query-representation-v2"
HYBRID = ROOT / "artifacts/evaluation/pdf-query-representation-v2-gate-d"
GATE_E = ROOT / "artifacts/evaluation/pdf-query-representation-v2-gate-e"


def test_concept_gate_is_issuer_disjoint_and_passes() -> None:
    split = json.loads((CONCEPT / "document-split-manifest.json").read_text())
    acceptance = json.loads((CONCEPT / "acceptance.json").read_text())
    assert split["strategy"] == "three_fold_leave_one_issuer_out"
    assert acceptance["concept_gate_passed"] is True
    assert acceptance["frozen_72_question_reads"] == 0
    assert acceptance["production_switch_allowed"] is False


def test_hybrid_gate_blocks_transfer_on_regressions() -> None:
    acceptance = json.loads((HYBRID / "acceptance.json").read_text())
    report = json.loads((HYBRID / "hybrid-funnel-results.json").read_text())
    assert report["selected_variant"] == "top_1_canonical_query"
    assert report["per_query_oracle_selection"] is False
    assert acceptance["final_recall_at_5_gain"] > 0.08
    assert acceptance["regressed_strict_hit_count"] > 1
    assert acceptance["gate_passed"] is False
    assert acceptance["frozen_transfer_allowed"] is False
    assert acceptance["frozen_72_question_reads"] == 0


def test_query_role_decoupling_and_dual_lane_close_safely() -> None:
    acceptance = json.loads((GATE_E / "acceptance.json").read_text())
    dual = json.loads((GATE_E / "gate-e2-dual-lane-results.json").read_text())
    assert acceptance["e1_development_strategy_passed"] is False
    assert acceptance["dual_lane_executed"] is True
    assert dual["equal_weights"] is True
    assert dual["rrf_k"] == 60
    assert dual["final_recall_at_5_gain"] < 0.08
    assert acceptance["decision"] == "canonical_query_retrieval_gain_not_regression_safe"
    assert acceptance["next_gate"] == "stop_query_representation_v2"
    assert acceptance["frozen_transfer_allowed"] is False


def test_gate_e_reports_missing_holdout_and_hard_no_answer_evidence() -> None:
    holdout = json.loads((GATE_E / "holdout-corpus-availability.json").read_text())
    no_answer = json.loads((GATE_E / "hard-no-answer-report.json").read_text())
    assert holdout["eligible_additional_nonbenchmark_issuer_count"] == 0
    assert holdout["holdout_validation_possible"] is False
    assert no_answer["status"] == "not_run"
    assert no_answer["abstention_f1"] is None
