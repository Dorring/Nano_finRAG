import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONCEPT = ROOT / "artifacts/evaluation/pdf-query-representation-v2"
HYBRID = ROOT / "artifacts/evaluation/pdf-query-representation-v2-gate-d"


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
