from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.canonical_metric_identity import (
    canonical_metric_id,
    canonical_metric_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-09-r5-2-r0"
R51 = BASE / "pdf-retrieval-v4-gate-09-r5-1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_canonical_metric_normalizes_path_order_and_closed_morphology() -> None:
    assert canonical_metric_tokens("Services net sales") == ("net", "sale", "service")
    assert canonical_metric_id("Services net sales") == canonical_metric_id("Net Sales / Services")
    assert canonical_metric_id("operating income") == canonical_metric_id("income / operations")


def test_canonical_metric_preserves_financial_modifiers() -> None:
    assert canonical_metric_id("gross margin") != canonical_metric_id("gross profit")
    assert canonical_metric_id("total assets") != canonical_metric_id("current assets")
    assert canonical_metric_id("GAAP revenue") != canonical_metric_id("non-GAAP revenue")


def test_canonical_metric_does_not_apply_subset_or_synonym_matching() -> None:
    assert canonical_metric_id("net sales") != canonical_metric_id("total net sales")
    assert canonical_metric_id("revenue") != canonical_metric_id("net sales")


def test_r0_is_postseal_only_and_r5_1_is_immutable() -> None:
    protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
    integrity = json.loads((OUT / "input-integrity.json").read_text(encoding="utf-8"))
    seal_path = R51 / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    assert protocol["stage"] == "post_seal_diagnostic_only"
    assert protocol["prediction_rerun"] == protocol["binder_mutation"] == 0
    assert protocol["retrieval_runs"] == protocol["reranker_calls"] == 0
    assert protocol["embedding_calls"] == protocol["calculator_calls"] == 0
    assert protocol["canonical_metric_used_for_binding"] is False
    assert integrity["r5_1_prediction_seal_sha256"] == sha256(seal_path)
    assert integrity["r5_1_output_sha256"] == {
        name: sha256(R51 / filename)
        for name, filename in {
            "classes_v2": "semantic-evidence-classes-v2.jsonl.gz",
            "metric_bindings": "metric-binding-candidates.jsonl.gz",
            "joint_bindings": "joint-operand-bindings.jsonl.gz",
            "projections_v2": "operand-projections-v2.jsonl.gz",
            "sets_v2": "evidence-set-predictions-v2.jsonl.gz",
        }.items()
    }
    assert integrity["r5_1_output_sha256"] == seal["output_sha256"]


def test_every_calculation_slot_has_one_first_failure() -> None:
    rows = read_gzip(OUT / "calculation-slot-audit.jsonl.gz")
    allowed = {
        "metric_representation_mismatch",
        "true_metric_absence",
        "multiple_operand_tuples",
        "period_or_dimension_conflict",
        "shape_conflict",
        "other",
    }
    assert len(rows) == 22
    assert len({(row["case_id"], row["slot_id"]) for row in rows}) == 22
    assert all(row["first_failure"] in allowed for row in rows)
    assert all(row["m2_diagnostic_only"] is True for row in rows)


def test_case_matrix_separates_logical_table_and_fragment_coherence() -> None:
    matrix = json.loads((OUT / "calculation-case-blocker-matrix.json").read_text(encoding="utf-8"))
    assert len(matrix["cases"]) == 11
    diagnostics = [item["joint_assignment_diagnostic"] for item in matrix["cases"]]
    assert all(
        item.get("logical_table_identity_is_distinct_from_fragment_identity", True)
        for item in diagnostics
    )
    assert any(
        item["same_logical_table_assignment_count"]
        != item["same_fragment_assignment_count"]
        for item in diagnostics
    )


def test_formal_r0_result_selects_structural_binder_ablation() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
    prereg = json.loads((OUT / "ablation-preregistration.json").read_text(encoding="utf-8"))
    assert acceptance["calculation_case_count"] == 11
    assert acceptance["required_slot_count"] == 22
    assert acceptance["first_failure_counts"] == {
        "multiple_operand_tuples": 12,
        "other": 4,
        "true_metric_absence": 6,
    }
    assert acceptance["metric_representation_significant"] is False
    assert acceptance["multiple_operand_tuples_significant"] is True
    assert acceptance["decision"] == "multiple_operand_tuple_failure_dominant"
    assert acceptance["next_gate"] == "structural_joint_binder_repair"
    assert acceptance["recommended_ablation"] == "B2"
    assert prereg["no_ablation_executed_in_r0"] is True
