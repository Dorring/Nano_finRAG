from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.structural_joint_binder_v2 import (
    bind_structural_operands_b2,
    canonical_row_label,
    hydrate_structural_provenance,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-09-r5-2-b2"
R51 = BASE / "pdf-retrieval-v4-gate-09-r5-1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def semantic_class(
    fact_id: str,
    period: str,
    value: str,
    *,
    row_id: str,
    fragment_id: str,
    raw_row_label: str,
    logical_id: str | None,
) -> dict:
    source = {
        "row_id": row_id,
        "table_fragment_id": fragment_id,
        "logical_table_id": logical_id,
        "logical_table_status": "resolved" if logical_id else "unresolved",
        "raw_row_label": raw_row_label,
        "canonical_row_label": canonical_row_label(raw_row_label),
        "canonical_row_identity": [logical_id, canonical_row_label(raw_row_label)]
        if logical_id
        else None,
    }
    return {
        "semantic_fact_id": fact_id,
        "document_id": "issuer_fy2025",
        "metric": "revenue",
        "period": period,
        "segment": None,
        "bucket": None,
        "value": value,
        "measurement_kind": "monetary_amount",
        "unit_context": {
            "scale": "1000000",
            "currency": "USD",
            "scale_status": "resolved",
            "currency_status": "resolved",
        },
        "supporting_candidate_keys": [f"candidate-{fact_id}"],
        "supporting_evidence_ids": [f"evidence-{fact_id}"],
        "physical_provenance": [source],
    }


def plan() -> dict:
    return {
        "task_type": "calculation_multi_operand",
        "operation": "growth_rate",
        "constraints": {"prefer_same_row": True, "prefer_same_logical_table": True},
    }


def options(current: list[dict], previous: list[dict]) -> list[dict]:
    return [
        {"slot": {"raw_metric_phrase": "revenue"}, "compatible_classes": current},
        {"slot": {"raw_metric_phrase": "revenue"}, "compatible_classes": previous},
    ]


def test_hydration_never_uses_fragment_as_logical_fallback() -> None:
    item = semantic_class(
        "fact", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id=None
    )
    item["physical_provenance"] = [{"row_id": "row-a", "table_fragment_id": "fragment-a"}]
    hydrated = hydrate_structural_provenance(item, {}, {"row-a": "Revenue"})
    source = hydrated["physical_provenance"][0]
    assert source["logical_table_id"] is None
    assert source["logical_table_status"] == "unresolved"
    assert source["canonical_row_identity"] is None


def test_same_logical_table_different_fragments_remains_coherent() -> None:
    current = semantic_class(
        "current", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    previous = semantic_class(
        "previous", "fy2024", "90", row_id="row-b", fragment_id="fragment-b", raw_row_label="Different label", logical_id="logical-a"
    )
    result = bind_structural_operands_b2(plan(), options([current], [previous]))
    assert result["logical_table_filter_applied"] is True
    assert result["binding_status"] == "deterministic_ready"
    assert result["selected_assignment"]["same_logical_table"] is True
    assert result["selected_assignment"]["same_table_fragment"] is False


def test_same_row_label_in_different_logical_tables_is_not_equivalent() -> None:
    current = semantic_class(
        "current", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Total revenues", logical_id="logical-a"
    )
    previous = semantic_class(
        "previous", "fy2024", "90", row_id="row-b", fragment_id="fragment-b", raw_row_label="Total revenue", logical_id="logical-b"
    )
    result = bind_structural_operands_b2(plan(), options([current], [previous]))
    assert result["canonical_row_filter_applied"] is False
    assert result["logical_table_filter_applied"] is False
    assert result["binding_status"] == "deterministic_ready"


def test_normalized_row_label_in_same_logical_table_is_canonical_row() -> None:
    current = semantic_class(
        "current", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Total Revenues", logical_id="logical-a"
    )
    previous = semantic_class(
        "previous", "fy2024", "90", row_id="row-b", fragment_id="fragment-b", raw_row_label="total revenue", logical_id="logical-a"
    )
    result = bind_structural_operands_b2(plan(), options([current], [previous]))
    assert result["canonical_row_filter_applied"] is True
    assert result["selected_assignment"]["same_canonical_row"] is True


def test_structural_filter_only_applies_when_complete_coherent_assignment_exists() -> None:
    current = semantic_class(
        "current", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue A", logical_id="logical-a"
    )
    previous = semantic_class(
        "previous", "fy2024", "90", row_id="row-b", fragment_id="fragment-b", raw_row_label="Revenue B", logical_id="logical-b"
    )
    result = bind_structural_operands_b2(plan(), options([current], [previous]))
    assert result["canonical_row_filter_applied"] is False
    assert result["logical_table_filter_applied"] is False
    assert result["assignment_lineage"]["before"] == {
        "physical_assignments": 1,
        "operand_tuples": 1,
    }
    assert result["binding_status"] == "deterministic_ready"


def test_different_operand_values_after_b2_remain_ambiguous_without_rank() -> None:
    current_a = semantic_class(
        "current-a", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    current_b = semantic_class(
        "current-b", "fy2025", "110", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    previous = semantic_class(
        "previous", "fy2024", "90", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    result = bind_structural_operands_b2(plan(), options([current_a, current_b], [previous]))
    assert result["binding_status"] == "runtime_operand_ambiguity"
    assert result["assignment_count"] == 2
    assert result["rank_used_to_resolve_ambiguity"] is False


def test_equivalent_physical_sources_collapse_to_one_operand_tuple() -> None:
    current_a = semantic_class(
        "current-a", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    current_b = semantic_class(
        "current-b", "fy2025", "100", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    previous = semantic_class(
        "previous", "fy2024", "90", row_id="row-a", fragment_id="fragment-a", raw_row_label="Revenue", logical_id="logical-a"
    )
    result = bind_structural_operands_b2(plan(), options([current_a, current_b], [previous]))
    assert result["binding_status"] == "deterministic_ready"
    assert result["assignment_lineage"]["before"]["physical_assignments"] == 2
    assert result["assignment_lineage"]["after_semantic_tuple_collapse"]["operand_tuples"] == 1
    assert result["selected_assignment"]["equivalent_semantic_fact_ids"][0] == [
        "current-a",
        "current-b",
    ]


def test_b2_seal_freezes_r5_1_and_disables_m2_rank_and_concepts() -> None:
    seal = json.loads((OUT / "prediction-seal.json").read_text(encoding="utf-8"))
    r51_seal_path = R51 / "prediction-seal.json"
    r51_seal = json.loads(r51_seal_path.read_text(encoding="utf-8"))
    assert seal["sealed"] is True
    assert seal["r5_1_prediction_seal_sha256"] == sha256(r51_seal_path)
    assert seal["r5_1_output_sha256"] == r51_seal["output_sha256"]
    assert seal["metric_contract"] == "R5.1_M0_M1_exact"
    assert seal["m2_canonical_metric"] is False
    assert seal["concept_candidate_deterministic"] is False
    assert seal["rank_resolves_ambiguity"] is False
    assert seal["gold_reads_before_seal"] == seal["strict_binding_reads_before_seal"] == 0
    assert seal["retrieval_runs"] == seal["reranker_calls"] == seal["embedding_calls"] == 0
    assert seal["unit_contract_source_sha256"] == r51_seal["source_sha256"][
        "operation_unit_contract.py"
    ]


def test_six_ambiguous_calculation_cases_have_complete_lineage() -> None:
    lineage = json.loads((OUT / "calculation-assignment-lineage.json").read_text(encoding="utf-8"))
    assert len(lineage["cases"]) == 6
    for record in lineage["cases"]:
        assert set(record) >= {
            "before",
            "after_same_canonical_row",
            "after_same_logical_table",
            "after_semantic_tuple_collapse",
            "final_status",
        }


def test_formal_b2_result_confirms_safe_gain() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
    false_binding = json.loads((OUT / "false-binding-audit.json").read_text(encoding="utf-8"))
    assert false_binding["false_slot_binding"] == 0
    assert acceptance["calculation_runtime_ready"] == "3/11"
    assert acceptance["calculation_runtime_ambiguous"] == "5/11"
    assert acceptance["calculation_undercovered"] == "3/11"
    assert acceptance["calculation_unit_blocked"] == "0/11"
    assert acceptance["runtime_ready_case_ids"] == [
        "ko_fy2025_006",
        "nvda_fy2025_006",
        "pfe_fy2024_006",
    ]
    assert acceptance["decision"] == "structural_joint_binding_gain_confirmed"
    assert acceptance["next_gate"] == "deterministic_calculator_shadow"


def test_every_b2_evidence_set_stays_within_budget() -> None:
    for record in read_gzip(OUT / "evidence-sets-b2.jsonl.gz"):
        assert record["evidence_item_count"] == len(record["selected_candidate_keys"])
        assert record["evidence_item_count"] <= 5
