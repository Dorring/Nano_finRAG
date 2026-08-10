from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.joint_operand_binder import bind_joint_operands
from src.pdf_retrieval_v4.metric_binding_contract_v2 import bind_metric
from src.pdf_retrieval_v4.operation_unit_contract import evaluate_operation_units
from src.pdf_retrieval_v4.unit_context_resolver import resolve_unit_context

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-09-r5-1"
R5 = BASE / "pdf-retrieval-v4-gate-09-r5"


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
    metric: str = "revenue",
    row: str = "row-1",
    table: str = "table-1",
    scale: str | None = "1000000",
    currency: str | None = "USD",
    measurement_kind: str = "monetary_amount",
) -> dict:
    return {
        "semantic_fact_id": fact_id,
        "document_id": "issuer_fy2025",
        "metric": metric,
        "period": period,
        "segment": None,
        "bucket": None,
        "value": value,
        "measurement_kind": measurement_kind,
        "supporting_candidate_keys": [f"candidate-{fact_id}"],
        "supporting_evidence_ids": [f"evidence-{fact_id}"],
        "physical_provenance": [{"row_id": row, "table_fragment_id": table}],
        "unit_context": {
            "scale": scale,
            "currency": currency,
            "scale_status": "resolved" if scale else "unresolved",
            "currency_status": "resolved" if currency else "unresolved",
        },
    }


def plan(operation: str = "growth_rate") -> dict:
    return {
        "task_type": "calculation_multi_operand",
        "operation": operation,
        "constraints": {"prefer_same_row": True, "prefer_same_logical_table": True},
    }


def test_concept_candidate_alone_cannot_create_deterministic_match() -> None:
    slot = {"raw_metric_phrase": "total sales", "concept_candidates": ["revenue"]}
    result = bind_metric(slot, {"metric": "revenue"})
    assert result["concept_hint_only"] is True
    assert result["deterministic_compatible"] is False


def test_exact_raw_metric_creates_deterministic_match() -> None:
    result = bind_metric(
        {"raw_metric_phrase": "revenue", "concept_candidates": []},
        {"metric": "operating results/revenue"},
    )
    assert result["metric_tier"] == "M1_exact_leaf"
    assert result["deterministic_compatible"] is True


def test_gross_margin_percentage_never_binds_gross_profit_amount() -> None:
    result = bind_metric(
        {"raw_metric_phrase": "GAAP gross margin", "concept_candidates": ["gross profit"]},
        {"metric": "gross profit"},
    )
    assert result["measurement_kind_conflict"] is True
    assert result["deterministic_compatible"] is False


def test_same_metric_two_periods_prefers_complete_same_row_assignment() -> None:
    current_same = semantic_class("current-same", "fy2025", "100", row="row-1")
    current_cross = semantic_class("current-cross", "fy2025", "101", row="row-2")
    previous = semantic_class("previous", "fy2024", "90", row="row-1")
    result = bind_joint_operands(
        plan(),
        [
            {"slot": {"raw_metric_phrase": "revenue"}, "compatible_classes": [current_same, current_cross]},
            {"slot": {"raw_metric_phrase": "revenue"}, "compatible_classes": [previous]},
        ],
    )
    assert result["same_row_filter_applied"] is True
    assert result["binding_status"] == "deterministic_ready"
    assert result["selected_assignment"]["semantic_fact_ids"] == ["current-same", "previous"]


def test_equivalent_physical_assignments_collapse_without_rank() -> None:
    current_a = semantic_class("current-a", "fy2025", "100")
    current_b = semantic_class("current-b", "fy2025", "100")
    previous_a = semantic_class("previous-a", "fy2024", "90")
    previous_b = semantic_class("previous-b", "fy2024", "90")
    result = bind_joint_operands(
        plan(),
        [
            {"slot": {"raw_metric_phrase": "revenue"}, "compatible_classes": [current_a, current_b]},
            {"slot": {"raw_metric_phrase": "revenue"}, "compatible_classes": [previous_a, previous_b]},
        ],
    )
    assert result["physical_assignment_count"] == 4
    assert result["assignment_count"] == 1
    assert result["binding_status"] == "deterministic_ready"
    assert result["rank_used_to_resolve_ambiguity"] is False
    assert result["selected_assignment"]["equivalent_semantic_fact_ids"] == [
        ["current-a", "current-b"],
        ["previous-a", "previous-b"],
    ]


def test_different_operand_tuples_remain_ambiguous() -> None:
    result = bind_joint_operands(
        plan(),
        [
            {
                "slot": {"raw_metric_phrase": "revenue"},
                "compatible_classes": [
                    semantic_class("current-a", "fy2025", "100"),
                    semantic_class("current-b", "fy2025", "110"),
                ],
            },
            {
                "slot": {"raw_metric_phrase": "revenue"},
                "compatible_classes": [semantic_class("previous", "fy2024", "90")],
            },
        ],
    )
    assert result["binding_status"] == "runtime_operand_ambiguity"
    assert result["rank_used_to_resolve_ambiguity"] is False


def test_growth_rate_shared_unknown_scale_and_currency_is_ready() -> None:
    operands = [
        semantic_class("current", "fy2025", "100", scale=None, currency=None),
        semantic_class("previous", "fy2024", "80", scale=None, currency=None),
    ]
    result = evaluate_operation_units("growth_rate", operands, same_row=True, same_table=True)
    assert result["ready"] is True
    assert result["scale_contract"] == "unresolved_shared_cancels"
    assert result["currency_contract"] == "unresolved_shared_cancels"


def test_growth_rate_different_known_scales_normalizes_values() -> None:
    operands = [
        semantic_class("current", "fy2025", "100", scale="1000000"),
        semantic_class("previous", "fy2024", "80000", scale="1000"),
    ]
    result = evaluate_operation_units("growth_rate", operands, same_row=True, same_table=True)
    assert result["ready"] is True
    assert result["normalized_values"] == ["1E+8", "8E+7"]


def test_difference_with_unresolved_scale_is_unit_blocked() -> None:
    operands = [
        semantic_class("current", "fy2025", "100", scale=None),
        semantic_class("previous", "fy2024", "80", scale=None),
    ]
    result = evaluate_operation_units("difference", operands, same_row=True, same_table=True)
    assert result == {"ready": False, "reason": "scale_required", "normalized_values": []}


def test_percentage_does_not_require_currency() -> None:
    operand = semantic_class(
        "margin", "fy2025", "74.4", metric="gross margin", currency=None, measurement_kind="percentage"
    )
    result = evaluate_operation_units(None, [operand], same_row=True, same_table=True)
    assert result["ready"] is True
    assert result["currency_contract"] == "not_required"


def test_monetary_difference_requires_currency() -> None:
    operands = [
        semantic_class("current", "fy2025", "100", currency=None),
        semantic_class("previous", "fy2024", "80", currency=None),
    ]
    result = evaluate_operation_units("difference", operands, same_row=True, same_table=True)
    assert result["ready"] is False
    assert result["reason"] == "currency_required_or_conflict"


def test_unit_context_prefers_authoritative_evidence_over_table_context() -> None:
    item = semantic_class("fact", "fy2025", "100", scale=None, currency=None)
    item["physical_provenance"] = [
        {
            "authoritative_evidence_id": "atomic:fact",
            "table_fragment_id": "table-1",
        }
    ]
    result = resolve_unit_context(
        item,
        {"table-1": {"scale": "1000", "scale_status": "resolved"}},
        {"table-1": {"currency_code": "EUR", "currency_status": "resolved"}},
        {"atomic:fact": {"scale": "1000000", "currency_code": "USD"}},
    )
    assert result["scale"] == "1000000"
    assert result["currency"] == "USD"
    assert result["scale_source"] == result["currency_source"] == "authoritative_evidence"


def test_r5_1_prediction_is_zero_gold_and_frozen_to_r5() -> None:
    seal = json.loads((OUT / "prediction-seal.json").read_text(encoding="utf-8"))
    assert seal["sealed"] is True
    assert seal["prediction_count"] == 72
    assert seal["gold_reads_before_seal"] == seal["strict_binding_reads_before_seal"] == 0
    assert seal["retrieval_runs"] == seal["reranker_calls"] == seal["embedding_calls"] == 0
    assert seal["candidate_mutation"] == seal["semantic_registry_mutation"] == 0
    assert seal["r5_access_universe_sha256"] == sha256(R5 / "evidence-access-universe.jsonl.gz")


def test_r5_1_sets_stay_inside_frozen_access_and_budget() -> None:
    access = {row["case_id"]: row for row in read_gzip(R5 / "evidence-access-universe.jsonl.gz")}
    for record in read_gzip(OUT / "evidence-set-predictions-v2.jsonl.gz"):
        allowed = {item["candidate_key"] for item in access[record["case_id"]]["candidates"]}
        assert set(record["selected_candidate_keys"]) <= allowed
        assert record["evidence_item_count"] <= 5


def test_r5_1_formal_result_is_fail_closed_without_false_binding() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
    false_binding = json.loads((OUT / "false-binding-audit.json").read_text(encoding="utf-8"))
    assert false_binding["false_slot_binding"] == 0
    assert acceptance["calculation_runtime_ready"] == "2/11"
    assert acceptance["decision"] == "deterministic_operand_binding_insufficient"
    assert acceptance["next_gate"] == "joint_operand_ambiguity_contract_repair"
