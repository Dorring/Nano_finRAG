from __future__ import annotations

from src.pdf_retrieval_v4.authoritative_evidence_attachment import rehydrate_evidence
from src.pdf_retrieval_v4.evidence_set_cover_v3 import build_sets
from src.pdf_retrieval_v4.evidence_slot_matcher_v3 import (
    match_slot,
    statement_compatibility,
)
from src.pdf_retrieval_v4.operand_projection_v2 import project


def slot(slot_id="current", period="FY2025"):
    return {
        "slot_id": slot_id,
        "required_evidence_shape": "atomic_fact",
        "raw_metric_phrase": "Revenue",
        "concept_candidates": [],
        "period": period,
        "segment_label": None,
        "bucket_label": None,
    }


def plan(*slots, hint=None, prefer_row=True, prefer_table=True):
    return {
        "plan_id": "p",
        "operation": "difference",
        "operand_slots": list(slots),
        "statement_hint": hint,
        "constraints": {
            "prefer_same_row": prefer_row,
            "prefer_same_logical_table": prefer_table,
        },
    }


def evidence(
    eid="a",
    kind="atomic_fact",
    periods=None,
    rank=1,
    table="t",
    row="r",
    statement="income_statement",
):
    if kind == "row_matrix":
        payload = {
            "semantic_fact_id": eid,
            "leaf_metric": "Revenue",
            "metric_path": "Revenue",
            "dimensions": [
                {"normalized_period": period, "value_normalized": str(index + 1)}
                for index, period in enumerate(periods or ["FY2025", "FY2024"])
            ],
            "scale": 1,
            "currency_code": "USD",
        }
    else:
        payload = {
            "semantic_fact_id": eid,
            "leaf_metric": "Revenue",
            "metric_path": "Revenue",
            "normalized_period": (periods or ["FY2025"])[0],
            "value_normalized": "1",
            "scale": 1,
            "currency_code": "USD",
        }
    return {
        "evidence_id": eid,
        "evidence_type": kind,
        "candidate_key": eid,
        "supporting_candidate_keys": [eid],
        "candidate_rank": rank,
        "document_id": "d",
        "semantic_payload": payload,
        "context": {
            "table_fragment_id": table,
            "row_id": row,
            "statement_type": statement,
            "table_title": statement,
        },
        "source_traceback": {"pdf_page": 1},
    }


def test_rehydrate_uses_authoritative_payload_and_context() -> None:
    frozen = {
        "evidence_id": "atomic:a",
        "evidence_type": "atomic_fact",
        "candidate_key": "c",
        "candidate_rank": 1,
        "supporting_candidate_keys": ["c"],
        "document_id": "d",
        "payload": {"metric": "wrong"},
        "source_traceback": [],
    }
    authoritative = {
        "atomic:a": {
            "semantic_fact_id": "atomic:a",
            "document_id": "d",
            "table_fragment_id": "t",
            "row_id": "r",
            "leaf_metric": "Revenue",
            "metric_path": "Revenue",
            "source_traceback": {"pdf_page": 1},
        }
    }
    item = rehydrate_evidence(
        frozen,
        authoritative,
        {"t": {"table_title": "Income", "statement_type": "income_statement"}},
        {"r": {"row_index": 2, "row_type": "metric"}},
        {},
    )
    assert item["semantic_payload"]["leaf_metric"] == "Revenue"
    assert item["context"]["table_title"] == "Income"


def test_statement_hint_compatible_conflict_and_unknown() -> None:
    item = evidence()
    assert (
        statement_compatibility(plan(slot(), hint="income statement"), item)[0]
        == "compatible"
    )
    assert (
        statement_compatibility(plan(slot(), hint="cash flow"), item)[0] == "conflict"
    )
    item["context"]["statement_type"] = None
    item["context"]["table_title"] = None
    assert statement_compatibility(plan(slot(), hint="cash flow"), item)[0] == "neutral"


def test_matrix_slots_bind_different_dimensions() -> None:
    item = evidence("m", "row_matrix")
    p = plan(slot(), slot("previous", "FY2024"))
    result = build_sets(p, [item])
    projection = project(p, result, [item])
    assert projection["typed_calculation_ready"]
    assert (
        projection["operands"]["current"]["dimension_identity"]
        != projection["operands"]["previous"]["dimension_identity"]
    )
    assert (
        projection["operands"]["current"]["value"]
        != projection["operands"]["previous"]["value"]
    )


def test_wrong_metric_matrix_rejected() -> None:
    item = evidence("m", "row_matrix")
    item["semantic_payload"]["leaf_metric"] = "Operating income"
    item["semantic_payload"]["metric_path"] = "Operating income"
    assert match_slot(plan(slot()), slot(), item) is None


def test_query_silent_collision_stays_ambiguous() -> None:
    result = build_sets(
        plan(slot(), hint=None),
        [evidence("a", table="t1", row="r1"), evidence("b", table="t2", row="r2")],
    )
    assert result["primary_status"] == "ambiguous"
    assert result["primary_set_id"] is None


def test_context_conflict_rejected_before_ranking() -> None:
    p = plan(slot(), hint="cash flow")
    assert match_slot(p, slot(), evidence(statement="balance_sheet")) is None
