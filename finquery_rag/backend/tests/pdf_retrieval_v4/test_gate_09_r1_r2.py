from __future__ import annotations

from src.pdf_retrieval_v4.evidence_set_cover import build_sets
from src.pdf_retrieval_v4.evidence_slot_matcher_v2 import match_slot, matrix_covers_slot
from src.pdf_retrieval_v4.operand_projection import project_operands


def slot(slot_id="current", shape="atomic_fact", metric="Revenue", period="FY2025"):
    return {
        "slot_id": slot_id,
        "required_evidence_shape": shape,
        "raw_metric_phrase": metric,
        "concept_candidates": [],
        "period": period,
        "segment_label": None,
        "bucket_label": None,
    }


def evidence(
    evidence_id="a",
    kind="atomic_fact",
    metric="Revenue",
    periods=None,
    rank=1,
    row="r",
    table="t",
):
    payload = {
        "metric": metric,
        "period": (periods or ["FY2025"])[0],
        "periods": periods or ["FY2025"],
        "parsed_value": "10",
        "scale": "million",
        "currency": "USD",
    }
    return {
        "evidence_id": evidence_id,
        "evidence_type": kind,
        "candidate_key": evidence_id,
        "supporting_candidate_keys": [evidence_id],
        "candidate_rank": rank,
        "document_id": "d",
        "payload": payload,
        "source_traceback": [
            {"row_id": row, "table_fragment_id": table, "pdf_page": 1}
        ],
    }


def plan(*slots):
    return {"plan_id": "p", "operation": "difference", "operand_slots": list(slots)}


def test_comparison_not_satisfied_by_atomic() -> None:
    assert match_slot(slot(shape="comparison_fact"), evidence()) is None


def test_row_matrix_shape_requires_matrix() -> None:
    assert match_slot(slot(shape="row_matrix"), evidence()) is None


def test_matrix_requires_metric_and_dimension() -> None:
    matrix = evidence(
        kind="row_matrix", metric="Operating income", periods=["FY2025", "FY2024"]
    )["payload"]
    assert not matrix_covers_slot(matrix, slot(metric="Revenue"))
    assert not matrix_covers_slot(
        matrix, slot(metric="Operating income", period="FY2023")
    )


def test_no_top20_truncation_and_complete_beats_partial() -> None:
    current = [evidence(f"wrong{i}", metric="Other", rank=i) for i in range(1, 30)]
    current.append(evidence("current", rank=39))
    previous = evidence("previous", periods=["FY2024"], rank=40)
    result = build_sets(
        plan(slot(), slot("previous", period="FY2024")), [*current, previous]
    )
    primary = next(
        item
        for item in result["sets"]
        if item["evidence_set_id"] == result["primary_set_id"]
    )
    assert primary["complete_slot_count"] == 2
    assert set(primary["evidence_ids"]) == {"current", "previous"}


def test_row_matrix_full_cover_preferred() -> None:
    matrix = evidence("matrix", "row_matrix", periods=["FY2025", "FY2024"], rank=10)
    result = build_sets(
        plan(slot(), slot("previous", period="FY2024")),
        [matrix, evidence("a"), evidence("b", periods=["FY2024"])],
    )
    assert result["primary_status"] == "unique"
    assert next(
        item
        for item in result["sets"]
        if item["evidence_set_id"] == result["primary_set_id"]
    )["row_matrix_full_cover"]


def test_ambiguous_primary_has_no_primary_id() -> None:
    left = evidence("a", rank=1, row="r1", table="t1")
    right = evidence("b", rank=1, row="r2", table="t2")
    result = build_sets(plan(slot()), [left, right])
    assert result["primary_status"] == "ambiguous"
    assert result["primary_set_id"] is None
    assert len(result["co_primary_set_ids"]) == 2


def test_calculation_requires_unique_primary() -> None:
    items = [evidence("a", rank=1, row="r1"), evidence("b", rank=1, row="r2")]
    result = build_sets(plan(slot()), items)
    assert not project_operands(plan(slot()), result, items)["typed_calculation_ready"]


def test_calculation_requires_all_slots_and_traceback() -> None:
    item = evidence()
    result = build_sets(plan(slot()), [item])
    assert project_operands(plan(slot()), result, [item])["typed_calculation_ready"]
    item["source_traceback"] = []
    assert not project_operands(plan(slot()), result, [item])["typed_calculation_ready"]


def test_segment_and_bucket_use_explicit_dimensions() -> None:
    item = evidence(kind="row_matrix")
    item["payload"]["dimensions"] = [{"labels": ["Americas", "1-3 years"]}]
    segment = slot()
    segment["segment_label"] = "Americas"
    bucket = slot(shape="bucket_fact")
    bucket["bucket_label"] = "1-3 years"
    assert match_slot(segment, item)
    assert match_slot(bucket, item)
    item["payload"]["dimensions"] = []
    assert match_slot(segment, item) is None
    assert match_slot(bucket, item) is None


def test_scale_conflict_blocks_calculation() -> None:
    item = evidence()
    item["payload"]["scale_status"] = "conflict"
    result = build_sets(plan(slot()), [item])
    assert not project_operands(plan(slot()), result, [item])["typed_calculation_ready"]
