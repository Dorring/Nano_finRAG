from __future__ import annotations

from src.pdf_retrieval_v4.candidate_evidence_attachment import (
    attach_candidate,
    canonicalize,
)
from src.pdf_retrieval_v4.evidence_set_generator import generate_evidence_sets
from src.pdf_retrieval_v4.evidence_slot_matcher import match_slot


def _plan(slots):
    return {"plan_id": "p", "operand_slots": slots}


def _slot(slot_id="s", period="FY2025"):
    return {
        "slot_id": slot_id,
        "raw_metric_phrase": "revenue",
        "concept_candidates": ["net sales"],
        "period": period,
        "required_evidence_shape": "atomic_fact",
        "segment_label": None,
        "bucket_label": None,
    }


def _evidence(eid="e", key="c", rank=1):
    return {
        "evidence_id": eid,
        "evidence_type": "atomic_fact",
        "candidate_key": key,
        "candidate_rank": rank,
        "document_id": "d",
        "payload": {"metric": "Revenue", "period": "FY2025", "value": "1"},
        "source_traceback": [{"pdf_page": 1}],
    }


def test_raw_candidate_never_typed() -> None:
    ref = canonicalize(attach_candidate("c", 1, None, ("d",)))[0]
    assert ref["evidence_type"] == "raw_candidate"
    assert match_slot(_slot(), ref) is None


def test_metric_and_period_exact() -> None:
    match = match_slot(_slot(), _evidence())
    assert match and match["metric_grade"] == "A_exact" and match["period_match"]


def test_period_conflict_fails_closed() -> None:
    assert match_slot(_slot(period="FY2024"), _evidence()) is None


def test_canonical_dedup_keeps_support() -> None:
    view = {
        "document_id": "d",
        "facts": [
            {
                "type": "atomic",
                "evidence_id": "e",
                "metric": "Revenue",
                "period": "FY2025",
            }
        ],
        "source_traceback": [{"pdf_page": 1}],
    }
    refs = attach_candidate("a", 2, view, ("d",)) + attach_candidate(
        "b", 1, view, ("d",)
    )
    item = canonicalize(refs)[0]
    assert item["candidate_key"] == "b" and item["supporting_candidate_keys"] == [
        "a",
        "b",
    ]


def test_multi_slot_complete_and_deterministic() -> None:
    plan = _plan([_slot("a"), _slot("b")])
    result = generate_evidence_sets(plan, [_evidence()])
    assert result["planner_complete"] and result == generate_evidence_sets(
        plan, [_evidence()]
    )
    assert result["sets"][0]["evidence_count"] == 1


def test_candidate_rank_breaks_tie() -> None:
    result = generate_evidence_sets(
        _plan([_slot()]), [_evidence("late", "b", 9), _evidence("early", "a", 2)]
    )
    assert result["sets"][0]["slot_mapping"]["s"]["evidence_id"] == "early"


def test_slot_match_preserves_canonical_supporting_candidates() -> None:
    evidence = _evidence()
    evidence["supporting_candidate_keys"] = ["c", "alias"]
    match = match_slot(_slot(), evidence)
    assert match and match["supporting_candidate_keys"] == ["c", "alias"]
