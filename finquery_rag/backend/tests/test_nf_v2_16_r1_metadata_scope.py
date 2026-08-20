import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.retrieval.metadata_scope import (
    FilterStrength,
    MetadataFilterPlannerV1,
    MetadataProvenance,
    apply_hard_scope,
    enforce_reranker_subset,
)


def _row(doc_id, **metadata):
    metadata.setdefault("doc_id", doc_id)
    return {"doc_id": doc_id, "content": "fact", "metadata": metadata}


def test_planner_marks_explicit_scope_hard_and_content_soft():
    scope = MetadataFilterPlannerV1().plan("MSFT 2024 Q2 revenue", authorization_scope={"user_id": 7})
    assert scope.hard_filters["tenant_id"] == ["7"]
    assert scope.hard_filters["ticker"] == ["MSFT"]
    assert scope.hard_filters["fiscal_year"] == ["2024"]
    assert scope.hard_filters["fiscal_quarter"] == ["Q2"]
    assert scope.hard_filters["period_semantics"] == ["QUARTER"]
    assert all(item.strength is FilterStrength.SOFT for item in scope.soft_conditions)


def test_missing_explicit_metadata_fails_closed_without_relaxation():
    scope = MetadataFilterPlannerV1().plan("MSFT FY2024 revenue", authorization_scope={"user_id": 1})
    accepted, metrics = apply_hard_scope([_row("missing", user_id=1, ticker="MSFT")], scope)
    assert accepted == []
    assert metrics["silent_hard_filter_relaxations"] == 0
    assert metrics["filter_invariant_violations"] == 0
    assert metrics["hard_filter_rejections"] == 1


def test_latest_uses_filing_date_not_created_at():
    scope = MetadataFilterPlannerV1().plan("MSFT latest annual report", authorization_scope={"user_id": 1})
    rows = [
        _row("old", user_id=1, ticker="MSFT", document_type="ANNUAL", period_semantics="ANNUAL", filing_date="2025-01-01", created_at="2030-01-01"),
        _row("new", user_id=1, ticker="MSFT", document_type="ANNUAL", period_semantics="ANNUAL", filing_date="2025-03-01", created_at="2024-01-01"),
    ]
    accepted, metrics = apply_hard_scope(rows, scope)
    assert [row["doc_id"] for row in accepted] == ["new"]
    assert metrics["created_at_temporal_misuse"] == 0


def test_reranker_output_is_subset_only():
    base = [_row("a", user_id=1)]
    selected, violations = enforce_reranker_subset(base, [_row("outside", user_id=2), base[0]])
    assert [row["doc_id"] for row in selected] == ["a"]
    assert violations == 1


def test_provenance_unknown_is_preserved():
    scope = MetadataFilterPlannerV1().plan("revenue")
    assert scope.soft_conditions
    row = _row("x", user_id=1, provenance={"fiscal_year": MetadataProvenance.UNKNOWN.value})
    assert row["metadata"]["provenance"]["fiscal_year"] == "UNKNOWN"




def test_distinct_metric_slots_are_not_temporal_conflicts():
    from rag_v2.adaptive import EvidenceConsistencyGateV1, EvidencePacketV1
    packets = [
        EvidencePacketV1(evidence_id="r", metric="revenue", value="100", period="FY2024", entity="Microsoft"),
        EvidencePacketV1(evidence_id="i", metric="operating income", value="25", period="FY2024", entity="Microsoft"),
    ]
    assert EvidenceConsistencyGateV1().evaluate(packets).decision.value != "UNRESOLVED_CONFLICT"
