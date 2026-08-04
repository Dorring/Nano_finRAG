from src.evaluation.nf_opt_14 import (
    candidate_slot_compatibility,
    deterministic_slot_selector,
    parse_query_slot_contract,
)


def test_growth_contract_binds_metric_and_previous_current_periods():
    contract = parse_query_slot_contract({"case_id": "x", "question": "How much did total net sales grow from FY2024 to FY2025?", "document_scope": ["doc"]})
    assert contract["contract_status"] == "complete"
    assert [(slot["role"], slot["period"]) for slot in contract["slots"]] == [("previous", "FY2024"), ("current", "FY2025")]
    assert contract["slots"][0]["normalized_metric_tokens"] == ["total", "net", "sales"]


def test_percentage_share_uses_distinct_same_period_metrics():
    contract = parse_query_slot_contract({"case_id": "x", "question": "What percentage of Total revenue was Cloud revenue in FY2025?", "document_scope": ["doc"]})
    assert [(slot["role"], slot["period"]) for slot in contract["slots"]] == [("numerator", "FY2025"), ("denominator", "FY2025")]
    assert contract["slots"][0]["metric_phrase"] != contract["slots"][1]["metric_phrase"]


def test_direct_multi_period_fact_has_two_slots():
    contract = parse_query_slot_contract({"case_id": "x", "question": "What was revenue reported in FY2024 and FY2025?", "document_scope": ["doc"]})
    assert contract["required_evidence_count"] == 2
    assert [slot["period"] for slot in contract["slots"]] == ["FY2024", "FY2025"]


def test_compatibility_requires_document_metric_and_period():
    slot = {"slot_id": "current", "normalized_metric_tokens": ["total", "revenue"], "period": "FY2025"}
    result = candidate_slot_compatibility(candidate={"candidate_key": "a", "canonical_document_id": "doc"}, candidate_text="Total revenue 2025", slot=slot, document_scope={"doc"})
    assert result["compatibility"] == "strict"
    assert candidate_slot_compatibility(candidate={"candidate_key": "a", "canonical_document_id": "doc"}, candidate_text="Total revenue 2024", slot=slot, document_scope={"doc"})["compatibility"] == "partial"


def test_selector_preserves_single_slot_baseline_and_allows_one_candidate_multi_slot():
    baseline = [{"candidate_key": "base"}]
    reranked = [{"candidate_key": "both"}, {"candidate_key": "base"}]
    single = deterministic_slot_selector(baseline_final=baseline, reranked=reranked, slots=[{"slot_id": "fact"}], matrix=[])
    assert single == baseline
    multiple = deterministic_slot_selector(baseline_final=baseline, reranked=reranked, slots=[{"slot_id": "first"}, {"slot_id": "second"}], matrix=[{"candidate_key": "both", "slot_id": "first", "compatibility": "strict"}, {"candidate_key": "both", "slot_id": "second", "compatibility": "strict"}])
    assert multiple[0]["candidate_key"] == "both"
