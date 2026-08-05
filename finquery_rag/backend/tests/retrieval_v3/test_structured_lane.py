from src.retrieval_v3.structured_lane import (
    append_structured_residual,
    enriched_retrieval_text,
    fixed_rrf,
    is_safe_structured_view,
)


def _view() -> dict:
    return {
        "candidate_key": "candidate:one",
        "evidence_id": "user_1_doc::page_1::table_1::row_1",
        "evidence_type": "table_row",
        "document_field": {"company": "Example", "fiscal_year": "FY2025"},
        "section_field": {"statement_title": "Statements of Income", "table_title": None},
        "metric_field": {"normalized_metric": "net sales"},
        "period_field": {"periods": ["FY2025"]},
        "unit_field": {"currency": "USD", "scale": "million"},
    }


def test_structured_candidate_identity_and_safe_text() -> None:
    view = _view()
    assert is_safe_structured_view(view)
    text = enriched_retrieval_text(view, "Net sales | 10")
    assert "metric net sales" in text
    assert "table periods FY2025" in text
    assert "cell" not in text.casefold()


def test_structured_query_scope_is_table_row_only() -> None:
    view = _view()
    view["evidence_type"] = "text"
    assert not is_safe_structured_view(view)


def test_raw_pool_protection_and_residual_merge() -> None:
    raw = [{"candidate_key": "raw-a", "score": 0.4}, {"candidate_key": "raw-b", "score": 0.3}]
    structured = [
        {"candidate_key": "raw-b", "structured_score": 0.9},
        {"candidate_key": "new-c", "structured_score": 0.8},
    ]
    result = append_structured_residual(raw, structured)
    assert result.raw_unchanged
    assert result.duplicate_count == 1
    assert result.combined[:2] == raw
    assert [item["candidate_key"] for item in result.combined] == ["raw-a", "raw-b", "new-c"]


def test_noneligible_pool_can_remain_byte_for_byte_equal() -> None:
    raw = [{"candidate_key": "raw-a", "score": 0.4}]
    result = append_structured_residual(raw, [])
    assert result.combined == raw


def test_fixed_rrf_is_identity_deduplicated_and_deterministic() -> None:
    first = fixed_rrf(["a", "b"], ["b", "c"], limit=20)
    second = fixed_rrf(["a", "b"], ["b", "c"], limit=20)
    assert first == second
    assert [item[0] for item in first] == ["b", "a", "c"]
