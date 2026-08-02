from __future__ import annotations

from src.evaluation.benchmark_source_binding import choose_candidate


def _label(*, value: str = "1000000", metric: str = "Revenue") -> dict:
    return {
        "expected_answer": {"canonical_value": value},
        "expected_sources": [{
            "document_id": "doc-1",
            "page": 3,
            "row_label": metric,
            "column_header": "2025",
            "unit": "currency",
            "scale": "1",
        }],
    }


def _candidate(
    *,
    key: str,
    document_id: str = "doc-1",
    page: int = 3,
    content: str = "Revenue | 1 | 2025",
    row_label: str = "Revenue",
) -> dict:
    return {
        "candidate_key": key,
        "benchmark_document_id": document_id,
        "page": page,
        "block_type": "table_row",
        "content": content,
        "metadata": {"row_label": row_label},
    }


def test_top1_candidate_is_not_automatically_bound():
    label = _label()
    source = label["expected_sources"][0]
    decision = choose_candidate(
        label=label,
        source=source,
        source_index=0,
        candidates=[_candidate(key="a"), _candidate(key="b")],
        by_id={},
        top20_keys={"a"},
    )
    assert decision.status == "ambiguous"
    assert decision.candidate is None


def test_candidate_document_must_match():
    label = _label()
    source = label["expected_sources"][0]
    decision = choose_candidate(
        label=label,
        source=source,
        source_index=0,
        candidates=[_candidate(key="wrong", document_id="doc-2")],
        by_id={},
        top20_keys=set(),
    )
    assert decision.status == "missing_from_index"


def test_candidate_page_requires_exact_or_verified_offset():
    label = _label()
    source = label["expected_sources"][0]
    decision = choose_candidate(
        label=label,
        source=source,
        source_index=0,
        candidates=[_candidate(key="wrong-page", page=4)],
        by_id={},
        top20_keys=set(),
    )
    assert decision.status == "missing_from_index"


def test_candidate_value_and_period_must_match():
    label = _label()
    source = label["expected_sources"][0]
    candidate = _candidate(key="wrong-value", content="Revenue | 2 | 2024")
    decision = choose_candidate(
        label=label,
        source=source,
        source_index=0,
        candidates=[candidate],
        by_id={},
        top20_keys=set(),
    )
    assert decision.status == "missing_from_index"


def test_missing_index_candidate_is_not_fabricated():
    label = _label()
    source = label["expected_sources"][0]
    decision = choose_candidate(
        label=label,
        source=source,
        source_index=0,
        candidates=[],
        by_id={},
        top20_keys=set(),
    )
    assert decision.status == "missing_from_index"
    assert decision.candidate is None
