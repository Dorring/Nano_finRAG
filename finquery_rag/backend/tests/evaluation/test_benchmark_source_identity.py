from __future__ import annotations

from src.evaluation.benchmark_source_binding import choose_candidate


def _candidate(key: str, value: str, row_label: str) -> dict:
    return {
        "candidate_key": key,
        "benchmark_document_id": "doc-1",
        "page": 5,
        "block_type": "table_row",
        "content": f"{row_label} | {value} | 2025",
        "metadata": {"row_label": row_label},
    }


def test_each_expected_source_has_separate_identity():
    label = {
        "calculation": {
            "operation": "growth_rate",
            "operands": [
                {"value": "1000000", "metric": "Revenue"},
                {"value": "1200000", "metric": "Revenue"},
            ],
        },
        "expected_sources": [
            {"document_id": "doc-1", "page": 5, "row_label": "Revenue", "column_header": "2025", "unit": "currency"},
            {"document_id": "doc-1", "page": 5, "row_label": "Revenue", "column_header": "2025", "unit": "currency"},
        ],
    }
    candidates = [_candidate("old", "1", "Revenue"), _candidate("new", "1.2", "Revenue")]
    first = choose_candidate(
        label=label, source=label["expected_sources"][0], source_index=0,
        candidates=candidates, by_id={}, top20_keys=set(),
    )
    second = choose_candidate(
        label=label, source=label["expected_sources"][1], source_index=1,
        candidates=candidates, by_id={}, top20_keys=set(),
    )
    assert first.status == "bound" and second.status == "bound"
    assert first.candidate["candidate_key"] != second.candidate["candidate_key"]


def test_calculation_operands_bind_to_expected_sources():
    label = {
        "calculation": {"operands": [{"value": "1000000", "metric": "Revenue"}]},
        "expected_sources": [{"document_id": "doc-1", "page": 5, "row_label": "Revenue", "column_header": "2025", "unit": "currency"}],
    }
    decision = choose_candidate(
        label=label, source=label["expected_sources"][0], source_index=0,
        candidates=[_candidate("row-1", "1", "Revenue")], by_id={}, top20_keys=set(),
    )
    assert decision.status == "bound"


def test_composite_answer_sources_are_all_bound():
    label = {
        "expected_answer": {"component_values": [
            {"canonical_value": "1000000", "metric": "Revenue"},
            {"canonical_value": "2000000", "metric": "Income"},
        ]},
        "expected_sources": [
            {"document_id": "doc-1", "page": 5, "row_label": "Revenue", "column_header": "2025", "unit": "currency"},
            {"document_id": "doc-1", "page": 5, "row_label": "Income", "column_header": "2025", "unit": "currency"},
        ],
    }
    candidates = [_candidate("revenue", "1", "Revenue"), _candidate("income", "2", "Income")]
    decisions = [
        choose_candidate(label=label, source=source, source_index=i, candidates=candidates, by_id={}, top20_keys=set())
        for i, source in enumerate(label["expected_sources"])
    ]
    assert [item.status for item in decisions] == ["bound", "bound"]


def test_tesla_total_automotive_is_not_automotive_sales():
    label = {"expected_answer": {"canonical_value": "69526000000"}, "expected_sources": [{
        "document_id": "tsla", "page": 69, "row_label": "Total automotive revenues", "column_header": "2025", "unit": "currency",
    }]}
    candidate = _candidate("sales", "65.821", "Automotive sales")
    candidate["benchmark_document_id"] = "tsla"
    candidate["page"] = 69
    decision = choose_candidate(
        label=label, source=label["expected_sources"][0], source_index=0,
        candidates=[candidate], by_id={}, top20_keys=set(),
    )
    assert decision.status == "missing_from_index"
