from __future__ import annotations

from pathlib import Path

from scripts.evaluation.benchmark_foundation import load_jsonl
from scripts.evaluation.draft_quality import quality_audit


ROOT = Path(__file__).parents[2]
DATA = ROOT / "benchmarks" / "financial_rag_v1" / "data"


def test_repaired_draft_quality_gate_is_clean_and_not_golden():
    result = quality_audit(
        load_jsonl(DATA / "questions.draft.jsonl"),
        load_jsonl(DATA / "labels.draft.jsonl"),
        load_jsonl(DATA / "review-status.jsonl"),
    )
    assert result["question_count"] == 72
    assert result["quality_valid"] is True
    assert result["golden_case_count"] == 0
    assert result["semantic_duplicate_count"] == 0
    assert result["question_label_period_mismatch"] == 0
    assert result["undefined_comparison_count"] == 0
    assert result["placeholder_section_count"] == 0
    assert result["placeholder_table_title_count"] == 0


def test_quality_gate_requires_explicit_multi_source_output_contract():
    questions = [{
        "case_id": "case-1",
        "company": "Example",
        "question": "Compare revenue with operating income in FY2025.",
        "answerable": True,
        "requires_multiple_sources": True,
    }]
    labels = [{
        "case_id": "case-1",
        "expected_answer": {"period": "FY2025"},
        "expected_sources": [],
    }]
    result = quality_audit(questions, labels, [{"case_id": "case-1"}])
    assert result["undefined_comparison_count"] == 1


def test_quality_gate_detects_question_label_period_mismatch():
    questions = [{
        "case_id": "case-1",
        "company": "Example",
        "question": "What was revenue in FY2024?",
        "requested_period": "FY2024",
        "answerable": True,
        "requires_multiple_sources": False,
    }]
    labels = [{
        "case_id": "case-1",
        "expected_answer": {"period": "FY2025"},
        "expected_sources": [{"period": "FY2025", "column_header": "2025"}],
        "review_plan": {"action": "rewrite"},
    }]
    result = quality_audit(questions, labels, [{"case_id": "case-1"}])
    assert result["question_label_period_mismatch"] == 1

