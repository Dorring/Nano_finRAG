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
    assert result["answer_key_missing_count"] == 0
    assert result["answer_key_status_missing_count"] == 0


def test_answer_keys_remain_unverified_draft_data():
    labels = load_jsonl(DATA / "labels.draft.jsonl")
    answerable = [label for label in labels if not label["expected_no_answer"]]
    no_answer = [label for label in labels if label["expected_no_answer"]]
    assert len(answerable) == 64
    assert all(label["label_status"] == "draft" for label in labels)
    assert all(label["review_status"] == "unreviewed" for label in labels)
    assert all(
        label["expected_answer"]["answer_key_status"] == "entered_unverified"
        for label in answerable
    )
    assert all(
        label["expected_answer"]["canonical_value"] is not None
        or label["expected_answer"].get("component_values")
        for label in answerable
    )
    assert all(
        label["expected_answer"]["answer_key_status"] == "pending_negative_evidence"
        for label in no_answer
    )
    assert all(
        not source["source_verified"]
        for label in labels
        for source in label["expected_sources"]
    )


def test_cross_type_replacements_have_distinct_audited_metrics():
    questions = {item["case_id"]: item for item in load_jsonl(DATA / "questions.draft.jsonl")}
    labels = {item["case_id"]: item for item in load_jsonl(DATA / "labels.draft.jsonl")}
    assert "operating income" in questions["msft_fy2025_005"]["question"].casefold()
    assert labels["msft_fy2025_005"]["expected_answer"]["canonical_value"] == "69773000000"
    assert "automotive revenue" in questions["nvda_fy2025_005"]["question"].casefold()
    assert labels["nvda_fy2025_005"]["expected_answer"]["canonical_value"] == "1694000000"
    assert "services and other" in questions["tsla_fy2025_002"]["question"].casefold()
    assert labels["tsla_fy2025_002"]["expected_answer"]["canonical_value"] == "12530000000"
    assert "regulatory credits" in questions["tsla_fy2025_003"]["question"].casefold()
    assert labels["tsla_fy2025_003"]["expected_answer"]["canonical_value"] == "1993000000"
    assert "total volume" in questions["v_fy2025_002"]["question"].casefold()
    assert labels["v_fy2025_002"]["expected_answer"]["canonical_value"] == "16700000000000"
    assert "gaap net income" in questions["v_fy2025_005"]["question"].casefold()
    assert labels["v_fy2025_005"]["expected_answer"]["canonical_value"] == "20058000000"


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
