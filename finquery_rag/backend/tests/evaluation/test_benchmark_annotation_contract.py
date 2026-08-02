from __future__ import annotations

from pathlib import Path

from scripts.evaluation.annotation_contract import (
    annotation_contract_report,
    build_annotation_worklist,
    ready_for_golden,
)
from scripts.evaluation.benchmark_foundation import load_jsonl


ROOT = Path(__file__).parents[2]
DATA = ROOT / "benchmarks" / "financial_rag_v1" / "data"


def _records():
    return (
        load_jsonl(DATA / "questions.draft.jsonl"),
        load_jsonl(DATA / "labels.draft.jsonl"),
        load_jsonl(DATA / "review-status.jsonl"),
    )


def test_repaired_question_has_no_stale_review_action():
    questions, labels, _ = _records()
    allowed = {"manual_answer_source_review", "manual_negative_evidence_review"}
    assert all(question["review_action"] in allowed for question in questions)
    assert all(label["review_action"] in allowed for label in labels)
    assert all(question["question_revision"] in {2, 3} for question in questions)
    assert all(question["question_revision_status"] == "ready_for_human_verification" for question in questions)
    assert all(question["superseded_review_action"] in {"keep", "rewrite", "replace", "manual_source_review"} for question in questions)


def test_question_label_review_status_are_consistent():
    questions, labels, reviews = _records()
    label_by_id = {item["case_id"]: item for item in labels}
    review_by_id = {item["case_id"]: item for item in reviews}
    for question in questions:
        case_id = question["case_id"]
        assert question["review_action"] == label_by_id[case_id]["review_action"]
        assert question["review_action"] == review_by_id[case_id]["review_action"]


def test_report_both_uses_composite_value_type():
    questions, labels, _ = _records()
    labels_by_id = {item["case_id"]: item for item in labels}
    report_both = [item for item in questions if item.get("output_contract") == "report_both"]
    assert len(report_both) == 5
    for question in report_both:
        answer = labels_by_id[question["case_id"]]["expected_answer"]
        assert answer["value_type"] == "composite"
        assert answer["canonical_value"] is None
        assert answer["currency"] is None
        assert answer["unit"] is None
        assert len(answer["component_values"]) >= 2


def test_composite_answer_has_no_top_level_currency():
    questions, labels, _ = _records()
    labels_by_id = {item["case_id"]: item for item in labels}
    for question in questions:
        if question.get("output_contract") == "report_both":
            answer = labels_by_id[question["case_id"]]["expected_answer"]
            assert answer["currency"] is None and answer["unit"] is None


def test_percentage_has_explicit_representation():
    questions, labels, _ = _records()
    labels_by_id = {item["case_id"]: item for item in labels}
    percentage_questions = [item for item in questions if item["answer_type"] == "percentage"]
    assert percentage_questions
    for question in percentage_questions:
        answer = labels_by_id[question["case_id"]]["expected_answer"]
        assert answer["percentage_representation"] == "percentage_points"
        assert answer["canonical_value"] is not None


def test_percentage_points_are_not_proportions():
    _, labels, _ = _records()
    answer = next(item["expected_answer"] for item in labels if item["case_id"] == "aapl_fy2025_003")
    assert answer["canonical_value"] == "46.9"
    assert answer["percentage_representation"] == "percentage_points"


def test_numeric_answer_requires_tolerance():
    questions, labels, _ = _records()
    labels_by_id = {item["case_id"]: item for item in labels}
    for question in questions:
        if question["answerable"] and question.get("output_contract") != "report_both":
            assert labels_by_id[question["case_id"]]["expected_answer"]["tolerance"] is not None


def test_source_gate_counts_source_records_not_cases():
    questions, labels, reviews = _records()
    report = annotation_contract_report(questions, labels, reviews)
    assert report["answerable_case_count"] == 64
    assert report["expected_source_record_count"] == 80
    assert report["verified_source_record_count"] == 0
    assert report["pdf_verified_source_record_count"] == 80
    assert report["candidate_identity_record_count"] == 0


def test_candidate_identity_required_for_golden():
    case = {
        "question_review_status": "reviewed",
        "answer_review_status": "reviewed",
        "source_review_status": "reviewed",
        "calculation_review_status": "not_applicable",
        "negative_evidence_review_status": "not_applicable",
        "expected_source_count": 1,
        "verified_source_count": 1,
        "all_sources_have_candidate_identity": False,
    }
    assert ready_for_golden(case) is False


def test_missing_row_id_is_allowed_when_limitation_recorded():
    questions, labels, reviews = _records()
    labels[0]["expected_sources"][0].update(
        {
            "candidate_key": "candidate:v1:test",
            "identity_granularity": "chunk",
            "identity_limitation": "row_identity_not_available",
            "source_verified": True,
        }
    )
    worklist = build_annotation_worklist(questions[:1], labels[:1], reviews[:1])
    assert worklist[0]["all_sources_have_candidate_identity"] is True


def test_negative_evidence_suggestions_are_not_completed_review():
    _, labels, reviews = _records()
    no_answer = next(item for item in labels if item["expected_no_answer"])
    review = next(item for item in reviews if item["case_id"] == no_answer["case_id"])
    assert no_answer["expected_answer"]["answer_key_status"] == "pending_negative_evidence"
    assert review["negative_evidence_review_status"] == "pending"
    assert review["full_document_search_completed"] is False


def test_all_golden_requirements_are_enforced():
    questions, labels, reviews = _records()
    worklist = build_annotation_worklist(questions, labels, reviews)
    assert sum(int(item["ready_for_golden"]) for item in worklist) == 0
