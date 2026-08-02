from __future__ import annotations

from pathlib import Path

from scripts.evaluation.annotation_contract import annotation_contract_report
from scripts.evaluation.benchmark_foundation import load_jsonl


ROOT = Path(__file__).parents[2]
DATA = ROOT / "benchmarks" / "financial_rag_v1" / "data"


def _records():
    questions = load_jsonl(DATA / "questions.draft.jsonl")
    labels = load_jsonl(DATA / "labels.draft.jsonl")
    reviews = load_jsonl(DATA / "review-status.jsonl")
    return questions, labels, reviews


def test_pdf_verified_pages_cover_all_answerable_source_records():
    _, labels, _ = _records()
    sources = [source for label in labels for source in label.get("expected_sources", [])]
    assert len(sources) == 80
    assert all(source["pdf_page_verified"] for source in sources)
    assert all(source["pdf_content_verified"] for source in sources)
    assert all(source["pdf_verification_method"] == "pdf_text_and_visual" for source in sources)


def test_pdf_verification_does_not_promote_source_or_golden_status():
    questions, labels, reviews = _records()
    sources = [source for label in labels for source in label.get("expected_sources", [])]
    assert all(source["source_verified"] is False for source in sources)
    assert all(source["candidate_identity_status"] == "pending" for source in sources)
    assert all(source["candidate_key"] is None for source in sources)
    report = annotation_contract_report(questions, labels, reviews)
    assert report["pdf_verified_source_record_count"] == 80
    assert report["verified_source_record_count"] == 0
    assert report["golden_case_count"] == 0


def test_pdf_audit_corrections_use_exact_displayed_table_names():
    questions, labels, _ = _records()
    question_by_id = {item["case_id"]: item for item in questions}
    label_by_id = {item["case_id"]: item for item in labels}
    assert "Revenue by End Market" in question_by_id["nvda_fy2025_004"]["question"]
    assert label_by_id["nvda_fy2025_004"]["expected_sources"][0]["page"] == 172
    assert label_by_id["nvda_fy2025_004"]["expected_sources"][0]["table_title"] == "Revenue by End Market"
    assert "Operational highlights" in question_by_id["v_fy2025_004"]["question"]
    assert label_by_id["v_fy2025_004"]["expected_sources"][0]["page"] == 4


def test_tesla_questions_match_the_verified_revenue_rows():
    questions, labels, _ = _records()
    question_by_id = {item["case_id"]: item for item in questions}
    label_by_id = {item["case_id"]: item for item in labels}
    table_case = question_by_id["tsla_fy2025_005"]
    assert "Results of Operations revenue table" in table_case["question"]
    source = label_by_id["tsla_fy2025_005"]["expected_sources"][0]
    assert source["page"] == 55
    assert source["row_label"] == "Energy generation and storage segment revenue"
    comparison = question_by_id["tsla_fy2025_007"]
    assert "Total automotive revenues" in comparison["question"]
    assert label_by_id["tsla_fy2025_007"]["expected_answer"]["canonical_value"] == "56755000000"


def test_changed_questions_have_revision_three_and_worklist_is_synchronized():
    questions, labels, reviews = _records()
    worklist = load_jsonl(DATA / "annotation-worklist.jsonl")
    changed = {
        "nvda_fy2025_004",
        "nvda_fy2025_005",
        "tsla_fy2025_005",
        "tsla_fy2025_007",
        "v_fy2025_003",
        "v_fy2025_004",
    }
    question_by_id = {item["case_id"]: item for item in questions}
    label_by_id = {item["case_id"]: item for item in labels}
    review_by_id = {item["case_id"]: item for item in reviews}
    worklist_by_id = {item["case_id"]: item for item in worklist}
    for case_id in changed:
        assert question_by_id[case_id]["question_revision"] == 3
        assert label_by_id[case_id]["question_revision"] == 3
        assert review_by_id[case_id]["question_revision"] == 3
        assert worklist_by_id[case_id]["question"] == question_by_id[case_id]["question"]


def test_no_answer_pdf_scan_remains_provisional():
    _, labels, reviews = _records()
    review_by_id = {item["case_id"]: item for item in reviews}
    no_answer = [label for label in labels if label["expected_no_answer"]]
    assert len(no_answer) == 8
    for label in no_answer:
        audit = label["no_answer_review"]
        assert audit["automated_pdf_full_text_scan_completed"] is True
        assert audit["automated_pdf_scan_status"] == "provisionally_supported"
        assert audit["human_negative_evidence_reviewed"] is False
        review = review_by_id[label["case_id"]]
        assert review["negative_evidence_review_status"] == "pending"
        assert review["full_document_search_completed"] is False
