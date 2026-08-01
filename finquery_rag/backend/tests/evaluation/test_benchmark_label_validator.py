from decimal import Decimal

from scripts.evaluation.benchmark_foundation import parse_decimal, validate_dataset


def test_numeric_value_is_decimal_safe():
    assert parse_decimal("1,234.50") == Decimal("1234.50")
    assert parse_decimal("not-a-number") is None


def test_duplicate_question_is_detected():
    corpus = {"benchmark_id": "financial-rag-v1", "documents": [{"document_id": "doc-1", "filename": "doc-1.pdf", "page_count": 5}]}
    question = {"case_id": "case-1", "benchmark_id": "financial-rag-v1", "company": "Example", "document_scope": ["doc-1"], "question": "What was revenue?", "answerable": False, "answer_type": "no_answer", "category": ["no_answer"], "difficulty": "easy", "requires_calculation": False, "requires_multiple_sources": False, "draft_status": "generated", "authoring_method": "human"}
    label = {"case_id": "case-1", "expected_answer": {"text": "N/A", "canonical_value": None, "currency": None, "unit": None, "scale": "1", "period": None, "tolerance": None}, "expected_sources": [], "calculation": None, "expected_no_answer": True, "label_status": "draft", "review_status": "unreviewed"}
    review = {"case_id": "case-1", "question_reviewed": False, "answer_reviewed": False, "source_reviewed": False, "calculation_reviewed": True, "ready_for_golden": False}
    result = validate_dataset(corpus=corpus, questions=[question, {**question, "case_id": "case-2"}], labels=[label, {**label, "case_id": "case-2"}], review_records=[review, {**review, "case_id": "case-2"}])
    assert result["warnings"]
