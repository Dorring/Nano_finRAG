from scripts.evaluation.benchmark_foundation import validate_dataset


def test_duplicate_case_id_is_rejected():
    corpus = {"benchmark_id": "financial-rag-v1", "documents": [{"document_id": "doc-1", "filename": "doc-1.pdf", "page_count": 2}]}
    question = {"case_id": "same", "benchmark_id": "financial-rag-v1", "company": "Example", "document_scope": ["doc-1"], "question": "What was revenue?", "answerable": False, "answer_type": "no_answer", "category": ["no_answer"], "difficulty": "easy", "requires_calculation": False, "requires_multiple_sources": False, "draft_status": "generated", "authoring_method": "human"}
    label = {"case_id": "same", "expected_answer": {"text": "N/A", "canonical_value": None, "currency": None, "unit": None, "scale": "1", "period": None, "tolerance": None}, "expected_sources": [], "calculation": None, "expected_no_answer": True, "label_status": "draft", "review_status": "unreviewed"}
    review = {"case_id": "same", "question_reviewed": False, "answer_reviewed": False, "source_reviewed": False, "calculation_reviewed": True, "ready_for_golden": False}
    result = validate_dataset(corpus=corpus, questions=[question, question], labels=[label], review_records=[review])
    assert not result["schema_valid"]
    assert result["duplicate_question_ids"] == ["same"]
