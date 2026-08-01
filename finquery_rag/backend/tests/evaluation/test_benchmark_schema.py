from scripts.evaluation.benchmark_foundation import validate_dataset


def _corpus():
    return {"benchmark_id": "financial-rag-v1", "documents": [{"document_id": "doc-1", "filename": "doc-1.pdf", "page_count": 5}]}


def _question(answerable=True):
    return {"case_id": "case-1", "benchmark_id": "financial-rag-v1", "company": "Example", "document_scope": ["doc-1"], "question": "What was revenue?", "answerable": answerable, "answer_type": "currency" if answerable else "no_answer", "category": ["single_document"], "difficulty": "easy", "requires_calculation": False, "requires_multiple_sources": False, "draft_status": "generated", "authoring_method": "human"}


def _label(answerable=True, include_sources=True):
    sources = (
        [{"document_id": "doc-1", "filename": "doc-1.pdf", "page": 1, "evidence_type": "text", "source_verified": False}]
        if answerable and include_sources
        else []
    )
    return {"case_id": "case-1", "expected_answer": {"text": None, "canonical_value": None, "currency": "USD", "unit": "currency", "scale": "1", "period": "FY2025", "tolerance": None}, "expected_sources": sources, "calculation": None, "expected_no_answer": not answerable, "label_status": "draft", "review_status": "unreviewed"}


def test_answerable_case_requires_source():
    result = validate_dataset(corpus=_corpus(), questions=[_question()], labels=[_label(include_sources=False)], review_records=[{"case_id": "case-1", "question_reviewed": False, "answer_reviewed": False, "source_reviewed": False, "calculation_reviewed": True, "ready_for_golden": False}])
    assert any("requires expected_sources" in error for error in result["errors"])


def test_no_answer_case_can_have_no_sources():
    result = validate_dataset(corpus=_corpus(), questions=[_question(False)], labels=[_label(False)], review_records=[{"case_id": "case-1", "question_reviewed": False, "answer_reviewed": False, "source_reviewed": False, "calculation_reviewed": True, "ready_for_golden": False}])
    assert result["schema_valid"]
