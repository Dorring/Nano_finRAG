from src.evaluation.financial_benchmark import validate_annotation_cases, validate_document_catalog


CATALOG = [{
    "document_id": "msft_2025",
    "company": "Microsoft",
    "fiscal_year": "2025",
    "official_landing_url": "https://example.com/report",
    "local_filename": "msft_2025.pdf",
    "source_kind": "issuer_investor_relations",
}]


def test_catalog_requires_issuer_source_and_identity_fields():
    assert validate_document_catalog(CATALOG) == []
    invalid = dict(CATALOG[0], source_kind="mirror")
    assert validate_document_catalog([invalid])[0].message == "source_kind must be issuer_investor_relations"


def test_answerable_case_requires_page_section_and_known_document():
    case = {
        "id": "msft_001", "type": "table_fact", "difficulty": "medium",
        "question": "What was revenue?", "answerable": True,
        "expected_sources": [{"document_id": "msft_2025", "page": 72, "section": "Segment Revenue"}],
        "review": {"status": "reviewed"},
    }
    assert validate_annotation_cases([case], allowed_document_ids={"msft_2025"}) == []


def test_no_answer_case_must_not_declare_source():
    case = {
        "id": "msft_002", "type": "no_answer", "difficulty": "easy",
        "question": "What is an unavailable metric?", "answerable": False,
        "expected_sources": [{"document_id": "msft_2025", "page": 1, "section": "Cover"}],
        "review": {"status": "draft"},
    }
    issues = validate_annotation_cases([case], allowed_document_ids={"msft_2025"})
    assert any(item.message == "no_answer case must not declare expected_sources" for item in issues)
