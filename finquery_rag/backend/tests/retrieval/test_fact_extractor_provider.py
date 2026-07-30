import pytest

from src.retrieval.fact_extractor_provider import (
    FactExtractorConfigurationError,
    StructuredShadowFactExtractor,
    build_fact_extractor_provider,
)


def _chunk(content: str) -> dict:
    return {
        "content": content,
        "metadata": {
            "candidate_key": "candidate:v1:test",
            "candidate_rank": 1,
            "document_id": "report.pdf",
            "page": 3,
            "type": "table",
        },
    }


def test_default_provider_is_current():
    assert build_fact_extractor_provider().name == "current"


def test_invalid_provider_fails_fast():
    with pytest.raises(FactExtractorConfigurationError):
        build_fact_extractor_provider("unsupported")


def test_structured_provider_preserves_table_period_and_provenance():
    facts = StructuredShadowFactExtractor().extract(
        question="cash 2020",
        evidence=(_chunk("Cash | 143,540 | 206,031\nTable column context: 2020 | 2019"),),
    ).facts
    numeric = [fact for fact in facts if fact.raw_value]
    assert [(fact.raw_value, fact.period) for fact in numeric] == [
        ("143,540", "2020"),
        ("206,031", "2019"),
    ]
    assert all(fact.candidate_key == "candidate:v1:test" for fact in numeric)


def test_structured_provider_is_explicitly_named_shadow():
    assert build_fact_extractor_provider("structured_shadow").name == "structured_shadow"
