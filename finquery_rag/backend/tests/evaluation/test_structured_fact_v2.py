from __future__ import annotations

import pytest

from src.evaluation.structured_fact_v2 import (
    EXCLUDED_ISSUER_CIKS,
    PINNED_V2_SOURCES,
    StructuredFactSource,
    fact_identity,
    normalize_concept_label,
    parse_numeric_value,
    parse_structured_fact_query,
    structured_fact_score,
    validate_v2_sources,
)


def test_v2_sources_are_issuer_disjoint_and_split_before_extraction() -> None:
    validate_v2_sources(PINNED_V2_SOURCES)
    assert not ({source.cik for source in PINNED_V2_SOURCES} & EXCLUDED_ISSUER_CIKS)
    assert sum(source.split == "holdout" for source in PINNED_V2_SOURCES) == 3


def test_v2_source_overlap_fails_closed() -> None:
    sources = list(PINNED_V2_SOURCES)
    sources[0] = StructuredFactSource("Apple", "320193", "2025-01-01", "2024-12-31", "x", "x.htm", "development")
    with pytest.raises(ValueError, match="overlaps"):
        validate_v2_sources(tuple(sources))


def test_fact_identity_is_stable_and_does_not_need_query_or_gold() -> None:
    first = fact_identity(document_id="doc", concept="us-gaap:Revenues", context_id="c1", unit_ref="USD", source_fact_id="f1")
    second = fact_identity(document_id="doc", concept="us-gaap:Revenues", context_id="c1", unit_ref="USD", source_fact_id="f1")
    assert first == second
    assert first.startswith("fact:v2:")


def test_numeric_value_respects_inline_xbrl_scale_and_sign() -> None:
    assert parse_numeric_value("416,161", sign=None, scale="6") == ("416161000000", 6)
    assert parse_numeric_value("(1,234)", sign=None, scale="3") == ("-1234000", 3)
    assert parse_numeric_value("—", sign=None, scale="6") == (None, 6)


def test_concept_label_normalization_is_generic() -> None:
    assert normalize_concept_label("us-gaap:NetIncomeLoss") == "net income loss"


def test_structured_fact_score_requires_document_and_metric_and_rewards_period() -> None:
    fact = {"issuer": "Issuer", "label": "net income loss", "period_end": "2025-12-31"}
    assert structured_fact_score(query_issuer="Issuer", query_metric="Net Income Loss", query_periods=("2025-12-31",), fact=fact) == 12.0
    assert structured_fact_score(query_issuer="Other", query_metric="Net Income Loss", query_periods=("2025-12-31",), fact=fact) is None
    assert structured_fact_score(query_issuer="Issuer", query_metric="Revenue", query_periods=("2025-12-31",), fact=fact) is None


def test_structured_query_parser_reads_question_text_only() -> None:
    assert parse_structured_fact_query(
        "According to Issuer's Form 10-K, what was net income loss for the periods ended 2024-12-31 and 2025-12-31?"
    ) == ("Issuer", "net income loss", ("2024-12-31", "2025-12-31"))
    assert parse_structured_fact_query(
        "According to Issuer's Form 10-K, what was nonexistent lunar reserve metric for 2099?"
    ) is None
