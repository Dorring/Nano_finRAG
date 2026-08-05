from __future__ import annotations

import pytest

from src.evaluation.structured_fact_v2 import (
    EXCLUDED_ISSUER_CIKS,
    PINNED_V2_SOURCES,
    StructuredFactSource,
    fact_identity,
    normalize_concept_label,
    parse_numeric_value,
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
