from __future__ import annotations

import pytest

from src.evaluation.nf_opt_17 import (
    PINNED_DEVELOPMENT_SOURCES,
    SecFilingSource,
    build_annotation_contract,
    source_manifest_hash,
    validate_annotation_record,
    validate_development_sources,
)


def test_pinned_sources_are_unique_sec_10ks_and_have_stable_urls() -> None:
    validate_development_sources(PINNED_DEVELOPMENT_SOURCES, frozen_filenames={"aapl_fy2025_10k.pdf"})
    assert len({source.cik for source in PINNED_DEVELOPMENT_SOURCES}) == 4
    assert all(source.archive_url.startswith("https://www.sec.gov/Archives/edgar/data/") for source in PINNED_DEVELOPMENT_SOURCES)
    assert source_manifest_hash(PINNED_DEVELOPMENT_SOURCES) == source_manifest_hash(PINNED_DEVELOPMENT_SOURCES)


def test_development_source_validation_rejects_duplicate_cik() -> None:
    duplicate = (
        PINNED_DEVELOPMENT_SOURCES[0],
        SecFilingSource(
            issuer="Duplicate",
            cik=PINNED_DEVELOPMENT_SOURCES[0].cik,
            filing_date="2026-01-01",
            accession_number="0000000000-26-000001",
            primary_document="duplicate.htm",
        ),
        *PINNED_DEVELOPMENT_SOURCES[1:3],
    )
    with pytest.raises(ValueError, match="CIKs must be unique"):
        validate_development_sources(duplicate, frozen_filenames=())


def test_annotation_contract_forbids_frozen_benchmark_fields() -> None:
    contract = build_annotation_contract()
    assert contract["training_allowed"] is False
    assert "expected_answer" in contract["forbidden_fields"]
    with pytest.raises(ValueError, match="forbidden"):
        validate_annotation_record(
            {
                "question": "What was revenue?",
                "candidate_key": "dev:1",
                "negative_type": "same_row_wrong_period",
                "expected_answer": "forbidden",
            }
        )


def test_annotation_record_requires_query_candidate_and_negative_type() -> None:
    with pytest.raises(ValueError, match="question"):
        validate_annotation_record({"candidate_key": "dev:1", "negative_type": "same_page_wrong_row"})
