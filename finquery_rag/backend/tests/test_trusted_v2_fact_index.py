"""Tests for source-only R4 views derived from the V2 fact-store contract."""

from __future__ import annotations

import pytest

from src.runtime.trusted_v2_fact_index import (
    FACT_CANDIDATE_INDEX_SCHEMA_VERSION,
    TrustedV2FactIndexError,
    build_fact_store_candidate_views,
)


def _fact(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_key": "v2fact:revenue-fy2024",
        "candidate_id": "v2fact:revenue-fy2024",
        "evidence_id": "fact:revenue-fy2024",
        "fact_id": "fact:revenue-fy2024",
        "provenance_complete": True,
        "document_id": "aapl_fy2024",
        "pdf_page": 28,
        "entity": "Apple",
        "metric": "Revenue",
        "period": "FY2024",
        "value": "391035",
        "currency": "USD",
        "scale": "million",
        "scope": "consolidated",
        "table_fragment_id": "table:operations",
        "row_id": "row:net-sales",
        "content": "Net sales were 391,035 in 2024.",
        "structural_context": {
            "metric_paths": ["Net sales"],
            "periods": ["FY2024", "FY2023"],
            "table_title": "Consolidated Statements of Operations",
        },
    }
    value.update(overrides)
    return value


def test_fact_store_views_share_candidate_keys_and_keep_source_only_context() -> None:
    pairs, summary = build_fact_store_candidate_views(
        [
            _fact(
                assistant_text="Revenue was $999B",
                model_generated_summary="invented financial narrative",
                conversation_history=[{"role": "assistant", "content": "ignore me"}],
            )
        ]
    )

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.candidate_key == "v2fact:revenue-fy2024"
    assert pair.raw_view.candidate_key == pair.structured_view.candidate_key
    assert pair.raw_view.fact_ids == ("fact:revenue-fy2024",)
    assert pair.structured_view.metric_paths == ("Revenue", "Net sales")
    assert pair.structured_view.periods == ("FY2024", "FY2023")
    assert "Net sales were 391,035" in pair.raw_view.retrieval_text
    rendered_structured = pair.structured_view.retrieval_text
    assert "Revenue was $999B" not in rendered_structured
    assert "invented financial narrative" not in rendered_structured
    assert "ignore me" not in rendered_structured
    assert summary.to_dict() == {
        "schema_version": FACT_CANDIDATE_INDEX_SCHEMA_VERSION,
        "source_fact_count": 1,
        "candidate_pair_count": 1,
        "structured_view_count": 1,
        "document_count": 1,
        "explicit_metadata_coverage": {
            "entity": {"populated_count": 1, "ratio": 1.0},
            "metric": {"populated_count": 1, "ratio": 1.0},
            "period": {"populated_count": 1, "ratio": 1.0},
            "scope": {"populated_count": 1, "ratio": 1.0},
            "unit": {"populated_count": 0, "ratio": 0.0},
            "currency": {"populated_count": 1, "ratio": 1.0},
            "scale": {"populated_count": 1, "ratio": 1.0},
            "statement_type": {"populated_count": 0, "ratio": 0.0},
        },
    }


def test_fact_store_views_supports_structured_fact_store_candidate_id_aliases() -> None:
    pairs, _ = build_fact_store_candidate_views(
        [
            _fact(
                candidate_key=None,
                candidate_id=None,
                candidate_ids=["v2fact:alias-primary", "v2fact:alias-secondary"],
            )
        ]
    )

    # StructuredFactStore uses the first declared candidate ID as its canonical
    # materialization key, so the generated R4 view must do the same.
    assert pairs[0].candidate_key == "v2fact:alias-primary"


def test_fact_store_views_reject_duplicate_candidate_keys() -> None:
    with pytest.raises(TrustedV2FactIndexError, match="duplicate_candidate_key"):
        build_fact_store_candidate_views([_fact(), _fact(evidence_id="fact:other")])


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"candidate_key": None, "candidate_id": None}, "missing_candidate_key"),
        ({"evidence_id": None, "fact_id": None}, "missing_evidence_id"),
        ({"document_id": None}, "missing_document_identity"),
        ({"provenance_complete": False}, "incomplete_provenance"),
    ],
)
def test_fact_store_views_reject_incomplete_source_contracts(
    overrides: dict[str, object],
    error: str,
) -> None:
    with pytest.raises(TrustedV2FactIndexError, match=error):
        build_fact_store_candidate_views([_fact(**overrides)])


def test_fact_store_summary_treats_unknown_metadata_as_unpopulated() -> None:
    _, summary = build_fact_store_candidate_views(
        [
            _fact(
                entity="UNKNOWN",
                metric="unknown",
                period=None,
                scope="N/A",
                currency=None,
                scale="null",
                statement_type="UNKNOWN",
            )
        ]
    )

    coverage = summary.to_dict()["explicit_metadata_coverage"]
    for name in ("entity", "metric", "period", "scope", "currency", "scale", "statement_type"):
        assert coverage[name] == {"populated_count": 0, "ratio": 0.0}
