"""Tests for deterministic, source-derived Trusted V2 fact enrichment."""

from __future__ import annotations

from src.runtime.trusted_v2_fact_enrichment import (
    ENRICHMENT_SCHEMA_VERSION,
    enrich_fact_record,
    enrich_fact_records,
)


def _fact(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "fact_id": "atomic:revenue-2024",
        "evidence_id": "atomic:revenue-2024",
        "candidate_key": "v2fact:revenue-2024",
        "entity": "Apple",
        "metric": "Revenue",
        "period": "FY2024",
        "value": "391035",
        "table_fragment_id": "table:income-statement",
        "row_id": "row:revenue",
        "provenance_complete": True,
    }
    value.update(overrides)
    return value


def _view(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "candidate_key": "candidate:view-revenue",
        "document_id": "aapl_fy2024",
        "pdf_page": 28,
        "table_title": "Consolidated Statements of Operations",
        "metric_paths": ["Net sales"],
        "periods": ["FY2024", "FY2023"],
        "section_path": ["Financial Statements"],
        "row_ids": ["row:revenue"],
        "semantic_evidence_ids": ["atomic:revenue-2024"],
        "facts": [
            {
                "type": "atomic",
                "evidence_id": "atomic:revenue-2024",
                "metric": "Revenue",
                "period": "FY2024",
                "value": "391035",
            }
        ],
        "source_traceback": [
            {
                "table_fragment_id": "table:income-statement",
                "row_id": "row:revenue",
                "raw_text": "raw source text must not be copied",
            }
        ],
        "raw_content": "raw view content must not be copied",
    }
    value.update(overrides)
    return value


def test_row_identity_enrichment_adds_only_compact_source_context() -> None:
    fact = _fact()
    enriched, linkage = enrich_fact_record(fact, [_view()])

    assert linkage == "EVIDENCE_ID"
    assert enriched["fact_id"] == fact["fact_id"]
    assert enriched["value"] == "391035"
    assert "structural_context" not in fact
    assert enriched["structural_context"] == {
        "structural_linkage": "EVIDENCE_ID",
        "source_view_ids": ["candidate:view-revenue"],
        "document_id": "aapl_fy2024",
        "pdf_page": 28,
        "table_title": "Consolidated Statements of Operations",
        "section_path": ["Financial Statements"],
        "row_ids": ["row:revenue"],
        "semantic_evidence_ids": ["atomic:revenue-2024"],
        "metric_paths": ["Net sales"],
        "periods": ["FY2024", "FY2023"],
    }
    rendered = str(enriched["structural_context"])
    assert "raw source text" not in rendered
    assert "raw view content" not in rendered


def test_table_only_linkage_never_contributes_row_metric_or_period() -> None:
    fact = _fact(row_id=None, evidence_id="atomic:unmapped")
    view = _view(
        row_ids=["row:other"],
        semantic_evidence_ids=["atomic:other"],
        facts=[{"evidence_id": "atomic:other"}],
    )

    enriched, linkage = enrich_fact_record(fact, [view])

    assert linkage == "TABLE_FRAGMENT"
    context = enriched["structural_context"]
    assert context["table_title"] == "Consolidated Statements of Operations"
    assert "metric_paths" not in context
    assert "periods" not in context


def test_existing_structural_context_is_never_overwritten() -> None:
    fact = _fact(structural_context={"table_title": "Frozen source title"})

    enriched, linkage = enrich_fact_record(fact, [_view()])

    assert linkage == "EVIDENCE_ID"
    assert enriched["structural_context"]["table_title"] == "Frozen source title"
    assert enriched["structural_context"]["metric_paths"] == ["Net sales"]


def test_bulk_enrichment_reports_linkage_without_gold_or_question_inputs() -> None:
    records, report = enrich_fact_records(
        [_fact(), _fact(fact_id="unmatched", evidence_id="unmatched", row_id=None, table_fragment_id=None)],
        [_view()],
    )

    assert len(records) == 2
    assert report == {
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "source_fact_count": 2,
        "structured_view_count": 1,
        "enriched_fact_count": 1,
        "unmatched_fact_count": 1,
        "linkage_counts": {"EVIDENCE_ID": 1},
        "index_key_counts": {"evidence_ids": 1, "row_ids": 1, "table_ids": 1},
    }
