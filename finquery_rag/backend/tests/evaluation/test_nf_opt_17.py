from __future__ import annotations

import pytest

from src.evaluation.nf_opt_17 import (
    FinancialFact,
    PINNED_DEVELOPMENT_SOURCES,
    SecFilingSource,
    build_hard_negative_annotation,
    build_annotation_contract,
    build_shadow_candidate_corpus,
    metric_label_for_concept,
    parse_context_period,
    source_manifest_hash,
    validate_annotation_record,
    validate_generated_annotation,
    validate_development_sources,
)
from src.evaluation.nf_opt_17_gate_d import (
    classify_pair_distinguishability,
    issuer_grouped_folds,
    rank_triplet,
    render_structured_reranker_input,
    shadow_candidate_key,
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


def test_generic_us_gaap_metric_normalizer_rejects_text_blocks() -> None:
    assert metric_label_for_concept("us-gaap:AccountsPayableCurrent") == "accounts payable current"
    assert metric_label_for_concept("us-gaap:RevenueRecognitionPolicyTextBlock") is None
    assert metric_label_for_concept("issuer:CompanySpecificRevenue") is None


def test_context_period_requires_a_valid_ordered_date_range() -> None:
    assert parse_context_period({"instant": "2025-12-31"}) == ("2025-12-31", "instant")
    assert parse_context_period({"start": "2025-01-01", "end": "2025-12-31"}) == ("2025-12-31", "duration")
    assert parse_context_period({"start": "2025-12-31", "end": "2025-01-01"}) is None


def test_generated_annotation_has_two_distinct_structural_hard_negatives() -> None:
    def fact(fact_id: str, metric: str, period: str, row: int) -> FinancialFact:
        return FinancialFact(
            source_cik="123",
            accession_number="0000000123-26-000001",
            primary_document="issuer-20251231.htm",
            issuer="Issuer Inc.",
            fact_id=fact_id,
            concept=f"us-gaap:{metric.replace(' ', '')}",
            metric=metric,
            context_id=f"ctx-{fact_id}",
            period_end=period,
            period_kind="duration",
            table_index=1,
            row_index=row,
            evidence_excerpt=f"{metric} {period}",
        )

    annotation = build_hard_negative_annotation(
        positive=fact("positive", "revenue", "2025-12-31", 3),
        wrong_period=fact("prior", "revenue", "2024-12-31", 3),
        wrong_metric=fact("other", "cost of revenue", "2025-12-31", 4),
    )
    validate_generated_annotation(annotation)
    assert annotation["expected_answer_stored"] is False


def test_ai_review_preserves_non_human_review_boundary() -> None:
    from scripts.evaluation.run_nf_opt_17_gate_c import _review_row

    def fact(fact_id: str, metric: str, period: str, row: int) -> FinancialFact:
        return FinancialFact(
            source_cik="123",
            accession_number="0000000123-26-000001",
            primary_document="issuer-20251231.htm",
            issuer="Issuer Inc.",
            fact_id=fact_id,
            concept=f"us-gaap:{metric.replace(' ', '')}",
            metric=metric,
            context_id=f"ctx-{fact_id}",
            period_end=period,
            period_kind="duration",
            table_index=1,
            row_index=row,
            evidence_excerpt=f"{metric} {period}",
        )

    annotation = build_hard_negative_annotation(
        positive=fact("positive", "revenue", "2025-12-31", 3),
        wrong_period=fact("prior", "revenue", "2024-12-31", 3),
        wrong_metric=fact("other", "cost of revenue", "2025-12-31", 4),
    )
    review = _review_row(annotation)
    assert review["ai_review_status"] == "structural_and_lexical_pass"
    assert review["human_review_status"] == "not_reviewed"


def test_shadow_candidate_corpus_retains_all_annotation_lineage() -> None:
    def fact(fact_id: str, metric: str, period: str, row: int) -> FinancialFact:
        return FinancialFact(
            source_cik="123",
            accession_number="0000000123-26-000001",
            primary_document="issuer-20251231.htm",
            issuer="Issuer Inc.",
            fact_id=fact_id,
            concept=f"us-gaap:{metric.replace(' ', '')}",
            metric=metric,
            context_id=f"ctx-{fact_id}",
            period_end=period,
            period_kind="duration",
            table_index=1,
            row_index=row,
            evidence_excerpt=f"{metric} {period}",
        )

    annotation = build_hard_negative_annotation(
        positive=fact("positive", "revenue", "2025-12-31", 3),
        wrong_period=fact("prior", "revenue", "2024-12-31", 3),
        wrong_metric=fact("other", "cost of revenue", "2025-12-31", 4),
    )
    candidates, lineage = build_shadow_candidate_corpus([annotation])
    assert len(candidates) == 3
    assert len(lineage) == 3
    assert all(candidate["candidate_key"].startswith("dev:sec:") for candidate in candidates)


def test_same_row_fact_labels_fail_closed_when_candidate_input_collides() -> None:
    candidate = {
        "shadow_candidate_key": "shadow:row",
        "content_sha256": "content",
        "metadata_sha256": "metadata",
        "reranker_input_sha256": "heuristic-input",
        "cross_encoder_input_sha256": "content",
    }
    result = classify_pair_distinguishability(positive=candidate, negative=dict(candidate))
    assert result["status"] == "candidate_identity_collision"
    assert result["trainable_with_current_heuristic"] is False


def test_distinct_table_rows_are_trainable_when_production_input_differs() -> None:
    positive = {
        "shadow_candidate_key": "shadow:positive",
        "content_sha256": "positive-content",
        "metadata_sha256": "positive-metadata",
        "reranker_input_sha256": "positive-input",
        "cross_encoder_input_sha256": "positive-content",
    }
    negative = {
        "shadow_candidate_key": "shadow:negative",
        "content_sha256": "negative-content",
        "metadata_sha256": "negative-metadata",
        "reranker_input_sha256": "negative-input",
        "cross_encoder_input_sha256": "negative-content",
    }
    result = classify_pair_distinguishability(positive=positive, negative=negative)
    assert result["status"] == "trainable_pair"
    assert result["trainable_with_current_heuristic"] is True


def test_candidate_identity_is_query_independent_and_fold_sets_are_disjoint() -> None:
    assert shadow_candidate_key(document_id="doc", table_index=1, row_index=2, content="row") == shadow_candidate_key(
        document_id="doc", table_index=1, row_index=2, content="row"
    )
    folds = issuer_grouped_folds(
        issuer_to_document_ids={"A": ["doc-a"], "B": ["doc-b"]},
        issuer_to_candidate_keys={"A": ["candidate-a"], "B": ["candidate-b"]},
        issuer_to_query_ids={"A": ["query-a"], "B": ["query-b"]},
    )
    assert len(folds) == 2
    assert all(fold["candidate_overlap_count"] == 0 for fold in folds)


def test_structured_reranker_input_uses_source_fields_without_expected_values() -> None:
    rendered = render_structured_reranker_input(
        {
            "issuer": "Issuer Inc.",
            "xbrl_concept": "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
            "period_end": "2025-12-31",
            "period_kind": "duration",
            "content": "Revenue 100 90",
        }
    )
    assert "period: 2025-12-31" in rendered
    assert "expected" not in rendered.casefold()


def test_triplet_ranking_reports_each_negative_type_separately() -> None:
    result = rank_triplet([0.9, 0.1, 0.8], ["same_row_wrong_period", "same_table_wrong_metric"])
    assert result["positive_top1"] is True
    assert result["pairwise"] == {
        "same_row_wrong_period": True,
        "same_table_wrong_metric": True,
    }
