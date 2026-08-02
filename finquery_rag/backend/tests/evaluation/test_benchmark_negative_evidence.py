from __future__ import annotations

from src.evaluation.benchmark_source_binding import negative_review_passes


def _review(**updates):
    value = {
        "human_negative_evidence_reviewed": True,
        "negative_evidence_reviewed": True,
        "full_document_search_completed": True,
        "reviewer": "reviewer",
        "review_notes": "No matching quantitative disclosure.",
        "positive_quantitative_match_count": 0,
    }
    value.update(updates)
    return value


def test_automatic_fulltext_scan_is_not_human_negative_review():
    assert negative_review_passes(None) is False
    assert negative_review_passes({"positive_match_count": 0}) is False


def test_negative_evidence_requires_search_log():
    assert negative_review_passes(_review(full_document_search_completed=False)) is False
    assert negative_review_passes(_review(positive_quantitative_match_count=1)) is False
    assert negative_review_passes(_review()) is True
