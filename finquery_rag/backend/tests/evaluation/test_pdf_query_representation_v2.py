from src.evaluation.pdf_query_representation_v2 import (
    canonical_key,
    char_score,
    concept_family,
    fixed_rrf,
    natural_phrase,
    normalize_label,
    token_bm25_scores,
)


def test_normalization_removes_layout_noise_not_meaning() -> None:
    assert normalize_label("Total revenue .......... $") == "total revenue"


def test_concept_family_is_generic() -> None:
    assert concept_family("Net cash provided by operating activities") == "cash_flow"
    assert concept_family("Total revenues") == "revenue"


def test_canonical_key_elides_only_generic_modifiers() -> None:
    assert canonical_key("Total net sales") == "revenue:sales"


def test_character_score_prefers_related_phrase() -> None:
    assert char_score("receivables", "accounts receivable") > char_score("receivables", "inventory")


def test_token_bm25_prefers_matching_concept() -> None:
    scores = token_bm25_scores("operating cash", ["operating cash flow", "total assets"])
    assert scores[0] > scores[1]


def test_fixed_rrf_has_no_tuned_weights() -> None:
    assert fixed_rrf([[1, 2], [2, 1]]) == fixed_rrf([[2, 1], [1, 2]])


def test_natural_phrase_does_not_copy_revenue_label() -> None:
    phrase, method = natural_phrase("Total net sales", 0)
    assert phrase == "money made from sales"
    assert method == "generic_financial_alias"
