from src.evaluation.nf_eval_04 import (
    VerifiedCandidateEquivalence,
    require_verified_equivalence,
)


def test_parent_child_mapping_requires_verified_relation():
    relation = VerifiedCandidateEquivalence(
        gold_candidate_key="gold",
        retrievable_candidate_key="parent",
        relation="row_to_parent_table",
        verification_source="golden-binding",
    )
    assert require_verified_equivalence(
        relation,
        relation="row_to_parent_table",
    )
    assert not require_verified_equivalence(
        None,
        relation="row_to_parent_table",
    )


def test_same_page_is_not_identity_equivalence():
    relation = VerifiedCandidateEquivalence(
        gold_candidate_key="gold",
        retrievable_candidate_key="same-page",
        relation="same_page",
        verification_source="page-only",
    )
    assert not require_verified_equivalence(
        relation,
        relation="row_to_parent_table",
    )
