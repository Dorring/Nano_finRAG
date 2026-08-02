from __future__ import annotations

from src.evaluation.benchmark_source_binding import golden_promotion_gate


def _kwargs(**updates):
    values = {
        "question_count": 72,
        "answerable_count": 64,
        "expected_source_count": 80,
        "bound_candidate_identity_count": 80,
        "negative_evidence_verified_count": 8,
        "ambiguous_identity_count": 0,
        "missing_from_index_count": 0,
        "out_of_scope_identity_count": 0,
        "unresolved_anomaly_count": 0,
        "all_cases_ready": True,
    }
    values.update(updates)
    return values


def test_golden_requires_eighty_bound_identities():
    assert golden_promotion_gate(**_kwargs(bound_candidate_identity_count=79)) is False
    assert golden_promotion_gate(**_kwargs()) is True


def test_golden_requires_eight_negative_reviews():
    assert golden_promotion_gate(**_kwargs(negative_evidence_verified_count=7)) is False


def test_golden_hash_gate_is_deterministic():
    values = _kwargs()
    assert golden_promotion_gate(**values) == golden_promotion_gate(**values)
