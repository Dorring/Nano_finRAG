from __future__ import annotations

from rag_v2.contracts.plan import SupervisorPlan
from rag_v2.evidence.binder_provider import _binding_from_payload
from rag_v2.evidence.binding_validator import validate_binding


def _plan() -> SupervisorPlan:
    return SupervisorPlan.from_dict({
        "intent": "DIRECT_FACT",
        "required_slots": [{"slot_id": "slot_1", "metric": "revenue", "period": "FY2025", "role": "value", "value_type": "numeric", "unit": None}],
        "operation": None,
        "next_action": "RETRIEVE",
    })


def _fact(fact_id: str = "f1", *, metric: str = "revenue", period: str = "FY2025", candidate: str = "candidate:1", provenance: bool = True) -> dict:
    return {"fact_id": fact_id, "normalized_metric": metric, "normalized_period": period, "candidate_id": candidate, "provenance_complete": provenance}


def _binding(status: str, *, slot_bindings=None, missing_slots=None, ambiguous_slots=None, invalid_reasons=None):
    return _binding_from_payload({
        "status": status,
        "slot_bindings": slot_bindings or {},
        "missing_slots": missing_slots or [],
        "ambiguous_slots": ambiguous_slots or [],
        "invalid_reasons": invalid_reasons or (["provider_invalid"] if status == "INVALID" else []),
    })


def test_valid_bound_passes() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["f1"]}), _plan(), (_fact(),))
    assert result.passed


def test_valid_missing_passes() -> None:
    result = validate_binding(_binding("MISSING", missing_slots=["slot_1"]), _plan(), (_fact(),))
    assert result.passed


def test_valid_ambiguous_passes() -> None:
    result = validate_binding(_binding("AMBIGUOUS", ambiguous_slots=["slot_1"]), _plan(), (_fact(), _fact("f2")))
    assert result.passed


def test_unknown_fact_fails() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["unknown"]}), _plan(), (_fact(),))
    assert not result.passed
    assert any(reason.startswith("unknown_fact") for reason in result.reasons)


def test_unknown_slot_fails() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"unknown": ["f1"]}), _plan(), (_fact(),))
    assert not result.passed
    assert any(reason.startswith("unknown_slot") for reason in result.reasons)


def test_cross_query_fact_fails_as_unknown_packet_fact() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["other_query_fact"]}), _plan(), (_fact(),))
    assert not result.passed


def test_non_provenance_fact_fails() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["f1"]}), _plan(), (_fact(provenance=False),))
    assert not result.passed


def test_semantically_wrong_metric_is_structurally_valid() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["f1"]}), _plan(), (_fact(metric="operating income"),))
    assert result.passed


def test_wrong_period_is_structurally_valid() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["f1"]}), _plan(), (_fact(period="FY2024"),))
    assert result.passed


def test_non_gold_source_is_structurally_valid() -> None:
    result = validate_binding(_binding("BOUND", slot_bindings={"slot_1": ["f1"]}), _plan(), (_fact(candidate="candidate:not_gold"),))
    assert result.passed
