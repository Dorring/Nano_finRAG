"""H2A-1E: a slot is bound only when its admissible candidates agree.

The readiness benchmark found three false releases, all one shape: several
admissible candidates for one required slot, one of them binds, and the
disagreement is never examined.

    slot revenue
      ├── CF1 = 100
      └── CF2 = 999
    -> SUFFICIENT, no reason codes, bound to CF1, RELEASE

"One candidate can be bound" was being treated as "this slot is settled".  It is
not the same claim, and the difference is exactly the class of defect the H1
review already found once in the calculator: existence is not admissibility.

These tests come first and are expected to fail.  The fix belongs at the
evidence-binding layer -- provider-independent, because a runtime that trusts its
binder to volunteer AMBIGUOUS is trusting a component it does not control.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_v2.adaptive.adaptive_contracts import (
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidencePacketV1,
    ReasonCode,
)
from rag_v2.evidence.binder_service import SemanticBinderService
from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability
from tests.test_trusted_v2_r4_binder import SelectingBinderProvider, _fact, _plan, _slot

QUERY = "What was Apple's FY2024 revenue?"


def _capability() -> SemanticEvidenceEvaluationCapability:
    return SemanticEvidenceEvaluationCapability(
        SemanticBinderService(SelectingBinderProvider())
    )


def _state(packets: list[dict[str, Any]]) -> AdaptiveRAGStateV1:
    state = AdaptiveRAGStateV1.new(
        "req-conflict",
        QUERY,
        required_slots=[{"slot_id": "revenue", "metric": "Revenue", "period": "FY2024"}],
    )
    state.plan = {"supervisor_plan": _plan(_slot("revenue")).to_dict()}
    state.add_evidence([EvidencePacketV1.from_mapping(p) for p in packets])
    return state


def _evaluate(packets: list[dict[str, Any]]) -> tuple[Any, Any]:
    capability = _capability()
    evaluation = capability.evaluate(_state(packets))
    return evaluation, capability


# --- must be flagged ---------------------------------------------------------


def test_candidates_that_disagree_for_one_slot_are_not_sufficient() -> None:
    """The reproduction.  Two admissible values for one slot, 100 and 999."""

    evaluation, capability = _evaluate(
        [
            _fact("CF1", slots=("revenue",), value="100"),
            _fact("CF2", slots=("revenue",), value="999"),
        ]
    )

    assert evaluation.decision is not EvidenceDecision.SUFFICIENT
    assert ReasonCode.EVIDENCE_CONFLICT in evaluation.reason_codes
    # The bound set must not present a conflicted slot as settled, whatever the
    # decision: a downstream consumer reading only `bound_evidence_ids` must not
    # be able to mistake this for a clean binding.
    assert "revenue" not in evaluation.supported_slots
    assert capability.last_bound_evidence_ids == ()


def test_a_conflicting_slot_is_not_repairable_by_retrying_the_same_thing() -> None:
    """Two admissible candidates in hand is not a retrieval gap.

    Marking it REPAIRABLE would spend a retrieval round on evidence that is
    already admissible, and the replanner would re-issue an unchanged query.
    """

    evaluation, _ = _evaluate(
        [
            _fact("CF1", slots=("revenue",), value="100"),
            _fact("CF2", slots=("revenue",), value="999"),
        ]
    )

    assert evaluation.decision is not EvidenceDecision.REPAIRABLE


# --- must NOT be flagged -----------------------------------------------------


def test_the_same_fact_retrieved_twice_is_not_a_conflict() -> None:
    """Exact duplicate: one source, the same evidence_id."""

    duplicate = _fact("SAME", slots=("revenue",), value="100")
    evaluation, _ = _evaluate([dict(duplicate), dict(duplicate)])

    assert evaluation.decision is EvidenceDecision.SUFFICIENT
    assert ReasonCode.EVIDENCE_CONFLICT not in evaluation.reason_codes


def test_two_sources_agreeing_is_corroboration_not_conflict() -> None:
    """Different evidence ids, same value: agreement, not disagreement."""

    evaluation, _ = _evaluate(
        [
            _fact("A", slots=("revenue",), value="100"),
            _fact("B", slots=("revenue",), value="100"),
        ]
    )

    assert evaluation.decision is EvidenceDecision.SUFFICIENT
    assert ReasonCode.EVIDENCE_CONFLICT not in evaluation.reason_codes


@pytest.mark.parametrize(
    "left,right",
    [
        (("1000", "million"), ("1", "billion")),
        (("100", "million"), ("100", "million")),
    ],
)
def test_representation_equivalent_values_are_not_conflicts(
    left: tuple[str, str], right: tuple[str, str]
) -> None:
    """1000 million and 1 billion are the same quantity, not a disagreement.

    Comparing raw strings would fail this, which is why the conflict gate has to
    canonicalise before it compares.
    """

    evaluation, _ = _evaluate(
        [
            {**_fact("L", slots=("revenue",), value=left[0]), "scale": left[1]},
            {**_fact("R", slots=("revenue",), value=right[0]), "scale": right[1]},
        ]
    )

    assert evaluation.decision is EvidenceDecision.SUFFICIENT
    assert ReasonCode.EVIDENCE_CONFLICT not in evaluation.reason_codes


def test_a_different_period_candidate_is_not_a_same_slot_conflict() -> None:
    """The FY2023 distractor is a period problem, not a same-slot disagreement.

    It must keep its existing reason code; folding it into conflict would
    collapse two distinct diagnostics into one.
    """

    evaluation, _ = _evaluate(
        [
            _fact("RIGHT", slots=("revenue",), value="100"),
            _fact("WRONG", period="FY2023", slots=("revenue",), value="90"),
        ]
    )

    assert ReasonCode.EVIDENCE_CONFLICT not in evaluation.reason_codes


# --- the value identity the gate compares ------------------------------------
#
# The gate is only as good as its notion of "these two facts agree".  A key that
# is too loose merges different quantities; one that is too tight manufactures a
# conflict from consistent evidence, and a fail-closed system then refuses to
# answer a question it could answer.  Both directions are pinned here.


#: The financial semantic contract, authored here rather than generated.
#:
#: The helper below calls the production `_fact_value_key`, so any expectation it
#: produced would be the function agreeing with itself.  These pairs state the
#: *domain* claim instead -- which quantities are the same and which are not --
#: and the test asks whether the runtime agrees.  They are the same distinctions
#: the conflict contract was sealed on, written down where the implementation
#: cannot reach them.
EQUIVALENT_QUANTITIES: tuple[tuple[dict, dict], ...] = (
    # 1000 million is 1 billion.  Written two ways, one quantity.
    ({"value": "1000", "scale": "million"}, {"value": "1", "scale": "billion"}),
    # Decimal precision is presentation, not quantity.
    ({"value": "1.000", "scale": "billion"}, {"value": "1000.0", "scale": "million"}),
    # Corroboration: two sources, the same figure.
    ({"value": "391"}, {"value": "391"}),
)

DISTINCT_QUANTITIES: tuple[tuple[dict, dict], ...] = (
    # A different currency is a different quantity, whatever the digits.
    ({"value": "1000", "scale": "million", "currency": "USD"},
     {"value": "1", "scale": "billion", "currency": "EUR"}),
    # A share count is not a currency amount.
    ({"value": "1", "scale": "billion", "unit": "shares"},
     {"value": "1", "scale": "billion", "unit": "USD"}),
    # A ratio is not its percentage form: equating them on numeric coincidence
    # would merge a ratio with a percentage point.
    ({"value": "0.12", "unit": "ratio", "scale": None},
     {"value": "12", "unit": "%", "scale": None}),
    # An unrecognised scale is a qualifier, not a magnitude.
    ({"value": "1", "scale": "adjusted-billion"}, {"value": "1", "scale": "billion"}),
    # A different figure is a different figure.
    ({"value": "391"}, {"value": "383"}),
)


def _key(value: str, *, unit: str = "USD", currency: str = "USD", scale: Any = "million") -> str:
    from src.runtime.trusted_v2_binder import SemanticEvidenceEvaluationCapability

    fact = {
        **_fact("K", slots=("revenue",), value=value),
        "unit": unit,
        "currency": currency,
        "scale": scale,
    }
    key = SemanticEvidenceEvaluationCapability._fact_value_key(fact)
    assert key is not None
    return key


def _key_for(spec: dict) -> str:
    return _key(
        spec.get("value", "100"),
        unit=spec.get("unit", "USD"),
        currency=spec.get("currency", "USD"),
        scale=spec.get("scale", "million"),
    )


@pytest.mark.parametrize("left,right", EQUIVALENT_QUANTITIES)
def test_quantities_the_domain_calls_equal_share_a_key(left: dict, right: dict) -> None:
    assert _key_for(left) == _key_for(right)


@pytest.mark.parametrize("left,right", DISTINCT_QUANTITIES)
def test_quantities_the_domain_calls_different_do_not_share_a_key(
    left: dict, right: dict
) -> None:
    assert _key_for(left) != _key_for(right)


def test_the_folded_magnitude_is_the_literal_quantity_not_a_derived_one() -> None:
    """The one expectation worth pinning as a value rather than a relation.

    Key equality says "these agree"; it does not say they agree on the *right*
    number.  A canonicalizer that folded every scale to the same constant would
    satisfy every equivalence above.  This pins the magnitude itself, against a
    literal written here.
    """

    assert _key("1000", scale="million").startswith("1e+9|")
    assert _key("1", scale="billion").startswith("1e+9|")
    # And an unscaled value is left alone rather than being folded by default.
    assert _key("391", scale=None).startswith("391|")


def test_a_single_admissible_candidate_is_still_sufficient() -> None:
    """The arity guard, pinned from below.

    Every "must not be flagged" case in this file supplies two facts, so the
    guard's lower bound was never tested: relaxing `len(admissible) < 2` to
    `< 1` passed the whole suite. A one-character change there would fail closed
    on entirely ordinary evidence -- one candidate for one slot -- with nothing
    objecting.
    """

    evaluation, capability = _evaluate([_fact("ONLY", slots=("revenue",), value="100")])

    assert evaluation.decision is EvidenceDecision.SUFFICIENT
    assert evaluation.supported_slots == ("revenue",)
    assert capability.last_bound_evidence_ids == ("ONLY",)
