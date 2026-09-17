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


def test_a_known_scale_is_folded_into_the_quantity() -> None:
    assert _key("1000", scale="million") == _key("1", scale="billion")


def test_a_different_currency_is_a_different_quantity() -> None:
    """Equal numbers are not equal amounts.

    1e9 USD and 1e9 EUR normalise to the same numeric value; treating them as
    agreement would let one currency's figure be bound for the other's slot.
    """

    assert _key("1000", scale="million", currency="USD") != _key(
        "1", scale="billion", currency="EUR"
    )


def test_a_different_unit_is_a_different_quantity() -> None:
    """A share count is not a currency amount."""

    assert _key("1", scale="billion", unit="shares") != _key("1", scale="billion", unit="USD")


def test_the_same_quantity_at_different_precision_is_the_same_quantity() -> None:
    """1.000 billion and 1000.0 million are one number written two ways.

    Decimal multiplication preserves significant digits, so a naive string
    comparison of the product sees "1000000000.000" and "1000000000.00" as
    different.
    """

    assert _key("1.000", scale="billion") == _key("1000.0", scale="million")


def test_an_unrecognised_scale_stays_a_difference() -> None:
    """'adjusted billion' is not a magnitude, it is a qualifier.

    Treating an unknown scale keyword as a multiplier would be a silent
    semantic conversion -- the thing scale normalisation must not become.
    """

    assert _key("1", scale="adjusted-billion") != _key("1", scale="billion")


def test_a_percentage_is_not_silently_equated_with_its_decimal_form() -> None:
    """0.12 and 12% may or may not be the same field.

    Normalising scale must not quietly become unit normalisation: equating these
    on numeric coincidence would merge a ratio with a percentage point.  If the
    financial contract ever declares them canonical, this changes deliberately.
    """

    assert _key("0.12", unit="ratio", scale=None) != _key("12", unit="%", scale=None)
