from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.evidence.selective_admission_v2 import admit_binding_v2


def _plan():
    return SupervisorPlan(
        intent=Intent.DIRECT_FACT,
        required_slots=(RequiredSlot(slot_id="slot-1", metric="revenue", period="FY2025", role="value", value_type="numeric"),),
        operation=None,
        next_action=Action.RETRIEVE,
    )


def _fact(fact_id: str, period: str, metric: str = "Revenue"):
    return {"fact_id": fact_id, "candidate_id": "candidate-1", "physical_source_id": "source-1", "provenance_complete": True, "raw_metric": metric, "normalized_metric": metric.casefold(), "raw_period": period, "normalized_period": period, "unit": None, "currency": None}


def test_runtime_v2_admits_when_all_competitors_have_explicit_period_conflicts():
    binding = EvidenceBinding(status=BindingStatus.BOUND.value, slot_bindings={"slot-1": ("fact-1",)})
    result = admit_binding_v2(binding, _plan(), [_fact("fact-1", "FY2025"), _fact("fact-2", "FY2024")], source_map={"candidate-1": {}})
    assert result.released is True
    assert result.binding.status == BindingStatus.BOUND.value


def test_runtime_v2_abstains_when_unknown_period_competitor_remains_plausible():
    binding = EvidenceBinding(status=BindingStatus.BOUND.value, slot_bindings={"slot-1": ("fact-1",)})
    result = admit_binding_v2(binding, _plan(), [_fact("fact-1", "FY2025"), _fact("fact-2", "FY")], source_map={"candidate-1": {}})
    assert result.released is False
    assert result.binding.status == BindingStatus.AMBIGUOUS.value


def test_runtime_v2_uses_linked_source_context_for_competitor_safety():
    binding = EvidenceBinding(status=BindingStatus.BOUND.value, slot_bindings={"slot-1": ("fact-1",)})
    facts = [_fact("fact-1", "FY2025", "Product revenue"), _fact("fact-2", "FY2025", "Worldwide")]
    source_map = {
        "candidate-1": {},
        "candidate-2": {"source_text": "Selected product discussion: Product revenue worldwide"},
    }
    facts[1]["candidate_id"] = "candidate-2"
    result = admit_binding_v2(binding, _plan(), facts, source_map=source_map)
    assert result.released is False
    assert result.binding.status == BindingStatus.AMBIGUOUS.value


def test_runtime_v2_never_promotes_missing_or_ambiguous():
    missing = EvidenceBinding(status=BindingStatus.MISSING.value, slot_bindings={}, missing_slots=("slot-1",))
    ambiguous = EvidenceBinding(status=BindingStatus.AMBIGUOUS.value, slot_bindings={}, ambiguous_slots=("slot-1",))
    facts = [_fact("fact-1", "FY2025")]
    assert admit_binding_v2(missing, _plan(), facts, source_map={"candidate-1": {}}).binding.status == BindingStatus.MISSING.value
    assert admit_binding_v2(ambiguous, _plan(), facts, source_map={"candidate-1": {}}).binding.status == BindingStatus.AMBIGUOUS.value
