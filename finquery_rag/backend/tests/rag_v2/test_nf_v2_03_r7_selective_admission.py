from rag_v2.contracts.evidence import BindingStatus
from rag_v2.evidence.selective_admission import admit_selective_binding


def _facts():
    return [{"fact_id": "fact-1", "provenance_complete": True}]


def test_selective_admission_releases_only_a_unique_valid_selection():
    result = admit_selective_binding(
        slot_ids=["slot-1"],
        slot_bindings={"slot-1": ["fact-1"]},
        packet_facts=_facts(),
        binding_validator_pass=True,
        unique_admissible_selection=True,
    )
    assert result.released is True
    assert result.binding.status == BindingStatus.BOUND.value


def test_selective_admission_fails_closed_on_missing_safety_gate():
    result = admit_selective_binding(
        slot_ids=["slot-1"],
        slot_bindings={"slot-1": ["fact-1"]},
        packet_facts=_facts(),
        binding_validator_pass=True,
        unique_admissible_selection=False,
    )
    assert result.released is False
    assert result.binding.status == BindingStatus.MISSING.value
    assert "selection_not_unique" in result.reasons


def test_selective_admission_rejects_unknown_or_unprovenanced_facts():
    unknown = admit_selective_binding(
        slot_ids=["slot-1"],
        slot_bindings={"slot-1": ["not-in-packet"]},
        packet_facts=_facts(),
        binding_validator_pass=True,
        unique_admissible_selection=True,
    )
    assert unknown.released is False
    assert any(reason.startswith("unknown_fact:") for reason in unknown.reasons)

    incomplete = admit_selective_binding(
        slot_ids=["slot-1"],
        slot_bindings={"slot-1": ["fact-2"]},
        packet_facts=[{"fact_id": "fact-2", "provenance_complete": False}],
        binding_validator_pass=True,
        unique_admissible_selection=True,
    )
    assert incomplete.released is False
    assert any(reason.startswith("incomplete_provenance:") for reason in incomplete.reasons)

