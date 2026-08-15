from __future__ import annotations

import pytest

from rag_v2.generation import (
    GenerationInputV1,
    GenerationRecoveryPolicyV1,
    MockGeneratorProviderV1,
    ProviderRegistryV1,
    RuntimeGenerationValidatorV1,
    TrustedGenerationStateMachineV1,
)
from rag_v2.contracts.plan import Action, Intent, SupervisorPlan
from rag_v2.runtime import (
    GeneratorRouteConfigV1,
    GeneratorRoutingPolicyV1,
    RuntimeRouteV1,
    SemanticClaimVerifierV1,
    TrustedRAGQueryV2,
    TrustedRAGRuntimeV2,
)


def _packet(metric: str) -> dict:
    return {
        "query_id": "q1",
        "route": "DIRECT",
        "validation_status": "VERIFIED",
        "allowed_citation_ids": ["E1"],
        "evidence_items": [{
            "citation_id": "E1",
            "fact_id": "fact-1",
            "source_id": "source-1",
            "provenance": {"physical_source_id": "physical-1"},
            "metric": metric,
            "period": "FY2025",
            "value": "16.7" if metric == "Total volume" else "115186",
            "unit": None,
            "currency": None,
            "scale": None,
        }],
    }


def _run(metric: str, answer: str):
    provider = MockGeneratorProviderV1(
        "primary",
        "fixture",
        {
            "query_id": "q1",
            "route": "DIRECT",
            "answer_text": answer,
            "citation_ids": ["E1"],
            "generator_model": "fixture",
        },
    )
    packet = _packet(metric)
    machine = TrustedGenerationStateMachineV1(
        ProviderRegistryV1({"primary": provider}),
        RuntimeGenerationValidatorV1(),
        GenerationRecoveryPolicyV1("primary", fallback_budget=0),
        semantic_verifier=SemanticClaimVerifierV1(),
    )
    return machine.run(GenerationInputV1("q1", "DIRECT", "question", packet))


@pytest.mark.parametrize(
    ("metric", "answer"),
    [
        ("Data Center revenue", "Data Center revenue was 115186 in FY2025 [E1]."),
        ("Automotive revenue", "Automotive revenue was 115186 in FY2025 [E1]."),
        ("Visa transactions", "Visa transactions were 115186 in FY2025 [E1]."),
    ],
)
def test_a2_safe_claims_remain_released(metric: str, answer: str) -> None:
    assert _run(metric, answer).released


def test_a2_unsupported_unit_claim_is_blocked() -> None:
    result = _run("Total volume", "Total volume was 16.7 cubic feet in FY2025 [E1].")
    assert not result.released
    assert result.validation_report is not None
    assert "SCV_UNIT_UNSUPPORTED" in result.validation_report.failure_codes


def test_runtime_default_wires_claim_verifier() -> None:
    packet = _packet("Total volume")
    provider = MockGeneratorProviderV1(
        "primary",
        "fixture",
        {
            "query_id": "q1",
            "route": "DIRECT",
            "answer_text": "Total volume was 16.7 cubic feet in FY2025 [E1].",
            "citation_ids": ["E1"],
            "generator_model": "fixture",
        },
    )
    plan = SupervisorPlan.from_dict({
        "intent": Intent.DIRECT_FACT.value,
        "operation": None,
        "next_action": Action.GENERATE.value,
        "required_slots": [{"slot_id": "slot_1", "metric": "Total volume",
                             "period": "FY2025", "role": "fact", "value_type": "number", "unit": None}],
    })
    runtime = TrustedRAGRuntimeV2(
        ProviderRegistryV1({"primary": provider}),
        GeneratorRoutingPolicyV1({RuntimeRouteV1.DIRECT_FACT: GeneratorRouteConfigV1("primary")}),
    )
    response = runtime.handle(TrustedRAGQueryV2("q1", "What was total volume?", plan, packet))
    assert not response.released
    assert any("SCV_UNIT_UNSUPPORTED" in codes for codes in response.trace.validator_codes)
