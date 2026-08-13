from __future__ import annotations

from rag_v2.generation import (
    AnswerEnvelopeV1,
    GenerationInputV1,
    GenerationRecoveryPolicyV1,
    MockGeneratorProviderV1,
    ProviderRegistryV1,
    RecoveryAction,
    RuntimeGenerationValidatorV1,
    TrustedGenerationStateMachineV1,
)
from rag_v2.generation.contracts import ValidationSeverity


def packet() -> dict:
    return {
        "query_id": "q1",
        "route": "DIRECT",
        "validation_status": "VERIFIED",
        "allowed_citation_ids": ["EV-1"],
        "evidence_items": [{"citation_id": "EV-1", "value": "100", "period": "FY2025",
                             "unit": "USD", "currency": "USD", "scale": "1"}],
    }


def envelope(text: str, citations: tuple[str, ...] = ("EV-1",)) -> AnswerEnvelopeV1:
    return AnswerEnvelopeV1("q1", "DIRECT", text, citations, "mock", "mock")


def test_citation_and_numeric_fidelity() -> None:
    validator = RuntimeGenerationValidatorV1()
    assert validator.validate(packet(), envelope("100 USD in FY2025 [EV-1].")).status is ValidationSeverity.PASS
    assert validator.validate(packet(), envelope("101 USD in FY2025 [EV-1].")).hard_fail
    assert validator.validate(packet(), envelope("100 USD in FY2025 [EV-404].", ("EV-404",))).hard_fail


def test_period_unit_and_scale_conflicts_are_hard_failures() -> None:
    validator = RuntimeGenerationValidatorV1()
    assert "GV4_PERIOD_FIDELITY" in validator.validate(packet(), envelope("100 USD in FY2024 [EV-1].")).failure_codes
    assert "GV5_UNIT_CURRENCY_SCALE_FIDELITY" in validator.validate(packet(), envelope("100 EUR in FY2025 [EV-1].")).failure_codes
    assert "GV5_UNIT_CURRENCY_SCALE_FIDELITY" in validator.validate(packet(), envelope("100 million USD in FY2025 [EV-1].")).failure_codes


def test_first_pass_pass_releases() -> None:
    provider = MockGeneratorProviderV1(response={"query_id": "q1", "route": "DIRECT",
        "answer_text": "100 USD in FY2025 [EV-1].", "citation_ids": ["EV-1"], "generator_model": "mock"})
    machine = TrustedGenerationStateMachineV1(ProviderRegistryV1({"primary": provider}),
        RuntimeGenerationValidatorV1(), GenerationRecoveryPolicyV1("primary", fallback_budget=0))
    result = machine.run(GenerationInputV1("q1", "DIRECT", "question", packet()))
    assert result.released and result.state.value == "RELEASED" and len(result.attempts) == 1


def test_failed_first_pass_without_fallback_abstains() -> None:
    provider = MockGeneratorProviderV1(response={"query_id": "q1", "route": "DIRECT",
        "answer_text": "101 USD in FY2025 [EV-1].", "citation_ids": ["EV-1"], "generator_model": "mock"})
    machine = TrustedGenerationStateMachineV1(ProviderRegistryV1({"primary": provider}),
        RuntimeGenerationValidatorV1(), GenerationRecoveryPolicyV1("primary", fallback_budget=0))
    result = machine.run(GenerationInputV1("q1", "DIRECT", "question", packet()))
    assert not result.released and result.state.value == "ABSTAINED" and len(result.attempts) == 1


def test_failed_first_pass_fallback_passes_once() -> None:
    first = MockGeneratorProviderV1(provider_id="primary", response={"query_id": "q1", "route": "DIRECT",
        "answer_text": "101 USD in FY2025 [EV-1].", "citation_ids": ["EV-1"], "generator_model": "mock"})
    fallback = MockGeneratorProviderV1(provider_id="fallback", response={"query_id": "q1", "route": "DIRECT",
        "answer_text": "100 USD in FY2025 [EV-1].", "citation_ids": ["EV-1"], "generator_model": "mock"})
    policy = GenerationRecoveryPolicyV1("primary", "fallback", RecoveryAction.FALLBACK_PROVIDER, fallback_budget=1)
    result = TrustedGenerationStateMachineV1(ProviderRegistryV1({"primary": first, "fallback": fallback}),
        RuntimeGenerationValidatorV1(), policy).run(GenerationInputV1("q1", "DIRECT", "question", packet()))
    assert result.released and len(result.attempts) == 2 and first.calls == fallback.calls == 1


def test_failed_first_pass_and_failed_fallback_abstains_with_two_attempts() -> None:
    bad = {"query_id": "q1", "route": "DIRECT", "answer_text": "101 USD in FY2025 [EV-1].",
           "citation_ids": ["EV-1"], "generator_model": "mock"}
    first = MockGeneratorProviderV1(provider_id="primary", response=bad)
    fallback = MockGeneratorProviderV1(provider_id="fallback", response=bad)
    policy = GenerationRecoveryPolicyV1("primary", "fallback", RecoveryAction.FALLBACK_PROVIDER, fallback_budget=1)
    result = TrustedGenerationStateMachineV1(ProviderRegistryV1({"primary": first, "fallback": fallback}),
        RuntimeGenerationValidatorV1(), policy).run(GenerationInputV1("q1", "DIRECT", "question", packet()))
    assert not result.released and result.state.value == "ABSTAINED"
    assert len(result.attempts) == 2 and max(item.attempt_index for item in result.attempts) == 1


def test_provider_error_and_no_answer_fail_closed() -> None:
    class Broken(MockGeneratorProviderV1):
        def generate(self, generation_input, generation_context):
            self.calls += 1
            raise RuntimeError("offline")
    provider = Broken()
    machine = TrustedGenerationStateMachineV1(ProviderRegistryV1({"primary": provider}),
        RuntimeGenerationValidatorV1(), GenerationRecoveryPolicyV1("primary", fallback_budget=0))
    assert machine.run(None, no_answer=True).state.value == "ABSTAINED" and provider.calls == 0
    result = machine.run(GenerationInputV1("q1", "DIRECT", "question", packet()))
    assert not result.released and result.state.value == "ABSTAINED" and provider.calls == 1
