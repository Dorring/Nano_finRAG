"""TV2-05 canonical validator/release gate integration tests."""
from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.contracts import Intent, SupervisorPlan
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    DeterministicCalculationCapability,
    TrustedFinancialRuntimeV2,
    TrustedReleaseValidationCapability,
    TrustedV2CapabilityPorts,
    TrustedV2FactoryError,
    TrustedV2GenerationCapability,
    V2ExecutionStatus,
    build_trusted_v2_runtime,
)
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator
from src.runtime.trusted_v2_generation import CandidateExecutionResult

from tests.test_trusted_v2_r4_binder import (
    SelectingBinderProvider,
    _fact,
    _plan,
    _real_capabilities,
    _request,
    _slot,
)


def _coordinator(
    query: str,
    plan: SupervisorPlan,
    retrieval: Any,
    binder: Any,
    *,
    calculation: Any = None,
    generation: Any = None,
    validator: Any = None,
) -> BoundedTrustedV2Coordinator:
    return BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({query: plan})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
            generation=generation,
            release_validator=validator,
        ),
        budget=AdaptiveRAGBudgetV1(
            max_replan_rounds=3,
            max_total_tool_calls=4,
            max_same_tool_retry=3,
        ),
    )


def test_fact_candidate_passes_canonical_release_gate() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = TrustedV2GenerationCapability()
    validator = TrustedReleaseValidationCapability()
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("What was revenue?", "tv2-05-fact"))
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert outcome.runtime_metadata["config_version"] == "tv2-05"
    assert outcome.validator_status == "PASS"
    assert outcome.evidence_ids == ["E1"]
    assert outcome.citation_ids == ["citation-E1"]
    trace = outcome.debug_metadata["trace"]
    assert trace["validation_passed"] is True
    assert trace["release_decision"] == "RELEASED"
    assert trace["renderer_release_authority"] if "renderer_release_authority" in trace else True
    assert generation.renderer_calls == 1


def test_calculation_candidate_passes_and_preserves_c1_provenance() -> None:
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT", "PRIOR"]], facts, SelectingBinderProvider()
    )
    calculation = DeterministicCalculationCapability()
    generation = TrustedV2GenerationCapability()
    validator = TrustedReleaseValidationCapability()
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )
    outcome = asyncio.run(
        _coordinator(
            "Compare years",
            plan,
            retrieval,
            binder,
            calculation=calculation,
            generation=generation,
            validator=validator,
        ).execute(_request("Compare years", "tv2-05-calc"))
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert outcome.release_status.value == "RELEASED"
    assert outcome.calculation_ids == [calculation.last_calculation_id]
    assert outcome.runtime_metadata["calculation_result_id"] == calculation.last_calculation_id
    assert generation.renderer_calls == 1
    assert calculation.calls == 1


class _Specialist:
    def __init__(self, answer: str, citation_ids: list[str]) -> None:
        self.answer = answer
        self.citation_ids = citation_ids
        self.calls = 0

    def generate(self, question: str, evidence_items: list[dict[str, Any]], calculation_result: dict[str, Any] | None):
        del question, evidence_items, calculation_result
        self.calls += 1
        return {"answer_text": self.answer, "citation_ids": self.citation_ids}


def test_specialist_candidate_uses_same_release_authority() -> None:
    facts = {
        "E1": _fact("E1", slots=("a",)),
        "E2": _fact("E2", slots=("b",)),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1", "E2"]], facts, SelectingBinderProvider()
    )
    specialist = _Specialist("Revenue [citation-E1]", ["citation-E1"])
    generation = TrustedV2GenerationCapability(specialist=specialist)
    validator = TrustedReleaseValidationCapability()
    plan = _plan(_slot("a"), _slot("b"))
    outcome = asyncio.run(
        _coordinator(
            "Summarize revenue",
            plan,
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("Summarize revenue", "tv2-05-specialist"))
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert specialist.calls == 1
    assert generation.specialist_calls == 1
    assert outcome.route == "MULTI"


class _BadGeneration:
    candidate_mode = True

    def __init__(self, candidate: CandidateExecutionResult) -> None:
        self.candidate = candidate
        self.calls = 0

    def generate(self, state):
        self.calls += 1
        return self.candidate


def _bad_candidate(answer: str, *, evidence: tuple[str, ...] = ("E1",), citations: tuple[str, ...] = ("citation-E1",)) -> CandidateExecutionResult:
    return CandidateExecutionResult(
        candidate_answer=answer,
        route="STRUCTURED_SINGLE",
        route_reason="fixture",
        bound_evidence_ids=evidence,
        citation_ids=citations,
    )


def test_numeric_mismatch_repairs_once_then_releases() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = _BadGeneration(_bad_candidate("Revenue (FY2024): 999 USD million [citation-E1]"))
    validator = TrustedReleaseValidationCapability()
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("What was revenue?", "tv2-05-repair"))
    )

    assert outcome.status is V2ExecutionStatus.READY_FOR_RELEASE
    assert "REPAIRED_ONCE" in outcome.reason_codes
    assert validator.repair_calls == 1
    assert validator.validation_calls == 2
    assert outcome.answer is not None and "100" in outcome.answer
    assert generation.calls == 1
    trace = outcome.debug_metadata["trace"]
    assert trace["repair_count"] == 1
    assert trace["revalidated"] is True


class _BadRepair:
    def can_repair(self, state, candidate, validation):
        del state, candidate, validation
        return True

    def repair(self, state, candidate, validation):
        del state, validation
        return candidate


class _CrashingRepair:
    def can_repair(self, state, candidate, validation):
        del state, candidate, validation
        return True

    def repair(self, state, candidate, validation):
        del state, candidate, validation
        raise RuntimeError("repair crashed")


def test_repair_crash_is_execution_error() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = _BadGeneration(_bad_candidate("Revenue (FY2024): 999 USD million [citation-E1]"))
    validator = TrustedReleaseValidationCapability(repairer=_CrashingRepair())
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("What was revenue?", "tv2-05-repair-error"))
    )

    assert outcome.status is V2ExecutionStatus.EXECUTION_ERROR
    assert outcome.release_status.value == "NOT_RELEASED"
    assert validator.repair_calls == 1
    assert "REPAIR_EXCEPTION" in outcome.reason_codes


def test_failed_revalidation_is_fail_closed_without_second_repair() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = _BadGeneration(_bad_candidate("Revenue (FY2024): 999 USD million [citation-E1]"))
    validator = TrustedReleaseValidationCapability(repairer=_BadRepair())
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("What was revenue?", "tv2-05-repair-fail"))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.release_status.value == "NOT_RELEASED"
    assert validator.repair_calls == 1
    assert validator.validation_calls == 2
    assert "REPAIR_REVALIDATION_FAILED" in outcome.reason_codes


def test_unbound_evidence_cannot_repair_or_release() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = _BadGeneration(_bad_candidate(
        "Revenue (FY2024): 100 USD million [citation-E1]",
        evidence=("E999",),
    ))
    validator = TrustedReleaseValidationCapability()
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("What was revenue?", "tv2-05-unbound"))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "BOUND_EVIDENCE_NOT_ADMITTED" in outcome.reason_codes
    assert validator.repair_calls == 0


class _BrokenValidator:
    candidate_mode = True

    def validate(self, state, candidate):
        del state, candidate
        raise RuntimeError("validator crashed")


def test_validator_crash_is_execution_error() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=TrustedV2GenerationCapability(),
            validator=_BrokenValidator(),
        ).execute(_request("What was revenue?", "tv2-05-validator-error"))
    )

    assert outcome.status is V2ExecutionStatus.EXECUTION_ERROR
    assert outcome.release_status.value == "NOT_RELEASED"
    assert "VALIDATOR_EXCEPTION" in outcome.reason_codes


def test_factory_is_complete_and_has_no_v1_fallback() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    capabilities = TrustedV2CapabilityPorts(
        retrieval=retrieval,
        evidence_evaluator=binder,
        calculation=DeterministicCalculationCapability(),
        generation=TrustedV2GenerationCapability(),
        release_validator=TrustedReleaseValidationCapability(),
    )
    supervisor = SupervisorService(
        DeterministicFallbackProvider({"What was revenue?": _plan(_slot("revenue"))})
    )
    runtime = build_trusted_v2_runtime(supervisor, capabilities=capabilities)
    assert isinstance(runtime, TrustedFinancialRuntimeV2)

    try:
        build_trusted_v2_runtime(
            supervisor,
            capabilities=TrustedV2CapabilityPorts(
                retrieval=retrieval,
                evidence_evaluator=binder,
                generation=capabilities.generation,
                release_validator=capabilities.release_validator,
            ),
        )
    except TrustedV2FactoryError:
        pass
    else:
        raise AssertionError("missing calculator must fail fast")


def test_candidate_text_does_not_create_provenance() -> None:
    facts = {"E1": _fact("E1", value="100")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = _BadGeneration(_bad_candidate(
        "Revenue (FY2024): 100 USD million [citation-E999]",
        citations=(),
    ))
    validator = TrustedReleaseValidationCapability()
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
            validator=validator,
        ).execute(_request("What was revenue?", "tv2-05-text-provenance"))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.citation_ids == []
    assert "UNBOUND_CITATION_METADATA" in outcome.reason_codes
