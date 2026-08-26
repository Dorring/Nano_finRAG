"""TV2-04 deterministic calculator and generator routing integration tests."""

from __future__ import annotations

import asyncio
from typing import Any

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.contracts import Action, Intent, SupervisorPlan
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.domain.calculation import CalculationStatus
from src.runtime import (
    DeterministicCalculationCapability,
    LocalSpecialistGenerationAdapter,
    TrustedV2CapabilityPorts,
    TrustedV2GenerationCapability,
    V2ExecutionStatus,
)
from src.runtime.trusted_v2_coordinator import BoundedTrustedV2Coordinator

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
    budget: AdaptiveRAGBudgetV1 | None = None,
) -> BoundedTrustedV2Coordinator:
    return BoundedTrustedV2Coordinator(
        SupervisorService(DeterministicFallbackProvider({query: plan})),
        capabilities=TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=binder,
            calculation=calculation,
            generation=generation,
        ),
        budget=budget
        or AdaptiveRAGBudgetV1(
            max_replan_rounds=3,
            max_total_tool_calls=4,
            max_same_tool_retry=3,
        ),
    )


def test_fact_routes_to_existing_deterministic_renderer_and_stays_unreleased() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    generation = TrustedV2GenerationCapability()
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=generation,
        ).execute(_request("What was revenue?"))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.release_status.value == "NOT_RELEASED"
    assert "FINAL_VALIDATION_NOT_WIRED" in outcome.reason_codes
    assert outcome.runtime_metadata["candidate_status"] == "CANDIDATE_READY_FOR_VALIDATION"
    trace = outcome.debug_metadata["trace"]
    assert trace["generation_route"] == "STRUCTURED_SINGLE"
    assert trace["renderer_invoked"] is True
    assert trace["specialist_invoked"] is False
    assert outcome.evidence_ids == ["E1"]
    assert outcome.citation_ids == ["citation-E1"]


def test_calculation_uses_real_nine_operation_executor_and_bound_lineage() -> None:
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="391"),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="383"),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT", "PRIOR"]],
        facts,
        SelectingBinderProvider(),
    )
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )
    calculation = DeterministicCalculationCapability()
    generation = TrustedV2GenerationCapability()
    outcome = asyncio.run(
        _coordinator(
            "Compare years",
            plan,
            retrieval,
            binder,
            calculation=calculation,
            generation=generation,
        ).execute(_request("Compare years"))
    )

    assert calculation.calls == 1
    assert calculation.last_result is not None
    assert calculation.last_result.status is CalculationStatus.EXECUTED
    assert calculation.last_operand_evidence_ids == ("CURRENT", "PRIOR")
    assert len(outcome.calculation_ids) == 1
    assert outcome.evidence_ids == ["CURRENT", "PRIOR"]
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "FINAL_VALIDATION_NOT_WIRED" in outcome.reason_codes
    assert "Growth Rate" in (outcome.answer or "")
    trace = outcome.debug_metadata["trace"]
    assert trace["generation_route"] == "CALCULATION_SIMPLE"
    assert trace["calculator_invoked"] is True
    assert trace["renderer_invoked"] is True
    assert trace["candidate_ready"] is True


def test_missing_operand_never_calls_calculator_or_generator() -> None:
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",)),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",)),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT"], ["CURRENT"]], facts, SelectingBinderProvider()
    )
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )
    calculation = DeterministicCalculationCapability()
    generation = TrustedV2GenerationCapability()
    outcome = asyncio.run(
        _coordinator(
            "Compare years",
            plan,
            retrieval,
            binder,
            calculation=calculation,
            generation=generation,
            budget=AdaptiveRAGBudgetV1(
                max_replan_rounds=1, max_total_tool_calls=2, max_same_tool_retry=2
            ),
        ).execute(_request("Compare years"))
    )

    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert calculation.calls == 0
    assert generation.route_calls == 0
    assert "MISSING_OPERAND" in outcome.reason_codes or "NO_PROGRESS" in outcome.reason_codes


def test_zero_denominator_is_fail_closed_without_specialist_fallback() -> None:
    facts = {
        "CURRENT": _fact("CURRENT", period="FY2024", slots=("current",), value="10"),
        "PRIOR": _fact("PRIOR", period="FY2023", slots=("prior",), value="0"),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["CURRENT", "PRIOR"]], facts, SelectingBinderProvider()
    )
    plan = _plan(
        _slot("current", period="FY2024", role="current"),
        _slot("prior", period="FY2023", role="prior"),
        intent=Intent.CALCULATION,
        operation="growth_rate",
    )
    calculation = DeterministicCalculationCapability()
    generation = TrustedV2GenerationCapability(
        specialist=LocalSpecialistGenerationAdapter(
            _FakeSpecialist("should-not-be-called")
        )
    )
    outcome = asyncio.run(
        _coordinator(
            "Compare years",
            plan,
            retrieval,
            binder,
            calculation=calculation,
            generation=generation,
        ).execute(_request("Compare years"))
    )

    assert calculation.calls == 1
    assert calculation.last_result is not None
    assert calculation.last_result.status is CalculationStatus.BLOCKED
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert "CALCULATION_INVALID" in outcome.reason_codes
    assert generation.route_calls == 0
    assert generation.specialist_calls == 0


class _FakeSpecialist:
    def __init__(self, answer: str, citation_ids: list[str] | None = None) -> None:
        self.answer = answer
        self.citation_ids = citation_ids or []
        self.calls = 0
        self.last_items: list[dict[str, Any]] = []

    def generate(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        self.calls += 1
        self.last_items = evidence_items
        return {"answer_text": self.answer, "citation_ids": self.citation_ids}


def test_qualitative_route_calls_specialist_with_bound_evidence_only() -> None:
    facts = {
        "E1": _fact("E1", slots=("cause_a",), metric="Operating Margin"),
        "E2": _fact("E2", slots=("cause_b",), metric="Operating Margin"),
    }
    retrieval, binder, _, _, _ = _real_capabilities(
        [["E1", "E2"]], facts, SelectingBinderProvider()
    )
    specialist = _FakeSpecialist("Margin declined because of costs.", ["unknown-X"])
    generation = TrustedV2GenerationCapability(
        specialist=LocalSpecialistGenerationAdapter(specialist)
    )
    plan = _plan(
        _slot("cause_a", metric="Operating Margin", role="operand"),
        _slot("cause_b", metric="Operating Margin", role="operand"),
        intent=Intent.MULTI_EVIDENCE,
    )
    outcome = asyncio.run(
        _coordinator(
            "Why did operating margin decline?",
            plan,
            retrieval,
            binder,
            generation=generation,
        ).execute(_request("Why did operating margin decline?"))
    )

    assert specialist.calls == 1
    assert {item["evidence_id"] for item in specialist.last_items} == {"E1", "E2"}
    assert outcome.status is V2ExecutionStatus.FAIL_CLOSED
    assert outcome.citation_ids == ["citation-E1", "citation-E2"]
    assert "unknown-X" not in outcome.citation_ids
    assert outcome.runtime_metadata["unknown_generated_citation_ids"] == ["unknown-X"]
    assert outcome.debug_metadata["trace"]["specialist_invoked"] is True


def test_candidate_paths_never_emit_released_status() -> None:
    facts = {"E1": _fact("E1")}
    retrieval, binder, _, _, _ = _real_capabilities([["E1"]], facts)
    outcome = asyncio.run(
        _coordinator(
            "What was revenue?",
            _plan(_slot("revenue")),
            retrieval,
            binder,
            generation=TrustedV2GenerationCapability(),
        ).execute(_request("What was revenue?"))
    )
    assert outcome.release_status.value == "NOT_RELEASED"
    assert outcome.status is not V2ExecutionStatus.READY_FOR_RELEASE


def test_all_nine_registry_operations_use_existing_executor() -> None:
    from rag_v2.adaptive import AdaptiveRAGStateV1
    from rag_v2.contracts import RequiredSlot
    from src.domain.calculation import CalculationOperation, CalculationStatus
    from src.finance.calculation_registry import CALCULATION_REGISTRY
    from src.runtime.trusted_v2_calculation import SUPPORTED_CALCULATION_OPERATIONS

    assert set(SUPPORTED_CALCULATION_OPERATIONS) == {
        operation.value for operation in CalculationOperation
    }
    assert set(CALCULATION_REGISTRY) == set(CalculationOperation)

    fixtures = {
        "difference": (("current", "previous"), ("10", "4"), {}),
        "growth_rate": (("current", "previous"), ("10", "4"), {}),
        "percentage_share": (("part", "total"), ("25", "100"), {}),
        "sum": (("left", "right"), ("10", "4"), {}),
        "average": (("left", "right"), ("10", "4"), {}),
        "gross_margin": (("revenue", "cogs"), ("10", "4"), {}),
        "net_margin": (("revenue", "net_income"), ("10", "4"), {}),
        "debt_ratio": (("total_liabilities", "total_assets"), ("4", "10"), {}),
        "scale_conversion": (
            ("value",),
            ("2",),
            {"source_scale": "million", "target_scale": "billion"},
        ),
    }

    for operation, (roles, values, requirements) in fixtures.items():
        slots = tuple(
            RequiredSlot(
                slot_id=f"slot-{role}",
                metric="Revenue",
                period="FY2024",
                role=role,
                value_type="numeric",
                unit=None,
            )
            for role in roles
        )
        plan = SupervisorPlan(
            intent=Intent.CALCULATION,
            required_slots=slots,
            operation=operation,
            next_action=Action.RETRIEVE,
        )
        state = AdaptiveRAGStateV1.new(
            request_id=f"calc-{operation}",
            query=operation,
            intent=Intent.CALCULATION.value,
            task_type=Intent.CALCULATION.value,
            required_slots=[slot.to_dict() for slot in slots],
            plan={"supervisor_plan": plan.to_dict()},
            calculation_requirements={
                "operation": operation,
                "operand_slots": [slot.slot_id for slot in slots],
                **requirements,
            },
        )
        state.evidence_packets = [
            _fact(
                f"E-{operation}-{index}",
                slots=(role,),
                value=value,
                period="FY2024",
            )
            for index, (role, value) in enumerate(zip(roles, values), start=1)
        ]
        state.bound_evidence_ids = [
            str(item["evidence_id"]) for item in state.evidence_packets
        ]
        state.bound_slot_bindings = {
            f"slot-{role}": [f"E-{operation}-{index}"]
            for index, role in enumerate(roles, start=1)
        }

        capability = DeterministicCalculationCapability()
        result = capability.calculate(state)

        assert capability.calls == 1, operation
        assert result.status is CalculationStatus.EXECUTED, (
            operation,
            result.error_code,
            result.error_message,
        )
        assert capability.last_calculation_id is not None
