from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import SupervisorPlan

from .binder_provider import BinderCallMetadata, BinderProvider, BinderProviderError, BinderProviderResult
from .binding_validator import BindingValidationResult, validate_binding


@dataclass(frozen=True)
class BinderRequest:
    question_id: str
    question: str
    plan: SupervisorPlan
    facts: tuple[Mapping[str, Any], ...]

    @property
    def required_slots(self):
        return self.plan.required_slots

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "intent": self.plan.intent.value,
            "operation": self.plan.operation,
            "required_slots": [slot.to_dict() for slot in self.plan.required_slots],
            "financial_facts": [dict(fact) for fact in self.facts],
        }


@dataclass(frozen=True)
class BinderRun:
    request: BinderRequest
    binding: EvidenceBinding
    validation: BindingValidationResult
    metadata: BinderCallMetadata | None
    skipped_no_fact_supply: bool = False
    raw_response: str | None = None
    schema_valid: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.request.question_id,
            "binder_model_status": self.binding.status,
            "binding": self.binding.to_dict(),
            "binding_schema_valid": self.schema_valid,
            "binding_validator_pass": self.validation.passed,
            "final_binding_status": self.validation.final_status,
            "selected_fact_ids": list(self.validation.selected_fact_ids),
            "missing_slots": list(self.binding.missing_slots),
            "ambiguous_slots": list(self.binding.ambiguous_slots),
            "invalid_reasons": list(self.binding.invalid_reasons),
            "validation_reasons": list(self.validation.reasons),
            "slots_requested": len(self.request.required_slots),
            "slots_bound": self.validation.bound_slot_count,
            "slots_missing": len(self.binding.missing_slots),
            "slots_ambiguous": len(self.binding.ambiguous_slots),
            "skipped_no_fact_supply": self.skipped_no_fact_supply,
            "metadata": self.metadata.to_dict() if self.metadata else None,
        }


def empty_fact_binding(plan: SupervisorPlan) -> EvidenceBinding:
    return EvidenceBinding(
        status=BindingStatus.MISSING.value,
        slot_bindings={},
        missing_slots=tuple(slot.slot_id for slot in plan.required_slots),
    )


class SemanticBinderService:
    """One-call semantic binder; it never retrieves, calculates, or retries."""

    def __init__(self, provider: BinderProvider, *, max_calls_per_query: int = 1) -> None:
        if max_calls_per_query != 1:
            raise ValueError("NF-V2-03 freezes max_binder_calls at 1")
        self.provider = provider
        self.max_calls_per_query = max_calls_per_query

    def bind(self, request: BinderRequest) -> BinderRun:
        if not request.facts:
            binding = empty_fact_binding(request.plan)
            validation = validate_binding(binding, request.plan, request.facts)
            return BinderRun(request, binding, validation, None, skipped_no_fact_supply=True, schema_valid=True)
        try:
            result: BinderProviderResult = self.provider.bind(request.to_dict())
            binding = result.binding
            if binding is None:
                raise BinderProviderError("provider returned no EvidenceBinding")
            validation = validate_binding(binding, request.plan, request.facts)
            return BinderRun(request, binding, validation, result.metadata, raw_response=result.raw_response, schema_valid=True)
        except BinderProviderError as exc:
            metadata = getattr(self.provider, "last_call", None)
            classification = getattr(exc, "classification", None)
            failure_reason = f"adapter_failure:{classification}" if classification else f"provider_failure:{type(exc).__name__}"
            binding = EvidenceBinding(
                status=BindingStatus.INVALID.value,
                slot_bindings={},
                invalid_reasons=(failure_reason,),
            )
            validation = validate_binding(binding, request.plan, request.facts)
            return BinderRun(request, binding, validation, metadata, schema_valid=bool(getattr(exc, "schema_valid", False)))
