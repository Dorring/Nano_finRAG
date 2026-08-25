from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag_v2.adaptive import (
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidenceEvaluationV1,
    ReasonCode,
)
from rag_v2.contracts.evidence import BindingStatus
from rag_v2.contracts.plan import Intent, SupervisorPlan
from rag_v2.evidence.binder_service import BinderRequest, BinderRun, SemanticBinderService


class SemanticBinderCapabilityError(RuntimeError):
    """Raised when the Binder contract itself cannot be evaluated safely."""


def _norm(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().casefold()
    return text or None


def _stable_unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return tuple(result)


class SemanticEvidenceEvaluationCapability:
    """TV2-02 EvidenceEvaluationCapability backed by SemanticBinderService.

    The Binder owns slot admission.  This adapter only maps its structured
    result to the bounded runtime's deterministic reason-code contract.
    """

    def __init__(self, binder: SemanticBinderService) -> None:
        if not isinstance(binder, SemanticBinderService):
            raise TypeError("binder must be SemanticBinderService")
        self.binder = binder
        self.calls = 0
        self.last_run: BinderRun | None = None
        self.last_bound_evidence_ids: tuple[str, ...] = ()
        self.last_citation_ids: tuple[str, ...] = ()
        self.last_bound_slot_bindings: dict[str, tuple[str, ...]] = {}
        self._trace: list[dict[str, Any]] = []

    @staticmethod
    def _plan(state: AdaptiveRAGStateV1) -> SupervisorPlan:
        try:
            return SupervisorPlan.from_dict(state.plan["supervisor_plan"])
        except Exception as exc:
            raise SemanticBinderCapabilityError(
                "invalid_supervisor_plan_for_binder"
            ) from exc

    @staticmethod
    def _facts(state: AdaptiveRAGStateV1) -> tuple[Mapping[str, Any], ...]:
        facts: list[Mapping[str, Any]] = []
        for raw in state.evidence_packets:
            if not isinstance(raw, Mapping):
                raise SemanticBinderCapabilityError("candidate_fact_must_be_mapping")
            fact = dict(raw)
            metadata = fact.get("metadata")
            if isinstance(metadata, Mapping):
                for key in ("fact_id", "provenance_complete", "physical_source_id"):
                    if key not in fact and key in metadata:
                        fact[key] = metadata[key]
            fact_id = fact.get("fact_id") or fact.get("evidence_id")
            if not fact_id:
                raise SemanticBinderCapabilityError("candidate_fact_missing_fact_id")
            fact["fact_id"] = str(fact_id)
            fact.setdefault("candidate_id", fact.get("candidate_key", fact["fact_id"]))
            fact.setdefault(
                "physical_source_id",
                fact.get("source") or fact.get("document_id") or fact["fact_id"],
            )
            fact["provenance_complete"] = bool(
                fact.get("provenance_complete", False)
            )
            facts.append(fact)
        return tuple(facts)

    @staticmethod
    def _missing_reason(
        plan: SupervisorPlan,
        missing_slots: tuple[str, ...],
        facts: tuple[Mapping[str, Any], ...],
    ) -> ReasonCode:
        if plan.intent is Intent.CALCULATION and missing_slots:
            return ReasonCode.MISSING_OPERAND
        by_slot = {slot.slot_id: slot for slot in plan.required_slots}
        for slot_id in missing_slots:
            slot = by_slot.get(slot_id)
            if slot is None:
                continue
            same_metric = [
                fact
                for fact in facts
                if _norm(fact.get("metric")) == _norm(slot.metric)
            ]
            if same_metric and all(
                _norm(fact.get("period")) != _norm(slot.period)
                for fact in same_metric
            ):
                return ReasonCode.WRONG_PERIOD
        return ReasonCode.MISSING_SLOT

    def evaluate(self, state: AdaptiveRAGStateV1) -> EvidenceEvaluationV1:
        plan = self._plan(state)
        facts = self._facts(state)
        self.calls += 1
        request = BinderRequest(
            question_id=state.request_id,
            question=state.normalized_query,
            plan=plan,
            facts=facts,
        )
        run = self.binder.bind(request)
        self.last_run = run
        status = getattr(run.binding.status, "value", str(run.binding.status))
        if status == BindingStatus.BOUND.value:
            if not run.validation.passed:
                raise SemanticBinderCapabilityError(
                    "bound_evidence_failed_structural_validation"
                )
            selected = _stable_unique(run.validation.selected_fact_ids)
            self.last_bound_evidence_ids = selected
            self.last_bound_slot_bindings = {
                str(slot_id): tuple(str(item) for item in fact_ids)
                for slot_id, fact_ids in run.binding.slot_bindings.items()
            }
            fact_by_id = {
                str(fact.get("fact_id")): fact
                for fact in facts
                if fact.get("fact_id")
            }
            self.last_citation_ids = _stable_unique(
                str(fact_by_id[fact_id].get("citation_id"))
                for fact_id in selected
                if fact_by_id.get(fact_id, {}).get("citation_id")
            )
            evaluation = EvidenceEvaluationV1(
                decision=EvidenceDecision.SUFFICIENT,
                requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                supported_slots=tuple(
                    str(slot_id) for slot_id in run.binding.slot_bindings
                ),
                supporting_evidence_ids=selected,
                temporal_status="BOUND",
                calculation_ready=False,
            )
        elif status == BindingStatus.MISSING.value:
            missing = tuple(run.binding.missing_slots)
            reason = self._missing_reason(plan, missing, facts)
            self.last_bound_evidence_ids = _stable_unique(
                run.validation.selected_fact_ids
            )
            self.last_bound_slot_bindings = {
                str(slot_id): tuple(str(item) for item in fact_ids)
                for slot_id, fact_ids in run.binding.slot_bindings.items()
            }
            self.last_citation_ids = ()
            evaluation = EvidenceEvaluationV1(
                decision=EvidenceDecision.REPAIRABLE,
                reason_codes=(reason,),
                requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                supported_slots=tuple(
                    str(slot_id) for slot_id in run.binding.slot_bindings
                ),
                missing_slots=missing,
                supporting_evidence_ids=self.last_bound_evidence_ids,
                temporal_status="MISSING",
                calculation_ready=False,
            )
        elif status == BindingStatus.AMBIGUOUS.value:
            self.last_bound_evidence_ids = ()
            self.last_bound_slot_bindings = {}
            self.last_citation_ids = ()
            evaluation = EvidenceEvaluationV1(
                decision=EvidenceDecision.UNRESOLVED_CONFLICT,
                reason_codes=(ReasonCode.EVIDENCE_CONFLICT,),
                requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                supported_slots=tuple(
                    str(slot_id) for slot_id in run.binding.slot_bindings
                ),
                missing_slots=tuple(run.binding.ambiguous_slots),
                conflicts=tuple(
                    {"slot_id": str(slot_id), "type": "AMBIGUOUS"}
                    for slot_id in run.binding.ambiguous_slots
                ),
                temporal_status="AMBIGUOUS",
                calculation_ready=False,
            )
        elif status == BindingStatus.INVALID.value:
            self.last_bound_evidence_ids = ()
            self.last_bound_slot_bindings = {}
            self.last_citation_ids = ()
            if not run.schema_valid:
                raise SemanticBinderCapabilityError(
                    "binder_returned_invalid_schema"
                )
            evaluation = EvidenceEvaluationV1(
                decision=EvidenceDecision.TERMINAL_INSUFFICIENT,
                reason_codes=(ReasonCode.EVIDENCE_CONFLICT,),
                requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                missing_slots=tuple(run.binding.missing_slots),
                conflicts=tuple(
                    {"reason": str(reason)}
                    for reason in (
                        *run.binding.invalid_reasons,
                        *run.validation.reasons,
                    )
                ),
                temporal_status="INVALID",
                calculation_ready=False,
            )
        else:
            raise SemanticBinderCapabilityError(
                f"unknown_binder_status:{run.binding.status}"
            )

        self._trace.append(
            {
                "round": self.calls - 1,
                "status": status,
                "bound_slot_ids": list(run.binding.slot_bindings),
                "missing_slot_ids": list(run.binding.missing_slots),
                "ambiguous_slot_ids": list(run.binding.ambiguous_slots),
                "bound_evidence_ids": list(self.last_bound_evidence_ids),
                "reason_codes": [item.value for item in evaluation.reason_codes],
                "selected_fact_ids": list(run.validation.selected_fact_ids),
            }
        )
        return evaluation

    def trace_snapshot(self) -> dict[str, Any]:
        records = [dict(item) for item in self._trace]
        return {
            "binder_status_per_round": [str(item["status"]) for item in records],
            "bound_slot_ids": [
                slot_id for item in records for slot_id in item["bound_slot_ids"]
            ],
            "missing_slot_ids": [
                slot_id for item in records for slot_id in item["missing_slot_ids"]
            ],
            "wrong_period_slots": [
                slot_id
                for item in records
                if "WRONG_PERIOD" in item["reason_codes"]
                for slot_id in item["missing_slot_ids"]
            ],
            "missing_operand_slots": [
                slot_id
                for item in records
                if "MISSING_OPERAND" in item["reason_codes"]
                for slot_id in item["missing_slot_ids"]
            ],
            "conflict_ids": [
                slot_id for item in records for slot_id in item["ambiguous_slot_ids"]
            ],
            "bound_evidence_ids": list(self.last_bound_evidence_ids),
            "bound_slot_bindings": {
                key: list(value) for key, value in self.last_bound_slot_bindings.items()
            },
            "binder_rounds": records,
        }


__all__ = [
    "SemanticBinderCapabilityError",
    "SemanticEvidenceEvaluationCapability",
]
