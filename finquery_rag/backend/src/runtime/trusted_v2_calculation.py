"""TV2-04 deterministic calculation capability.

This module is a thin adapter around the frozen finance calculation domain.
It never reads query text or conversation history to invent operands: only
Binder-admitted evidence IDs and the structured SupervisorPlan are accepted.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Callable

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.contracts.plan import Intent, SupervisorPlan

from src.domain.calculation import (
    CalculationOperation,
    CalculationOperand,
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
)
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_registry import get_operation_entry
from src.finance.primitive_tools import parse_financial_number


class DeterministicCalculationCapabilityError(RuntimeError):
    """Raised when the structured calculator boundary is invalid."""


SUPPORTED_CALCULATION_OPERATIONS = tuple(
    operation.value for operation in CalculationOperation
)


def _stable_unique(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _operation(value: Any) -> CalculationOperation:
    try:
        return value if isinstance(value, CalculationOperation) else CalculationOperation(str(value))
    except (TypeError, ValueError) as exc:
        raise DeterministicCalculationCapabilityError(
            f"unsupported_calculation_operation:{value}"
        ) from exc


def _normalise_role(role: str) -> str:
    value = str(role or "").strip().casefold().replace("-", "_")
    aliases = {
        "prior": "previous",
        "prev": "previous",
        "old": "previous",
        "new": "current",
        "minuend": "current",
        "subtrahend": "previous",
    }
    return aliases.get(value, value)


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    value = candidate.get("fact_id") or candidate.get("evidence_id") or candidate.get("candidate_id")
    if value is None or not str(value).strip():
        raise DeterministicCalculationCapabilityError("bound_candidate_missing_id")
    return str(value).strip()


def _parse_candidate_value(
    candidate: Mapping[str, Any],
    *,
    apply_scale: bool = True,
) -> Decimal | None:
    """Use structured candidate fields, never answer text, for numeric input."""

    parsed = candidate.get("parsed_numeric_value")
    if parsed is not None:
        result = parse_financial_number(parsed)
    else:
        raw = candidate.get("value")
        if raw is None:
            raw = candidate.get("raw_value")
        if raw is None:
            return None
        result = parse_financial_number(
            raw,
            scale=candidate.get("scale") if apply_scale else None,
        )
    return result.value if result.ok else None


def _source_text(candidate: Mapping[str, Any]) -> str:
    for key in ("source_text", "row_label", "metric", "normalized_metric"):
        value = candidate.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "bound evidence"


def _page(candidate: Mapping[str, Any]) -> int | None:
    value = candidate.get("page", candidate.get("pdf_page"))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _calculation_id(operation: CalculationOperation, result: CalculationResult) -> str:
    payload = {
        "operation": operation.value,
        "formula_version": result.formula_version,
        "value": str(result.value) if result.value is not None else None,
        "unit": result.unit,
        "operands": [
            {
                "name": operand.name,
                "value": str(operand.value),
                "evidence_chunk_id": operand.evidence_chunk_id,
            }
            for operand in result.operands
        ],
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"C1-{digest}"


class DeterministicCalculationCapability:
    """Adapt the existing 9-operation executor to the TV2 capability port."""

    candidate_mode = True

    def __init__(
        self,
        executor: Callable[[CalculationPlan], CalculationResult] = execute_plan,
    ) -> None:
        self.executor = executor
        self.calls = 0
        self.last_result: CalculationResult | None = None
        self.last_calculation_id: str | None = None
        self.last_operand_evidence_ids: tuple[str, ...] = ()
        self._last_error: str | None = None

    @staticmethod
    def _plan(state: AdaptiveRAGStateV1) -> SupervisorPlan:
        try:
            return SupervisorPlan.from_dict(state.plan["supervisor_plan"])
        except Exception as exc:
            raise DeterministicCalculationCapabilityError(
                "invalid_supervisor_plan_for_calculator"
            ) from exc

    @staticmethod
    def _bound_candidates(state: AdaptiveRAGStateV1) -> dict[str, Mapping[str, Any]]:
        allowed = {
            str(value).strip()
            for value in getattr(state, "bound_evidence_ids", ())
            if str(value).strip()
        }
        if not allowed:
            return {}
        result: dict[str, Mapping[str, Any]] = {}
        for raw in state.evidence_packets:
            if not isinstance(raw, Mapping):
                raise DeterministicCalculationCapabilityError(
                    "candidate_evidence_must_be_mapping"
                )
            candidate = dict(raw)
            identity = _candidate_id(candidate)
            if identity in allowed:
                result[identity] = candidate
        return result

    @staticmethod
    def _slot_bindings(state: AdaptiveRAGStateV1) -> dict[str, tuple[str, ...]]:
        raw = getattr(state, "bound_slot_bindings", {})
        if not isinstance(raw, Mapping):
            return {}
        return {
            str(slot_id): tuple(str(item) for item in values)
            for slot_id, values in raw.items()
            if isinstance(values, (list, tuple, set))
        }

    @classmethod
    def _operand_for_slot(
        cls,
        slot_id: str,
        role: str,
        bindings: Mapping[str, tuple[str, ...]],
        candidates: Mapping[str, Mapping[str, Any]],
        operation: CalculationOperation,
    ) -> CalculationOperand | None:
        ids = bindings.get(slot_id, ())
        if len(ids) != 1:
            return None
        candidate = candidates.get(ids[0])
        if candidate is None:
            return None
        value = _parse_candidate_value(
            candidate,
            apply_scale=operation is not CalculationOperation.SCALE_CONVERSION,
        )
        if value is None:
            return None
        return CalculationOperand(
            name=role,
            value=value,
            unit=(candidate.get("unit") or candidate.get("currency")),
            scale=(str(candidate["scale"]) if candidate.get("scale") is not None else None),
            source_text=_source_text(candidate),
            evidence_chunk_id=_candidate_id(candidate),
            document_name=(candidate.get("document_name") or candidate.get("document_id")),
            page=_page(candidate),
        )

    @classmethod
    def _build_operands(
        cls,
        plan: SupervisorPlan,
        state: AdaptiveRAGStateV1,
    ) -> tuple[CalculationOperand, ...]:
        operation = _operation(plan.operation)
        entry = get_operation_entry(operation)
        if entry is None:
            raise DeterministicCalculationCapabilityError(
                f"calculation_registry_entry_missing:{operation.value}"
            )
        candidates = cls._bound_candidates(state)
        bindings = cls._slot_bindings(state)
        if not candidates or not bindings:
            return ()

        slot_by_role: dict[str, str] = {}
        for slot in plan.required_slots:
            slot_by_role.setdefault(_normalise_role(slot.role), slot.slot_id)
            slot_by_role.setdefault(_normalise_role(slot.slot_id), slot.slot_id)

        operands: list[CalculationOperand] = []
        for index, role in enumerate(entry.operand_roles):
            normalized_role = _normalise_role(role)
            slot_id = slot_by_role.get(normalized_role)
            if slot_id is None and index < len(plan.required_slots):
                slot_id = plan.required_slots[index].slot_id
            if slot_id is None:
                return ()
            operand = cls._operand_for_slot(
                slot_id, role, bindings, candidates, operation
            )
            if operand is None:
                return ()
            operands.append(operand)

        if not entry.operand_roles:
            for slot in plan.required_slots:
                operand = cls._operand_for_slot(
                    slot.slot_id,
                    _normalise_role(slot.role) or slot.slot_id,
                    bindings,
                    candidates,
                    operation,
                )
                if operand is None:
                    return ()
                operands.append(operand)
        return tuple(operands)

    def calculate(self, state: AdaptiveRAGStateV1) -> CalculationResult:
        """Execute only on Binder-admitted complete structured operands."""

        plan = self._plan(state)
        if plan.intent is not Intent.CALCULATION:
            raise DeterministicCalculationCapabilityError(
                "calculator_called_for_non_calculation_plan"
            )
        operands = self._build_operands(plan, state)
        operation = _operation(plan.operation)
        entry = get_operation_entry(operation)
        if entry is None:
            raise DeterministicCalculationCapabilityError(
                f"unsupported_calculation_operation:{operation.value}"
            )
        if len(operands) < entry.min_operands:
            result = CalculationResult(
                status=CalculationStatus.BLOCKED,
                operation=operation,
                formula=entry.formula,
                formula_version=entry.formula_version,
                target_metric=operation.value,
                operands=operands,
                error_code="INSUFFICIENT_OPERANDS",
                error_message="Binder-admitted operands are incomplete",
            )
            self.last_result = result
            self.last_calculation_id = None
            self.last_operand_evidence_ids = tuple(
                operand.evidence_chunk_id for operand in operands
            )
            return result

        requirements = state.calculation_requirements
        if not isinstance(requirements, Mapping):
            raise DeterministicCalculationCapabilityError(
                "calculation_requirements_must_be_mapping"
            )
        raw_precision = requirements.get("precision", 4)
        try:
            precision = max(0, int(raw_precision))
        except (TypeError, ValueError) as exc:
            raise DeterministicCalculationCapabilityError(
                "invalid_calculation_precision"
            ) from exc
        source_scale = requirements.get("source_scale")
        if source_scale is None and operands:
            source_scale = operands[0].scale
        target_scale = requirements.get("target_scale")
        target_metric = requirements.get("target_metric") or operation.value
        calculation_plan = CalculationPlan(
            operation=operation,
            operands=operands,
            formula_version=entry.formula_version,
            target_metric=str(target_metric),
            precision=precision,
            label=(
                str(requirements["label"])
                if requirements.get("label") is not None
                else None
            ),
            source_scale=(
                str(source_scale) if source_scale is not None else None
            ),
            target_scale=(
                str(target_scale) if target_scale is not None else None
            ),
            status=CalculationStatus.READY,
        )
        self.calls += 1
        try:
            result = self.executor(calculation_plan)
        except Exception as exc:
            self._last_error = type(exc).__name__
            raise
        if not isinstance(result, CalculationResult):
            raise DeterministicCalculationCapabilityError(
                "calculator_must_return_CalculationResult"
            )
        self.last_result = result
        self.last_operand_evidence_ids = tuple(
            operand.evidence_chunk_id for operand in result.operands
        )
        self.last_calculation_id = (
            _calculation_id(operation, result)
            if result.status is CalculationStatus.EXECUTED
            else None
        )
        state.calculation_result = result.to_dict()
        state.calculation_result_id = self.last_calculation_id
        state._calculation_result_obj = result
        return result

    def trace_snapshot(self) -> dict[str, Any]:
        result = self.last_result
        return {
            "calculator_invoked": self.calls > 0,
            "calculator_call_count": self.calls,
            "calculation_result_id": self.last_calculation_id,
            "operand_evidence_ids": list(self.last_operand_evidence_ids),
            "calculation_status": result.status.value if result else None,
            "calculation_error_code": result.error_code if result else None,
        }


__all__ = [
    "DeterministicCalculationCapability",
    "DeterministicCalculationCapabilityError",
    "SUPPORTED_CALCULATION_OPERATIONS",
]