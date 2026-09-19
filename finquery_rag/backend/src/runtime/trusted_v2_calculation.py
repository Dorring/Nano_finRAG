"""TV2-04 deterministic calculation capability.

This module is a thin adapter around the frozen finance calculation domain.
It never reads query text or conversation history to invent operands: only
Binder-admitted evidence IDs and the structured SupervisorPlan are accepted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from decimal import Decimal
from typing import Any, Callable

from rag_v2.adaptive import AdaptiveRAGStateV1
from rag_v2.contracts.plan import SupervisorPlan

from src.domain.calculation import (
    CalculationOperation,
    CalculationOperand,
    CalculationPlan,
    CalculationResult,
    CalculationStatus,
)
from src.finance.calculation_executor import execute_plan
from src.finance.calculation_registry import CALCULATION_REGISTRY, get_operation_entry
from src.finance.operand_ambiguity import CoordinateStatus, coordinate_status
from src.finance.primitive_tools import parse_financial_number
from src.finance.relational_directory import (
    RelationalOperandDirectory,
    RelationalOperandRef,
)


class DeterministicCalculationCapabilityError(RuntimeError):
    """Raised when the structured calculator boundary is invalid."""


#: The operations this runtime can execute, projected from the registry rather
#: than listed here.
#:
#: The registry is the executable-capability authority: an operation is
#: executable exactly when it has an entry describing how to execute it.  A
#: second list here -- even one derived from the enum rather than hand-typed --
#: would answer the same question a second time, and the two would disagree the
#: first time an operation gained a registry entry without one, or the reverse.
#: Deriving it keeps the answer in one place; the tuple is a compatibility
#: projection for callers that want a flat sequence of names.
SUPPORTED_CALCULATION_OPERATIONS = tuple(
    operation.value for operation in CALCULATION_REGISTRY
)


def build_relational_operand_directory(
    plan: SupervisorPlan,
) -> RelationalOperandDirectory:
    """The ``slot_id -> operand`` projection a relational result is stated in.

    A *controlled* projection of `required_slots`: only the coordinates a
    relation is named with, and deliberately no evidence id, chunk, retrieval
    score or source text.  Resolving a relation must not need evidence, because
    binding already happened and a second lookup would be a second bind.

    Built here rather than beside the type because this is the layer that
    already depends on the supervisor; the type stays dependency-free so the
    renderer can hold it.  The renderer and the validator both take the
    directory and never the plan, so neither can grow a `slot_id -> entity`
    lookup of its own that disagrees with the other's.
    """

    return RelationalOperandDirectory(
        refs=tuple(
            RelationalOperandRef(
                slot_id=slot.slot_id,
                entity=slot.entity,
                entity_id=slot.entity_id,
                metric=slot.metric,
                period=slot.period,
                role=slot.role,
            )
            for slot in plan.required_slots
        )
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
    # The operand as the record states it.  `ratio_value` would divide a
    # percentage by 100 here and the calculator would then divide it again.
    return result.points_value if result.ok else None


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


class DeterministicCalculationCapability:
    """Adapt the existing 9-operation executor to the TV2 capability port."""

    candidate_mode = True

    def __init__(
        self,
        executor: Callable[[CalculationPlan], CalculationResult] = execute_plan,
        *,
        fact_store: Any | None = None,
    ) -> None:
        self.executor = executor
        #: The authoritative fact store, when the caller has one.  Used only to
        #: ask whether a bound operand's coordinate identifies one value -- see
        #: `src/finance/operand_ambiguity.py`.  `None` disables the guard, which
        #: is what the deterministic unit tests want and what a caller without a
        #: store has to accept.
        self.fact_store = fact_store
        self.calls = 0
        self.last_result: CalculationResult | None = None
        self.last_operand_evidence_ids: tuple[str, ...] = ()
        self._last_error: str | None = None

    @property
    def last_calculation_id(self) -> str | None:
        """The last calculation's identity, derived from the result that owns it.

        H2A-2D-3B.  This was a stored field kept in step with `last_result` by
        hand -- assigned the id on success and reset to ``None`` on failure --
        which is a second truth with a lifecycle.  A paired assignment is only
        correct while both halves are remembered, and the failure mode is not a
        wrong value but a *stale* one: an id from a previous execution surviving
        into a run whose calculation never produced an id.

        Reading through to the result removes the lifecycle entirely.  A BLOCKED
        or FAILED result has no id to read, so the reset is not a step anyone can
        forget.
        """

        return None if self.last_result is None else self.last_result.calculation_id

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
        *,
        slot: Any | None = None,
        fact_store: Any | None = None,
    ) -> CalculationOperand | None:
        ids = bindings.get(slot_id, ())
        if not ids:
            return None
        # H2A-2C-2: a slot may carry several independent supports of one
        # canonical fact.  They are provenance, not operands -- the calculator
        # runs once, on the quantity, and must not be handed three rows that
        # say the same thing.  The first id is the canonical representative
        # chosen by the binding's admission order, and the validator has already
        # proved every id in the tuple states the same quantity, so which one is
        # read cannot change the arithmetic.  Computing once per support would
        # both triple the work and invite a sum or an average of one number.
        candidate = candidates.get(ids[0])
        if candidate is None:
            return None

        # P1.6-0: is this operand uniquely identifiable at all?
        #
        # `rank-005` bound The Coca-Cola Company / Interest rate contracts /
        # FY2025 to `-2` while the store holds four distinct values under that
        # coordinate and the gold is `16`.  The deterministic ranking then
        # faithfully rendered the wrong operands and released.  Nothing
        # downstream could have caught it: the executor did what it was given
        # and the release invariant saw an admissible relational result.
        #
        # Asked of the *store*, not the packet.  A packet-scoped guard would be
        # safe only on the days top-K happens to surface a competitor, which
        # makes safety a function of retrieval depth rather than of whether the
        # fact is identifiable.
        #
        # Returning `None` is the whole mechanism: `_build_operands` then
        # returns `()`, and the existing `INSUFFICIENT_OPERANDS` path blocks the
        # calculation.  Refusing is the honest answer -- this cannot know which
        # of the competing values is right, and separating them is P1.6-A.
        if slot is not None and fact_store is not None:
            status = coordinate_status(
                candidate,
                fact_store.facts_at_coordinate(
                    getattr(slot, "entity", None),
                    getattr(slot, "metric", None),
                    getattr(slot, "period", None),
                ),
            )
            if status is CoordinateStatus.CONFLICTING_VALUES:
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
            # The operand's semantic identity, and what a relational result
            # refers to.  The evidence id above is provenance -- one canonical
            # fact may be supported by several rows -- so an ordering keyed on
            # the evidence would move when a different support was bound.
            slot_id=slot_id,
        )

    @classmethod
    def _build_operands(
        cls,
        plan: SupervisorPlan,
        state: AdaptiveRAGStateV1,
        *,
        fact_store: Any | None = None,
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

        slots_by_id = {slot.slot_id: slot for slot in plan.required_slots}
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
                slot_id,
                role,
                bindings,
                candidates,
                operation,
                slot=slots_by_id.get(slot_id),
                fact_store=fact_store,
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
                    slot=slot,
                    fact_store=fact_store,
                )
                if operand is None:
                    return ()
                operands.append(operand)
        return tuple(operands)

    def calculate(self, state: AdaptiveRAGStateV1) -> CalculationResult:
        """Execute only on Binder-admitted complete structured operands."""

        plan = self._plan(state)
        # Execution eligibility is a property of the *operation*, decided by the
        # registry, and deliberately not of the plan's intent.
        #
        # This gate used to read ``plan.intent is Intent.CALCULATION``.  That
        # made an entire stratum unreachable: cross-entity comparison and
        # ranking are planned as ``MULTI_EVIDENCE``, so however clearly such a
        # plan named an operation, the calculator refused it -- and their
        # answers stayed prose that no validator had a structured result to
        # check.
        #
        # ``Intent`` still decides *routing*: whether the coordinator offers a
        # plan to the calculator at all.  That is the question intent answers.
        # "Can this be executed deterministically" is a different one, and the
        # registry is where it is answered.
        operation = _operation(plan.operation)
        entry = get_operation_entry(operation)
        if entry is None:
            raise DeterministicCalculationCapabilityError(
                f"unsupported_calculation_operation:{operation.value}"
            )
        operands = self._build_operands(plan, state, fact_store=self.fact_store)
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
        # The identity belongs to the result, and the result only hands one
        # out when it is admissible.  This used to be computed here from a
        # status check made at the call site -- the same fact, but living
        # beside the result rather than in it, so any other producer could
        # pair a BLOCKED result with a non-None id.
        state.calculation_result = result.to_dict()
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