from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

from .errors import ContractError


class Intent(str, Enum):
    """Supervisor intent; no-answer is intentionally not a query intent."""

    DIRECT_FACT = "DIRECT_FACT"
    MULTI_EVIDENCE = "MULTI_EVIDENCE"
    CALCULATION = "CALCULATION"


class Action(str, Enum):
    """Actions that a validated plan may propose to the state machine."""

    RETRIEVE = "RETRIEVE"
    BIND = "BIND"
    REPAIR_RETRIEVAL = "REPAIR_RETRIEVAL"
    CALCULATE = "CALCULATE"
    GENERATE = "GENERATE"
    REPAIR_GENERATION = "REPAIR_GENERATION"
    ABSTAIN = "ABSTAIN"
    STOP = "STOP"


_OPERATION_VALUES = frozenset(
    {
        "difference",
        "growth_rate",
        "percentage_share",
        "sum",
        "average",
        "gross_margin",
        "net_margin",
        "debt_ratio",
        "scale_conversion",
        # Relational operations: their answer is a relation between operands
        # rather than a quantity.  Cross-entity comparison and ranking used to
        # be planned with no operation at all, which left their answers as prose
        # that no validator had a structured result to check against.
        #
        # This set is necessarily a literal: it is the contract layer, and
        # `rag_v2` must never import `src`, where the executable registry lives.
        # The two are kept in agreement by the tests that assert every
        # registered operation is expressible here and every name here is
        # executable.
        "comparison",
        "ranking",
    }
)
_PERIOD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ./_:-]{0,63}$")


def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be a non-empty string")
    return value.strip()


def is_valid_period(value: str) -> bool:
    """Return whether a period is a bounded canonical text token.

    V2-00 does not add period aliases.  It only rejects empty/control-heavy
    values; semantic period extraction remains a later supervisor concern.
    """

    return isinstance(value, str) and bool(_PERIOD_RE.fullmatch(value.strip()))


def _optional_text(value: str | None, field_name: str) -> str | None:
    """Return a trimmed optional text, rejecting an empty-but-present one.

    ``None`` and ``""`` would otherwise mean different things in a record that
    only ever tests for truthiness, so the empty string is refused where it is
    written instead of being silently equivalent to absence everywhere it is
    read.
    """

    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be None or a non-empty string")
    return value.strip()


#: The fields a slot payload must carry.  Unchanged since V2-00.
_REQUIRED_SLOT_FIELDS = ("slot_id", "metric", "period", "role", "value_type", "unit")

#: Additive coordinates.  Optional on the wire so a V2-00 payload -- and a
#: supervisor that has not learned to emit them -- still parses unchanged.
_OPTIONAL_SLOT_FIELDS = ("entity", "entity_id", "scope", "scope_id")


def slot_key_error(slot: Mapping[str, Any]) -> str | None:
    """Why a slot payload is not a slot, or ``None`` when it is one.

    Shared by the supervisor providers rather than written out in each.  They
    held the same six-name literal twice, which is exactly the shape of change
    that gets applied to one of them: adding a field to the contract would have
    left one provider accepting the new slot and the other rejecting the model's
    own output, with neither failing until a run.

    The six V2-00 fields must be present; the coordinates are optional so a
    supervisor that has not learned to emit them still parses; nothing else is
    allowed, because an unrecognised key is a misunderstanding rather than an
    extension.
    """

    if not isinstance(slot, Mapping):
        return "slot must be an object"
    keys = set(slot)
    missing = set(_REQUIRED_SLOT_FIELDS) - keys
    if missing:
        return f"slot missing required fields: {sorted(missing)}"
    extra = keys - set(_REQUIRED_SLOT_FIELDS) - set(_OPTIONAL_SLOT_FIELDS)
    if extra:
        return f"slot carries unknown fields: {sorted(extra)}"
    return None


@dataclass(frozen=True)
class RequiredSlot:
    """A single evidence requirement emitted by the supervisor.

    ``metric`` and ``period`` say *what quantity* is wanted.  ``entity`` narrows
    that to *whose*, and ``scope`` to *which part of the filing* -- without them
    "Apple's FY2025 revenue" and "Visa's FY2025 revenue" are one requirement
    written twice, which is what made twenty comparison fixtures unanswerable
    and unscoreable.

    ``entity`` preserves the open-world semantic mention: whatever the plan
    actually said, with no dependency on any vocabulary.  ``entity_id`` is an
    optional normalised identity the Harness *derived* from it, and the
    relationship is one-way -- mention, then canonicalisation, then id.  There
    is no second authority and no back-fill.

    ``entity_id is None`` therefore means *the mention is constrained and the
    ontology cannot name it*.  It never means *there is no entity constraint*.
    That distinction is the whole reason the two fields are separate, and it is
    what keeps a company the vocabulary has never heard of from being silently
    dropped.
    """

    slot_id: str
    metric: str
    period: str
    role: str
    value_type: str
    unit: str | None = None
    entity: str | None = None
    entity_id: str | None = None
    scope: str | None = None
    scope_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("slot_id", "metric", "role", "value_type"):
            _required_text(getattr(self, name), name)
        period = _required_text(self.period, "period")
        if not is_valid_period(period):
            raise ContractError(f"invalid period token: {self.period!r}")
        for name in _OPTIONAL_SLOT_FIELDS:
            _optional_text(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "metric": self.metric,
            "period": self.period,
            "role": self.role,
            "value_type": self.value_type,
            "unit": self.unit,
            "entity": self.entity,
            "entity_id": self.entity_id,
            "scope": self.scope,
            "scope_id": self.scope_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiredSlot":
        if not isinstance(payload, Mapping):
            raise ContractError("required slot must be an object")
        missing = set(_REQUIRED_SLOT_FIELDS) - payload.keys()
        if missing:
            raise ContractError(f"required slot missing fields: {sorted(missing)}")
        return cls(
            **{name: payload[name] for name in _REQUIRED_SLOT_FIELDS},
            # Read with ``get``: a payload written before these fields existed
            # carries none of them and must keep parsing.
            **{name: payload.get(name) for name in _OPTIONAL_SLOT_FIELDS},
        )


@dataclass(frozen=True)
class OperandRef:
    """A reference to an input operand in an ExecutionDAG step.

    ref_type must be either 'slot' (referencing a RequiredSlot in the plan)
    or 'step' (referencing an earlier OperationStep in the DAG).
    """

    ref_type: str
    ref_id: str
    role: str | None = None

    def __post_init__(self) -> None:
        normalized_type = _required_text(self.ref_type, "ref_type").lower()
        if normalized_type not in {"slot", "step"}:
            raise ContractError(f"ref_type must be 'slot' or 'step', got {self.ref_type!r}")
        object.__setattr__(self, "ref_type", normalized_type)
        object.__setattr__(self, "ref_id", _required_text(self.ref_id, "ref_id"))
        if self.role is not None:
            object.__setattr__(self, "role", _optional_text(self.role, "role"))

    @property
    def is_slot(self) -> bool:
        return self.ref_type == "slot"

    @property
    def is_step(self) -> bool:
        return self.ref_type == "step"

    @property
    def slot_id(self) -> str | None:
        return self.ref_id if self.ref_type == "slot" else None

    @property
    def step_id(self) -> str | None:
        return self.ref_id if self.ref_type == "step" else None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "ref_type": self.ref_type,
            "ref_id": self.ref_id,
        }
        if self.role is not None:
            data["role"] = self.role
        return data

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperandRef":
        if not isinstance(payload, Mapping):
            raise ContractError("OperandRef must be an object")
        if "slot_id" in payload and payload.get("slot_id"):
            return cls(ref_type="slot", ref_id=str(payload["slot_id"]), role=payload.get("role"))
        if "step_id" in payload and payload.get("step_id"):
            return cls(ref_type="step", ref_id=str(payload["step_id"]), role=payload.get("role"))
        ref_type = payload.get("ref_type")
        ref_id = payload.get("ref_id")
        if not ref_type or not ref_id:
            raise ContractError("OperandRef missing ref_type or ref_id")
        return cls(ref_type=str(ref_type), ref_id=str(ref_id), role=payload.get("role"))


@dataclass(frozen=True)
class OperationStep:
    """An atomic calculation step in an ExecutionDAG."""

    step_id: str
    operation: str
    operands: tuple[OperandRef, ...]
    output_unit: str | None = None
    target_metric: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_id", _required_text(self.step_id, "step_id"))
        op = _required_text(self.operation, "operation")
        if op not in _OPERATION_VALUES:
            raise ContractError(f"unsupported operation in step: {op}")
        object.__setattr__(self, "operation", op)
        if not self.operands:
            raise ContractError("OperationStep operands must not be empty")
        for operand in self.operands:
            if not isinstance(operand, OperandRef):
                raise ContractError("OperationStep operand must be an OperandRef")
        if self.output_unit is not None:
            object.__setattr__(self, "output_unit", _optional_text(self.output_unit, "output_unit"))
        if self.target_metric is not None:
            object.__setattr__(self, "target_metric", _optional_text(self.target_metric, "target_metric"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "operation": self.operation,
            "operands": [op.to_dict() for op in self.operands],
            "output_unit": self.output_unit,
            "target_metric": self.target_metric,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OperationStep":
        if not isinstance(payload, Mapping):
            raise ContractError("OperationStep must be an object")
        for key in ("step_id", "operation", "operands"):
            if key not in payload:
                raise ContractError(f"OperationStep missing field: {key}")
        raw_ops = payload["operands"]
        if not isinstance(raw_ops, (list, tuple)):
            raise ContractError("OperationStep operands must be an array")
        return cls(
            step_id=str(payload["step_id"]),
            operation=str(payload["operation"]),
            operands=tuple(OperandRef.from_dict(item) for item in raw_ops),
            output_unit=payload.get("output_unit"),
            target_metric=payload.get("target_metric"),
        )


@dataclass(frozen=True)
class ExecutionDAG:
    """An immutable directed acyclic graph of deterministic financial steps."""

    steps: tuple[OperationStep, ...]
    output_step_id: str
    target_metric: str | None = None
    definition_version: str | None = None

    def __post_init__(self) -> None:
        if not self.steps:
            raise ContractError("ExecutionDAG steps must not be empty")
        object.__setattr__(self, "output_step_id", _required_text(self.output_step_id, "output_step_id"))
        for step in self.steps:
            if not isinstance(step, OperationStep):
                raise ContractError("ExecutionDAG step must be an OperationStep")
        step_ids = [s.step_id for s in self.steps]
        if len(set(step_ids)) != len(step_ids):
            raise ContractError("ExecutionDAG steps must have unique step_ids")
        if self.output_step_id not in step_ids:
            raise ContractError(f"output_step_id {self.output_step_id!r} not found in steps")
        if self.target_metric is not None:
            object.__setattr__(self, "target_metric", _optional_text(self.target_metric, "target_metric"))
        if self.definition_version is not None:
            object.__setattr__(self, "definition_version", _optional_text(self.definition_version, "definition_version"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "output_step_id": self.output_step_id,
            "target_metric": self.target_metric,
            "definition_version": self.definition_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionDAG":
        if not isinstance(payload, Mapping):
            raise ContractError("ExecutionDAG must be an object")
        for key in ("steps", "output_step_id"):
            if key not in payload:
                raise ContractError(f"ExecutionDAG missing field: {key}")
        raw_steps = payload["steps"]
        if not isinstance(raw_steps, (list, tuple)):
            raise ContractError("ExecutionDAG steps must be an array")
        return cls(
            steps=tuple(OperationStep.from_dict(item) for item in raw_steps),
            output_step_id=str(payload["output_step_id"]),
            target_metric=payload.get("target_metric"),
            definition_version=payload.get("definition_version"),
        )


@dataclass(frozen=True)
class SupervisorPlan:
    """The only output the supervisor may send to the control plane."""

    intent: Intent
    required_slots: tuple[RequiredSlot, ...]
    operation: str | None
    next_action: Action
    execution_dag: ExecutionDAG | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.intent, Intent):
            raise ContractError("intent must be an Intent enum")
        if not self.required_slots:
            raise ContractError("required_slots must not be empty")
        if len({slot.slot_id for slot in self.required_slots}) != len(self.required_slots):
            raise ContractError("required_slots must have unique slot_id values")
        if self.operation is not None:
            operation = _required_text(self.operation, "operation")
            if operation not in _OPERATION_VALUES:
                raise ContractError(f"unsupported operation: {operation}")
        if self.execution_dag is not None and not isinstance(self.execution_dag, ExecutionDAG):
            raise ContractError("execution_dag must be an ExecutionDAG or None")
        if not isinstance(self.next_action, Action):
            raise ContractError("next_action must be an Action enum")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "intent": self.intent.value,
            "required_slots": [slot.to_dict() for slot in self.required_slots],
            "operation": self.operation,
            "next_action": self.next_action.value,
        }
        if self.execution_dag is not None:
            payload["execution_dag"] = self.execution_dag.to_dict()
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SupervisorPlan":
        if not isinstance(payload, Mapping):
            raise ContractError("supervisor plan must be an object")
        required = {"intent", "required_slots", "operation", "next_action"}
        missing = required - payload.keys()
        if missing:
            raise ContractError(f"supervisor plan missing fields: {sorted(missing)}")
        try:
            intent = Intent(payload["intent"])
            action = Action(payload["next_action"])
        except (TypeError, ValueError) as exc:
            raise ContractError("invalid intent or next_action enum") from exc
        slots = payload["required_slots"]
        if not isinstance(slots, (list, tuple)):
            raise ContractError("required_slots must be an array")
        dag_payload = payload.get("execution_dag")
        dag = ExecutionDAG.from_dict(dag_payload) if dag_payload is not None else None
        return cls(
            intent=intent,
            required_slots=tuple(RequiredSlot.from_dict(slot) for slot in slots),
            operation=payload["operation"],
            next_action=action,
            execution_dag=dag,
        )
