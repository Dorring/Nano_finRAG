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
class SupervisorPlan:
    """The only output the supervisor may send to the control plane."""

    intent: Intent
    required_slots: tuple[RequiredSlot, ...]
    operation: str | None
    next_action: Action

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
        if not isinstance(self.next_action, Action):
            raise ContractError("next_action must be an Action enum")

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent.value,
            "required_slots": [slot.to_dict() for slot in self.required_slots],
            "operation": self.operation,
            "next_action": self.next_action.value,
        }

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
        return cls(
            intent=intent,
            required_slots=tuple(RequiredSlot.from_dict(slot) for slot in slots),
            operation=payload["operation"],
            next_action=action,
        )
