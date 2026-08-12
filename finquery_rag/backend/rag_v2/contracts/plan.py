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


@dataclass(frozen=True)
class RequiredSlot:
    """A single evidence requirement emitted by the supervisor."""

    slot_id: str
    metric: str
    period: str
    role: str
    value_type: str
    unit: str | None = None

    def __post_init__(self) -> None:
        for name in ("slot_id", "metric", "role", "value_type"):
            _required_text(getattr(self, name), name)
        period = _required_text(self.period, "period")
        if not is_valid_period(period):
            raise ContractError(f"invalid period token: {self.period!r}")
        if self.unit is not None:
            _required_text(self.unit, "unit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "metric": self.metric,
            "period": self.period,
            "role": self.role,
            "value_type": self.value_type,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RequiredSlot":
        if not isinstance(payload, Mapping):
            raise ContractError("required slot must be an object")
        required = {"slot_id", "metric", "period", "role", "value_type", "unit"}
        missing = required - payload.keys()
        if missing:
            raise ContractError(f"required slot missing fields: {sorted(missing)}")
        return cls(
            slot_id=payload["slot_id"],
            metric=payload["metric"],
            period=payload["period"],
            role=payload["role"],
            value_type=payload["value_type"],
            unit=payload["unit"],
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
