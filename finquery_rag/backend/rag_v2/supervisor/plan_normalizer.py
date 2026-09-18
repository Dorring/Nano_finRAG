"""Conservative, deterministic normalization for contradictory supervisor plans.

The supervisor is allowed one model call, but a structured response can still
contain an internally contradictory combination such as ``MULTI_EVIDENCE``
with ``operation=growth_rate``.  This module repairs only that narrow class of
contradiction when the user query supplies explicit, unambiguous evidence for
the calculation.  It never invents a metric, period, entity, or operation and
the result is always passed through the normal plan validator afterwards.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from rag_v2.contracts.plan import Intent, RequiredSlot, SupervisorPlan

from .semantic_alignment import (
    canonical_entity_id,
    canonical_operation_id,
    canonical_period_id,
    extract_query_semantic_frame,
)


_ROLE_BY_OPERATION = {
    "growth_rate": ("current_period", "base_period"),
    "difference": ("minuend", "subtrahend"),
}
_SUPPORTED_NORMALIZATIONS = frozenset(_ROLE_BY_OPERATION)
_PERIOD_RE = re.compile(r"^FY(?P<year>\d{4})(?:-Q(?P<quarter>[1-4]))?$")


@dataclass(frozen=True)
class PlanNormalization:
    """Auditable description of a deterministic plan repair."""

    strategy: str
    changes: tuple[str, ...]
    explicit_operation: str
    explicit_periods: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "changes": list(self.changes),
            "explicit_operation": self.explicit_operation,
            "explicit_periods": list(self.explicit_periods),
        }


def _period_sort_key(period_id: str) -> tuple[int, int]:
    match = _PERIOD_RE.fullmatch(period_id)
    if match is None:
        raise ValueError(f"unsupported canonical period: {period_id}")
    return int(match.group("year")), int(match.group("quarter") or 0)


def _ordered_periods(periods: tuple[str, ...]) -> tuple[str, str] | None:
    """Return oldest/newest only for exactly two distinct annual periods."""

    canonical = tuple(dict.fromkeys(canonical_period_id(item) for item in periods))
    if len(canonical) != 2 or any(item is None for item in canonical):
        return None
    try:
        ordered = sorted((str(item) for item in canonical), key=_period_sort_key)
    except ValueError:
        return None
    if ordered[0] == ordered[1]:
        return None
    return ordered[0], ordered[1]


def _remap_roles(
    plan: SupervisorPlan,
    operation: str,
    *,
    oldest_period: str,
    newest_period: str,
) -> tuple[tuple[RequiredSlot, ...], tuple[str, ...]] | None:
    """Map two period slots to the operation's chronological roles.

    Role repair is intentionally limited to two slots whose periods are the
    two explicit query periods.  If a provider emits extra/implicit slots or
    opaque periods, the normalizer declines to guess.
    """

    desired_roles = _ROLE_BY_OPERATION[operation]
    if len(plan.required_slots) != 2:
        return None
    slot_periods: list[tuple[RequiredSlot, str]] = []
    for slot in plan.required_slots:
        period_id = canonical_period_id(slot.period)
        if period_id is None:
            return None
        slot_periods.append((slot, period_id))
    if {period for _, period in slot_periods} != {oldest_period, newest_period}:
        return None
    remapped: list[RequiredSlot] = []
    changes: list[str] = []
    for slot, period_id in slot_periods:
        desired = desired_roles[0] if period_id == newest_period else desired_roles[1]
        if slot.role.strip().casefold() != desired:
            changes.append(f"slot_role:{slot.slot_id}:{slot.role}->{desired}")
            remapped.append(replace(slot, role=desired))
        else:
            remapped.append(slot)
    return tuple(remapped), tuple(changes)


def normalize_supervisor_plan(
    question: str,
    plan: SupervisorPlan,
) -> tuple[SupervisorPlan, PlanNormalization | None]:
    """Repair one narrow, explicitly evidenced plan contradiction.

    The function returns the original plan when the query does not contain a
    single recognized operation and exactly two explicit periods.  It never
    turns a generic comparison into a calculation and never repairs metric or
    entity semantics; those remain the responsibility of semantic alignment.
    """

    if not isinstance(plan, SupervisorPlan):
        raise TypeError("plan must be SupervisorPlan")
    frame = extract_query_semantic_frame(question)
    if len(frame.operation_ids) != 1 or not frame.metric_ids:
        return plan, None
    operation = frame.operation_ids[0]
    if operation not in _SUPPORTED_NORMALIZATIONS:
        return plan, None
    ordered_periods = _ordered_periods(frame.period_ids)
    if ordered_periods is None:
        return plan, None
    oldest_period, newest_period = ordered_periods
    proposed_operation = canonical_operation_id(plan.operation)
    if proposed_operation != operation:
        # Do not infer or overwrite an operation chosen by the model.  A
        # contradiction remains fail-closed via the normal semantic/plan
        # validators rather than being silently changed here.
        return plan, None

    changes: list[str] = []
    normalized_slots = plan.required_slots
    remapped = _remap_roles(
        plan,
        operation,
        oldest_period=oldest_period,
        newest_period=newest_period,
    )
    if remapped is not None:
        normalized_slots, role_changes = remapped
        changes.extend(role_changes)

    normalized_intent = plan.intent
    if plan.intent is not Intent.CALCULATION:
        normalized_intent = Intent.CALCULATION
        changes.append(f"intent:{plan.intent.value}->CALCULATION")

    normalized = replace(
        plan,
        intent=normalized_intent,
        required_slots=normalized_slots,
    )
    if not changes:
        return plan, None
    return normalized, PlanNormalization(
        strategy="explicit_query_calculation_alignment",
        changes=tuple(changes),
        explicit_operation=operation,
        explicit_periods=(oldest_period, newest_period),
    )


def derive_slot_identities(plan: SupervisorPlan) -> SupervisorPlan:
    """Fill each slot's derived coordinates from the mentions it carries.

    The contract keeps ``entity`` as an open-world mention and ``entity_id`` as
    an identity the Harness *derived* from it, one way: mention, then
    canonicalisation, then id.  This is the derivation step.  It is not a
    repair and it is not conditional -- every plan that reaches the runtime goes
    through it, because a plan whose ids were derived only sometimes would make
    the binder's behaviour depend on which path the plan arrived by.

    The mention is authoritative.  Deriving from it *overwrites* any id that
    came with the plan, so a plan cannot assert an identity its own mention
    contradicts; and it is deterministic and vocabulary-only, with no model and
    no fuzzy matching anywhere in it.

    A mention the vocabulary cannot name keeps ``entity_id=None``, which means
    *constrained and unnamed* and never *unconstrained*.  The mention is
    therefore kept exactly as written -- dropping it because its id could not be
    derived would turn "we cannot name this company" into "any company", and
    that failure would look like a wider recall rather than a lost constraint.
    """

    if not isinstance(plan, SupervisorPlan):
        raise TypeError("plan must be SupervisorPlan")

    slots: list[RequiredSlot] = []
    changed = False
    for slot in plan.required_slots:
        if not slot.entity:
            slots.append(slot)
            continue
        entity_id = canonical_entity_id(slot.entity)
        if slot.entity_id != entity_id:
            slots.append(replace(slot, entity_id=entity_id))
            changed = True
        else:
            slots.append(slot)

    if not changed:
        return plan
    return replace(plan, required_slots=tuple(slots))


__all__ = [
    "PlanNormalization",
    "derive_slot_identities",
    "normalize_supervisor_plan",
]
