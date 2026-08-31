"""Deterministic claim-level provenance for Trusted Financial Runtime V2.

The helper consumes only Supervisor slots and Binder-admitted runtime state.
It intentionally has no answer-text input: a generated number or citation can
never create evidence or calculation lineage on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag_v2.contracts.plan import SupervisorPlan

from .runtime_contract import ClaimProvenance, ReleaseStatus


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _identity(item: Mapping[str, Any]) -> str:
    for key in ("evidence_id", "fact_id", "candidate_id"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _citation(item: Mapping[str, Any]) -> str:
    value = item.get("citation_id")
    return str(value).strip() if value is not None else ""


def _state_bindings(state: Any) -> dict[str, tuple[str, ...]]:
    raw = getattr(state, "bound_slot_bindings", {})
    if not isinstance(raw, Mapping):
        return {}
    bindings: dict[str, tuple[str, ...]] = {}
    for slot_id, evidence_ids in raw.items():
        if isinstance(evidence_ids, (str, bytes)) or not isinstance(
            evidence_ids,
            Iterable,
        ):
            continue
        bindings[str(slot_id)] = _stable_unique(evidence_ids)
    return bindings


def _state_packets(state: Any) -> dict[str, Mapping[str, Any]]:
    packets = getattr(state, "evidence_packets", ())
    if isinstance(packets, (str, bytes)) or not isinstance(packets, Iterable):
        return {}
    by_id: dict[str, Mapping[str, Any]] = {}
    for packet in packets:
        if isinstance(packet, Mapping):
            identity = _identity(packet)
            if identity and identity not in by_id:
                by_id[identity] = packet
    return by_id


def build_claim_provenance(
    *,
    plan: SupervisorPlan | None,
    state: Any | None,
    evidence_ids: Iterable[str] = (),
    citation_ids: Iterable[str] = (),
    calculation_ids: Iterable[str] = (),
    release_status: ReleaseStatus = ReleaseStatus.NOT_RELEASED,
    validator_status: str | None = None,
) -> tuple[ClaimProvenance, ...]:
    """Build stable claim → slot → admitted-evidence lineage.

    ``evidence_ids`` is intersected with ``state.bound_evidence_ids`` when
    that admission set is available.  Retrieval candidates therefore cannot
    leak into final provenance merely because they happened to be present in
    the runtime state.  Calculation lineage is represented by one claim per
    structured calculation ID and never inferred from the answer text.
    """

    if plan is None or state is None:
        return ()

    requested_ids = _stable_unique(evidence_ids)
    admitted_ids = _stable_unique(getattr(state, "bound_evidence_ids", ()))
    if admitted_ids:
        admitted_set = set(admitted_ids)
        bound_ids = tuple(item for item in requested_ids if item in admitted_set)
    else:
        bound_ids = requested_ids
    if not bound_ids:
        return ()

    bindings = _state_bindings(state)
    packets = _state_packets(state)
    bound_set = set(bound_ids)
    claims: list[ClaimProvenance] = []
    all_bound_ids: list[str] = []
    all_citation_ids: list[str] = []

    for slot in plan.required_slots:
        slot_ids = tuple(
            item for item in bindings.get(slot.slot_id, ()) if item in bound_set
        )
        if not slot_ids:
            continue
        slot_citations = _stable_unique(
            _citation(packets[item])
            for item in slot_ids
            if item in packets and _citation(packets[item])
        )
        claims.append(
            ClaimProvenance(
                claim_id=f"slot:{slot.slot_id}",
                required_slot_ids=(slot.slot_id,),
                bound_evidence_ids=slot_ids,
                citation_ids=slot_citations,
                release_status=release_status,
                validator_status=validator_status,
            )
        )
        all_bound_ids.extend(slot_ids)
        all_citation_ids.extend(slot_citations)

    calculation_id_list = _stable_unique(calculation_ids)
    if calculation_id_list:
        calculation_evidence_ids = _stable_unique(all_bound_ids) or bound_ids
        calculation_citations = _stable_unique(all_citation_ids)
        required_slot_ids = tuple(
            slot.slot_id
            for slot in plan.required_slots
            if slot.slot_id in bindings
            and any(item in bound_set for item in bindings[slot.slot_id])
        )
        for calculation_id in calculation_id_list:
            claims.append(
                ClaimProvenance(
                    claim_id=f"calculation:{calculation_id}",
                    required_slot_ids=required_slot_ids,
                    bound_evidence_ids=calculation_evidence_ids,
                    citation_ids=calculation_citations,
                    calculation_ids=(calculation_id,),
                    release_status=release_status,
                    validator_status=validator_status,
                )
            )

    if not claims:
        # Keep a deterministic fallback for older injected evaluators that
        # expose admitted IDs but not a slot map.  It still contains only
        # structured IDs and is explicitly answer-scoped, never text-derived.
        claims.append(
            ClaimProvenance(
                claim_id="answer",
                bound_evidence_ids=bound_ids,
                citation_ids=_stable_unique(citation_ids),
                calculation_ids=calculation_id_list,
                release_status=release_status,
                validator_status=validator_status,
            )
        )
    return tuple(claims)


__all__ = ["build_claim_provenance"]
