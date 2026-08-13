"""Runtime-safe, Gold-independent selective admission for NF-V2-03.

The V2 contract proves uniqueness against the complete current fact packet,
not against a Top-K shortlist.  It admits a Binder selection only when every
other packet fact has an explicit deterministic material conflict.  Candidate
source serialization is used only as already-linked lexical context; it does
not introduce query-conditioned or Gold-derived semantics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import SupervisorPlan

from .binding_validator import BindingValidationResult, validate_binding


CONTRACT_NAME = "SelectiveBindingAdmissionV2"


@dataclass(frozen=True)
class RuntimeAdmissionEvidence:
    """Deterministic proof/failure record for one candidate selection."""

    selected_fact_id: str
    selected_admissible: bool
    competitor_conflicts: Mapping[str, tuple[str, ...]]
    plausible_competitors: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def uniquely_admissible(self) -> bool:
        return self.selected_admissible and not self.plausible_competitors and not self.reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_fact_id": self.selected_fact_id,
            "selected_admissible": self.selected_admissible,
            "competitor_conflicts": {key: list(value) for key, value in self.competitor_conflicts.items()},
            "plausible_competitors": list(self.plausible_competitors),
            "reasons": list(self.reasons),
            "uniquely_admissible": self.uniquely_admissible,
        }


@dataclass(frozen=True)
class SelectiveAdmissionV2Result:
    contract: str
    binding: EvidenceBinding
    validation: BindingValidationResult
    released: bool
    slot_evidence: Mapping[str, RuntimeAdmissionEvidence]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "released": self.released,
            "binding": self.binding.to_dict(),
            "validation": self.validation.to_dict(),
            "slot_evidence": {key: value.to_dict() for key, value in self.slot_evidence.items()},
            "reasons": list(self.reasons),
        }


def _tokens(value: Any) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", str(value or "").casefold()) if token}


def _context_tokens(
    fact: Mapping[str, Any],
    source_meta: Mapping[str, Any] | None = None,
) -> set[str]:
    values: list[Any] = [
        fact.get("raw_metric"),
        fact.get("normalized_metric"),
        fact.get("row_label"),
        fact.get("row_path"),
        fact.get("row_hierarchy"),
        fact.get("column_label"),
        fact.get("column_header"),
        fact.get("column_header_path"),
        fact.get("multi_level_column_headers"),
    ]
    # Candidate serialization is frozen, source-derived context.  Include it
    # only as a lexical context signal; it is never a question- or Gold-
    # conditioned semantic label.  This matters for facts whose atomic row
    # metric is a terse child/header label (for example, a table row labelled
    # "Worldwide") while the surrounding candidate serialization preserves
    # the product/statement identity.
    if source_meta:
        values.extend(
            [
                source_meta.get("source_text"),
                source_meta.get("table_title"),
                source_meta.get("statement_id"),
                source_meta.get("row_label"),
                source_meta.get("column_header"),
            ]
        )
    tokens: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            for item in value:
                tokens |= _tokens(item)
        else:
            tokens |= _tokens(value)
    return tokens


def _exact_period(value: Any) -> str | None:
    match = re.search(r"\bfy\s*(\d{4})\b", str(value or "").casefold())
    return f"fy{match.group(1)}" if match else None


def _known_conflict_reasons(
    slot: Any,
    fact: Mapping[str, Any],
    *,
    slot_metric_tokens: set[str],
    source_meta: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    reasons: list[str] = []
    slot_period = _exact_period(getattr(slot, "period", None))
    fact_period = _exact_period(fact.get("normalized_period") or fact.get("raw_period"))
    if slot_period and fact_period and slot_period != fact_period:
        reasons.append("explicit_period_conflict")
    slot_unit = str(getattr(slot, "unit", None) or "").casefold().strip()
    fact_unit = str(fact.get("unit") or "").casefold().strip()
    if slot_unit and fact_unit and slot_unit != fact_unit:
        reasons.append("explicit_unit_conflict")
    # Currency is material only when both source facts make it explicit.  A
    # missing currency is uncertainty, not a conflict.
    slot_currency = str(getattr(slot, "currency", None) or "").casefold().strip()
    fact_currency = str(fact.get("currency") or "").casefold().strip()
    if slot_currency and fact_currency and slot_currency != fact_currency:
        reasons.append("explicit_currency_conflict")
    if slot_metric_tokens and not slot_metric_tokens.intersection(_context_tokens(fact, source_meta)):
        reasons.append("explicit_metric_context_conflict")
    return tuple(reasons)


def _selected_compatibility(
    slot: Any,
    selected: Mapping[str, Any],
    *,
    source_map: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if selected.get("provenance_complete") is not True:
        reasons.append("fact_not_provenance_complete")
    candidate_id = str(selected.get("candidate_id") or "")
    if not candidate_id or source_map.get(candidate_id) is None:
        reasons.append("source_relation_failure")
    slot_period = _exact_period(getattr(slot, "period", None))
    fact_period = _exact_period(selected.get("normalized_period") or selected.get("raw_period"))
    if slot_period and fact_period != slot_period:
        reasons.append("selected_period_not_exact")
    slot_unit = str(getattr(slot, "unit", None) or "").casefold().strip()
    fact_unit = str(selected.get("unit") or "").casefold().strip()
    if slot_unit and fact_unit != slot_unit:
        reasons.append("selected_unit_not_compatible")
    metric_tokens = _tokens(getattr(slot, "metric", None))
    source_meta = source_map.get(candidate_id)
    if metric_tokens and not metric_tokens.intersection(_context_tokens(selected, source_meta)):
        reasons.append("selected_metric_context_not_compatible")
    return not reasons, reasons


def evaluate_slot(
    slot: Any,
    selected_fact_id: str,
    facts: Iterable[Mapping[str, Any]],
    *,
    source_map: Mapping[str, Mapping[str, Any]],
) -> RuntimeAdmissionEvidence:
    fact_map = {str(fact.get("fact_id")): fact for fact in facts if fact.get("fact_id")}
    selected = fact_map.get(str(selected_fact_id))
    if selected is None:
        return RuntimeAdmissionEvidence(str(selected_fact_id), False, {}, (), ("fact_not_in_query_packet",))
    metric_tokens = _tokens(getattr(slot, "metric", None))
    selected_ok, selected_reasons = _selected_compatibility(slot, selected, source_map=source_map)
    conflicts: dict[str, tuple[str, ...]] = {}
    plausible: list[str] = []
    for fact_id, fact in fact_map.items():
        if fact_id == str(selected_fact_id):
            continue
        source_meta = source_map.get(str(fact.get("candidate_id") or ""))
        reasons = _known_conflict_reasons(slot, fact, slot_metric_tokens=metric_tokens, source_meta=source_meta)
        if reasons:
            conflicts[fact_id] = reasons
        else:
            plausible.append(fact_id)
    return RuntimeAdmissionEvidence(str(selected_fact_id), selected_ok, conflicts, tuple(plausible), tuple(selected_reasons))


def _binding_from_input(binding: EvidenceBinding | Mapping[str, Any]) -> EvidenceBinding:
    if isinstance(binding, EvidenceBinding):
        return binding
    return EvidenceBinding(
        status=str(binding.get("status")),
        slot_bindings={key: tuple(value) for key, value in (binding.get("slot_bindings") or {}).items()},
        missing_slots=tuple(binding.get("missing_slots") or ()),
        ambiguous_slots=tuple(binding.get("ambiguous_slots") or ()),
        invalid_reasons=tuple(binding.get("invalid_reasons") or ()),
    )


def admit_binding_v2(
    binding: EvidenceBinding | Mapping[str, Any],
    plan: SupervisorPlan,
    facts: Iterable[Mapping[str, Any]],
    *,
    source_map: Mapping[str, Mapping[str, Any]],
) -> SelectiveAdmissionV2Result:
    """Apply A1-A8 and return a fail-closed EvidenceBinding."""

    normalized_binding = _binding_from_input(binding)
    fact_list = tuple(facts)
    validation = validate_binding(normalized_binding, plan, fact_list)
    expected_slots = tuple(slot.slot_id for slot in plan.required_slots)
    slot_evidence: dict[str, RuntimeAdmissionEvidence] = {}
    reasons: list[str] = []
    if normalized_binding.status == BindingStatus.BOUND.value:
        for slot in plan.required_slots:
            selected = normalized_binding.slot_bindings.get(slot.slot_id, ())
            if len(selected) != 1:
                reasons.append(f"{slot.slot_id}:unique_selection_required")
                continue
            evidence = evaluate_slot(slot, selected[0], fact_list, source_map=source_map)
            slot_evidence[slot.slot_id] = evidence
            if not evidence.selected_admissible:
                reasons.extend(f"{slot.slot_id}:{reason}" for reason in evidence.reasons)
            if evidence.plausible_competitors:
                reasons.append(f"{slot.slot_id}:plausible_competitor")
        if not validation.passed:
            reasons.append("binding_validator_failed")
        if not reasons and all(slot_id in slot_evidence and slot_evidence[slot_id].uniquely_admissible for slot_id in expected_slots):
            return SelectiveAdmissionV2Result(CONTRACT_NAME, normalized_binding, validation, True, slot_evidence, ())
        ambiguous = tuple(slot_id for slot_id, evidence in slot_evidence.items() if evidence.plausible_competitors)
        if ambiguous:
            safe_binding = EvidenceBinding(status=BindingStatus.AMBIGUOUS.value, slot_bindings={}, ambiguous_slots=ambiguous)
        else:
            safe_binding = EvidenceBinding(status=BindingStatus.MISSING.value, slot_bindings={}, missing_slots=expected_slots)
        return SelectiveAdmissionV2Result(CONTRACT_NAME, safe_binding, validation, False, slot_evidence, tuple(reasons))

    # Existing MISSING/AMBIGUOUS results are never promoted by V2.
    if normalized_binding.status == BindingStatus.AMBIGUOUS.value:
        safe_binding = normalized_binding
    elif normalized_binding.status == BindingStatus.MISSING.value:
        safe_binding = normalized_binding
    else:
        safe_binding = EvidenceBinding(status=BindingStatus.INVALID.value, slot_bindings={}, invalid_reasons=("non_admissible_input_status",))
    return SelectiveAdmissionV2Result(CONTRACT_NAME, safe_binding, validation, False, slot_evidence, ("binder_did_not_return_BOUND",))
