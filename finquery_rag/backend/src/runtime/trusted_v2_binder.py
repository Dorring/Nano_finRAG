from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rag_v2.adaptive import (
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidenceEvaluationV1,
    ReasonCode,
)
from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.plan import Intent, SupervisorPlan
from rag_v2.evidence.binder_service import (
    BinderRequest,
    BinderRun,
    SemanticBinderService,
)
from rag_v2.evidence.binding_validator import validate_binding
from rag_v2.supervisor import (
    BoundEvidenceSemanticCheck,
    align_bound_evidence_to_query,
    canonical_entity_id,
    canonical_metric_id,
    canonical_period_id,
    classify_evidence_scope,
    extract_query_semantic_frame,
)
from rag_v2.supervisor import EvidenceScope, query_allows_evidence_scope


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
        self.last_semantic_check: BoundEvidenceSemanticCheck | None = None
        self.last_semantic_repair: dict[str, Any] | None = None
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
            fact["provenance_complete"] = bool(fact.get("provenance_complete", False))
            facts.append(fact)
        return tuple(facts)

    @staticmethod
    def _facts_for_binding(
        state: AdaptiveRAGStateV1,
        plan: SupervisorPlan,
        facts: tuple[Mapping[str, Any], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        """Build a bounded, slot-relevant Binder view of the candidate pool.

        R4 deliberately keeps a broad candidate packet for audit and
        recovery. Sending every unrelated row to the semantic provider makes
        a one-slot decision unnecessarily noisy (and can cause a provider to
        return ``MISSING`` even when the admitted aggregate is present). The
        Binder still remains the authority: this is only deterministic
        context shaping, never slot admission or answer generation.
        """

        expected_metrics = {
            metric_id
            for metric_id in (
                canonical_metric_id(slot.metric) for slot in plan.required_slots
            )
            if metric_id
        }
        if not expected_metrics:
            return facts
        relevant = tuple(
            fact
            for fact in facts
            if canonical_metric_id(
                fact.get("metric")
                or fact.get("normalized_metric")
                or fact.get("raw_metric")
            )
            in expected_metrics
        )
        # Preserve the full packet when the source schema did not expose a
        # metric at all; the Binder can then make an explicit MISSING decision
        # instead of receiving an empty, misleading request.
        if not relevant:
            return facts

        # For an unqualified direct fact, company-total scope is the safe
        # convention. If a source-verified aggregate is available, do not
        # ask the model to choose between it and segment rows. Unknown-scope
        # candidates remain visible for conservative fallback and are checked
        # again by the deterministic semantic firewall after binding.
        try:
            frame = extract_query_semantic_frame(state.normalized_query)
        except (TypeError, ValueError):
            frame = None
        if (
            frame is not None
            and plan.intent is Intent.DIRECT_FACT
            and not frame.scope_ids
        ):
            classifications = [classify_evidence_scope(fact) for fact in relevant]
            known_segment_labels = tuple(
                classification.scope_label
                for classification in classifications
                if classification.scope is EvidenceScope.SEGMENT
                and classification.scope_label
            )
            explicit_segment_requested = any(
                query_allows_evidence_scope(
                    state.normalized_query,
                    frame,
                    classification,
                    known_segment_labels=known_segment_labels,
                )
                for classification in classifications
                if classification.scope is EvidenceScope.SEGMENT
            )
            if (
                any(
                    item.scope is EvidenceScope.CONSOLIDATED for item in classifications
                )
                and not explicit_segment_requested
            ):
                filtered = tuple(
                    fact
                    for fact, classification in zip(relevant, classifications)
                    if classification.scope is not EvidenceScope.SEGMENT
                )
                if filtered:
                    return filtered
        return relevant

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
            expected_metric = canonical_metric_id(slot.metric)
            same_metric = [
                fact
                for fact in facts
                if canonical_metric_id(fact.get("metric")) == expected_metric
            ]
            if same_metric and all(
                _norm(fact.get("period")) != _norm(slot.period) for fact in same_metric
            ):
                return ReasonCode.WRONG_PERIOD
        return ReasonCode.MISSING_SLOT

    @staticmethod
    def _fact_value_key(fact: Mapping[str, Any]) -> str | None:
        """Build a conservative structured value identity for consensus repair.

        This is deliberately not a parser for answer text.  It only compares
        the value/unit/scale fields already present on an R4 candidate.  A
        differing scale or unit therefore remains a conflict instead of being
        silently treated as the same fact.
        """

        value = fact.get("parsed_numeric_value")
        if value is None:
            value = fact.get("value")
        if value is None:
            value = fact.get("raw_value")
        if value is None or not str(value).strip():
            return None

        def normalize(item: Any) -> str:
            return "".join(str(item).casefold().split()).replace(",", "")

        parts = (
            normalize(value),
            normalize(fact.get("unit")) if fact.get("unit") is not None else "",
            normalize(fact.get("currency")) if fact.get("currency") is not None else "",
            normalize(fact.get("scale")) if fact.get("scale") is not None else "",
        )
        return "|".join(parts)

    @classmethod
    def _fact_matches_slot(
        cls,
        query: str,
        frame: Any,
        slot: Any,
        fact: Mapping[str, Any],
        known_segment_labels: tuple[str, ...],
    ) -> bool:
        expected_metric = canonical_metric_id(slot.metric)
        expected_period = canonical_period_id(slot.period) or _norm(slot.period)
        fact_metric = canonical_metric_id(
            fact.get("metric")
            or fact.get("normalized_metric")
            or fact.get("raw_metric")
        )
        fact_period = canonical_period_id(
            fact.get("period")
            or fact.get("normalized_period")
            or fact.get("raw_period")
        ) or _norm(
            fact.get("period")
            or fact.get("normalized_period")
            or fact.get("raw_period")
        )
        if expected_metric is None or expected_period is None:
            return False
        if fact_metric != expected_metric or fact_period != expected_period:
            return False
        if frame.entity_ids:
            fact_entity = canonical_entity_id(
                fact.get("entity") or fact.get("company") or fact.get("ticker")
            )
            if fact_entity is None or fact_entity not in set(frame.entity_ids):
                return False
        return query_allows_evidence_scope(
            query,
            frame,
            classify_evidence_scope(fact),
            known_segment_labels=known_segment_labels,
        )

    @classmethod
    def _consensus_fact_for_slot(
        cls,
        query: str,
        frame: Any,
        slot: Any,
        facts: tuple[Mapping[str, Any], ...],
        *,
        require_explicit_source: bool = False,
    ) -> tuple[str, int, str] | None:
        """Return one fact only when candidate evidence has a safe consensus.

        A Binder provider may select a structurally valid duplicate whose
        period/value columns are misaligned in an extracted table.  We may
        correct that choice only from the same candidate packet: exact metric,
        period, entity and requested scope must match, and conflicting
        structured values must have a strict consensus.  Ties are intentionally
        left unresolved so the runtime still fails closed.
        """

        known_segment_labels = tuple(
            label
            for fact in facts
            for label in (classify_evidence_scope(fact).scope_label,)
            if label
        )
        groups: dict[str, list[tuple[str, str]]] = {}
        seen_fact_ids: set[str] = set()
        for fact in facts:
            if fact.get("provenance_complete") is not True:
                continue
            fact_id = fact.get("fact_id") or fact.get("evidence_id")
            if not fact_id:
                continue
            fact_id = str(fact_id).strip()
            if not fact_id or fact_id in seen_fact_ids:
                continue
            if not cls._fact_matches_slot(
                query,
                frame,
                slot,
                fact,
                known_segment_labels,
            ):
                continue
            value_key = cls._fact_value_key(fact)
            if value_key is None:
                continue
            source_value = fact.get("physical_source_id") or fact.get("source_id")
            if require_explicit_source and not source_value:
                continue
            source_id = str(source_value or fact_id).strip()
            if not source_id:
                continue
            # The same physical source can be represented by multiple atomic
            # IDs. Count it once so duplicate extraction rows cannot manufacture
            # a false majority.
            if any(
                source_id == prior_source
                for values in groups.values()
                for _, prior_source in values
            ):
                continue
            groups.setdefault(value_key, []).append((fact_id, source_id))
            seen_fact_ids.add(fact_id)

        if not groups:
            return None
        ranked = sorted(
            groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        winner_key, winner_values = ranked[0]
        second_count = len(ranked[1][1]) if len(ranked) > 1 else 0
        if len(ranked) > 1 and len(winner_values) <= second_count:
            return None
        if len(ranked) > 1 and len(winner_values) < 2:
            return None
        return winner_values[0][0], len(winner_values), winner_key

    @classmethod
    def _repair_semantic_binding(
        cls,
        query: str,
        plan: SupervisorPlan,
        facts: tuple[Mapping[str, Any], ...],
        run: BinderRun,
        semantic_check: BoundEvidenceSemanticCheck,
    ) -> tuple[BinderRun, BoundEvidenceSemanticCheck, dict[str, Any]] | None:
        """Repair a provider selection using only deterministic packet facts."""

        if plan.intent not in {Intent.DIRECT_FACT, Intent.CALCULATION}:
            return None
        try:
            frame = extract_query_semantic_frame(query)
        except (TypeError, ValueError):
            return None

        replacements: dict[str, tuple[str, ...]] = {}
        consensus_sizes: dict[str, int] = {}
        for slot in plan.required_slots:
            resolved = cls._consensus_fact_for_slot(query, frame, slot, facts)
            if resolved is None:
                return None
            fact_id, consensus_size, value_key = resolved
            replacements[slot.slot_id] = (fact_id,)
            consensus_sizes[slot.slot_id] = consensus_size

        original = {
            str(slot_id): tuple(str(item) for item in fact_ids)
            for slot_id, fact_ids in run.binding.slot_bindings.items()
        }
        fact_by_id = {
            str(fact.get("fact_id")): fact for fact in facts if fact.get("fact_id")
        }
        known_segment_labels = tuple(
            label
            for fact in facts
            for label in (classify_evidence_scope(fact).scope_label,)
            if label
        )
        # Keep a provider-selected duplicate when it is semantically exact and
        # carries the same structured value as the consensus winner.  This
        # preserves the provider's stable provenance choice for equivalent
        # duplicate rows while still replacing a wrong-period or wrong-value
        # selection.
        for slot in plan.required_slots:
            slot_id = slot.slot_id
            current_ids = original.get(slot_id, ())
            winner = cls._consensus_fact_for_slot(query, frame, slot, facts)
            if winner is None:
                return None
            winner_id, _consensus_size, winner_value_key = winner
            if len(current_ids) == 1:
                current_fact = fact_by_id.get(current_ids[0])
                if (
                    current_fact is not None
                    and cls._fact_matches_slot(
                        query,
                        frame,
                        slot,
                        current_fact,
                        known_segment_labels,
                    )
                    and cls._fact_value_key(current_fact) == winner_value_key
                ):
                    replacements[slot_id] = current_ids
        if all(
            original.get(slot_id) == fact_ids
            for slot_id, fact_ids in replacements.items()
        ):
            return None
        new_bindings = dict(original)
        new_bindings.update(replacements)
        try:
            binding = EvidenceBinding(
                status=BindingStatus.BOUND.value,
                slot_bindings=new_bindings,
            )
        except Exception:
            return None
        validation = validate_binding(binding, plan, facts)
        if not validation.passed:
            return None
        repaired_check = align_bound_evidence_to_query(
            query,
            plan,
            facts,
            binding.slot_bindings,
            selected_fact_ids=validation.selected_fact_ids,
        )
        if not repaired_check.allowed:
            return None
        repaired_run = BinderRun(
            request=run.request,
            binding=binding,
            validation=validation,
            metadata=run.metadata,
            skipped_no_fact_supply=run.skipped_no_fact_supply,
            raw_response=run.raw_response,
            schema_valid=run.schema_valid,
        )
        info = {
            "strategy": "deterministic_candidate_consensus",
            "original_mismatches": list(semantic_check.mismatches),
            "replaced_slot_bindings": {
                slot_id: {
                    "from": list(original.get(slot_id, ())),
                    "to": list(fact_ids),
                }
                for slot_id, fact_ids in replacements.items()
                if original.get(slot_id) != fact_ids
            },
            "consensus_size_by_slot": consensus_sizes,
        }
        return repaired_run, repaired_check, info

    @classmethod
    def _repair_ambiguous_binding(
        cls,
        query: str,
        plan: SupervisorPlan,
        facts: tuple[Mapping[str, Any], ...],
        run: BinderRun,
    ) -> tuple[BinderRun, BoundEvidenceSemanticCheck, dict[str, Any]] | None:
        """Resolve a provider ambiguity only with independently corroborated facts.

        ``AMBIGUOUS`` remains a fail-closed result by default. This narrow
        recovery is available only when each ambiguous requested slot has a
        unique, exact structured-value consensus across at least two
        independent physical sources. A provider may preserve already-bound
        non-ambiguous slots, but missing slots, unknown slot IDs, overlapping
        bindings, duplicate R4 lanes, answer text, and a single source never
        qualify as corroboration.
        """

        if plan.intent not in {Intent.DIRECT_FACT, Intent.CALCULATION}:
            return None
        if not run.schema_valid:
            return None
        expected_slot_ids = {slot.slot_id for slot in plan.required_slots}
        ambiguous_slot_ids = set(
            _stable_unique(str(slot_id) for slot_id in run.binding.ambiguous_slots)
        )
        existing_bindings = {
            str(slot_id): tuple(str(fact_id) for fact_id in fact_ids)
            for slot_id, fact_ids in run.binding.slot_bindings.items()
        }
        bound_slot_ids = set(existing_bindings)
        missing_slot_ids = {str(slot_id) for slot_id in run.binding.missing_slots}
        if not expected_slot_ids or not ambiguous_slot_ids:
            return None
        if not ambiguous_slot_ids.issubset(expected_slot_ids):
            return None
        if missing_slot_ids or bound_slot_ids & ambiguous_slot_ids:
            return None
        if bound_slot_ids | ambiguous_slot_ids != expected_slot_ids:
            return None
        try:
            frame = extract_query_semantic_frame(query)
        except (TypeError, ValueError):
            return None
        if not frame.entity_ids:
            return None

        replacements = dict(existing_bindings)
        consensus_sizes: dict[str, int] = {}
        normalized_slot_bindings: dict[str, dict[str, list[str]]] = {}
        fact_by_id = {
            str(fact.get("fact_id") or fact.get("evidence_id")): fact
            for fact in facts
            if fact.get("fact_id") or fact.get("evidence_id")
        }
        known_segment_labels = tuple(
            label
            for fact in facts
            for label in (classify_evidence_scope(fact).scope_label,)
            if label
        )
        for slot in plan.required_slots:
            slot_id = slot.slot_id
            current_ids = existing_bindings.get(slot_id, ())
            if slot_id not in ambiguous_slot_ids:
                # ``BOUND`` permits exactly one fact per slot.  An ambiguous
                # provider response may still carry duplicate, equal-valued
                # rows for a different slot.  Collapse those rows only when
                # every provider-selected row matches the same independently
                # corroborated packet consensus; a conflicting or malformed
                # selection remains fail-closed.
                if len(current_ids) <= 1:
                    continue
                resolved = cls._consensus_fact_for_slot(
                    query,
                    frame,
                    slot,
                    facts,
                    require_explicit_source=True,
                )
                if resolved is None:
                    return None
                fact_id, consensus_size, value_key = resolved
                if consensus_size < 2:
                    return None
                for current_id in current_ids:
                    current_fact = fact_by_id.get(current_id)
                    if (
                        current_fact is None
                        or not cls._fact_matches_slot(
                            query,
                            frame,
                            slot,
                            current_fact,
                            known_segment_labels,
                        )
                        or cls._fact_value_key(current_fact) != value_key
                    ):
                        return None
                replacements[slot_id] = (fact_id,)
                consensus_sizes[slot_id] = consensus_size
                normalized_slot_bindings[slot_id] = {
                    "from": list(current_ids),
                    "to": [fact_id],
                }
                continue
            resolved = cls._consensus_fact_for_slot(
                query,
                frame,
                slot,
                facts,
                require_explicit_source=True,
            )
            if resolved is None:
                return None
            fact_id, consensus_size, _value_key = resolved
            if consensus_size < 2:
                return None
            replacements[slot_id] = (fact_id,)
            consensus_sizes[slot_id] = consensus_size

        if set(replacements) != expected_slot_ids:
            return None
        try:
            binding = EvidenceBinding(
                status=BindingStatus.BOUND.value,
                slot_bindings=replacements,
            )
        except Exception:
            return None
        validation = validate_binding(binding, plan, facts)
        if not validation.passed:
            return None
        semantic_check = align_bound_evidence_to_query(
            query,
            plan,
            facts,
            binding.slot_bindings,
            selected_fact_ids=validation.selected_fact_ids,
        )
        if not semantic_check.allowed:
            return None
        repaired_run = BinderRun(
            request=run.request,
            binding=binding,
            validation=validation,
            metadata=run.metadata,
            skipped_no_fact_supply=run.skipped_no_fact_supply,
            raw_response=run.raw_response,
            schema_valid=run.schema_valid,
        )
        return (
            repaired_run,
            semantic_check,
            {
                "strategy": (
                    "deterministic_ambiguous_packet_consensus"
                    if not existing_bindings
                    else "deterministic_partial_ambiguous_packet_consensus"
                ),
                "original_ambiguous_slots": list(run.binding.ambiguous_slots),
                "preserved_slot_bindings": {
                    slot_id: list(fact_ids)
                    for slot_id, fact_ids in existing_bindings.items()
                },
                "replaced_slot_bindings": {
                    slot_id: {"to": list(fact_ids)}
                    for slot_id, fact_ids in replacements.items()
                    if slot_id in ambiguous_slot_ids
                },
                "normalized_slot_bindings": normalized_slot_bindings,
                "consensus_size_by_slot": consensus_sizes,
            },
        )

    def evaluate(self, state: AdaptiveRAGStateV1) -> EvidenceEvaluationV1:
        plan = self._plan(state)
        all_facts = self._facts(state)
        facts = self._facts_for_binding(state, plan, all_facts)
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
        self.last_semantic_check = None
        self.last_semantic_repair = None
        if status == BindingStatus.AMBIGUOUS.value:
            repaired = self._repair_ambiguous_binding(
                state.normalized_query,
                plan,
                facts,
                run,
            )
            if repaired is not None:
                run, semantic_check, repair_info = repaired
                self.last_run = run
                self.last_semantic_check = semantic_check
                self.last_semantic_repair = repair_info
                status = BindingStatus.BOUND.value
        if status == BindingStatus.BOUND.value:
            semantic_check: BoundEvidenceSemanticCheck
            # Providers occasionally return every equivalent duplicate row for
            # a slot instead of the one-fact-per-slot shape required by the
            # domain contract. Treat that narrow cardinality violation as a
            # repairable provider-shape problem: the existing deterministic
            # packet-only consensus repair may collapse it to one fact per
            # slot. Unknown slots/facts, incomplete provenance, duplicate
            # facts across different slots, and any other structural problem
            # remain hard capability errors and never get silently repaired.
            structural_repairable = not run.validation.passed and set(
                run.validation.reasons
            ) == {"bound_fact_cardinality_mismatch"}
            if not run.validation.passed and not structural_repairable:
                raise SemanticBinderCapabilityError(
                    "bound_evidence_failed_structural_validation"
                )
            try:
                semantic_check = align_bound_evidence_to_query(
                    state.normalized_query,
                    plan,
                    facts,
                    run.binding.slot_bindings,
                    selected_fact_ids=run.validation.selected_fact_ids,
                )
            except (TypeError, ValueError) as exc:
                if structural_repairable:
                    raise SemanticBinderCapabilityError(
                        "bound_evidence_failed_semantic_alignment"
                    ) from exc
                raise
            self.last_semantic_check = semantic_check
            # A provider can select a wrong-period candidate (caught by the
            # alignment check) or a same-period duplicate whose value was
            # mis-extracted from a neighboring table column (not caught by
            # metric/period equality alone).  Attempt the bounded, packet-only
            # consensus repair in both cases.  It never invents evidence and
            # returns ``None`` on conflict, so the existing fail-closed path
            # remains authoritative when no safe correction exists.
            repaired = self._repair_semantic_binding(
                state.normalized_query,
                plan,
                facts,
                run,
                semantic_check,
            )
            if repaired is not None:
                run, semantic_check, repair_info = repaired
                self.last_run = run
                self.last_semantic_check = semantic_check
                self.last_semantic_repair = repair_info
            if not semantic_check.allowed:
                self.last_bound_evidence_ids = ()
                self.last_bound_slot_bindings = {}
                self.last_citation_ids = ()
                evaluation = EvidenceEvaluationV1(
                    decision=EvidenceDecision.TERMINAL_INSUFFICIENT,
                    reason_codes=(ReasonCode.QUERY_EVIDENCE_SEMANTIC_MISMATCH,),
                    requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                    supported_slots=(),
                    missing_slots=tuple(slot.slot_id for slot in plan.required_slots),
                    conflicts=tuple(
                        {"reason": reason} for reason in semantic_check.mismatches
                    ),
                    temporal_status="SEMANTIC_MISMATCH",
                    calculation_ready=False,
                )
                self._trace.append(
                    {
                        "round": self.calls - 1,
                        "status": status,
                        "bound_slot_ids": [],
                        "missing_slot_ids": [
                            slot.slot_id for slot in plan.required_slots
                        ],
                        "ambiguous_slot_ids": [],
                        "bound_evidence_ids": [],
                        "reason_codes": [
                            item.value for item in evaluation.reason_codes
                        ],
                        "selected_fact_ids": list(run.validation.selected_fact_ids),
                        "candidate_fact_count": len(all_facts),
                        "binding_fact_count": len(facts),
                        "semantic_check": semantic_check.to_dict(),
                        "semantic_repair": self.last_semantic_repair,
                    }
                )
                return evaluation
            selected = _stable_unique(run.validation.selected_fact_ids)
            self.last_bound_evidence_ids = selected
            self.last_bound_slot_bindings = {
                str(slot_id): tuple(str(item) for item in fact_ids)
                for slot_id, fact_ids in run.binding.slot_bindings.items()
            }
            fact_by_id = {
                str(fact.get("fact_id")): fact for fact in facts if fact.get("fact_id")
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
                raise SemanticBinderCapabilityError("binder_returned_invalid_schema")
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
                "candidate_fact_count": len(all_facts),
                "binding_fact_count": len(facts),
                "semantic_check": (
                    self.last_semantic_check.to_dict()
                    if self.last_semantic_check is not None
                    else None
                ),
                "semantic_repair": self.last_semantic_repair,
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
            "semantic_checks": [
                item.get("semantic_check")
                for item in records
                if item.get("semantic_check") is not None
            ],
            "binder_rounds": records,
        }


__all__ = [
    "SemanticBinderCapabilityError",
    "SemanticEvidenceEvaluationCapability",
]
