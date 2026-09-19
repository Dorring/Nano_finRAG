from __future__ import annotations

from collections.abc import Iterable, Mapping
import os
from typing import Any

from rag_v2.adaptive import (
    AdaptiveRAGStateV1,
    EvidenceDecision,
    EvidenceEvaluationV1,
    ReasonCode,
)
from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding
from rag_v2.contracts.financial_semantics import (
    canonical_quantity,
    quantity_identity,
    text_identity,
)
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
    metric_identity,
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


#: Why a candidate cannot participate in binding a slot.  Structural facts about
#: the candidate and the slot's declared requirements -- never a judgement about
#: relevance, which is what the Binder is for.
CANDIDATE_PERIOD_ABSENT = "candidate_period_absent"
CANDIDATE_PERIOD_MISMATCH = "candidate_period_mismatch"
CANDIDATE_VALUE_NOT_NUMERIC = "candidate_numeric_value_absent"


def _fact_field(fact: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = fact.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _metric_is_admissible(
    fact: Mapping[str, Any],
    expected_metrics: set[str],
) -> bool:
    """Whether the metric prefilter may drop this candidate.

    It may drop a fact whose metric the ontology *names* and the plan does not
    ask for.  It may not drop one whose metric the ontology cannot name at all,
    and that distinction is not a nicety -- conflating them is a regression this
    filter has already caused twice.

    ``_facts_for_binding`` runs only when at least one slot's metric resolves.
    Before ``total_operating_expenses`` was added to the ontology, a plan naming
    it and ``Leasehold improvements`` resolved *nothing*, the early return handed
    the Binder the whole packet, and the case passed.  Adding the one metric made
    the filter fire -- and it then dropped every ``Leasehold improvements`` fact,
    because an unnamed metric is not in a set of names.  The Binder was left
    without the fact for one of its two slots and reported MISSING, over a packet
    that contained it.  ``pctshare-004`` is the same shape with the roles
    swapped: ``Cost of sales`` resolved, ``Impact of the State Aid Decision`` did
    not, and the State Aid rows were removed before the Binder saw anything.

    So an unnamed metric is retained.  It is genuinely unknown to the plan's
    vocabulary, which is a fact about this ontology and not evidence of
    irrelevance, and the Binder is the component that decides relevance.
    """

    named = canonical_metric_id(
        fact.get("metric")
        or fact.get("normalized_metric")
        or fact.get("raw_metric")
    )
    return named is None or named in expected_metrics


def candidate_eligible_for_slot(
    fact: Mapping[str, Any],
    slot: Any,
) -> tuple[bool, tuple[str, ...]]:
    """Whether a candidate meets the slot's *structural* minimum, and why not.

    This is not the complement of "relevant"; it is the complement of "complete
    enough to be considered".  A slot that declares FY2025 and a numeric value
    cannot be satisfied by a row with neither, and that is a deterministic
    reading of the slot's own contract rather than a guess about meaning.

    It exists because the two were being given to the model together.  A
    retrieval packet is allowed to be broad -- it carries rows for audit and
    recovery that no particular slot can use -- and sending those into a slot's
    candidate set asks the model to rule them out.  The model does it
    inconsistently: the case that motivated this bound its denominator on one run
    and reported both slots missing on the next, over a byte-identical packet.

    What this is NOT: a filter on the packet.  The packet keeps every row,
    because a period-less or value-less row may be exactly what a later
    qualitative, heading or metric-discovery task wants.  Retrieval-admissible
    and slot-binding-eligible are different properties, and only the second is
    decided here.

    Silence is never a rejection.  A slot that declares no period does not
    thereby reject a period-less candidate, and a slot that does not ask for a
    numeric value does not reject one that has none.  Only a requirement the slot
    actually states can be unmet -- otherwise this becomes the global tidy-up of
    untidy data that it is written not to be.

    ``canonical_quantity`` is the numeric test rather than ``canonical_decimal``
    because a percentage is a value: ``21%`` is what an income-tax-rate row
    states, and a rule that called it a non-value would exclude the very
    candidates these fixtures exist to bind.
    """

    reasons: list[str] = []

    required_period = canonical_period_id(slot.period) or _norm(slot.period)
    if required_period is not None:
        stated = _fact_field(fact, "period", "normalized_period", "raw_period")
        candidate_period = canonical_period_id(stated) or _norm(stated)
        if candidate_period is None:
            reasons.append(CANDIDATE_PERIOD_ABSENT)
        elif candidate_period != required_period:
            reasons.append(CANDIDATE_PERIOD_MISMATCH)

    if _norm(getattr(slot, "value_type", None)) == "numeric":
        value = _fact_field(fact, "value", "parsed_numeric_value", "raw_value")
        if (
            canonical_quantity(
                value,
                scale=_fact_field(fact, "scale"),
                unit=_fact_field(fact, "unit"),
                currency=_fact_field(fact, "currency"),
            )
            is None
        ):
            reasons.append(CANDIDATE_VALUE_NOT_NUMERIC)

    return (not reasons), tuple(reasons)


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

    _METADATA_PROMOTED_KEYS: tuple[str, ...] = (
        "fact_id",
        "provenance_complete",
        "physical_source_id",
        "metric",
        "normalized_metric",
        "raw_metric",
        "period",
        "normalized_period",
        "raw_period",
        "entity",
        "company",
        "ticker",
        "scope",
        "statement_type",
        "unit",
        "currency",
        "scale",
    )

    @classmethod
    def _facts(cls, state: AdaptiveRAGStateV1) -> tuple[Mapping[str, Any], ...]:
        facts: list[Mapping[str, Any]] = []
        for raw in state.evidence_packets:
            if not isinstance(raw, Mapping):
                raise SemanticBinderCapabilityError("candidate_fact_must_be_mapping")
            fact = dict(raw)
            metadata = fact.get("metadata")
            if isinstance(metadata, Mapping):
                for key in cls._METADATA_PROMOTED_KEYS:
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
            if _metric_is_admissible(fact, expected_metrics)
            # The metric prefilter is left in place; eligibility is added
            # alongside it.  This step changes one thing -- a candidate
            # structurally incapable of filling a slot is no longer offered to
            # the Binder for that slot.
            and any(
                candidate_eligible_for_slot(fact, slot)[0]
                for slot in plan.required_slots
            )
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
            expected_metric = metric_identity(slot.metric)
            same_metric = [
                fact
                for fact in facts
                if metric_identity(fact.get("metric")) == expected_metric
            ]
            if same_metric and all(
                _norm(fact.get("period")) != _norm(slot.period) for fact in same_metric
            ):
                return ReasonCode.WRONG_PERIOD
        return ReasonCode.MISSING_SLOT

    @classmethod
    def _fact_value_key(cls, fact: Mapping[str, Any]) -> str | None:
        """The structured quantity identity of an R4 candidate.

        This is deliberately not a parser for answer text.  It compares the
        value/unit/currency/scale fields already present on the candidate, and
        it does so by asking the shared financial semantics what those fields
        mean -- the same primitive the content fingerprint uses, so the two
        cannot disagree about whether two sources state the same quantity.

        Three consequences follow from that, and all three are the point:

        * A differing scale or unit stays a difference.  ``1 billion USD`` and
          ``1 billion EUR`` are different quantities, and two admissible
          candidates that state them are in conflict.
        * A *known* scale is folded into the value, so ``1000 million`` and
          ``1 billion`` key the same -- otherwise two sources writing one
          quantity in different units read as a disagreement, and a fail-closed
          gate then refuses evidence that agrees.
        * An *unrecognised* scale is not folded and is not a magnitude.  It
          stays a literal part of the key, so ``adjusted-billion`` remains a
          different fact from ``billion`` rather than being silently equated
          with it.

        Arbitration -- how many agreeing candidates are enough, what counts as
        corroboration -- is not here.  This method answers "what quantity does
        this record state"; the conflict gate answers "do these agree enough to
        bind".
        """

        value = fact.get("parsed_numeric_value")
        if value is None:
            value = fact.get("value")
        if value is None:
            value = fact.get("raw_value")
        if value is None or not str(value).strip():
            return None

        return quantity_identity(
            value,
            scale=fact.get("scale"),
            unit=fact.get("unit"),
            currency=fact.get("currency"),
        )

    @staticmethod
    def _frame_allows_entity(fact_entity_id: str | None, frame: Any) -> bool:
        """Whether the query's own mentions permit a fact about this company.

        A query that names no company constrains no company, which is the
        behaviour this had before slots could carry an entity at all.
        """

        entity_ids = set(getattr(frame, "entity_ids", ()) or ())
        if not entity_ids:
            return True
        return fact_entity_id is not None and fact_entity_id in entity_ids

    @classmethod
    def _entity_matches_slot(
        cls,
        slot: Any,
        fact: Mapping[str, Any],
        frame: Any,
    ) -> bool:
        """Whether this fact is about the company *this slot* asked for.

        Entity used to be matched against the query's mentions alone, once for
        the whole plan (`frame.entity_ids`).  In a two-company question that let
        either company's fact satisfy either slot, which is precisely the
        question a comparison asks -- so a comparison could be answered by
        binding the same company twice and never noticed.

        Three cases, in the order the contract defines them, and no fourth:

        * **the slot carries a canonical id.**  The fact must canonicalise to
          the same id.  A fact that cannot be canonicalised at all is *not* a
          match: the slot declared an identity, and degrading to a text
          comparison would quietly answer a question the plan did not ask.  The
          query's own mentions are still consulted here, because both sides are
          in the vocabulary and a plan should not be able to introduce a company
          the question never mentioned.
        * **the slot carries only a mention.**  Strict normalised text equality
          and nothing else.  ``Pfizer`` matches ``Pfizer``; it does not match
          ``Pfizer Inc.``.  Inferring that is ontology work, and doing it here
          would make the answer depend on which spellings happened to be
          written down.  The query's mentions are deliberately *not* consulted:
          they come from a twelve-company vocabulary, and gating on them would
          reject exactly the companies the mention exists to support.
        * **the slot carries neither.**  The query-level test is all there was
          before, and it stays in charge.
        """

        slot_entity = getattr(slot, "entity", None)
        slot_entity_id = getattr(slot, "entity_id", None)
        fact_entity = fact.get("entity") or fact.get("company") or fact.get("ticker")

        if os.environ.get("P15R_ENFORCE", "on") == "off":
            # P1.5-R diagnostic only.  Reproduces what this matcher did before
            # the entity contract reached it -- the query-level check alone --
            # so the ablation can tell "the model can see the entity" apart from
            # "the entity is enforced here".  The two are one field read at two
            # places, and only turning one of them off isolates them.
            return cls._frame_allows_entity(canonical_entity_id(fact_entity), frame)

        if not slot_entity and not slot_entity_id:
            return cls._frame_allows_entity(canonical_entity_id(fact_entity), frame)

        if slot_entity_id:
            fact_entity_id = canonical_entity_id(fact_entity)
            if fact_entity_id != slot_entity_id:
                return False
            return cls._frame_allows_entity(fact_entity_id, frame)

        if fact_entity is None:
            return False
        return text_identity(slot_entity) == text_identity(fact_entity)

    @classmethod
    def _fact_matches_slot(
        cls,
        query: str,
        frame: Any,
        slot: Any,
        fact: Mapping[str, Any],
        known_segment_labels: tuple[str, ...],
    ) -> bool:
        # The wider identity, so a metric the ontology cannot name is still
        # comparable.  This used to return False for both sides, which made any
        # slot naming such a metric unsatisfiable while the fact existed.
        expected_metric = metric_identity(slot.metric)
        expected_period = canonical_period_id(slot.period) or _norm(slot.period)
        fact_metric = metric_identity(
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
        if not cls._entity_matches_slot(slot, fact, frame):
            return False
        return query_allows_evidence_scope(
            query,
            frame,
            classify_evidence_scope(fact),
            known_segment_labels=known_segment_labels,
        )

    @classmethod
    def _unresolved_conflict_slots(
        cls,
        query: str,
        plan: SupervisorPlan,
        facts: tuple[Mapping[str, Any], ...],
    ) -> tuple[str, ...]:
        """Slots whose admissible candidates disagree with no safe consensus.

        Deliberately reuses ``_consensus_fact_for_slot`` rather than adding a
        second comparison rule: that helper already counts an exact duplicate
        once, accepts a strict majority of *distinct physical sources*, and
        returns ``None`` for a tie or an uncorroborated winner.  Writing the
        conflict test separately would be a second place for the definition of
        "these two facts agree" to live.

        A slot is reported only when more than one candidate is admissible for
        it *and* no consensus exists among them, so a single candidate, an exact
        duplicate, two sources reporting the same value, and a clear majority
        are all still bound normally.
        """

        try:
            frame = extract_query_semantic_frame(query)
        except (TypeError, ValueError):
            return ()
        known_segment_labels = tuple(
            label
            for fact in facts
            for label in (classify_evidence_scope(fact).scope_label,)
            if label
        )
        conflicted: list[str] = []
        for slot in plan.required_slots:
            admissible = [
                fact
                for fact in facts
                if cls._fact_matches_slot(
                    query,
                    frame,
                    slot,
                    fact,
                    known_segment_labels,
                )
            ]
            if len(admissible) < 2:
                continue
            if cls._consensus_fact_for_slot(query, frame, slot, facts) is None:
                conflicted.append(str(slot.slot_id))
        return tuple(conflicted)

    @classmethod
    def _consensus_fact_for_slot(
        cls,
        query: str,
        frame: Any,
        slot: Any,
        facts: tuple[Mapping[str, Any], ...],
        *,
        require_explicit_source: bool = False,
    ) -> tuple[str, tuple[str, ...], str] | None:
        """Return the admitted support set when candidate evidence has a safe consensus.

        A Binder provider may select a structurally valid duplicate whose
        period/value columns are misaligned in an extracted table.  We may
        correct that choice only from the same candidate packet: exact metric,
        period, entity and requested scope must match, and conflicting
        structured values must have a strict consensus.  Ties are intentionally
        left unresolved so the runtime still fails closed.

        Returns ``(canonical_fact_id, support_ids, value_key)``.  The support
        set is every fact in the winning canonical group, deduplicated by
        physical source -- one id per independent witness, however many
        extraction rows that witness produced.  Until H2A-2C-2 this returned
        only ``winner_values[0][0]`` and a count, so the corroboration it had
        just computed was discarded at the door: the binding layer could not
        keep two sources agreeing, which is the one thing a corroboration set is
        for.  The count is now ``len(support_ids)`` and the caller reads it from
        the set rather than beside it, so the two cannot disagree.
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
        # Every id in the winning group is an admitted independent support:
        # the group was built by discarding a fact whose physical source had
        # already been counted, so its size is a number of *witnesses*, not a
        # number of rows.  The first is the canonical representative, chosen by
        # the order the facts were admitted, so the value read from it is
        # stable and -- since the group agrees by construction -- the same one
        # any other member would give.
        return (
            winner_values[0][0],
            tuple(fact_id for fact_id, _source_id in winner_values),
            winner_key,
        )

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
            _fact_id, support_ids, _value_key = resolved
            replacements[slot.slot_id] = support_ids
            consensus_sizes[slot.slot_id] = len(support_ids)

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
        #
        # H2A-2C-2: the replacement is now the winning group rather than one
        # representative of it.  When the provider's pick is a member of that
        # group it is preserved -- as a member -- and so are its independent
        # corroborators, which the old single-id replacement dropped.
        for slot in plan.required_slots:
            slot_id = slot.slot_id
            current_ids = original.get(slot_id, ())
            winner = cls._consensus_fact_for_slot(query, frame, slot, facts)
            if winner is None:
                return None
            _winner_id, support_ids, winner_value_key = winner
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
                    replacements[slot_id] = support_ids
                    consensus_sizes[slot_id] = len(support_ids)
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
            if fact_ids
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
                # An ambiguous provider response may carry duplicate,
                # equal-valued rows for a different slot.  Collapse those rows
                # only when every provider-selected row matches the same
                # independently corroborated packet consensus; a conflicting or
                # malformed selection remains fail-closed.
                #
                # H2A-2C-2: "collapse" now means "bind the winning group",
                # which may itself hold several independent sources.  Every
                # provider-selected row was verified to state the winning
                # value, so nothing equivalent is being discarded -- only a
                # second copy of a witness already counted, or a row whose
                # stated value differs.
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
                _fact_id, support_ids, value_key = resolved
                if len(support_ids) < 2:
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
                replacements[slot_id] = support_ids
                consensus_sizes[slot_id] = len(support_ids)
                normalized_slot_bindings[slot_id] = {
                    "from": list(current_ids),
                    "to": list(support_ids),
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
            _fact_id, support_ids, _value_key = resolved
            if len(support_ids) < 2:
                return None
            replacements[slot_id] = support_ids
            consensus_sizes[slot_id] = len(support_ids)

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

    @classmethod
    def _repair_missing_binding(
        cls,
        query: str,
        plan: SupervisorPlan,
        facts: tuple[Mapping[str, Any], ...],
        run: BinderRun,
    ) -> tuple[BinderRun, BoundEvidenceSemanticCheck, dict[str, Any]] | None:
        """Recover a fully missing provider result only from packet consensus.

        This is intentionally narrower than ordinary retry. It is available
        only for direct facts and calculations when the provider returned a
        structurally valid complete missing response, every required slot has
        one exact, provenance-complete packet value with no competing structured
        value, and the reconstructed binding passes both contract validation
        and query-to-evidence semantic alignment. Multiple R4 views of one
        physical table do not fabricate corroboration, but a unique exact fact
        remains usable evidence. This never reads answer text,
        conversation history, or facts outside the current R4 packet.
        """

        if plan.intent not in {Intent.DIRECT_FACT, Intent.CALCULATION}:
            return None
        if not run.schema_valid:
            return None
        expected_slot_ids = {slot.slot_id for slot in plan.required_slots}
        missing_slot_ids = {
            str(slot_id) for slot_id in _stable_unique(run.binding.missing_slots)
        }
        existing_bindings = {
            str(slot_id): tuple(str(fact_id) for fact_id in fact_ids)
            for slot_id, fact_ids in run.binding.slot_bindings.items()
            if fact_ids
        }
        if (
            not expected_slot_ids
            or missing_slot_ids != expected_slot_ids
            or existing_bindings
            or run.binding.ambiguous_slots
        ):
            return None
        try:
            frame = extract_query_semantic_frame(query)
        except (TypeError, ValueError):
            return None
        if not frame.entity_ids:
            return None

        replacements: dict[str, tuple[str, ...]] = {}
        consensus_sizes: dict[str, int] = {}
        for slot in plan.required_slots:
            resolved = cls._consensus_fact_for_slot(
                query,
                frame,
                slot,
                facts,
                require_explicit_source=True,
            )
            if resolved is None:
                return None
            _fact_id, support_ids, _value_key = resolved
            replacements[slot.slot_id] = support_ids
            consensus_sizes[slot.slot_id] = len(support_ids)

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
                "strategy": "deterministic_missing_packet_consensus",
                "original_missing_slots": list(run.binding.missing_slots),
                "replaced_slot_bindings": {
                    slot_id: {"to": list(fact_ids)}
                    for slot_id, fact_ids in replacements.items()
                },
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
        if status == BindingStatus.MISSING.value:
            repaired = self._repair_missing_binding(
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
            # A provider that binds one of several *disagreeing* admissible
            # candidates has answered "which single fact can be bound", not
            # "which value is correct".  Those are different claims, and the
            # runtime must not accept the first as the second.
            #
            # ``_consensus_fact_for_slot`` already draws exactly the right
            # distinctions -- an exact duplicate is counted once, corroborating
            # sources agree, a strict majority wins -- and returns ``None`` when
            # there is no safe consensus.  What was missing was this caller
            # treating that ``None`` as "no repair found, proceed" instead of
            # "this slot is unresolved".  The three readiness cases that
            # released a confident answer from contradicting evidence all came
            # through here.
            conflicted = self._unresolved_conflict_slots(
                state.normalized_query,
                plan,
                facts,
            )
            if conflicted:
                self.last_bound_evidence_ids = ()
                self.last_bound_slot_bindings = {}
                self.last_citation_ids = ()
                evaluation = EvidenceEvaluationV1(
                    decision=EvidenceDecision.UNRESOLVED_CONFLICT,
                    reason_codes=(ReasonCode.EVIDENCE_CONFLICT,),
                    requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                    supported_slots=(),
                    missing_slots=tuple(slot.slot_id for slot in plan.required_slots),
                    supporting_evidence_ids=(),
                    temporal_status="CONFLICTED",
                    conflicts=tuple({"slot_id": slot_id} for slot_id in conflicted),
                    calculation_ready=False,
                )
                # Record the round before returning.  Every other exit from this
                # method writes a binder round, and a fail-closed path that is
                # invisible in the trace is the one path nobody can diagnose.
                self._trace.append(
                    {
                        "round": self.calls - 1,
                        "status": status,
                        "bound_slot_ids": [],
                        "missing_slot_ids": [
                            slot.slot_id for slot in plan.required_slots
                        ],
                        "ambiguous_slot_ids": list(conflicted),
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
                if fact_ids
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
                    str(slot_id)
                    for slot_id, fact_ids in run.binding.slot_bindings.items()
                    if fact_ids
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
                if fact_ids
            }
            self.last_citation_ids = ()
            evaluation = EvidenceEvaluationV1(
                decision=EvidenceDecision.REPAIRABLE,
                reason_codes=(reason,),
                requested_slots=tuple(slot.slot_id for slot in plan.required_slots),
                supported_slots=tuple(
                    str(slot_id)
                    for slot_id, fact_ids in run.binding.slot_bindings.items()
                    if fact_ids
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
                    str(slot_id)
                    for slot_id, fact_ids in run.binding.slot_bindings.items()
                    if fact_ids
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

        bound_slots = [
            str(slot_id)
            for slot_id, fact_ids in run.binding.slot_bindings.items()
            if fact_ids
        ]
        self._trace.append(
            {
                "round": self.calls - 1,
                "status": status,
                "bound_slot_ids": bound_slots,
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
