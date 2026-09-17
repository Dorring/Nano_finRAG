"""Serializable contracts for the NF-V2-16 bounded adaptive RAG experiment.

The contracts are deliberately independent of a model provider.  They carry
only the structured evidence state needed by a supervisor/control plane; a
Financial Specialist is never asked to decide evidence sufficiency.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from rag_v2.contracts.financial_semantics import quantity_identity, text_identity


#: What the evidence says.  Deliberately not the instance id, the citation,
#: the document, the page, the source or the metadata bag -- see
#: ``EvidencePacketV1.content_fingerprint``.
CONTENT_IDENTITY_FIELDS: tuple[str, ...] = (
    "metric",
    "period",
    "entity",
    "scope",
    "value",
    "unit",
    "currency",
    "scale",
)

#: Fields that are one quantity, folded as a quantity.
#:
#: These are deliberately not folded one at a time as text.  A magnitude is not
#: a label to be compared with other labels: ``1000 million`` and ``1 billion``
#: are the same quantity written two ways, and a field-by-field text comparison
#: can only ever see two different strings.  They go through the shared
#: financial semantics together, which is also what conflict consensus uses --
#: so the two cannot disagree about whether two sources state the same figure.
QUANTITY_IDENTITY_FIELDS: tuple[str, ...] = ("value", "unit", "currency", "scale")

#: Fields that are text, folded as text.  Derived rather than listed, so the two
#: halves cannot stop partitioning :data:`CONTENT_IDENTITY_FIELDS` -- a field
#: dropped from both would leave the fingerprint quietly ignoring it.
TEXT_IDENTITY_FIELDS: tuple[str, ...] = tuple(
    field for field in CONTENT_IDENTITY_FIELDS if field not in QUANTITY_IDENTITY_FIELDS
)


def _canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(_canon(value).encode("utf-8")).hexdigest()


class AdaptivePhase(str, Enum):
    PLAN = "PLAN"
    ACT = "ACT"
    OBSERVE = "OBSERVE"
    EVALUATE = "EVALUATE"
    REPLAN = "REPLAN"
    CALCULATE = "CALCULATE"
    READY_TO_GENERATE = "READY_TO_GENERATE"
    GENERATE = "GENERATE"
    VERIFY = "VERIFY"
    REPAIR = "REPAIR"
    RELEASE = "RELEASE"
    FAIL_CLOSED = "FAIL_CLOSED"


class EvidenceDecision(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    REPAIRABLE = "REPAIRABLE"
    TERMINAL_INSUFFICIENT = "TERMINAL_INSUFFICIENT"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"


class ReasonCode(str, Enum):
    QUERY_PLAN_SEMANTIC_MISMATCH = "QUERY_PLAN_SEMANTIC_MISMATCH"
    QUERY_PLAN_SEMANTIC_AMBIGUOUS = "QUERY_PLAN_SEMANTIC_AMBIGUOUS"
    QUERY_PLAN_SEMANTIC_UNKNOWN = "QUERY_PLAN_SEMANTIC_UNKNOWN"
    QUERY_EVIDENCE_SEMANTIC_MISMATCH = "QUERY_EVIDENCE_SEMANTIC_MISMATCH"
    MISSING_SLOT = "MISSING_SLOT"
    MISSING_OPERAND = "MISSING_OPERAND"
    WRONG_PERIOD = "WRONG_PERIOD"
    AMBIGUOUS_PERIOD = "AMBIGUOUS_PERIOD"
    WRONG_ENTITY_SCOPE = "WRONG_ENTITY_SCOPE"
    LOW_EVIDENCE_COVERAGE = "LOW_EVIDENCE_COVERAGE"
    EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
    VERSION_CONFLICT = "VERSION_CONFLICT"
    MISSING_AUTHORITATIVE_VERSION = "MISSING_AUTHORITATIVE_VERSION"
    LOW_SOURCE_QUALITY = "LOW_SOURCE_QUALITY"
    NO_PROGRESS = "NO_PROGRESS"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    TOOL_ERROR = "TOOL_ERROR"
    UNSUPPORTED_TOOL_ROUTE = "UNSUPPORTED_TOOL_ROUTE"
    STRUCTURAL_NOT_READY = "STRUCTURAL_NOT_READY"
    CALCULATOR_NOT_WIRED = "CALCULATOR_NOT_WIRED"
    CALCULATION_ERROR = "CALCULATION_ERROR"
    GENERATOR_NOT_WIRED = "GENERATOR_NOT_WIRED"
    GENERATION_ERROR = "GENERATION_ERROR"
    VERIFICATION_ERROR = "VERIFICATION_ERROR"
    VERIFICATION_NOT_WIRED = "VERIFICATION_NOT_WIRED"


class PeriodSemantics(str, Enum):
    INSTANT = "INSTANT"
    QUARTER = "QUARTER"
    YTD = "YTD"
    ANNUAL = "ANNUAL"
    UNKNOWN = "UNKNOWN"


class TemporalRelation(str, Enum):
    SAME_FACT_SCOPE = "SAME_FACT_SCOPE"
    DIFFERENT_PERIOD = "DIFFERENT_PERIOD"
    DIFFERENT_SOURCE_SCOPE = "DIFFERENT_SOURCE_SCOPE"
    TEMPORAL_SUCCESSION = "TEMPORAL_SUCCESSION"
    VERSION_SUPERSEDED = "VERSION_SUPERSEDED"
    AMBIGUOUS_SCOPE = "AMBIGUOUS_SCOPE"


class ConsistencyDecision(str, Enum):
    CONSISTENT = "CONSISTENT"
    TEMPORAL_SUCCESSION = "TEMPORAL_SUCCESSION"
    MULTI_SOURCE_COMPATIBLE = "MULTI_SOURCE_COMPATIBLE"
    SUPERSEDED = "SUPERSEDED"
    UNRESOLVED_CONFLICT = "UNRESOLVED_CONFLICT"
    AMBIGUOUS = "AMBIGUOUS"


class ToolCapability(str, Enum):
    LEXICAL_RETRIEVAL = "LEXICAL_RETRIEVAL"
    SEMANTIC_RETRIEVAL = "SEMANTIC_RETRIEVAL"
    STRUCTURED_FINANCIAL_LOOKUP = "STRUCTURED_FINANCIAL_LOOKUP"
    DOCUMENT_METADATA_LOOKUP = "DOCUMENT_METADATA_LOOKUP"
    EXTERNAL_WEB_SEARCH = "EXTERNAL_WEB_SEARCH"
    CALCULATION = "CALCULATION"


@dataclass(frozen=True)
class TemporalEvidenceV1:
    entity: str | None = None
    document_id: str | None = None
    document_type: str | None = None
    fiscal_year: str | None = None
    fiscal_quarter: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_semantics: PeriodSemantics = PeriodSemantics.UNKNOWN
    report_date: str | None = None
    filing_date: str | None = None
    version: str | None = None
    is_amended: bool = False
    supersedes_document_id: str | None = None
    source: str | None = None
    scope: str | None = None
    metric: str | None = None
    value: str | None = None
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    created_at: str | None = None
    ingested_at: str | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "TemporalEvidenceV1":
        raw = raw or {}
        period_semantics = raw.get("period_semantics", PeriodSemantics.UNKNOWN)
        try:
            period_semantics = PeriodSemantics(period_semantics)
        except (TypeError, ValueError):
            period_semantics = PeriodSemantics.UNKNOWN
        values = {
            key: raw.get(key)
            for key in cls.__dataclass_fields__
            if key != "period_semantics"
        }
        values["period_semantics"] = period_semantics
        values["is_amended"] = bool(raw.get("is_amended", False))
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["period_semantics"] = self.period_semantics.value
        return payload


def _coerce_page(value: Any) -> Any:
    """Preserve a page value verbatim, without inventing one.

    Integers stay integers, including ``0`` -- truthiness would silently drop a
    legitimate first page.  An absent page stays absent: provenance that is
    missing must read as missing, not as page 0 or 1.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


@dataclass(frozen=True)
class EvidencePacketV1:
    evidence_id: str
    metric: str | None = None
    value: str | None = None
    period: str | None = None
    entity: str | None = None
    scope: str | None = None
    unit: str | None = None
    currency: str | None = None
    scale: str | None = None
    citation_id: str | None = None
    source: str | None = None
    document_id: str | None = None
    #: Source location as the extractor recorded it.  A first-class field rather
    #: than a key inside ``metadata``: it used to be demoted there, and the
    #: citation projection reads the top level, so every public citation lost
    #: its page.  Promoting it also re-enables the calculation renderer's
    #: long-standing ``doc-X, p.N`` suffix, which had never once rendered.
    #:
    #: Semantics are the extractor's, unchanged and unrenumbered: a physical
    #: page value, typed, with ``0`` valid and absence preserved as absence.
    page: int | None = None
    slots: tuple[str, ...] = ()
    operands: tuple[str, ...] = ()
    calculation_result: str | None = None
    temporal: TemporalEvidenceV1 = field(default_factory=TemporalEvidenceV1)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "EvidencePacketV1":
        temporal_raw = raw.get("temporal")
        if temporal_raw is None:
            temporal_raw = raw
        slots = raw.get("slots", raw.get("slot_ids", ())) or ()
        operands = raw.get("operands", ()) or ()
        consumed = {
            "evidence_id", "id", "metric", "value", "period", "entity", "scope",
            "unit", "currency", "scale", "citation_id", "source", "document_id",
            "slots", "slot_ids", "operands", "calculation_result", "temporal",
        }
        return cls(
            evidence_id=str(raw.get("evidence_id", raw.get("id", ""))),
            metric=raw.get("metric"), value=None if raw.get("value") is None else str(raw.get("value")),
            period=raw.get("period"), entity=raw.get("entity"), scope=raw.get("scope"),
            unit=raw.get("unit"), currency=raw.get("currency"), scale=raw.get("scale"),
            citation_id=raw.get("citation_id"), source=raw.get("source"),
            document_id=raw.get("document_id"),
            # ``pdf_page`` is the extractor's name and ``page`` the fact store's
            # canonical alias; accept either and keep the value's type.
            # Accept either name, and treat an explicit ``None`` as absent --
            # ``dict.get(k, default)`` returns the default only when the key is
            # *missing*, so a producer emitting both with ``page=None`` would
            # silently discard a present ``pdf_page``.
            page=_coerce_page(
                raw["page"] if raw.get("page") is not None else raw.get("pdf_page")
            ),
            slots=tuple(str(x) for x in slots),
            operands=tuple(str(x) for x in operands),
            calculation_result=None if raw.get("calculation_result") is None else str(raw.get("calculation_result")),
            temporal=TemporalEvidenceV1.from_mapping(temporal_raw),
            metadata={key: value for key, value in raw.items() if key not in consumed},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id, "metric": self.metric, "value": self.value,
            "period": self.period, "entity": self.entity, "scope": self.scope,
            "unit": self.unit, "currency": self.currency, "scale": self.scale,
            "citation_id": self.citation_id, "source": self.source,
            "document_id": self.document_id, "page": self.page,
            "slots": list(self.slots),
            "operands": list(self.operands), "calculation_result": self.calculation_result,
            "temporal": self.temporal.to_dict(), "metadata": dict(self.metadata),
        }

    @property
    def content_fingerprint(self) -> str:
        """Identity of what this evidence *says*, not of which object it is.

        Excludes ``evidence_id``, ``citation_id``, ``document_id``, ``source``,
        ``page`` and ``metadata``.  Those are instance identity and provenance:
        two documents reporting the same figure are the same content and
        different provenance, and a fingerprint that merged them would make
        corroborating sources look like one source.

        The previous version hashed ``to_dict()``, which includes the evidence
        id, so it was a *serialized-record* hash: identical content under a new
        id hashed differently, and the progress detector read that as new
        information.

        The version before this one folded every field through one text helper
        that removed commas.  That is wrong in both directions and could not be
        made right, because the helper had no way to know which kind of field it
        held: for a metric, ``Revenue, net`` and ``Revenue net`` are different
        claims and were merged; for a value, ``1,5`` and ``15`` are different
        numbers and were merged too.  So the two kinds are separated here.  Text
        fields are folded as text; the numeric fields are handed to the shared
        financial semantics as the one quantity they describe, which is what
        makes ``1000 million USD`` and ``1 billion USD`` fingerprint alike.
        """

        payload = {
            field: text_identity(self._canonical(field))
            for field in TEXT_IDENTITY_FIELDS
        }
        payload["quantity"] = quantity_identity(
            self._canonical("value"),
            scale=self._canonical("scale"),
            unit=self._canonical("unit"),
            currency=self._canonical("currency"),
        )
        return stable_hash(payload)

    def _canonical(self, field: str) -> Any:
        """Prefer a normalised form when the packet carries one."""

        normalized = self.metadata.get(f"normalized_{field}")
        if normalized is not None and str(normalized).strip():
            return normalized
        return getattr(self, field, None)


@dataclass(frozen=True)
class ReplanActionV1:
    capability: ToolCapability
    query: str
    reason_code: ReasonCode
    target_slots: tuple[str, ...] = ()
    constraints: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability.value, "query": self.query,
            "reason_code": self.reason_code.value, "target_slots": list(self.target_slots),
            "constraints": dict(self.constraints),
        }


@dataclass(frozen=True)
class EvidenceEvaluationV1:
    decision: EvidenceDecision
    reason_codes: tuple[ReasonCode, ...] = ()
    requested_slots: tuple[str, ...] = ()
    supported_slots: tuple[str, ...] = ()
    missing_slots: tuple[str, ...] = ()
    supporting_evidence_ids: tuple[str, ...] = ()
    temporal_status: str = "UNKNOWN"
    conflicts: tuple[Mapping[str, Any], ...] = ()
    calculation_ready: bool = False
    recommended_action: ReplanActionV1 | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reason_codes": [item.value for item in self.reason_codes],
            "requested_slots": list(self.requested_slots),
            "supported_slots": list(self.supported_slots),
            "missing_slots": list(self.missing_slots),
            "supporting_evidence_ids": list(self.supporting_evidence_ids),
            "temporal_status": self.temporal_status,
            "conflicts": [dict(item) for item in self.conflicts],
            "calculation_ready": self.calculation_ready,
            "recommended_action": self.recommended_action.to_dict() if self.recommended_action else None,
        }


@dataclass
class AdaptiveRAGStateV1:
    request_id: str
    original_query: str
    normalized_query: str
    intent: str
    task_type: str
    plan: dict[str, Any]
    current_step: str = "PLAN"
    required_slots: list[dict[str, Any]] = field(default_factory=list)
    filled_slots: dict[str, list[str]] = field(default_factory=dict)
    missing_slots: list[str] = field(default_factory=list)
    evidence_packets: list[dict[str, Any]] = field(default_factory=list)
    evidence_hashes: list[str] = field(default_factory=list)
    scope_groups: dict[str, list[str]] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    calculation_requirements: dict[str, Any] = field(default_factory=dict)
    calculation_ready: bool = False
    # Set once the CALCULATE phase has run.  Its job is coupling: the coordinator
    # has two call sites that reach the candidate stage, and both must agree on
    # whether the calculation already happened inside the loop, so that the
    # result is validated rather than re-run.  It is *not* a guard against
    # recalculating forever -- EVALUATE cannot be re-entered after CALCULATE
    # (CALCULATE -> READY_TO_GENERATE -> GENERATE -> VERIFY -> RELEASE|REPAIR),
    # so there is no such cycle to prevent.
    calculation_attempted: bool = False
    # TV2-04 downstream fields are populated only after Binder admission.
    # They remain structured state, never answer-text-derived facts.
    bound_evidence_ids: list[str] = field(default_factory=list)
    bound_slot_bindings: dict[str, list[str]] = field(default_factory=dict)
    calculation_result: dict[str, Any] | None = None
    generation_route: str | None = None
    route_reason: str | None = None
    candidate_answer: str | None = None
    candidate_generation_id: str | None = None
    candidate_status: str | None = None
    validation_pending: bool = False
    tool_history: list[dict[str, Any]] = field(default_factory=list)
    query_history: list[str] = field(default_factory=list)
    iteration: int = 0
    tool_calls: int = 0
    replan_rounds: int = 0
    same_tool_retries: dict[str, int] = field(default_factory=dict)
    last_action: dict[str, Any] | None = None
    last_observation: dict[str, Any] | None = None
    progress_signatures: list[str] = field(default_factory=list)
    status: str = "PLAN"
    stop_reason: str | None = None
    transitions: list[dict[str, Any]] = field(default_factory=list)
    # One record per action the controller actually took, in order.  Unlike
    # ``transitions`` (which logs every phase spin, including no-op ones) this is
    # the run's decision trace: what was done, why, and what came back.
    turns: list[dict[str, Any]] = field(default_factory=list)

    @property
    def calculation_result_id(self) -> str | None:
        """The calculation's identity, derived from the result that owns it.

        H2A-2D-3B.  This was a durable field, assigned from
        ``CalculationResult.calculation_id`` at two write sites and read back as
        though it were a truth of its own -- a second stored copy that could
        drift from the result it described.  The result already holds the id and
        already lives on this state, so the field was a copy of something
        reachable from here.

        Reading through to the authority means there is nothing left to drift:
        an inadmissible result has no id to read, so ``BLOCKED`` and ``FAILED``
        cannot project one either, without the property needing to know that
        rule.  The name is unchanged, so every reader keeps working.

        The object is kept in ``__dict__`` rather than declared as a field
        because the calculator attaches it dynamically; ``getattr`` with a
        default keeps a state that never ran a calculation readable.
        """

        result = self.__dict__.get("_calculation_result_obj")
        return None if result is None else result.calculation_id

    @classmethod
    def new(
        cls,
        request_id: str,
        query: str,
        *,
        intent: str = "DIRECT_FACT",
        task_type: str = "DIRECT",
        required_slots: list[dict[str, Any]] | None = None,
        plan: Mapping[str, Any] | None = None,
        calculation_requirements: Mapping[str, Any] | None = None,
    ) -> "AdaptiveRAGStateV1":
        slots = list(required_slots or [])
        return cls(
            request_id=request_id, original_query=query,
            normalized_query=" ".join(query.split()), intent=intent,
            task_type=task_type, plan=dict(plan or {"initial_action": "retrieve"}),
            required_slots=slots,
            missing_slots=[str(slot.get("slot_id")) for slot in slots if slot.get("slot_id")],
            calculation_requirements=dict(calculation_requirements or {}),
        )

    def transition(self, to_state: AdaptivePhase, reason: str) -> None:
        self.transitions.append({"from": self.status, "to": to_state.value, "reason": reason})
        self.status = to_state.value
        self.current_step = to_state.value

    def record_turn(
        self,
        action: str,
        reason_code: str | None = None,
        *,
        query: str | None = None,
    ) -> dict[str, Any]:
        """Open a turn record for one action the controller is about to take."""

        record: dict[str, Any] = {
            "turn": len(self.turns) + 1,
            "action": str(action),
            "reason_code": reason_code,
            "query": query,
            "outcome": None,
        }
        self.turns.append(record)
        return record

    def observe_turn(self, outcome: Mapping[str, Any]) -> None:
        """Attach what the most recent turn produced, without opening a new one."""

        if self.turns:
            self.turns[-1]["outcome"] = dict(outcome)

    @property
    def action_trace(self) -> list[str]:
        """The ordered actions this run took."""

        return [str(record.get("action")) for record in self.turns]

    def add_evidence(self, packets: list[EvidencePacketV1]) -> None:
        by_id = {item.get("evidence_id"): item for item in self.evidence_packets}
        for packet in packets:
            by_id[packet.evidence_id] = packet.to_dict()
        self.evidence_packets = [by_id[key] for key in sorted(by_id)]
        # Semantic identity, not instance identity: this field has no reader today,
        # and if one appears it should see the same evidence the progress detector
        # does rather than one entry per retrieved row.
        self.evidence_hashes = [
            EvidencePacketV1.from_mapping(item).content_fingerprint
            for item in self.evidence_packets
        ]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
