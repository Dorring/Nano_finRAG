"""Financial calculation domain objects.

These types form the typed boundary between the RAG orchestrator and the
deterministic calculation pipeline introduced in Phase 3. They are
deliberately dependency-free (stdlib + ``decimal`` only) so that the
``domain`` layer does not import from ``finance``, ``application``, or
``services``.

Dependency direction: ``domain -> finance -> application -> services``.

Key invariants enforced by these types:
- Every ``CalculationOperand`` MUST cite ``source_text`` and
  ``evidence_chunk_id`` so the calculation is auditable end-to-end.
- Every formula carries a ``formula_version`` string for traceability across
  releases.
- ``CalculationStatus`` drives the orchestrator's LLM-bypass decision:
  - ``EXECUTED``  -> skip LLM, return deterministic answer.
  - ``BLOCKED``   -> skip LLM, return deterministic refusal.
  - ``FAILED``    -> skip LLM, return safe failure (never fall back to
    LLM to avoid reintroducing numeric hallucinations).
  - ``NOT_APPLICABLE`` -> continue normal RAG flow (no calculation attempted).
  - ``READY``     -> transient state between plan builder and executor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from decimal import Decimal
from enum import Enum
from typing import Any


class CalculationOperation(str, Enum):
    """The set of deterministic financial operations supported in Phase 3 v1.

    ``ROE`` and ``CAGR`` are deliberately excluded from v1 because they
    require additional evidence disambiguation (average equity / multi-period
    compounding) that is out of scope for this phase.
    """

    DIFFERENCE = "difference"
    GROWTH_RATE = "growth_rate"
    PERCENTAGE_SHARE = "percentage_share"
    SUM = "sum"
    AVERAGE = "average"
    GROSS_MARGIN = "gross_margin"
    NET_MARGIN = "net_margin"
    DEBT_RATIO = "debt_ratio"
    SCALE_CONVERSION = "scale_conversion"

    #: Relational operations.  Their answer is a *relation between operands*
    #: rather than a quantity: "Visa is larger than Apple", or "Visa > Apple >
    #: Tesla".  ``value`` is only a projection of that (the sign of a
    #: difference, or ``None``), and a consumer asking which operand won must
    #: read ``relation`` / ``ordering_groups`` rather than infer it from a
    #: number and a role order.
    #:
    #: They live in this enum because whether something can be executed
    #: deterministically is a property of the operation, not of the plan's
    #: intent.  Cross-entity questions used to be planned with no operation at
    #: all, so their answers were prose no validator could check.
    COMPARISON = "comparison"
    RANKING = "ranking"


class ComparisonRelation(str, Enum):
    """Which way a two-operand comparison resolved.

    Deliberately not the sign in ``CalculationResult.value``.  ``-1`` is a
    number; this is a claim about two named things.  A validator checking
    whether an answer said "Visa" needs the relation, and deriving it from the
    sign would also require knowing which operand was the left one -- two
    authorities for one fact.
    """

    LHS_GT_RHS = "lhs_gt_rhs"
    RHS_GT_LHS = "rhs_gt_lhs"
    EQUAL = "equal"


#: Operations whose answer is a relation between operands rather than a
#: quantity.  Named here so `relational_result_is_well_formed` and the identity
#: digest cannot disagree about which those are -- a second list would be a
#: second authority for the same fact.
RELATIONAL_OPERATIONS = frozenset(
    {CalculationOperation.COMPARISON, CalculationOperation.RANKING}
)


class CalculationStatus(str, Enum):
    """Lifecycle status of a calculation attempt.

    The orchestrator inspects this to decide whether to bypass the LLM.
    """

    NOT_APPLICABLE = "not_applicable"
    READY = "ready"
    EXECUTED = "executed"
    BLOCKED = "blocked"
    FAILED = "failed"


def _safe_excerpt(text: str, max_chars: int = 240) -> str:
    """Truncate text to ``max_chars`` with an ellipsis indicator.

    Used by public serialization to avoid leaking full evidence text into
    API responses while still providing enough context for the user to
    locate the source.
    """
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


@dataclass(frozen=True)
class CalculationOperand:
    """A single numeric input bound to retrieved evidence.

    Every operand MUST cite the exact ``source_text`` substring from a
    retrieved ``EvidenceItem`` so the calculation is auditable. The
    ``evidence_chunk_id`` ties back to the retrieval pipeline's evidence set;
    ``document_name`` and ``page`` are denormalized for display without
    requiring a re-lookup of the evidence item.
    """

    name: str
    value: Decimal
    unit: str | None = None
    scale: str | None = None
    source_text: str = ""
    evidence_chunk_id: str = ""
    document_name: str | None = None
    page: int | None = None
    #: The ``RequiredSlot.slot_id`` this operand was built for.  This is the
    #: operand's *semantic* identity, and it is what a relational result refers
    #: to -- never ``evidence_chunk_id``.  One canonical fact may be supported
    #: by several independent evidence rows, so an ordering keyed on evidence
    #: would change when a different support was bound even though the ranking
    #: it describes had not.  Appended with a default rather than placed beside
    #: ``name`` so existing positional construction keeps working.
    slot_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for internal / trace use.

        Includes the full ``source_text``. This is for internal diagnostics
        only; public API responses must use ``to_public_dict`` instead.
        """
        return {
            "name": self.name,
            "value": str(self.value),
            "unit": self.unit,
            "scale": self.scale,
            "source_text": self.source_text,
            "evidence_chunk_id": self.evidence_chunk_id,
            "document_name": self.document_name,
            "page": self.page,
            "slot_id": self.slot_id,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize to a dict safe for public API responses.

        Excludes the full ``source_text``; replaces it with a truncated
        ``evidence_excerpt`` (max 240 chars) so the user can locate the
        source without exposing the entire retrieved chunk.
        """
        return {
            "name": self.name,
            "value": str(self.value),
            "unit": self.unit,
            "scale": self.scale,
            "evidence_chunk_id": self.evidence_chunk_id,
            "document_name": self.document_name,
            "page": self.page,
            "evidence_excerpt": _safe_excerpt(self.source_text),
            "slot_id": self.slot_id,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize to a compact dict for trace logging.

        Excludes ``source_text`` entirely to minimize PII / content leakage
        into trace storage. Only structural metadata is retained.
        """
        return {
            "name": self.name,
            "value": str(self.value),
            "unit": self.unit,
            "scale": self.scale,
            "evidence_chunk_id": self.evidence_chunk_id,
            "document_name": self.document_name,
            "page": self.page,
        }


@dataclass(frozen=True)
class CalculationPlan:
    """An immutable plan describing a single deterministic calculation.

    The ``formula_version`` pins the exact formula used so results are
    reproducible and auditable across releases (e.g. ``"gross_margin.v1"``).
    ``precision`` controls the number of decimal places in the result.
    """

    operation: CalculationOperation
    operands: tuple[CalculationOperand, ...]
    formula_version: str
    target_metric: str
    precision: int = 4
    label: str | None = None
    source_scale: str | None = None
    target_scale: str | None = None
    status: CalculationStatus = CalculationStatus.READY
    block_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "operands": [op.to_dict() for op in self.operands],
            "formula_version": self.formula_version,
            "target_metric": self.target_metric,
            "precision": self.precision,
            "label": self.label,
            "source_scale": self.source_scale,
            "target_scale": self.target_scale,
            "status": self.status.value,
            "block_reason": self.block_reason,
        }


@dataclass(frozen=True)
class CalculationResult:
    """The outcome of executing (or attempting) a ``CalculationPlan``.

       - ``EXECUTED``  -> ``value`` is populated; orchestrator bypasses LLM.
       - ``BLOCKED``   -> plan could not be built or operands are insufficient;
         orchestrator bypasses LLM and returns a deterministic refusal.
       - ``FAILED``    -> plan was built but execution raised an error;
         orchestrator bypasses the LLM and returns a safe failure message.
    The internal error/stack is never exposed to the user.
       - ``NOT_APPLICABLE`` -> question was not a calculation; orchestrator
         continues the normal RAG flow.
       - ``READY``     -> transient; only used between plan builder and executor.
    """

    status: CalculationStatus
    operation: CalculationOperation | None = None
    value: Decimal | None = None
    unit: str | None = None
    formula: str | None = None
    formula_version: str | None = None
    target_metric: str | None = None
    operands: tuple[CalculationOperand, ...] = ()
    #: How a ``COMPARISON`` resolved.  Present because ``value`` alone is a
    #: projection: ``-1`` does not say which operand was larger without also
    #: consulting the operand order, and one fact should have one authority.
    relation: ComparisonRelation | None = None
    #: The ``RANKING`` result, as groups of equal operands in descending order::
    #:
    #:     (("s2",), ("s1", "s3"))     s2 > s1 = s3
    #:
    #: Groups are used from the start rather than a flat ordering, so a tie does
    #: not need a schema migration to express.  The refs are ``slot_id`` values
    #: -- semantic operands, never evidence ids: several evidence rows may
    #: support one canonical fact, and a result keyed on which support happened
    #: to be bound would change while the ranking it describes did not.
    #:
    #: ``operands`` keeps its provenance meaning and its order carries none.
    ordering_groups: tuple[tuple[str, ...], ...] | None = None
    error_code: str | None = None
    error_message: str | None = None

    @property
    def is_admissible(self) -> bool:
        """Whether this is a usable financial result, not merely a return value.

        ``EXECUTED`` is the only status that means the calculation produced a
        value that may be reasoned over.  ``BLOCKED`` and ``FAILED`` are
        successful *invocations* -- the calculator ran and reported honestly --
        that produced no admissible result, and ``NOT_APPLICABLE``/``READY`` are
        not results at all.

        Keeping this distinct from "the capability returned" is the point: a
        function returning normally says nothing about whether what it returned
        may be used.
        """

        return self.status is CalculationStatus.EXECUTED

    @property
    def relational_result_is_well_formed(self) -> bool:
        """Whether a relational result's structure can be used at all.

        One definition of a usable ordering, here rather than in each consumer.
        The rules are the same for both relational operations:

        * every operand carries a ``slot_id``, because an operand that cannot
          be referred to cannot appear in an ordering that is checkable;
        * ``COMPARISON`` states a ``relation`` over exactly two operands;
        * ``RANKING`` states ``ordering_groups`` naming every required operand
          **exactly once** and no ref it was not built from.

        ``sorted`` rather than a set comparison on purpose: a repeated ref is a
        different multiset from the required one, so one rule catches a
        duplicate and a missing ref together, and a set would hide the
        repetition.

        This addresses the *structure* only.  Whether the refs are in the right
        order is the executor's business, and whether the answer said so is the
        validator's.
        """

        if not self.is_admissible:
            return False

        required = [operand.slot_id for operand in self.operands]
        if not required or any(not ref for ref in required):
            return False

        if self.operation is CalculationOperation.COMPARISON:
            return self.relation is not None and len(required) == 2

        if self.operation is CalculationOperation.RANKING:
            groups = self.ordering_groups
            if not groups or any(not group for group in groups):
                return False
            refs = [ref for group in groups for ref in group]
            return sorted(refs) == sorted(required)

        return False

    @property
    def calculation_id(self) -> str | None:
        """This result's identity, or ``None`` when it is not admissible.

        The identity belongs to the result rather than living beside it, so
        there is one truth and it cannot disagree with the status.  A caller
        cannot obtain an id for a BLOCKED or FAILED result, because there is no
        id to obtain -- which is what makes admissibility a property of the
        contract rather than a check every caller has to remember.

        The digest covers only the arithmetic and its operand provenance, so it
        is stable across runs for the same inputs.
        """

        if not self.is_admissible or self.operation is None:
            return None
        payload: dict[str, Any] = {
            "operation": self.operation.value,
            "formula_version": self.formula_version,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "operands": self._identity_operands(),
        }
        if self.operation in RELATIONAL_OPERATIONS:
            # The relation *is* the answer, so it belongs to the identity; for a
            # quantity it would be redundant with `value`.
            payload["relation"] = self.relation.value if self.relation else None
            payload["ordering_groups"] = self._ordering_groups_payload()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"C1-{digest}"

    def _identity_operands(self) -> list[dict[str, Any]]:
        """The operands as the result's identity sees them.

        A quantity's identity is tied to the evidence it was read from, so the
        existing digest keeps ``evidence_chunk_id`` and operand order.

        A relational result's identity is its *semantics*: which required
        operands stood in what relation.  Three consequences, all deliberate:

        * the ref is ``slot_id``, because one canonical fact may be supported by
          several evidence rows and binding a different support must not make
          the same ranking a different result;
        * the list is sorted by ``slot_id``, so the order the operands happened
          to be built in cannot change the identity -- ``operands`` is
          provenance, and provenance order is not a claim;
        * ``name`` is dropped for the same reason as the evidence id: it is a
          role label from the binding, not part of what was ranked.
        """

        if self.operation in RELATIONAL_OPERATIONS:
            return [
                {"slot_id": operand.slot_id, "value": str(operand.value)}
                for operand in sorted(self.operands, key=lambda item: item.slot_id)
            ]
        return [
            {
                "name": operand.name,
                "value": str(operand.value),
                "evidence_chunk_id": operand.evidence_chunk_id,
            }
            for operand in self.operands
        ]

    def _ordering_groups_payload(self) -> list[list[str]] | None:
        """A JSON-safe form of the ordering, or ``None`` when there is none.

        Tuples become lists at the serialization boundary; the in-memory type
        stays a tuple so a result cannot be mutated through a reference it
        handed out.
        """

        if self.ordering_groups is None:
            return None
        return [list(group) for group in self.ordering_groups]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for internal diagnostics.

        Includes the full ``error_message`` and operand ``source_text``.
        This is for internal use only; public API responses must use
        ``to_public_dict`` and trace logging must use ``to_trace_dict``.
        """
        payload: dict[str, Any] = {
            "status": self.status.value,
            "operation": self.operation.value if self.operation else None,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "formula": self.formula,
            "formula_version": self.formula_version,
            "target_metric": self.target_metric,
            "operands": [op.to_dict() for op in self.operands],
            "relation": self.relation.value if self.relation else None,
            "ordering_groups": self._ordering_groups_payload(),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }
        return payload

    def to_public_dict(self) -> dict[str, Any]:
        """Serialize to a dict safe for public API responses.

        Key differences from ``to_dict``:
        - ``error_message`` is NEVER included. The internal exception text
          (e.g. ``str(exc)``) must not leak to the user. The frontend maps
          ``error_code`` to a user-visible message.
        - Operand ``source_text`` is replaced by ``evidence_excerpt``
          (max 240 chars) via ``CalculationOperand.to_public_dict``.
        """
        payload: dict[str, Any] = {
            "status": self.status.value,
            "operation": self.operation.value if self.operation else None,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "formula": self.formula,
            "formula_version": self.formula_version,
            "target_metric": self.target_metric,
            "operands": [op.to_public_dict() for op in self.operands],
            "relation": self.relation.value if self.relation else None,
            "ordering_groups": self._ordering_groups_payload(),
            "error_code": self.error_code,
        }
        return payload

    def to_trace_dict(self) -> dict[str, Any]:
        """Serialize to a compact dict for trace logging.

        Excludes ``error_message`` and operand ``source_text`` to minimize
        content leakage into trace storage. Only structural metadata needed
        for debugging is retained.
        """
        payload: dict[str, Any] = {
            "status": self.status.value,
            "operation": self.operation.value if self.operation else None,
            "value": str(self.value) if self.value is not None else None,
            "unit": self.unit,
            "formula_version": self.formula_version,
            "target_metric": self.target_metric,
            "operand_count": len(self.operands),
            "operands": [op.to_trace_dict() for op in self.operands],
            "relation": self.relation.value if self.relation else None,
            "ordering_groups": self._ordering_groups_payload(),
            "error_code": self.error_code,
        }
        return payload


NOT_APPLICABLE_RESULT = CalculationResult(status=CalculationStatus.NOT_APPLICABLE)
"""Sentinel returned by the pipeline when the question is not a calculation."""
