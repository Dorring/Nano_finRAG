"""A3-W1: the period-binding contract, and the provenance it must not lose.

Types and enums only.  **Nothing imports this yet and no behaviour changes** -- that is the
point of W1: the contract lands, is tested, and the pipeline runs exactly as it did until
W2 starts populating it.  A contract commit that moves behaviour means introduction and
migration were mixed, and then a 47-slot regression cannot be attributed to either.

Three responsibilities, kept apart on purpose:

    PeriodBindingV2     owns WHEN
    TemporalKind        owns temporal semantics
    EmissionAdmission   owns whether the fact is complete enough to store

The separations this line of work already paid for:

    TemporalKind != EmissionAdmission
    Admission    != authority

`A3-1d` is why the first is written down: a fully traceable `YEAR(2025)` was being dropped
for the same reason as a table of junk, because temporal kind was deciding admission.
`A2B-19/20` is why the second is: `table_role` owns company-level authority and nothing
here may reach over it.

**Status and temporal kind are different enums and must stay so.**  A binding is
`UNRESOLVED` when no source evidence was found; a temporal kind is `UNKNOWN` when the
source stated no shape.  Coca-Cola's equity statement is a `PARTIAL` binding with an
`UNKNOWN` kind -- identity resolved, shape unstated -- and if both were spelled
`UNRESOLVED` the next reader would take it for a failure and drop it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PeriodBindingStatus(str, Enum):
    """How much of the period the source actually stated.

    `PARTIAL` is the one that did not exist before and is the reason this enum is not a
    boolean: `normalized_period` and `granularity` are resolved while the temporal shape
    is not, which is a usable binding and not a failure.
    """

    RESOLVED = "RESOLVED"
    PARTIAL = "PARTIAL"
    CONFLICT = "CONFLICT"
    UNRESOLVED = "UNRESOLVED"


class PeriodGranularity(str, Enum):
    """How fine the source's claim is.  A year does not pin a day."""

    DAY = "DAY"
    YEAR = "YEAR"
    UNKNOWN = "UNKNOWN"


class PeriodTargetScope(str, Enum):
    """What the binding covers, read off the source's own geometry."""

    COLUMN = "COLUMN"
    ROW = "ROW"
    CELL_GROUP = "CELL_GROUP"


class PeriodBindingMethod(str, Enum):
    """Where the period came from.

    This is provenance, **not** a ranking.  A `DIRECT DECLARATION` shadows inherited
    context because the target row states its own period -- scope semantics -- and for no
    other reason.  Nothing may order these values and pick a winner; two bindings of the
    same class that disagree are a `CONFLICT`.
    """

    #: A. DIRECT DECLARATION -- the target row states WHEN itself.
    INLINE_PERIOD_DATA_ROW = "INLINE_PERIOD_DATA_ROW"

    #: B. INHERITED CONTEXT -- carried down from a header or group above.
    DIRECT_HEADER = "DIRECT_HEADER"
    HEADER_ROW_SELECTION_EXTEND = "HEADER_ROW_SELECTION_EXTEND"
    ADJACENT_YEAR_JOIN = "ADJACENT_YEAR_JOIN"
    YEAR_ONLY_PERIOD = "YEAR_ONLY_PERIOD"


#: The two classes of period evidence.  Membership decides shadowing, and nothing else.
DIRECT_DECLARATIONS = frozenset({PeriodBindingMethod.INLINE_PERIOD_DATA_ROW})

INHERITED_CONTEXT = frozenset({
    PeriodBindingMethod.DIRECT_HEADER,
    PeriodBindingMethod.HEADER_ROW_SELECTION_EXTEND,
    PeriodBindingMethod.ADJACENT_YEAR_JOIN,
    PeriodBindingMethod.YEAR_ONLY_PERIOD,
})


class TemporalKind(str, Enum):
    """What the source says about the *shape* of the time.

    `UNKNOWN` means the source stated a period but no shape -- Coca-Cola's bare `2025`.
    It is deliberately not spelled `UNRESOLVED`, which this module reserves for a binding
    that found nothing at all.
    """

    POINT = "point"
    DURATION = "duration"
    COMPARISON = "comparison"
    SEGMENT = "segment"
    BUCKET = "bucket"
    CATEGORY = "category"
    NON_TEMPORAL = "non_temporal"
    YEAR = "YEAR"
    UNKNOWN = "UNKNOWN"


class TemporalKindMethod(str, Enum):
    """Why a temporal kind has the value it has, so it is traceable and not asserted."""

    COLUMN_HEADER_CELL = "COLUMN_HEADER_CELL"
    AXIS_METADATA = "AXIS_METADATA"
    PERIOD_BINDING = "PERIOD_BINDING"
    #: The catalogue of legacy sources, kept so a kind carried over from the old path can
    #: say so rather than claim a provenance it does not have.
    LEGACY_CASCADE = "LEGACY_CASCADE"


@dataclass(frozen=True)
class SourceCell:
    """One actual HTML cell, addressed the way the grid addresses it.

    A provenance record that cannot be resolved back to a cell is not provenance, so this
    carries the table and document too -- `(row, column)` alone is only unique within one.
    """

    document_id: str
    table_id: str
    row: int
    column: int
    text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "table_id": self.table_id,
                "row": self.row, "column": self.column, "text": self.text}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SourceCell":
        return cls(document_id=str(payload.get("document_id") or ""),
                   table_id=str(payload.get("table_id") or ""),
                   row=int(payload.get("row") or 0),
                   column=int(payload.get("column") or 0),
                   text=str(payload.get("text") or ""))


@dataclass(frozen=True)
class TemporalKindEvidence:
    """Why a kind was assigned, and from which cells."""

    kind: TemporalKind
    method: TemporalKindMethod
    source_cells: tuple[SourceCell, ...] = ()
    matched_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "method": self.method.value,
                "source_cells": [c.to_dict() for c in self.source_cells],
                "matched_text": self.matched_text}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TemporalKindEvidence":
        return cls(kind=TemporalKind(str(payload.get("kind") or "UNKNOWN")),
                   method=TemporalKindMethod(
                       str(payload.get("method") or "LEGACY_CASCADE")),
                   source_cells=tuple(SourceCell.from_dict(c)
                                      for c in payload.get("source_cells") or ()),
                   matched_text=payload.get("matched_text"))


@dataclass(frozen=True)
class PeriodBindingV2:
    """WHEN, where it came from, and how much of it the source stated.

    A `CONFLICT` binding is a first-class outcome: it carries no single certain
    `normalized_period` and retains every competing candidate in `conflict_candidates`.
    Whether that is still worth storing is `EmissionAdmission`'s question and is not
    answered here -- deciding it in this layer is what welded the two together before.
    """

    normalized_period: str | None = None
    granularity: PeriodGranularity = PeriodGranularity.UNKNOWN
    status: PeriodBindingStatus = PeriodBindingStatus.UNRESOLVED
    method: PeriodBindingMethod | None = None
    target_scope: PeriodTargetScope | None = None
    source_cells: tuple[SourceCell, ...] = ()
    temporal: TemporalKindEvidence | None = None
    conflict_candidates: tuple["PeriodBindingV2", ...] = field(default=())

    @property
    def is_direct_declaration(self) -> bool:
        return self.method in DIRECT_DECLARATIONS

    @property
    def is_inherited_context(self) -> bool:
        return self.method in INHERITED_CONTEXT

    @property
    def is_usable(self) -> bool:
        """Whether a consumer may use this period at all.

        `PARTIAL` is usable: the identity is resolved even though the shape is not.  What
        it may be *matched against* is the compatibility function's question.
        """
        return (self.status in (PeriodBindingStatus.RESOLVED, PeriodBindingStatus.PARTIAL)
                and self.normalized_period is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "normalized_period": self.normalized_period,
            "granularity": self.granularity.value,
            "status": self.status.value,
            "method": self.method.value if self.method else None,
            "target_scope": self.target_scope.value if self.target_scope else None,
            "source_cells": [c.to_dict() for c in self.source_cells],
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "conflict_candidates": [c.to_dict() for c in self.conflict_candidates],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PeriodBindingV2":
        return cls(
            normalized_period=payload.get("normalized_period"),
            granularity=PeriodGranularity(str(payload.get("granularity") or "UNKNOWN")),
            status=PeriodBindingStatus(str(payload.get("status") or "UNRESOLVED")),
            method=(PeriodBindingMethod(str(payload["method"]))
                    if payload.get("method") else None),
            target_scope=(PeriodTargetScope(str(payload["target_scope"]))
                          if payload.get("target_scope") else None),
            source_cells=tuple(SourceCell.from_dict(c)
                               for c in payload.get("source_cells") or ()),
            temporal=(TemporalKindEvidence.from_dict(payload["temporal"])
                      if payload.get("temporal") else None),
            conflict_candidates=tuple(cls.from_dict(c)
                                      for c in payload.get("conflict_candidates") or ()),
        )


@dataclass(frozen=True)
class Conflict:
    """Two bindings of the same class that disagree about one cell.

    Returned rather than resolved.  The caller may not pick a winner; if it could, a
    genuine disagreement would be indistinguishable from a precedence rule.
    """

    key: str
    candidates: tuple[PeriodBindingV2, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key,
                "candidates": [c.to_dict() for c in self.candidates]}


def resolve_period_evidence(
    declared: PeriodBindingV2 | None,
    inherited: list[PeriodBindingV2],
) -> PeriodBindingV2 | Conflict:
    """Combine the period evidence for one target cell.

    A legal `DIRECT DECLARATION` shadows inherited context -- **scope semantics, not a
    tie-break**.  Pfizer's `Balance, December 31, 2022` legitimately overrides the column
    context above it because the row states its own period, and the distinction is exactly
    why that case is not a conflict while a genuine two-source disagreement is.

    Disagreement within a class is a `Conflict`, never a choice.  Note there is no
    comparison of `method` values anywhere in this function: an ordering over methods would
    turn provenance into an authority ranking, which is the defect this contract exists to
    avoid.
    """

    if declared is not None and declared.status != PeriodBindingStatus.UNRESOLVED:
        return declared

    live = [b for b in inherited if b.status != PeriodBindingStatus.UNRESOLVED]
    if not live:
        return PeriodBindingV2()
    if len(live) == 1:
        return live[0]

    distinct = {(b.normalized_period, b.granularity.value) for b in live}
    if len(distinct) == 1:
        # Agreement is not a conflict; the evidence is merged, not chosen between.
        first = live[0]
        return PeriodBindingV2(
            normalized_period=first.normalized_period,
            granularity=first.granularity,
            status=first.status,
            method=first.method,
            target_scope=first.target_scope,
            source_cells=tuple(c for b in live for c in b.source_cells),
            temporal=first.temporal,
        )
    return Conflict(key="|".join(sorted(f"{b.normalized_period}" for b in live)),
                    candidates=tuple(live))


def periods_are_compatible(
    source: PeriodBindingV2 | None,
    requested_granularity: PeriodGranularity,
    requested_period: str | None = None,
) -> bool:
    """Whether a source period may answer a request at the requested granularity.

    One function, outside the resolver.  The rule the seal recorded: `YEAR(2025)` answers
    `FY2025` and does **not** answer `2025-12-31` -- a year does not pin a day, and
    fabricating one so that string equality then succeeds is the failure this replaces.
    """

    if source is None or not source.is_usable:
        return False
    if source.granularity == PeriodGranularity.UNKNOWN:
        return False
    if source.granularity == requested_granularity:
        if requested_period is None:
            return True
        return source.normalized_period == requested_period
    # A finer source may answer a coarser request; a coarser source may not answer a finer.
    order = {PeriodGranularity.YEAR: 0, PeriodGranularity.DAY: 1}
    return order[source.granularity] > order[requested_granularity]
