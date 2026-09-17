"""Deterministic numeric fidelity for a model-cleaned financial table.

The verifier's whole job is one question, asked against an oracle that is not a
model: *did the transformation alter, invent, remove or reassign financially
meaningful numeric content?*

Two properties make the answer worth having.

**The oracle is the source.**  The authoritative pre-model markdown is the other
side of every comparison, so the check cannot be satisfied by a model agreeing
with itself.  The prompt already says "Do NOT change numeric values", and that
sentence is not enforcement -- it is a request, made to a component whose
compliance is not a property anyone can test.  This module is the test.

**Association is checked, not just the multiset.**  A bag-of-numbers comparison
would call this transformation perfect:

    source          derived
    Revenue  100    Revenue   80
    Cost      80    Cost     100

Every number survived and every count matches.  What moved is *which figure
belongs to which label*, which is the thing a reader of a financial table
actually relies on -- and the reason ``NUMERIC_VALUE_REASSIGNED`` is a separate
reason from ``NUMERIC_VALUE_CHANGED``.

The comparison is over pipe-table structure only, because that is the strongest
authoritative structure this pipeline possesses: the ingestion path turns a
Camelot dataframe into markdown and keeps no cell coordinates through to here,
so inventing a row/column model the extraction never had would be inventing
semantics rather than using them.  A row is keyed by its label, a figure by the
row label and the column header above it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from rag_v2.contracts.financial_semantics import canonical_decimal, text_identity

from .contracts import (
    AdmissionOutcomeV1,
    AdmissionReasonV1,
    ArtifactAdmissionResultV1,
    ModelDerivedArtifactV1,
)

__all__ = [
    "NumericFactV1",
    "numeric_facts_of",
    "verify_table_fidelity",
]

#: A markdown alignment row: `---`, `:---`, `---:`, `:---:` with any run of dashes.
_ALIGNMENT_ROW = re.compile(r"^:?-{3,}:?$")


@dataclass(frozen=True)
class NumericFactV1:
    """One numeric cell, and the structured position that gives it meaning."""

    row_label: str
    column_label: str
    value: Decimal

    @property
    def key(self) -> tuple[str, str]:
        return (self.row_label, self.column_label)

    @property
    def location(self) -> str:
        """A bounded, source-derived name for where this sits.

        Row label and column header, both taken from the table itself.  It
        answers "where" in an admission record without carrying model content
        into a log -- the labels are the source's, not the model's.
        """

        return f"{self.row_label} / {self.column_label}"


def _rows(markdown: str) -> list[list[str]] | None:
    """The pipe rows of a markdown table, or ``None`` if there are none."""

    rows: list[list[str]] = []
    for line in (markdown or "").splitlines():
        stripped = line.strip()
        if not stripped or "|" not in stripped:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not any(cells):
            continue
        if all(_ALIGNMENT_ROW.fullmatch(cell) for cell in cells):
            continue
        rows.append(cells)
    return rows or None


def numeric_facts_of(markdown: str) -> tuple[NumericFactV1, ...] | None:
    """Every numeric cell, keyed by the structure that gives it meaning.

    ``None`` means the structure could not be established -- not a table, ragged
    rows, or nothing but a header -- and every caller treats that as a failure to
    verify rather than as a pass.  An empty tuple would mean "a well-formed table
    with no numbers in it", which is a different and legitimate answer.

    A cell is numeric when the shared financial semantics say so:
    ``canonical_decimal`` folds digit-grouping separators, currency edge symbols,
    accounting negatives and footnote markers, so ``1,000`` and ``1000`` are the
    same quantity and ``(25)`` is minus twenty-five.  Prose is *not* pushed
    through it -- a cell that will not canonicalise is a label, and labels are
    compared as text.
    """

    rows = _rows(markdown)
    if rows is None or len(rows) < 2:
        return None
    if len({len(row) for row in rows}) != 1:
        # Ragged rows have no stable column mapping, so a figure cannot be
        # attributed to a position at all.
        return None

    header = [
        text_identity(cell) or f"col{index}" for index, cell in enumerate(rows[0])
    ]

    facts: list[NumericFactV1] = []
    for row_index, row in enumerate(rows[1:], start=1):
        label = ""
        for cell in row:
            if canonical_decimal(cell) is None and text_identity(cell):
                label = text_identity(cell)
                break
        if not label:
            # A row of nothing but numbers has no label of its own; its
            # position is the only identity it has, and saying so is better
            # than borrowing a label from somewhere else.
            label = f"row{row_index}"
        for column_index, cell in enumerate(row):
            value = canonical_decimal(cell)
            if value is None:
                continue
            facts.append(
                NumericFactV1(
                    row_label=label,
                    column_label=header[column_index],
                    value=value,
                )
            )
    return tuple(facts)


def _result(
    candidate: ModelDerivedArtifactV1,
    outcome: AdmissionOutcomeV1,
    reason: AdmissionReasonV1,
    location: str = "",
) -> ArtifactAdmissionResultV1:
    return ArtifactAdmissionResultV1(
        artifact_id=candidate.artifact_id,
        outcome=outcome,
        reason=reason,
        source_reference=candidate.source_reference,
        location=location,
    )


def _by_key(
    facts: tuple[NumericFactV1, ...],
) -> dict[tuple[str, str], list[Decimal]]:
    grouped: dict[tuple[str, str], list[Decimal]] = {}
    for fact in facts:
        grouped.setdefault(fact.key, []).append(fact.value)
    return grouped


def verify_table_fidelity(
    candidate: ModelDerivedArtifactV1,
    authoritative_markdown: str,
) -> ArtifactAdmissionResultV1:
    """Decide whether a cleaned table preserved the source's numeric content.

    The source is examined in row order and the model's table only afterwards,
    so the *reported* reason is deterministic: for a transformation that both
    changed a value and invented another, the first discrepancy in source order
    is the one named.  Callers should treat every refusal the same way -- the
    reason is for the reader of a log, not for a caller choosing a repair.

    One transformation is refused for a reason that reads oddly and is worth
    stating: renaming a column header changes every key beneath it, so it
    surfaces as a removal and an invention rather than as a structural error.
    That is the honest classification available from table structure alone --
    the contract asks the model to preserve headers -- and the consequence of
    over-refusing is that the authoritative table is used, which is the safe
    direction.
    """

    source = numeric_facts_of(authoritative_markdown)
    derived = numeric_facts_of(candidate.content)
    if source is None or derived is None:
        return _result(
            candidate,
            AdmissionOutcomeV1.REJECTED,
            AdmissionReasonV1.STRUCTURE_UNVERIFIABLE,
        )

    source_by_key = _by_key(source)
    derived_by_key = _by_key(derived)
    source_values = {fact.value for fact in source}

    for fact in source:
        mine = source_by_key[fact.key]
        theirs = derived_by_key.get(fact.key)
        if theirs is None or len(theirs) < len(mine):
            return _result(
                candidate,
                AdmissionOutcomeV1.REJECTED,
                AdmissionReasonV1.NUMERIC_VALUE_REMOVED,
                fact.location,
            )
        if len(theirs) > len(mine):
            # The same figure stated more times than the source states it is an
            # extra fact about the world, however familiar its value looks.
            return _result(
                candidate,
                AdmissionOutcomeV1.REJECTED,
                AdmissionReasonV1.NUMERIC_VALUE_INVENTED,
                fact.location,
            )
        if theirs != mine:
            # A value that exists *elsewhere* in the source has moved rather
            # than been edited, and the two are different failures.
            reason = (
                AdmissionReasonV1.NUMERIC_VALUE_REASSIGNED
                if all(value in source_values for value in theirs)
                else AdmissionReasonV1.NUMERIC_VALUE_CHANGED
            )
            return _result(
                candidate, AdmissionOutcomeV1.REJECTED, reason, fact.location
            )

    for fact in derived:
        if fact.key not in source_by_key:
            return _result(
                candidate,
                AdmissionOutcomeV1.REJECTED,
                AdmissionReasonV1.NUMERIC_VALUE_INVENTED,
                fact.location,
            )

    return _result(
        candidate,
        AdmissionOutcomeV1.ADMITTED,
        AdmissionReasonV1.FIDELITY_PRESERVED,
    )
