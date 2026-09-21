"""Derive a canonical concept from a stored fact's metric path.

`build_canonical_fact_store` files a fact under its breadcrumb path rather than
its leaf label, so Apple's operating-income row is stored as
``'Operating expenses: / Research and development'``.  A concept derived from
that path is what a cross-filing comparison needs -- no two filings write the
breadcrumb the same way -- but the path is not reliably ``section / leaf``.

NVIDIA's compensation table files the *column headers* as the path:

    'KEVAN PAREKH / CHRIS KONDO / WANDA AUSTIN / ALEX GORSKY / ANDREA JUNG / /s/ Arthur D. Levi'

whose leaf is ``/s/ Arthur D. Levi``.  Taking the last segment there yields a
signature, and a resolver that guessed would file a person's name as a metric --
which is how ``NVIDIA / Colette M. Kress`` came to be a coordinate in the first
place.  So this refuses rather than guesses, and it needs **two independent
signals to agree** before it resolves anything.

**Shadow only.**  Nothing here writes to the store or changes retrieval.  The
result is a derivation with its own status and provenance, so that "concept
recovery" and "production behaviour changed" stay two separately measurable
deltas -- the same discipline as the two tracks in `p1_2_dual_track`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

#: Strings that mark a segment as a caption, a signature or a name rather than a
#: concept.  Listed rather than pattern-matched so the exclusions are auditable.
_NON_CONCEPT_MARKERS = (
    "/s/",
    "signature",
    "director",
    "compensation",
    "(in millions)",
    "(in thousands)",
    "(in billions)",
)

#: A concept naming a quantity is short.  `Gross profit total automotive &
#: services and other segment` is 60-odd characters and legitimate; past this it
#: is prose, a whole table header, or a concatenation of several labels.
_MAX_CONCEPT_CHARS = 90

_LEADING_NUMBERING = re.compile(r"^(?:\(?\d+[\.\)]?\s+)+")
_TRAILING_CAPTION = re.compile(r"\s*\((?:in|except)[^)]*\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ConceptDerivationV1:
    """What was derived, how, and whether it may be trusted."""

    status: str  # RESOLVED | UNRESOLVED
    concept: str | None
    source: str
    raw_metric_path: str
    content_label: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _fold(text: object) -> str:
    """Casefold and collapse whitespace, so two spellings of one label match."""

    return " ".join(str(text or "").split()).casefold()


def _content_label(content: object) -> str | None:
    """The row label a fact's content opens with, or ``None``.

    Stored content reads ``'Research and development | 34,550 | 31,370 |
    29,915'`` -- the row's cells joined, label first.  Only the first field is
    taken; the rest are values, not names.
    """

    text = str(content or "").strip()
    if not text:
        return None
    label = text.split("|", 1)[0].strip()
    return label or None


def _path_leaf(raw_metric_path: object) -> str | None:
    """The last segment of a breadcrumb path, or ``None`` if there is no path."""

    text = str(raw_metric_path or "").strip()
    if not text:
        return None
    if " / " not in text:
        return text
    leaf = text.rsplit(" / ", 1)[-1].strip()
    return leaf or None


def _rejection(candidate: str) -> str | None:
    """Why a candidate label cannot be a concept, or ``None`` if it can."""

    folded = candidate.casefold()
    for marker in _NON_CONCEPT_MARKERS:
        if marker in folded:
            return f"contains {marker!r}, a caption or signature marker"
    if len(candidate) > _MAX_CONCEPT_CHARS:
        return f"longer than {_MAX_CONCEPT_CHARS} characters"
    if not any(character.isalpha() for character in candidate):
        return "no letters"
    # A label that is entirely upper case and more than one word is a name or a
    # heading rather than a metric: `WANDA AUSTIN`, `NET OPERATING REVENUES`.
    words = candidate.split()
    if len(words) > 1 and candidate.isupper():
        return "all upper case, several words -- a name or a heading"
    return None


def _clean(candidate: str) -> str:
    """Drop leading numbering and a trailing unit caption.

    `Net interest income (in millions)` names a concept and annotates its units;
    the caption is not part of the name.  Cleaning happens *before* the
    rejection check, not after -- `(in millions)` is itself a caption marker, so
    checking first refused every legitimate row that carried one.
    """

    stripped = _TRAILING_CAPTION.sub("", candidate).strip()
    return _LEADING_NUMBERING.sub("", stripped).strip()


def derive_fact_concept(
    raw_metric_path: object, content: object
) -> ConceptDerivationV1:
    """Reconcile the path's leaf with the content's row label.

    Resolution requires the two to agree after folding.  One signal alone is not
    enough: the path's leaf is a signature in NVIDIA's table, and the content's
    label is the row whose cells were emitted as several facts -- each of which
    is exactly the ambiguity this exists to remove.  Agreement of two independent
    derivations is the cheapest evidence that the label names a concept.
    """

    raw = str(raw_metric_path or "")
    leaf = _path_leaf(raw)
    label = _content_label(content)

    def unresolved(reason: str) -> ConceptDerivationV1:
        return ConceptDerivationV1(
            status="UNRESOLVED", concept=None, source="none",
            raw_metric_path=raw, content_label=label, reason=reason,
        )

    if not raw.strip():
        return unresolved("empty metric path")
    if leaf is None:
        return unresolved("metric path has no leaf")
    if label is None:
        return unresolved("content has no row label")

    cleaned_leaf, cleaned_label = _clean(leaf), _clean(label)
    if not cleaned_leaf:
        return unresolved("label is only numbering or a caption")

    problem = _rejection(cleaned_leaf)
    if problem is not None:
        return unresolved(f"metric-path leaf rejected: {problem}")
    problem = _rejection(cleaned_label)
    if problem is not None:
        return unresolved(f"content label rejected: {problem}")

    if _fold(cleaned_leaf) != _fold(cleaned_label):
        return unresolved(
            f"the two signals disagree: path leaf {leaf!r} vs content label {label!r}"
        )

    return ConceptDerivationV1(
        status="RESOLVED",
        concept=cleaned_leaf,
        source="metric_path_leaf+content_label",
        raw_metric_path=raw,
        content_label=label,
        reason="the path's leaf and the content's row label agree",
    )
