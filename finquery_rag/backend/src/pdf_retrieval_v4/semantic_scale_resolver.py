"""Gate 03 R2 Pass D — Semantic Scale Resolver.

Resolves numeric scale for each table using a strict level hierarchy::

    S0  cell explicit           → auto-resolve
    S1  row explicit             → auto-resolve
    S2  table caption/title      → auto-resolve
    S3  same logical table       → auto-resolve
    S4  same section compatible  → auto-resolve (with conditions)
    S5  same page nearby         → candidate-only
    S6  adjacent page            → candidate-only

Conflict (e.g. "in millions" + "in thousands" simultaneously):
    scale = None, scale_status = "conflict"

宁可 Recall 少一点，也不要 False Scale Binding.
"""

from __future__ import annotations

from typing import Any

from rag_v2.contracts.financial_semantics import (
    MAGNITUDE_SYNONYMS,
    magnitude_canonical_name,
    magnitude_multiplier,
    magnitude_of,
)
from src.pdf_retrieval_v4.semantic_graph_models import ScaleResolution

# ---------------------------------------------------------------------------
# Scale vocabulary
# ---------------------------------------------------------------------------
#
# The words are not defined here.  They belong to the shared financial-semantic
# authority (`rag_v2/contracts/financial_semantics.py`), which this module now
# reads instead of maintaining its own copy -- the copy here and the two in
# `rag_v2` enumerated the same words with no shared source, so any of the three
# could drift and the other two would keep looking correct.
#
# What stays local is the *retrieval* decision: which of the known phrasings a
# table caption contains, and what the level hierarchy does about it.  That is a
# lexical and procedural choice, not a semantic one, and it is the reason this
# module is a consumer of the vocabulary rather than a second definition of it.
#
# `MAGNITUDE_SYNONYMS` is keyed by lower-cased, whitespace-normalised token and
# is scanned as substrings below, so its order is load-bearing: longer phrasings
# precede the bare words they contain.

_MAGNITUDE_KEYWORDS: tuple[str, ...] = tuple(MAGNITUDE_SYNONYMS)


def _resolve_scale_keyword(keyword: str) -> tuple[float, str] | None:
    """Resolve a single scale keyword to (factor, unit)."""
    scale = magnitude_of(keyword)
    if scale is None:
        return None
    # A float factor is this module's published shape; the shared authority
    # carries an exact Decimal, so this is the boundary where it is widened.
    return (float(magnitude_multiplier(scale)), magnitude_canonical_name(scale))


def resolve_scale_keyword(keyword: str) -> tuple[float, str] | None:
    """Public alias for :func:`_resolve_scale_keyword`."""
    return _resolve_scale_keyword(keyword)


def _detect_scale_conflict(
    candidates: list[str],
) -> bool:
    """Return True if candidates contain conflicting scale keywords."""
    units: set[str] = set()
    for kw in candidates:
        resolved = _resolve_scale_keyword(kw)
        if resolved:
            units.add(resolved[1])
    return len(units) > 1


def resolve_table_scale(
    table: dict[str, Any],
    table_fragment_id: str,
    section_scale_candidates: list[str] | None = None,
    page_scale_candidates: list[str] | None = None,
    adjacent_page_scale_candidates: list[str] | None = None,
) -> ScaleResolution:
    """Resolve scale for a single table.

    Parameters
    ----------
    table
        Table fragment dict from adapter-predictions.
    table_fragment_id
        The table's fragment id.
    section_scale_candidates
        Scale candidates from the same section (for S4).
    page_scale_candidates
        Scale candidates from the same page (for S5).
    adjacent_page_scale_candidates
        Scale candidates from adjacent pages (for S6).
    """
    # Collect candidates from different levels
    raw_candidates: list[str] = []

    # S0/S1: cell and row level — scan cell resolved_text and row metric_text
    cell_scale_candidates: list[str] = []
    for cell in table.get("cells") or []:
        for sc in cell.get("scale_candidates") or []:
            cell_scale_candidates.append(str(sc))
        # Also check cell text directly
        text = str(cell.get("resolved_text") or "").lower()
        for kw in _MAGNITUDE_KEYWORDS:
            if kw in text:
                cell_scale_candidates.append(kw)

    row_scale_candidates: list[str] = []
    for row in table.get("rows") or []:
        text = str(row.get("resolved_text") or "").lower()
        for kw in _MAGNITUDE_KEYWORDS:
            if kw in text:
                row_scale_candidates.append(kw)

    # S2: table caption/title
    title_scale_candidates: list[str] = []
    header_texts = table.get("header_texts") or []
    for ht in header_texts:
        text = str(ht).lower()
        for kw in _MAGNITUDE_KEYWORDS:
            if kw in text:
                title_scale_candidates.append(kw)

    # S3: table-level scale_candidates (from adapter)
    table_scale_candidates = [str(s) for s in (table.get("scale_candidates") or [])]

    # S4: section-level
    section_cands = section_scale_candidates or []

    # S5: page-level
    page_cands = page_scale_candidates or []

    # S6: adjacent-page
    adjacent_cands = adjacent_page_scale_candidates or []

    raw_candidates = (
        cell_scale_candidates
        + row_scale_candidates
        + title_scale_candidates
        + table_scale_candidates
        + section_cands
        + page_cands
        + adjacent_cands
    )

    # Deduplicate while preserving order
    seen: set[str] = set()
    deduped: list[str] = []
    for c in raw_candidates:
        cl = c.lower()
        if cl not in seen:
            seen.add(cl)
            deduped.append(c)

    raw_tuple = tuple(deduped)

    # Check for conflict across ALL candidates
    if _detect_scale_conflict(deduped):
        return ScaleResolution(
            table_fragment_id=table_fragment_id,
            scale=None,
            scale_unit=None,
            scale_level="S0",  # conflict reported at highest level
            scale_status="conflict",
            raw_candidates=raw_tuple,
            source="conflict_detected",
        )

    # Try auto-resolve levels S0 → S4
    for level, candidates in [
        ("S0", cell_scale_candidates),
        ("S1", row_scale_candidates),
        ("S2", title_scale_candidates),
        ("S3", table_scale_candidates),
    ]:
        for kw in candidates:
            resolved = _resolve_scale_keyword(kw)
            if resolved:
                factor, unit = resolved
                return ScaleResolution(
                    table_fragment_id=table_fragment_id,
                    scale=factor,
                    scale_unit=unit,
                    scale_level=level,
                    scale_status="resolved",
                    raw_candidates=raw_tuple,
                    source=f"{level}:{kw}",
                )

    # S4: section-level — only if no competing scale
    if section_cands and not _detect_scale_conflict(section_cands):
        for kw in section_cands:
            resolved = _resolve_scale_keyword(kw)
            if resolved:
                factor, unit = resolved
                return ScaleResolution(
                    table_fragment_id=table_fragment_id,
                    scale=factor,
                    scale_unit=unit,
                    scale_level="S4",
                    scale_status="resolved",
                    raw_candidates=raw_tuple,
                    source=f"S4:{kw}",
                )

    # S5/S6: candidate-only — never auto-resolve
    if page_cands or adjacent_cands:
        # Return the first candidate as candidate-only
        candidate_kw = None
        for cands in (page_cands, adjacent_cands):
            for kw in cands:
                resolved = _resolve_scale_keyword(kw)
                if resolved:
                    candidate_kw = kw
                    break
            if candidate_kw:
                break
        if candidate_kw:
            resolved = _resolve_scale_keyword(candidate_kw)
            if resolved:
                factor, unit = resolved
                return ScaleResolution(
                    table_fragment_id=table_fragment_id,
                    scale=factor,
                    scale_unit=unit,
                    scale_level="S5" if page_cands else "S6",
                    scale_status="candidate",
                    raw_candidates=raw_tuple,
                    source=f"{'S5' if page_cands else 'S6'}:{candidate_kw}",
                )

    # No scale candidates at all
    return ScaleResolution(
        table_fragment_id=table_fragment_id,
        scale=None,
        scale_unit=None,
        scale_level="S0",
        scale_status="missing",
        raw_candidates=raw_tuple,
        source=None,
    )
