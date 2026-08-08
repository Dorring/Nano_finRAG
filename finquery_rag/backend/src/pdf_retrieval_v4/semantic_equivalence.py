"""Gate 03 R2 — Semantic Equivalence handling.

Handles the 3 Tesla ``equivalent_set`` cases from R3.2 R1: duplicate
physical rows from repeated table fragments must collapse into a single
semantic evidence identity, preventing duplicate semantic facts.

Core principle::

    Physical identities may differ
    Semantic evidence identity must not double count

This module loads the R3.2 R1 ambiguity-closure results to identify
equivalent-set groups, then provides helpers to assign
``equivalent_group_id`` and collapse duplicate facts into canonical
identities.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pdf_retrieval_v4.semantic_graph_models import (
    build_equivalent_group_id,
    canonical_semantic_fact_id,
)


def load_equivalent_sets(
    ambiguity_closure_path: Path,
) -> list[dict[str, Any]]:
    """Load equivalent_set records from R3.2 R1 ambiguity-closure.json.

    Returns the list of records where ``alignment_status == "equivalent_set"``.
    Each record contains ``physical_row_ids`` (list of row_ids in the set).
    """
    if not ambiguity_closure_path.is_file():
        return []
    data = json.loads(ambiguity_closure_path.read_text(encoding="utf-8"))
    records = data.get("records") or []
    return [r for r in records if r.get("alignment_status") == "equivalent_set"]


def build_equivalence_map(
    equivalent_sets: list[dict[str, Any]],
) -> dict[str, str]:
    """Build a mapping from physical row_id → equivalent_group_id.

    For each equivalent_set, all physical row_ids map to the same
    ``equivalent_group_id``.
    """
    mapping: dict[str, str] = {}
    for record in equivalent_sets:
        row_ids = record.get("physical_row_ids") or []
        if not row_ids:
            # R3.2 R1 ambiguity-closure.json uses "equivalent_row_ids"
            row_ids = record.get("equivalent_row_ids") or []
        if not row_ids:
            # Try to extract from the record's physical_sources
            sources = record.get("physical_sources") or []
            row_ids = [s.get("row_id") for s in sources if s.get("row_id")]
        if not row_ids:
            # Try row_evidence list (R3.2 R1 format)
            row_evidence = record.get("row_evidence") or []
            row_ids = [re_.get("row_id") for re_ in row_evidence if re_.get("row_id")]
        if len(row_ids) < 2:
            continue
        group_id = build_equivalent_group_id(row_ids)
        for rid in row_ids:
            mapping[rid] = group_id
    return mapping


def get_equivalent_group_id(
    row_id: str,
    equivalence_map: dict[str, str],
) -> str | None:
    """Return the equivalent_group_id for a row_id, or None if not in any set."""
    return equivalence_map.get(row_id)


def collapse_equivalent_facts(
    fact_ids_by_group: dict[str, list[str]],
) -> dict[str, str]:
    """Collapse multiple physical fact IDs from equivalent sets into canonical IDs.

    Returns a mapping from physical_fact_id → canonical_fact_id.
    Facts not in any equivalent set are not in the mapping.
    """
    result: dict[str, str] = {}
    for group_id, fact_ids in fact_ids_by_group.items():
        if len(fact_ids) < 2:
            continue
        canonical = canonical_semantic_fact_id(fact_ids)
        for fid in fact_ids:
            result[fid] = canonical
    return result


def detect_equivalent_set_double_counting(
    facts: list[dict[str, Any]],
    equivalence_map: dict[str, str],
) -> int:
    """Detect if any equivalent set produces duplicate semantic facts.

    After collapsing, each equivalent_group_id should have exactly one
    canonical fact per (metric_path, temporal_kind, normalized_period).
    This function counts violations.

    Returns the count of duplicate semantic facts (0 = no double counting).
    """
    # Group facts by (equivalent_group_id, metric_path, temporal_kind, normalized_period)
    seen: dict[tuple, str] = {}
    duplicates = 0

    for fact in facts:
        group_id = fact.get("equivalent_group_id")
        if not group_id:
            continue
        key = (
            group_id,
            fact.get("metric_path") or "",
            fact.get("temporal_kind") or "",
            fact.get("normalized_period") or "",
        )
        if key in seen:
            duplicates += 1
        else:
            seen[key] = fact.get("semantic_fact_id") or ""

    return duplicates
