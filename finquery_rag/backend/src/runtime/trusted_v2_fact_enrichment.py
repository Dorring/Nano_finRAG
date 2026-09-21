"""Deterministic structural enrichment for a Trusted V2 fact-store artifact.

The production fact store already contains the numeric fact and its stable
identity.  Older extraction artifacts, however, can keep table/row structure
in a separate structured-view artifact.  This module joins those two frozen,
source-derived artifacts without looking at questions, Gold labels, answers,
or model output.

The result is an optional *new fact-store artifact*.  It never changes an
existing fact's value, period, metric, candidate identity, or provenance flag;
it adds only a compact ``structural_context`` when an exact evidence, row, or
(table-only) source linkage exists.  A table-only linkage is deliberately
restricted to table-level fields and never contributes row metric/period data.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


ENRICHMENT_SCHEMA_VERSION = "trusted-v2-fact-structural-enrichment-v1"


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _stable_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _mapping_sequence(value: Any) -> tuple[Mapping[str, Any], ...]:
    if isinstance(value, Mapping):
        return (value,)
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _record_ids(record: Mapping[str, Any], *keys: str) -> set[str]:
    values: list[Any] = []
    for key in keys:
        value = record.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(value)
        else:
            values.append(value)
    for trace in _mapping_sequence(record.get("source_traceback")):
        for key in keys:
            values.append(trace.get(key))
    return set(_stable_unique(values))


def _view_evidence_ids(view: Mapping[str, Any]) -> set[str]:
    values: list[Any] = list(view.get("semantic_evidence_ids") or ())
    for fact in _mapping_sequence(view.get("facts")):
        values.append(fact.get("evidence_id"))
        values.append(fact.get("fact_id"))
    return set(_stable_unique(values))


def _view_id(view: Mapping[str, Any], position: int) -> str:
    return _text(view.get("view_id")) or _text(view.get("candidate_key")) or f"view:{position}"


def _build_view_indexes(
    structured_views: Iterable[Mapping[str, Any]],
) -> tuple[
    tuple[tuple[int, Mapping[str, Any]], ...],
    dict[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    dict[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    dict[str, tuple[tuple[int, Mapping[str, Any]], ...]],
]:
    """Build stable source-identity indexes once for a fact-store run."""

    frozen_views = tuple(
        (position, dict(view))
        for position, view in enumerate(structured_views, 1)
        if isinstance(view, Mapping)
    )
    evidence_index: defaultdict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    row_index: defaultdict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    table_index: defaultdict[str, list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for item in frozen_views:
        position, view = item
        for source_id in _view_evidence_ids(view):
            evidence_index[source_id].append(item)
        for source_id in _record_ids(view, "row_id", "row_ids"):
            row_index[source_id].append(item)
        for source_id in _record_ids(view, "table_fragment_id", "table_id"):
            table_index[source_id].append(item)
    return (
        frozen_views,
        {key: tuple(value) for key, value in evidence_index.items()},
        {key: tuple(value) for key, value in row_index.items()},
        {key: tuple(value) for key, value in table_index.items()},
    )


def _matches_for_ids(
    index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    source_ids: set[str],
) -> list[tuple[int, Mapping[str, Any]]]:
    """Merge indexed hits in source-view order without duplicate views."""

    by_position: dict[int, Mapping[str, Any]] = {}
    for source_id in source_ids:
        for position, view in index.get(source_id, ()):
            by_position.setdefault(position, view)
    return [(position, by_position[position]) for position in sorted(by_position)]


def _best_view_matches(
    fact: Mapping[str, Any],
    evidence_index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    row_index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    table_index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
) -> tuple[str | None, list[tuple[int, Mapping[str, Any]]]]:
    """Return matches at one strongest source-identity level only."""

    evidence_matches = _matches_for_ids(
        evidence_index,
        _record_ids(fact, "fact_id", "evidence_id"),
    )
    if evidence_matches:
        return "EVIDENCE_ID", evidence_matches
    row_matches = _matches_for_ids(row_index, _record_ids(fact, "row_id", "row_ids"))
    if row_matches:
        return "ROW_ID", row_matches
    table_matches = _matches_for_ids(
        table_index,
        _record_ids(fact, "table_fragment_id", "table_id"),
    )
    if table_matches:
        return "TABLE_FRAGMENT", table_matches
    return None, []

def _single_value_or_none(values: Iterable[Any]) -> Any | None:
    """Return one source scalar without changing its source type."""

    unique: list[Any] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, (str, int, float, bool)):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        identity = json.dumps(value, ensure_ascii=False, sort_keys=True)
        if identity not in seen:
            unique.append(value)
            seen.add(identity)
    return copy.deepcopy(unique[0]) if len(unique) == 1 else None


def _aggregate_context(
    matches: list[tuple[int, Mapping[str, Any]]],
    linkage: str,
) -> dict[str, Any]:
    """Aggregate only source-derived structured fields from matched views."""

    views = [view for _, view in matches]
    context: dict[str, Any] = {
        "structural_linkage": linkage,
        "source_view_ids": [_view_id(view, position) for position, view in matches],
    }
    documents = _single_value_or_none(view.get("document_id") for view in views)
    if documents is not None:
        context["document_id"] = documents
    pages = _single_value_or_none(view.get("pdf_page") for view in views)
    if pages is not None:
        context["pdf_page"] = pages
    table_titles = _single_value_or_none(view.get("table_title") for view in views)
    if table_titles is not None:
        context["table_title"] = table_titles
    sections: list[Any] = []
    rows: list[Any] = []
    evidence_ids: list[Any] = []
    for view in views:
        sections.extend(view.get("section_path") or ())
        rows.extend(view.get("row_ids") or ())
        evidence_ids.extend(view.get("semantic_evidence_ids") or ())
    if sections:
        context["section_path"] = _stable_unique(sections)
    if rows:
        context["row_ids"] = _stable_unique(rows)
    if evidence_ids:
        context["semantic_evidence_ids"] = _stable_unique(evidence_ids)

    # A table fragment can contain several rows.  Only row/evidence identity
    # linkage may enrich row-level metric and period context.
    if linkage != "TABLE_FRAGMENT":
        metric_paths: list[Any] = []
        periods: list[Any] = []
        for view in views:
            metric_paths.extend(view.get("metric_paths") or ())
            periods.extend(view.get("periods") or ())
        if metric_paths:
            context["metric_paths"] = _stable_unique(metric_paths)
        if periods:
            context["periods"] = _stable_unique(periods)
    return context


def _enrich_fact_with_indexes(
    fact: Mapping[str, Any],
    evidence_index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    row_index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
    table_index: Mapping[str, tuple[tuple[int, Mapping[str, Any]], ...]],
) -> tuple[dict[str, Any], str | None]:
    if not isinstance(fact, Mapping):
        raise TypeError("fact must be a mapping")
    linkage, matches = _best_view_matches(
        fact,
        evidence_index,
        row_index,
        table_index,
    )
    enriched = copy.deepcopy(dict(fact))
    if linkage is None:
        return enriched, None

    existing = enriched.get("structural_context")
    context = copy.deepcopy(dict(existing)) if isinstance(existing, Mapping) else {}
    derived = _aggregate_context(matches, linkage)
    for key, value in derived.items():
        # Existing fact-store metadata is authoritative. An enrichment pass may
        # add absent context but must never overwrite a frozen source field.
        context.setdefault(key, value)
    enriched["structural_context"] = context
    return enriched, linkage


def enrich_fact_record(
    fact: Mapping[str, Any],
    structured_views: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], str | None]:
    """Return one immutable-preserving fact copy plus its linkage level."""

    _, evidence_index, row_index, table_index = _build_view_indexes(structured_views)
    return _enrich_fact_with_indexes(fact, evidence_index, row_index, table_index)


def enrich_fact_records(
    facts: Iterable[Mapping[str, Any]],
    structured_views: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Enrich a fact stream in O(facts + views + indexed matches)."""

    frozen_views, evidence_index, row_index, table_index = _build_view_indexes(
        structured_views
    )
    enriched: list[dict[str, Any]] = []
    linkages: Counter[str] = Counter()
    for fact in facts:
        record, linkage = _enrich_fact_with_indexes(
            fact,
            evidence_index,
            row_index,
            table_index,
        )
        enriched.append(record)
        if linkage is not None:
            linkages[linkage] += 1
    return enriched, {
        "schema_version": ENRICHMENT_SCHEMA_VERSION,
        "source_fact_count": len(enriched),
        "structured_view_count": len(frozen_views),
        "enriched_fact_count": sum(linkages.values()),
        "unmatched_fact_count": len(enriched) - sum(linkages.values()),
        "linkage_counts": dict(sorted(linkages.items())),
        "index_key_counts": {
            "evidence_ids": len(evidence_index),
            "row_ids": len(row_index),
            "table_ids": len(table_index),
        },
    }

def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL/JSONL.GZ artifact without accepting executable content."""

    source = Path(path)
    opener = gzip.open if source.name.casefold().endswith(".gz") else open
    rows: list[dict[str, Any]] = []
    with opener(source, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{source}:{line_number}") from exc
            if not isinstance(value, Mapping):
                raise ValueError(f"jsonl_record_must_be_object:{source}:{line_number}")
            rows.append(dict(value))
    return rows


def write_jsonl(path: Path | str, records: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "ENRICHMENT_SCHEMA_VERSION",
    "enrich_fact_record",
    "enrich_fact_records",
    "read_jsonl",
    "sha256_file",
    "write_jsonl",
]
