"""NF-V2-18A-R2 shared financial-table semantics adapter.

This module is deliberately an adapter at the physical-source boundary.
It converts frozen A4 SEC HTML/iXBRL tables into the repository's existing
semantic-graph contracts.  It does not define a parallel table/cell schema.
"""
from __future__ import annotations

import copy
import re
from collections import Counter, defaultdict
from typing import Any

from src.pdf_retrieval_v4.financial_table_classifier import classify_table
from src.pdf_retrieval_v4.metric_path_builder import build_metric_paths
from src.pdf_retrieval_v4.semantic_currency_resolver import resolve_table_currency
from src.pdf_retrieval_v4.semantic_graph_models import AtomicFact
from src.pdf_retrieval_v4.semantic_row_classifier import classify_table_rows
from src.pdf_retrieval_v4.semantic_scale_resolver import resolve_table_scale
from src.pdf_retrieval_v4.temporal_axis_graph import build_axis_bindings
from src.pdf_retrieval_v4.typed_evidence_emitters import emit_atomic_facts

_DATE = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
_NUMBER = re.compile(r"(?<![A-Za-z])(?:\(?\s*[-+]?\$?\s*\d[\d,]*(?:\.\d+)?\s*\)?)(?:%|[A-Za-z])?")
_VOCAB = (
    (re.compile(r"\bthree(?:[- ]month|[- ]months?)\b|\b3[- ]months?\b", re.I), "QUARTER"),
    (re.compile(r"\bsix(?:[- ]month|[- ]months?)\b|\b6[- ]months?\b", re.I), "YTD"),
    (re.compile(r"\bnine(?:[- ]month|[- ]months?)\b|\b9[- ]months?\b", re.I), "YTD"),
    (re.compile(r"\btwelve(?:[- ]month|[- ]months?)\b|\b12[- ]months?\b", re.I), "ANNUAL"),
    (re.compile(r"\byear(?:s)?\s+ended\b|\bannual\b", re.I), "ANNUAL"),
    (re.compile(r"\bas\s+of\b|\bat\s+[A-Za-z]+\s+\d{1,2}", re.I), "INSTANT"),
    (re.compile(r"\byear[- ]to[- ]date\b|\bytd\b", re.I), "YTD"),
)
_SEM_TO_KIND = {"INSTANT": "instant", "QUARTER": "duration", "YTD": "duration", "ANNUAL": "duration"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(_text(x) for x in value)
    return str(value)


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", _text(value)).strip()


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        value = _clean(value)
        if value and value.lower() not in seen:
            out.append(value)
            seen.add(value.lower())
    return out


def html_period_semantics(header: str, fallback: str | None = None) -> str:
    """Map SEC HTML vocabulary conservatively to the existing period contract."""
    text = _clean(header)
    for pattern, semantic in _VOCAB:
        if pattern.search(text):
            return semantic
    value = _clean(fallback).upper()
    return value if value in {"INSTANT", "QUARTER", "YTD", "ANNUAL"} else "UNKNOWN"


def _normal_number(raw: Any, normalized: Any) -> str | None:
    if normalized in (None, ""):
        return None
    value = _clean(raw)
    if not value or not _NUMBER.search(value):
        return None
    if isinstance(normalized, bool):
        return None
    return _clean(normalized)


def _physical_header_grid(rows: list[Any]) -> list[list[str]]:
    """Expand optional rowspan/colspan cells without resolving semantics."""
    if not rows:
        return []
    occupied: dict[tuple[int, int], str] = {}
    max_col = 0
    for row_index, row in enumerate(rows):
        col = 0
        for raw_cell in row or []:
            while (row_index, col) in occupied:
                col += 1
            if isinstance(raw_cell, dict):
                text = _clean(raw_cell.get("text") or raw_cell.get("value") or raw_cell.get("label"))
                rowspan = max(1, int(raw_cell.get("rowspan") or 1))
                colspan = max(1, int(raw_cell.get("colspan") or 1))
            else:
                text, rowspan, colspan = _clean(raw_cell), 1, 1
            for rr in range(row_index, row_index + rowspan):
                for cc in range(col, col + colspan):
                    occupied[(rr, cc)] = text
                    max_col = max(max_col, cc + 1)
            col += colspan
    return [
        [occupied.get((row_index, col), "") for col in range(max_col)]
        for row_index in range(len(rows))
    ]


def _header_paths(table: dict[str, Any], width: int) -> list[list[str]]:
    """Reconstruct per-column paths from the physical header grid."""
    rows = _physical_header_grid(table.get("header_rows") or [])
    headers = table.get("column_headers") or []
    result: list[list[str]] = []
    for col in range(width):
        parts: list[str] = []
        for row in rows:
            if col < len(row):
                parts.append(_text(row[col]))
        if col < len(headers):
            parts.append(_text(headers[col]))
        result.append(_dedupe(parts))
    return result


def _adapt_table(table: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    """Convert one A4 table into the input shape expected by shared passes."""
    did = _text(document.get("document_id") or table.get("document_id"))
    tid = _text(table.get("table_id"))
    rows0 = table.get("rows") or []
    cells0 = table.get("cells") or []
    width = max(
        [int(c.get("column_index") or 0) + 1 for c in cells0]
        + [len(table.get("column_headers") or [])]
        + [1]
    )
    physical_header_rows = _physical_header_grid(table.get("header_rows") or [])
    width = max(width, max((len(row) for row in physical_header_rows), default=0))
    paths = _header_paths({"header_rows": physical_header_rows, "column_headers": table.get("column_headers") or []}, width)
    period_columns = table.get("period_columns") or []
    sem_by_col: dict[int, str] = {}
    end_by_col: dict[int, Any] = {}
    start_by_col: dict[int, Any] = {}
    for col, binding in enumerate(period_columns):
        b = binding or {}
        header = _text(b.get("header_text"))
        sem = html_period_semantics(header, b.get("period_semantics"))
        sem_by_col[col] = sem
        end_by_col[col] = b.get("period_end")
        start_by_col[col] = b.get("period_start")
    cells: list[dict[str, Any]] = []
    row_cells: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell0 in cells0:
        c = copy.deepcopy(cell0)
        col = int(c.get("column_index") or 0)
        header = _clean(c.get("column_header")) or " / ".join(paths[col] if col < len(paths) else [])
        fallback_sem = c.get("period_semantics") or sem_by_col.get(col)
        sem = html_period_semantics(header, fallback_sem)
        p_end = c.get("period_end") or end_by_col.get(col)
        p_start = c.get("period_start") or start_by_col.get(col)
        raw = c.get("raw_value")
        norm_value = c.get("normalized_value")
        c["raw_text"] = _clean(raw)
        c["resolved_text"] = _clean(raw)
        c["header_path"] = paths[col] if col < len(paths) else [header]
        c["normalized_period"] = p_end or (f"FY{document.get('fiscal_year')}" if sem == "ANNUAL" else None)
        c["period_kind"] = _SEM_TO_KIND.get(sem)
        c["period_start"] = p_start
        c["period_end"] = p_end
        c["period_semantics"] = sem
        c["source_period_semantics"] = sem
        c["scale_candidates"] = [_text(table.get("scale"))] if table.get("scale") else []
        c["cell_bbox"] = None
        numeric = _normal_number(raw, norm_value)
        c["parsed_numeric"] = [{"normalized": numeric}] if numeric is not None else []
        provenance = dict(c.get("source_provenance") or {})
        provenance.update({"source_type": "SEC_HTML", "document_id": did, "table_id": tid})
        c["source_provenance"] = provenance
        cells.append(c)
        row_cells[_text(c.get("row_id"))].append(c)
    rows: list[dict[str, Any]] = []
    for row0 in rows0:
        row = copy.deepcopy(row0)
        rid = _text(row.get("row_id"))
        label = _clean(row.get("row_label"))
        row["metric_text"] = label
        row["resolved_text"] = label
        row["row_bbox"] = None
        row["cells"] = row_cells.get(rid, row.get("cells") or [])
        rows.append(row)
    header_texts = _dedupe(
        [_text(table.get("table_title"))]
        + [_text(x) for row in physical_header_rows for x in row]
        + [_text(x) for x in table.get("column_headers") or []]
    )
    scale_candidates = [_text(table.get("scale"))] if table.get("scale") else []
    return {
        "table_fragment_id": tid,
        "document_id": did,
        "pdf_page": 0,
        "table_index": int(table.get("source_order") or 0),
        "row_count": len(rows),
        "column_count": width,
        "table_title": _clean(table.get("table_title")),
        "header_texts": header_texts,
        "header_rows": physical_header_rows,
        "column_headers": table.get("column_headers") or [],
        "rows": rows,
        "cells": cells,
        "table_text": table.get("table_text") or "",
        "section_type": table.get("section_type") or "UNKNOWN",
        "scale_candidates": scale_candidates,
        "currency": table.get("currency"),
        "scale": table.get("scale"),
        "footnotes": table.get("footnotes") or [],
        "table_bbox": [],
        "source_type": "SEC_HTML",
        "source_document_metadata": {
            "document_id": did,
            "accession_number": document.get("accession_number"),
            "raw_sha256": document.get("source_raw_sha256"),
        },
    }


def _fact_period_semantics(fact: AtomicFact, cell_by_id: dict[str, dict[str, Any]]) -> str:
    return _text(cell_by_id.get(fact.cell_id, {}).get("source_period_semantics") or "UNKNOWN").upper()


def _build_ix_index(ix_facts: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in ix_facts:
        context = item.get("context") or {}
        pend = _text(context.get("period_end"))
        sem = _text(context.get("period_semantics") or "UNKNOWN").upper()
        for value in (item.get("raw_value"), item.get("normalized_value")):
            value_text = _clean(value)
            if value_text:
                index[(value_text, pend, sem)].append(item)
    return index


def _match_ixbrl(
    fact: AtomicFact,
    cell: dict[str, Any],
    ix_index: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[str, Any] | None:
    """Attach iXBRL only for a unique value+period match."""
    raw = _clean(cell.get("raw_value"))
    norm = _clean(cell.get("normalized_value"))
    pend = _text(fact.period_end)
    sem = _fact_period_semantics(fact, {fact.cell_id: cell})
    candidates: list[dict[str, Any]] = []
    for value in _dedupe([raw, norm]):
        for candidate_sem in (sem, "UNKNOWN"):
            candidates.extend(ix_index.get((value, pend, candidate_sem), []))
    unique = {str(x.get("fact_id")): x for x in candidates if x.get("fact_id")}
    if len(unique) != 1:
        return None
    item = next(iter(unique.values()))
    return {
        "fact_id": item.get("fact_id"),
        "concept": item.get("concept"),
        "context_ref": item.get("context_ref"),
        "unit_ref": item.get("unit_ref"),
        "unit": item.get("unit"),
        "decimals": item.get("decimals"),
        "period_context": item.get("context"),
        "source": "inline_xbrl_unique_value_period_match",
    }


def adapt_document_tables(document: dict[str, Any], ixbrl_facts: list[dict[str, Any]]) -> dict[str, Any]:
    """Build shared semantic objects for all tables in one A4 document."""
    logical_tables: list[Any] = []
    semantic_rows: list[Any] = []
    metric_paths: list[Any] = []
    axis_bindings: list[Any] = []
    atomic_facts: list[Any] = []
    # Scope every physical identity by document. SEC table/row labels are
    # only locally stable; using table_id alone would silently overwrite
    # same-numbered tables across filings.
    table_meta: dict[tuple[str, str], dict[str, Any]] = {}
    row_meta: dict[tuple[str, str, str], dict[str, Any]] = {}
    fact_meta: dict[str, dict[str, Any]] = {}
    cell_by_id: dict[str, dict[str, Any]] = {}
    did = _text(document.get("document_id"))
    mapping_failures: list[dict[str, Any]] = []
    ix_matches = 0
    ix_index = _build_ix_index(ixbrl_facts)
    tables = document.get("tables") or []
    for table0 in tables:
        table = _adapt_table(table0, document)
        tid = table["table_fragment_id"]
        table_meta[(did, tid)] = table
        for c in table["cells"]:
            cell_by_id[c["cell_id"]] = c
        lt = classify_table(table)
        logical_tables.append(lt)
        srs = classify_table_rows(table, tid, table["document_id"], 0)
        semantic_rows.extend(srs)
        mps = build_metric_paths(srs)
        metric_paths.extend(mps)
        axes = build_axis_bindings(table["cells"], tid)
        axis_bindings.extend(axes)
        scale = resolve_table_scale(table, tid)
        currency = resolve_table_currency(table, tid)
        facts = emit_atomic_facts(srs, mps, axes, table["cells"], scale, currency, {})
        atomic_facts.extend(facts)
        for fact in facts:
            cell = cell_by_id.get(fact.cell_id, {})
            ix = _match_ixbrl(fact, cell, ix_index)
            if ix:
                ix_matches += 1
            fact_meta[fact.semantic_fact_id] = {
                "period_semantics": _fact_period_semantics(fact, cell_by_id),
                "header_path": list(cell.get("header_path") or []),
                "source_period_semantics": cell.get("source_period_semantics") or "UNKNOWN",
                "canonical_row_id": fact.row_id,
                "canonical_table_id": fact.table_fragment_id,
                "canonical_cell_id": fact.cell_id,
                "ixbrl": ix,
                "metric_path": fact.metric_path,
            }
        for sr in srs:
            row_meta[(did, tid, sr.row_id)] = {
                "semantic_row": sr,
                "metric_path": next((m for m in mps if m.row_id == sr.row_id), None),
                "atomic_facts": [f for f in facts if f.row_id == sr.row_id],
                "axis_bindings": [a for a in axes if a.row_id == sr.row_id],
            }
        for row in table["rows"]:
            if not _clean(row.get("row_label")):
                continue
            if not any(m.row_id == row.get("row_id") for m in mps):
                continue
    return {
        "logical_tables": logical_tables,
        "semantic_rows": semantic_rows,
        "metric_paths": metric_paths,
        "axis_bindings": axis_bindings,
        "atomic_facts": atomic_facts,
        "table_meta": table_meta,
        "row_meta": row_meta,
        "fact_meta": fact_meta,
        "mapping_failures": mapping_failures,
        "ixbrl_matches": ix_matches,
        "documents": 1,
    }


def build_semantic_corpus(parsed_documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Adapt all frozen A4 documents without reading questions or Gold."""
    corpus = {
        "logical_tables": [],
        "semantic_rows": [],
        "metric_paths": [],
        "axis_bindings": [],
        "atomic_facts": [],
        "table_meta": {},
        "row_meta": {},
        "fact_meta": {},
        "mapping_failures": [],
        "ixbrl_matches": 0,
        "documents": 0,
    }
    for document in parsed_documents:
        part = adapt_document_tables(document, document.get("ixbrl_facts") or [])
        for key in ("logical_tables", "semantic_rows", "metric_paths", "axis_bindings", "atomic_facts"):
            corpus[key].extend(part[key])
        corpus["table_meta"].update(part["table_meta"])
        corpus["row_meta"].update(part["row_meta"])
        corpus["fact_meta"].update(part["fact_meta"])
        corpus["mapping_failures"].extend(part["mapping_failures"])
        corpus["ixbrl_matches"] += part["ixbrl_matches"]
        corpus["documents"] += 1
    physical_tables = sum(len(document.get("tables") or []) for document in parsed_documents)
    physical_rows = sum(
        len(table.get("rows") or [])
        for document in parsed_documents
        for table in (document.get("tables") or [])
    )
    corpus["stats"] = {
        "documents": corpus["documents"],
        "physical_tables": physical_tables,
        "tables_mapped": len(corpus["table_meta"]),
        "duplicate_table_occurrences": max(0, physical_tables - len(corpus["table_meta"])),
        "physical_rows": physical_rows,
        "rows_mapped": len(corpus["row_meta"]),
        "duplicate_row_occurrences": max(0, physical_rows - len(corpus["row_meta"])),
        "semantic_rows": len(corpus["semantic_rows"]),
        "metric_paths": len(corpus["metric_paths"]),
        "axis_bindings": len(corpus["axis_bindings"]),
        "atomic_facts": len(corpus["atomic_facts"]),
        "ixbrl_deterministic_matches": corpus["ixbrl_matches"],
        "mapping_failures": len(corpus["mapping_failures"]),
        "period_semantics": dict(Counter(_text(corpus["fact_meta"].get(f.semantic_fact_id, {}).get("period_semantics") or "UNKNOWN") for f in corpus["atomic_facts"])),
    }
    return corpus


def semantic_text(record: dict[str, Any], row_info: dict[str, Any] | None) -> str:
    """Serialize shared semantic fields for retrieval, without new evidence types."""
    values: list[str] = [
        _text(record.get("ticker")),
        _text(record.get("company")),
        _text(record.get("document_type")),
        _text(record.get("fiscal_year")),
        _text(record.get("fiscal_quarter")),
        _text(record.get("report_period_end")),
        _text(record.get("section_type")),
        _text(record.get("table_title")),
        _text(record.get("row_label")),
    ]
    if row_info:
        mp = row_info.get("metric_path")
        if mp:
            values += [mp.metric_path, mp.leaf_metric, " ".join(mp.metric_path_segments)]
        for fact in row_info.get("atomic_facts") or []:
            meta = record.get("_fact_meta", {}).get(fact.semantic_fact_id, {})
            values += [
                _text(meta.get("period_semantics")),
                _text(meta.get("header_path")),
                _text(fact.period_start),
                _text(fact.period_end),
                _text(fact.normalized_period),
                _text(fact.value_raw),
                _text(fact.value_normalized),
                _text(fact.currency_code),
                _text(fact.scale_unit),
            ]
    return _clean(" ".join(x for x in values if x))


def attach_semantics(records: list[dict[str, Any]], corpus: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, Any]]]:
    """Attach existing semantic objects while retaining canonical chunk IDs."""
    row_records: dict[tuple[str, str, str], dict[str, Any]] = {}
    out: list[dict[str, Any]] = []
    for record0 in records:
        record = dict(record0)
        did = _text(record.get("document_id"))
        tid = _text(record.get("table_id"))
        rid = _text(record.get("row_id"))
        info = corpus["row_meta"].get((did, tid, rid)) if did and tid and rid else None
        if info:
            record["logical_table_id"] = tid
            record["semantic_row"] = info["semantic_row"].to_dict()
            record["metric_path"] = info["metric_path"].to_dict() if info.get("metric_path") else None
            record["semantic_facts"] = []
            for fact in info.get("atomic_facts") or []:
                fd = fact.to_dict()
                fd["period_semantics"] = corpus["fact_meta"].get(fact.semantic_fact_id, {}).get("period_semantics", "UNKNOWN")
                fd["header_path"] = corpus["fact_meta"].get(fact.semantic_fact_id, {}).get("header_path", [])
                fd["ixbrl"] = corpus["fact_meta"].get(fact.semantic_fact_id, {}).get("ixbrl")
                fd["canonical_evidence_id"] = record.get("chunk_id")
                record["semantic_facts"].append(fd)
            record["_fact_meta"] = corpus["fact_meta"]
            record["semantic_retrieval_text"] = semantic_text(record, info)
            row_records[(did, tid, rid)] = record
        elif record.get("content_type") == "TABLE":
            record["logical_table_id"] = tid
            record["semantic_retrieval_text"] = semantic_text(record, None)
        else:
            record["semantic_retrieval_text"] = record.get("retrieval_text_v2") or record.get("content") or ""
        out.append(record)
    return out, row_records
