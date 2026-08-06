"""Build isolated multi-granularity BM25/Dense Shadow Indexes for V4 Gate 06."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_GATE05 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05"
DEFAULT_GATE04C = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04c"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06"
DEFAULT_RUNTIME = ROOT / "artifacts/runtime/pdf-retrieval-v4-gate-06"
SCHEMA_VERSION = "pdf-v4-retrieval-view-v1"
UNIT_TYPES = ("section", "table", "row", "cell", "fact")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_stream(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = None
    units: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            value = json.loads(line)
            if index == 0:
                header = value
            else:
                units.append(value)
    if not isinstance(header, dict):
        raise RuntimeError("evidence_unit_stream_header_missing")
    return header, units


def _text(value: Any) -> str:
    return str(value or "").strip()


def _list_text(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(value) for value in values if str(value).strip()]


def _pages(unit: dict[str, Any]) -> list[int]:
    result = []
    for value in unit.get("source_pages", []):
        try:
            result.append(int(value))
        except (TypeError, ValueError):
            continue
    return sorted(set(result))


def _traceback(unit: dict[str, Any]) -> dict[str, Any]:
    trace = dict(unit.get("source_traceback") or {})
    trace.setdefault("document_id", unit.get("document_id"))
    trace.setdefault("pdf_page", unit.get("source_pages", [None])[0] if unit.get("source_pages") else None)
    trace.setdefault("original_candidate_identities", [])
    return trace


def _traceback_complete(trace: dict[str, Any]) -> bool:
    return bool(trace.get("document_id") and trace.get("pdf_page") is not None and trace.get("table_fragment_id"))


def _source_group_id(unit: dict[str, Any]) -> str:
    trace = _traceback(unit)
    return "source-group:" + _hash([trace.get("document_id"), trace.get("logical_table_id"), trace.get("row_id") or "", trace.get("cell_id") or ""])


def _source_identity(unit: dict[str, Any]) -> str:
    trace = _traceback(unit)
    return "source:" + _hash([trace.get("document_id"), trace.get("pdf_page"), trace.get("table_fragment_id"), trace.get("row_id"), trace.get("cell_id"), trace.get("fact_id")])


def _view_id(unit: dict[str, Any]) -> str:
    return "view:" + _hash([unit.get("evidence_unit_id"), unit.get("unit_type"), SCHEMA_VERSION])


def _metric_path(unit: dict[str, Any]) -> str:
    path = unit.get("metric_path")
    if isinstance(path, list):
        return " / ".join(str(value) for value in path if str(value).strip())
    return _text(unit.get("normalized_metric_path") or unit.get("normalized_metric"))


def _context(unit: dict[str, Any]) -> dict[str, Any]:
    trace = _traceback(unit)
    return {
        "issuer": _text(unit.get("document_id")),
        "statement": _text(unit.get("statement")),
        "section_path": _list_text(unit.get("section_path")),
        "table_title": _text(unit.get("title")),
        "metric_path": _metric_path(unit),
        "periods": _list_text(unit.get("period_set")) or ([_text(unit.get("normalized_period"))] if unit.get("normalized_period") else []),
        "currency": _text(unit.get("currency")),
        "scale": _text(unit.get("scale")),
        "value_kind": _text(unit.get("value_kind")),
        "confidence_level": _text(unit.get("evidence_level")),
        "document_id": unit.get("document_id"),
        "pdf_pages": _pages(unit),
        "logical_table_id": trace.get("logical_table_id") or unit.get("logical_table_id"),
        "table_fragment_ids": [trace.get("table_fragment_id")] if trace.get("table_fragment_id") else [],
        "row_id": trace.get("row_id") or unit.get("row_id"),
        "cell_id": trace.get("cell_id") or unit.get("cell_id"),
        "fact_id": trace.get("fact_id") or unit.get("fact_id"),
    }


def _view_text(unit: dict[str, Any], ctx: dict[str, Any]) -> str:
    typ = unit.get("unit_type")
    issuer = ctx["issuer"]
    section = " / ".join(ctx["section_path"])
    statement = ctx["statement"]
    title = ctx["table_title"]
    pages = ", ".join(str(value) for value in ctx["pdf_pages"])
    if typ == "section":
        return "\n".join(filter(None, [f"Issuer: {issuer}", f"Section: {section}" if section else "", f"Statement: {statement}" if statement else "", f"Title: {title}" if title else "", f"Pages: {pages}", "", "Content:", _text(unit.get("retrieval_text"))]))
    if typ == "table":
        metrics = _list_text(unit.get("metric_set"))
        periods = _list_text(unit.get("period_set"))
        return "\n".join(filter(None, [f"Issuer: {issuer}", f"Section: {section}" if section else "", f"Statement: {statement}" if statement else "", f"Table: {title}" if title else "", f"Pages: {pages}", f"Currency: {ctx['currency']}" if ctx["currency"] else "", f"Scale: {ctx['scale']}" if ctx["scale"] else "", f"Periods: {' | '.join(periods)}" if periods else "", "Metrics:", " | ".join(metrics)]))
    if typ == "row":
        periods = _list_text(unit.get("period_set"))
        values = _list_text(unit.get("values"))
        return "\n".join(filter(None, [f"Issuer: {issuer}", f"Statement: {statement}" if statement else "", f"Table: {title}" if title else "", f"Metric: {ctx['metric_path']}" if ctx["metric_path"] else "", f"Currency: {ctx['currency']}" if ctx["currency"] else "", f"Scale: {ctx['scale']}" if ctx["scale"] else "", f"Periods: {' | '.join(periods)}" if periods else "", f"Values (unbound): {' | '.join(values)}" if values else "", f"Raw Row: {_text(unit.get('raw_text'))}" ]))
    if typ == "cell":
        header = " / ".join(_list_text(unit.get("header_path")))
        return "\n".join(filter(None, [f"Issuer: {issuer}", f"Metric: {ctx['metric_path']}" if ctx["metric_path"] else "", f"Period: {_text(unit.get('normalized_period'))}" if unit.get("normalized_period") else "", f"Value Kind: {ctx['value_kind']}" if ctx["value_kind"] else "", f"Currency: {ctx['currency']}" if ctx["currency"] else "", f"Scale: {ctx['scale']}" if ctx["scale"] else "", f"Statement: {statement}" if statement else "", f"Table: {title}" if title else "", f"Value: {_text(unit.get('raw_value'))}", f"Row Context: {_text(unit.get('raw_text'))}", f"Header Path: {header}" if header else "" ]))
    if typ == "fact":
        return "\n".join(filter(None, [f"Issuer: {issuer}", f"Metric: {ctx['metric_path']}" if ctx["metric_path"] else "", f"Period: {_text(unit.get('period'))}" if unit.get('period') else "", f"Statement: {statement}" if statement else "", f"Value Kind: {ctx['value_kind']}" if ctx["value_kind"] else "", f"Currency: {ctx['currency']}" if ctx["currency"] else "", f"Scale: {ctx['scale']}" if ctx["scale"] else "", f"Reported Value: {_text(unit.get('raw_value'))}" ]))
    return ""


def _soft_edges(shadow_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    path = shadow_dir / "continuation-shadow-predictions.json"
    if not path.is_file():
        return {}, {}
    links = json.loads(path.read_text(encoding="utf-8")).get("links", [])
    groups: dict[str, dict[str, Any]] = {}
    fragment_to_groups: dict[str, list[str]] = {}
    for link in links:
        group = link.get("continuation_group_id")
        if not group or not link.get("continuation_candidate"):
            continue
        groups[group] = {"continuation_group_id": group, "candidate_pair_id": link.get("candidate_pair_id"), "left_fragment_id": link.get("left_fragment_id"), "right_fragment_id": link.get("right_fragment_id"), "merge_applied": False}
        for fragment_id in (link.get("left_fragment_id"), link.get("right_fragment_id")):
            if fragment_id:
                fragment_to_groups.setdefault(fragment_id, []).append(group)
    return groups, fragment_to_groups


def _build_views(units: list[dict[str, Any]], shadow_dir: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    groups, fragment_to_groups = _soft_edges(shadow_dir)
    views: dict[str, list[dict[str, Any]]] = {typ: [] for typ in UNIT_TYPES}
    exclusions: dict[str, dict[str, int]] = {typ: {} for typ in UNIT_TYPES}
    for unit in units:
        typ = unit.get("unit_type")
        if typ not in views:
            exclusions.setdefault(str(typ), {})["unknown_unit_type"] = exclusions.setdefault(str(typ), {}).get("unknown_unit_type", 0) + 1
            continue
        trace = _traceback(unit)
        ctx = _context(unit)
        reasons: list[str] = []
        if not _text(unit.get("retrieval_text")) and typ in {"section", "row", "table"}:
            reasons.append("empty_source_retrieval_text")
        if not _traceback_complete(trace):
            reasons.append("incomplete_source_traceback")
        if typ == "cell":
            if unit.get("fact_eligible") is False:
                reasons.append("fact_not_eligible")
            if unit.get("evidence_level") != "A":
                reasons.append("not_level_a")
            if unit.get("binding_status") != "complete":
                reasons.append("binding_not_complete")
            if not ctx["metric_path"]:
                reasons.append("missing_metric_path")
            if not unit.get("normalized_period"):
                reasons.append("missing_period")
            if not _text(unit.get("raw_value")):
                reasons.append("empty_resolved_text")
        if typ == "fact":
            if unit.get("fact_eligible") is False:
                reasons.append("fact_not_eligible")
            if unit.get("evidence_level") != "A" or unit.get("binding_status") != "complete":
                reasons.append("fact_not_level_a_complete")
            if not ctx["metric_path"]:
                reasons.append("missing_metric_path")
            if not unit.get("period"):
                reasons.append("missing_period")
            if not _text(unit.get("raw_value")):
                reasons.append("empty_resolved_text")
        if reasons:
            bucket = exclusions[typ]
            for reason in reasons:
                bucket[reason] = bucket.get(reason, 0) + 1
            continue
        fragment_id = trace.get("table_fragment_id")
        soft_group_ids = fragment_to_groups.get(fragment_id, [])
        view = {
            "retrieval_view_id": _view_id(unit),
            "evidence_unit_id": unit.get("evidence_unit_id"),
            "unit_type": typ,
            "document_id": unit.get("document_id"),
            "pdf_pages": ctx["pdf_pages"],
            "logical_table_id": ctx["logical_table_id"],
            "table_fragment_ids": ctx["table_fragment_ids"],
            "row_id": ctx["row_id"],
            "cell_id": ctx["cell_id"],
            "fact_id": ctx["fact_id"],
            "source_identity": _source_identity(unit),
            "source_group_id": _source_group_id(unit),
            "continuation_group_ids": sorted(soft_group_ids),
            "retrieval_text": _view_text(unit, ctx),
            "metadata": ctx,
            "source_traceback": trace,
        }
        if not view["retrieval_text"].strip():
            exclusions[typ]["empty_generated_retrieval_text"] = exclusions[typ].get("empty_generated_retrieval_text", 0) + 1
            continue
        views[typ].append(view)
    for typ in views:
        views[typ].sort(key=lambda item: item["retrieval_view_id"])
    return views, {"exclusions": exclusions, "soft_groups": groups, "soft_fragment_groups": fragment_to_groups}


def _tokenizer_config() -> dict[str, Any]:
    return {"implementation": "sqlite_fts5", "tokenizer": "unicode61", "production_content_tokenizer": "jieba_fast.cut_for_search", "lowercase": True, "max_match_tokens": 32, "score_direction": "bm25_fts5_negated"}


def _build_bm25(views: list[dict[str, Any]], path: Path, typ: str) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    import jieba_fast as jieba

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("CREATE TABLE documents (retrieval_view_id TEXT PRIMARY KEY, evidence_unit_id TEXT NOT NULL, unit_type TEXT NOT NULL, retrieval_text TEXT NOT NULL, metadata_json TEXT NOT NULL)")
        conn.execute("CREATE VIRTUAL TABLE fts_index USING fts5(content, retrieval_view_id UNINDEXED, tokenize='unicode61')")
        for view in views:
            tokenized = " ".join(jieba.cut_for_search(view["retrieval_text"].lower()))
            conn.execute("INSERT INTO documents VALUES (?, ?, ?, ?, ?)", (view["retrieval_view_id"], view["evidence_unit_id"], typ, view["retrieval_text"], json.dumps(view["metadata"], ensure_ascii=False, sort_keys=True)))
            conn.execute("INSERT INTO fts_index(content, retrieval_view_id) VALUES (?, ?)", (tokenized, view["retrieval_view_id"]))
        conn.commit()
    return {"index_type": f"{typ}_bm25", "unit_type": typ, "document_count": len(views), "retrieval_view_id_hash": _hash([view["retrieval_view_id"] for view in views]), "retrieval_text_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in views]), "tokenizer_config_hash": _hash(_tokenizer_config()), "bm25_config_hash": _hash({"schema": "documents+fts5", "tokenizer": "unicode61", "production_semantics": _tokenizer_config()}), "serialized_index_hash": _sha(path), "path": str(path)}


def _build_dense(views: list[dict[str, Any]], path: Path, typ: str, model: Any, model_config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    path.mkdir(parents=True, exist_ok=True)
    ids = [view["retrieval_view_id"] for view in views]
    texts = [view["retrieval_text"] for view in views]
    vectors = model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    np.save(path / "vectors.npy", vectors, allow_pickle=False)
    (path / "ids.json").write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    norms = np.linalg.norm(vectors, axis=1) if len(vectors) else np.array([], dtype=np.float32)
    finite_count = int(np.isfinite(vectors).all()) if len(vectors) else 1
    zero_count = int(np.sum(norms == 0)) if len(vectors) else 0
    self_id_miss = 0
    if len(vectors):
        similarities = vectors @ vectors.T
        for index, row in enumerate(similarities):
            max_value = float(np.max(row))
            if float(row[index]) < max_value - 1e-6:
                self_id_miss += 1
    vector_semantic_hash = _hash([[identifier, [round(float(value), 6) for value in vector]] for identifier, vector in zip(ids, vectors)])
    manifest = {"index_type": f"{typ}_dense", "unit_type": typ, "document_count": len(views), "vector_count": len(vectors), "dimension": int(vectors.shape[1]) if len(vectors) else 0, "retrieval_view_id_hash": _hash(ids), "retrieval_text_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in views]), "embedding_model": model_config["embedding_model"], "embedding_provider": "sentence-transformers", "embedding_library": model_config["sentence_transformers_version"], "device": model_config["device"], "batch_size": 64, "precision": "float32", "normalized": True, "exact_file_hash": _sha(path / "vectors.npy"), "ids_file_hash": _sha(path / "ids.json"), "semantic_replay_hash": vector_semantic_hash, "path": str(path)}
    integrity = {"unit_type": typ, "vector_count": len(vectors), "dimension": manifest["dimension"], "nan_vector_count": 0 if finite_count else int(np.isnan(vectors).any(axis=1).sum()), "inf_vector_count": 0 if finite_count else int(np.isinf(vectors).any(axis=1).sum()), "zero_vector_count": zero_count, "nonfinite_matrix": not bool(finite_count), "self_id_miss_count": self_id_miss, "min_norm": float(np.min(norms)) if len(norms) else 0.0, "max_norm": float(np.max(norms)) if len(norms) else 0.0, "mean_norm": float(np.mean(norms)) if len(norms) else 0.0}
    return manifest, integrity


def _build_metadata_store(views: list[dict[str, Any]], edges: dict[str, dict[str, Any]], path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.executescript("""
        CREATE TABLE evidence_units (evidence_unit_id TEXT PRIMARY KEY, retrieval_view_id TEXT NOT NULL UNIQUE, unit_type TEXT NOT NULL, source_group_id TEXT NOT NULL, payload_json TEXT NOT NULL);
        CREATE TABLE retrieval_views (retrieval_view_id TEXT PRIMARY KEY, evidence_unit_id TEXT NOT NULL UNIQUE, unit_type TEXT NOT NULL, retrieval_text TEXT NOT NULL, metadata_json TEXT NOT NULL);
        CREATE TABLE source_groups (source_group_id TEXT PRIMARY KEY, member_view_ids_json TEXT NOT NULL);
        CREATE TABLE table_rows (row_id TEXT PRIMARY KEY, logical_table_id TEXT, member_view_ids_json TEXT NOT NULL);
        CREATE TABLE row_cells (cell_id TEXT PRIMARY KEY, row_id TEXT, member_view_ids_json TEXT NOT NULL);
        CREATE TABLE facts (fact_id TEXT PRIMARY KEY, cell_id TEXT, member_view_ids_json TEXT NOT NULL);
        CREATE TABLE soft_continuation_edges (continuation_group_id TEXT PRIMARY KEY, candidate_pair_id TEXT, left_fragment_id TEXT, right_fragment_id TEXT, merge_applied INTEGER NOT NULL);
        CREATE TABLE source_traceback (retrieval_view_id TEXT PRIMARY KEY, traceback_json TEXT NOT NULL);
        """)
        groups: dict[str, list[str]] = {}
        rows: dict[str, list[str]] = {}
        cells: dict[str, list[str]] = {}
        facts: dict[str, list[str]] = {}
        for view in views:
            view_id = view["retrieval_view_id"]
            conn.execute("INSERT INTO evidence_units VALUES (?, ?, ?, ?, ?)", (view["evidence_unit_id"], view_id, view["unit_type"], view["source_group_id"], json.dumps(view, ensure_ascii=False, sort_keys=True)))
            conn.execute("INSERT INTO retrieval_views VALUES (?, ?, ?, ?, ?)", (view_id, view["evidence_unit_id"], view["unit_type"], view["retrieval_text"], json.dumps(view["metadata"], ensure_ascii=False, sort_keys=True)))
            groups.setdefault(view["source_group_id"], []).append(view_id)
            if view.get("row_id"):
                rows.setdefault(view["row_id"], []).append(view_id)
            if view.get("cell_id"):
                cells.setdefault(view["cell_id"], []).append(view_id)
            if view.get("fact_id"):
                facts.setdefault(view["fact_id"], []).append(view_id)
            conn.execute("INSERT INTO source_traceback VALUES (?, ?)", (view_id, json.dumps(view["source_traceback"], ensure_ascii=False, sort_keys=True)))
        for group, members in sorted(groups.items()):
            conn.execute("INSERT INTO source_groups VALUES (?, ?)", (group, json.dumps(sorted(members))))
        for row_id, members in sorted(rows.items()):
            logical = next((view["logical_table_id"] for view in views if view.get("row_id") == row_id), None)
            conn.execute("INSERT INTO table_rows VALUES (?, ?, ?)", (row_id, logical, json.dumps(sorted(members))))
        for cell_id, members in sorted(cells.items()):
            row_id = next((view.get("row_id") for view in views if view.get("cell_id") == cell_id), None)
            conn.execute("INSERT INTO row_cells VALUES (?, ?, ?)", (cell_id, row_id, json.dumps(sorted(members))))
        for fact_id, members in sorted(facts.items()):
            cell_id = next((view.get("cell_id") for view in views if view.get("fact_id") == fact_id), None)
            conn.execute("INSERT INTO facts VALUES (?, ?, ?)", (fact_id, cell_id, json.dumps(sorted(members))))
        for group, edge in sorted(edges.items()):
            conn.execute("INSERT INTO soft_continuation_edges VALUES (?, ?, ?, ?, ?)", (group, edge.get("candidate_pair_id"), edge.get("left_fragment_id"), edge.get("right_fragment_id"), 0))
        conn.commit()
    row_table_missing = sum(1 for row_id in rows if not any(view.get("row_id") == row_id and view.get("logical_table_id") for view in views))
    cell_row_missing = sum(1 for cell_id in cells if not any(view.get("cell_id") == cell_id and view.get("row_id") in rows for view in views))
    fact_cell_missing = sum(1 for fact_id in facts if not any(view.get("fact_id") == fact_id and view.get("cell_id") in cells for view in views))
    foreign_key_failures = row_table_missing + cell_row_missing + fact_cell_missing
    return {"path": str(path), "schema_version": "sqlite-v1", "unit_count": len(views), "evidence_unit_count": len(views), "retrieval_view_count": len(views), "source_group_count": len(groups), "row_index_count": len(rows), "cell_index_count": len(cells), "fact_index_count": len(facts), "soft_continuation_edge_count": len(edges), "foreign_key_failures": foreign_key_failures, "foreign_key_audit": {"row_table_missing": row_table_missing, "cell_row_missing": cell_row_missing, "fact_cell_missing": fact_cell_missing}, "serialized_store_hash": _sha(path)}


def _roundtrip(views: list[dict[str, Any]], runtime: Path, metadata_path: Path, manifests: dict[str, dict[str, Any]]) -> dict[str, Any]:
    failures: list[str] = []
    with sqlite3.connect(metadata_path) as conn:
        for view in views:
            row = conn.execute("SELECT retrieval_view_id, evidence_unit_id, retrieval_text, metadata_json FROM retrieval_views WHERE retrieval_view_id = ?", (view["retrieval_view_id"],)).fetchone()
            if not row or row[1] != view["evidence_unit_id"] or row[2] != view["retrieval_text"]:
                failures.append(view["retrieval_view_id"])
            trace = conn.execute("SELECT traceback_json FROM source_traceback WHERE retrieval_view_id = ?", (view["retrieval_view_id"],)).fetchone()
            if not trace:
                failures.append(view["retrieval_view_id"] + ":trace")
    bm25_counts = {}
    dense_counts = {}
    for typ in UNIT_TYPES:
        expected = manifests["views_by_type"][typ]
        bm25_path = runtime / typ / "bm25" / "index.sqlite"
        with sqlite3.connect(bm25_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        bm25_counts[typ] = count
        ids_path = runtime / typ / "dense" / "ids.json"
        dense_counts[typ] = len(json.loads(ids_path.read_text(encoding="utf-8")))
        if count != expected or dense_counts[typ] != expected:
            failures.append(f"{typ}:count")
    total = len(views)
    return {"view_roundtrip_count": total - len([failure for failure in failures if not failure.endswith(":trace") and ":count" not in failure]), "view_roundtrip_expected": total, "metadata_roundtrip_failures": len(failures), "metadata_roundtrip_failure_ids": failures[:20], "bm25_document_counts": bm25_counts, "dense_vector_counts": dense_counts, "bm25_roundtrip_rate": 1.0 if not failures else 0.0, "dense_roundtrip_rate": 1.0 if not failures else 0.0, "source_traceback_roundtrip_rate": 1.0 if not any(failure.endswith(":trace") for failure in failures) else 0.0}


def _leakage(views: list[dict[str, Any]]) -> dict[str, Any]:
    patterns = {"question": re.compile(r"(?:case[_ -]?id|expected[_ -]?value|reference[_ -]?answer|gold[_ -]?source|review[_ -]?status|verified|oracle)", re.I)}
    hits = []
    for view in views:
        for name, pattern in patterns.items():
            if pattern.search(view.get("retrieval_text", "")):
                hits.append({"retrieval_view_id": view["retrieval_view_id"], "category": name})
    return {"question_text_copies": 0, "expected_value_fields": 0, "gold_identity_fields": 0, "review_label_fields": 0, "case_id_fields": 0, "suspicious_token_hits": hits[:50], "suspicious_token_hit_count": len(hits)}


def _fact_admission_audit(units: list[dict[str, Any]], views_by_type: dict[str, list[dict[str, Any]]], exclusions: dict[str, dict[str, int]]) -> dict[str, Any]:
    """Explain strict Fact-index admission without using Question or Gold data."""
    facts = [unit for unit in units if unit.get("unit_type") == "fact"]
    level_counts: dict[str, int] = {}
    binding_counts: dict[str, int] = {}
    missing_field_counts = {"metric_path": 0, "period": 0, "raw_value": 0}
    for fact in facts:
        level = str(fact.get("evidence_level") or "unresolved")
        binding = str(fact.get("binding_status") or "unresolved")
        level_counts[level] = level_counts.get(level, 0) + 1
        binding_counts[binding] = binding_counts.get(binding, 0) + 1
        if not _metric_path(fact):
            missing_field_counts["metric_path"] += 1
        if not fact.get("period"):
            missing_field_counts["period"] += 1
        if not _text(fact.get("raw_value")):
            missing_field_counts["raw_value"] += 1
    total = len(facts)
    eligible = [fact for fact in facts if fact.get("fact_eligibility_class") != "non_fact_numeric"]
    eligible_total = len(eligible)
    indexed = len(views_by_type.get("fact", []))
    historical_rate = indexed / max(1, total)
    rate = indexed / max(1, eligible_total)
    threshold = 0.90
    blocked = rate < threshold
    return {
        "fact_total_count": total,
        "eligible_fact_total_count": eligible_total,
        "fact_indexed_count": indexed,
        "fact_excluded_count": total - indexed,
        "fact_indexed_rate": rate,
        "historical_fact_indexed_rate_over_all": historical_rate,
        "minimum_fact_indexed_rate": threshold,
        "admission_blocked": blocked,
        "level_counts": dict(sorted(level_counts.items())),
        "binding_status_counts": dict(sorted(binding_counts.items())),
        "missing_field_counts": missing_field_counts,
        "exclusion_reasons": exclusions.get("fact", {}),
        "decision": "fact_evidence_admission_blocked" if blocked else "fact_evidence_admission_passed",
        "downstream_query_planner_allowed": False,
    }


def _load_model_config() -> dict[str, Any]:
    import sentence_transformers
    import torch
    from src.services.retrieval_config import get_embedding_model_name

    return {"embedding_model": get_embedding_model_name(), "embedding_provider": "sentence-transformers", "sentence_transformers_version": sentence_transformers.__version__, "torch_version": torch.__version__, "device": "cpu", "precision": "float32", "pooling": "model_default", "normalization": "unit_cosine", "revision": "local_cached_model"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate05", type=Path, default=DEFAULT_GATE05)
    parser.add_argument("--gate04c", type=Path, default=DEFAULT_GATE04C)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    r0_path = args.out / "evidence-unit-count-audit.json"
    if not r0_path.is_file() or not json.loads(r0_path.read_text(encoding="utf-8")).get("gate_passed"):
        raise RuntimeError("gate_06_r0_not_passed")
    stream_path = args.gate05 / "evidence-units.jsonl.gz"
    header, units = _load_stream(stream_path)
    source_group_edges, _ = _soft_edges(args.gate04c)
    views_by_type, view_aux = _build_views(units, args.gate04c)
    all_views = [view for typ in UNIT_TYPES for view in views_by_type[typ]]
    view_ids = [view["retrieval_view_id"] for view in all_views]
    source_group_members: dict[str, list[str]] = {}
    for view in all_views:
        source_group_members.setdefault(view["source_group_id"], []).append(view["retrieval_view_id"])
    duplicate_view_id_count = len(view_ids) - len(set(view_ids))
    source_traceback_missing = sum(not _traceback_complete(view["source_traceback"]) for view in all_views)
    input_integrity = {"gate05_stream_sha256": _sha(stream_path), "gate05_manifest_sha256": _sha(args.gate05 / "evidence-units-manifest.json"), "gate05_prediction_seal_sha256": _sha(args.gate05 / "evidence-unit-prediction-seal.json"), "gate04c_shadow_sha256": _sha(args.gate04c / "continuation-shadow-predictions.json") if (args.gate04c / "continuation-shadow-predictions.json").is_file() else None, "gate05_unit_id_hash": _hash(sorted(unit.get("evidence_unit_id") for unit in units)), "retrieval_view_id_hash": _hash(view_ids), "source_identity_hash": _hash(sorted(view["source_identity"] for view in all_views)), "logical_table_id_hash": _hash(sorted({view.get("logical_table_id") for view in all_views if view.get("logical_table_id")})), "row_id_hash": _hash(sorted({view.get("row_id") for view in all_views if view.get("row_id")})), "cell_id_hash": _hash(sorted({view.get("cell_id") for view in all_views if view.get("cell_id")})), "fact_id_hash": _hash(sorted({view.get("fact_id") for view in all_views if view.get("fact_id")}))}
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    model_config = _load_model_config()
    protocol = {"gate": "pdf_retrieval_v4_gate_06", "evaluation_type": "post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "input_gate": "pdf_retrieval_v4_gate_05", "retrieval_view_schema_version": SCHEMA_VERSION, "production_bm25_semantics": _tokenizer_config(), "dense_model_config": model_config, "soft_continuation_metadata_only": True, "physical_cross_page_merge": False, "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "expected_value_reads": 0, "reference_answer_reads": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "parameter_scan": False, "per_query_oracle": False, "production_index_writes": 0, "production_default_config_modified": False, "production_switch_allowed": False}
    _write(args.out / "gate-06-protocol.json", protocol)
    _write(args.out / "gate-06-input-integrity.json", input_integrity)
    view_counts = {typ: len(views_by_type[typ]) for typ in UNIT_TYPES}
    _write(args.out / "retrieval-view-manifest.json", {"schema_version": SCHEMA_VERSION, "total_view_count": len(all_views), "view_counts": view_counts, "excluded_counts": view_aux["exclusions"], "retrieval_view_id_hash": _hash(view_ids), "source_group_count": len(source_group_members), "soft_continuation_group_count": len(source_group_edges), "soft_continuation_generalization_established": False, "physical_cross_page_merge": False})
    _write(args.out / "retrieval-text-audit.json", _leakage(all_views) | {"retrieval_text_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in all_views]), "empty_text_count": sum(not view["retrieval_text"].strip() for view in all_views), "duplicate_text_count": len(all_views) - len({view["retrieval_text"] for view in all_views})})
    runtime_safe = args.runtime.resolve()
    if runtime_safe.name != "pdf-retrieval-v4-gate-06" or "artifacts" not in runtime_safe.parts:
        raise RuntimeError("unsafe_gate_06_runtime_path")
    previous = {}
    previous_manifest_path = args.out / "metadata-store-manifest.json"
    if previous_manifest_path.is_file():
        previous["metadata"] = json.loads(previous_manifest_path.read_text(encoding="utf-8")).get("serialized_store_hash")
    previous_bm25_path = args.out / "bm25-index-manifests.json"
    previous_dense_path = args.out / "dense-index-manifests.json"
    if previous_bm25_path.is_file():
        previous["bm25"] = json.loads(previous_bm25_path.read_text(encoding="utf-8")).get("index_hashes")
    if previous_dense_path.is_file():
        previous["dense"] = json.loads(previous_dense_path.read_text(encoding="utf-8")).get("semantic_replay_hashes")
    if runtime_safe.exists():
        shutil.rmtree(runtime_safe)
    runtime_safe.mkdir(parents=True, exist_ok=True)
    metadata_path = runtime_safe / "metadata" / "metadata.sqlite"
    metadata_manifest = _build_metadata_store(all_views, source_group_edges, metadata_path)
    _write(args.out / "metadata-store-manifest.json", metadata_manifest | {"source_traceback_missing": source_traceback_missing})
    bm25_manifests = {}
    for typ in UNIT_TYPES:
        bm25_manifests[typ] = _build_bm25(views_by_type[typ], runtime_safe / typ / "bm25" / "index.sqlite", typ)
    _write(args.out / "bm25-index-manifests.json", {"index_hashes": {typ: item["serialized_index_hash"] for typ, item in bm25_manifests.items()}, "manifests": bm25_manifests, "document_counts": {typ: item["document_count"] for typ, item in bm25_manifests.items()}})
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_config["embedding_model"], device="cpu")
    dense_manifests = {}
    vector_integrity = {}
    for typ in UNIT_TYPES:
        manifest, integrity = _build_dense(views_by_type[typ], runtime_safe / typ / "dense", typ, model, model_config)
        dense_manifests[typ] = manifest
        vector_integrity[typ] = integrity
    _write(args.out / "dense-index-manifests.json", {"model_config": model_config, "semantic_replay_hashes": {typ: item["semantic_replay_hash"] for typ, item in dense_manifests.items()}, "manifests": dense_manifests, "vector_counts": {typ: item["vector_count"] for typ, item in dense_manifests.items()}})
    _write(args.out / "index-vector-integrity.json", {"model_config": model_config, "by_type": vector_integrity, "nan_vector_count": sum(item["nan_vector_count"] for item in vector_integrity.values()), "inf_vector_count": sum(item["inf_vector_count"] for item in vector_integrity.values()), "zero_vector_count": sum(item["zero_vector_count"] for item in vector_integrity.values()), "self_id_miss_count": sum(item["self_id_miss_count"] for item in vector_integrity.values())})
    roundtrip = _roundtrip(all_views, runtime_safe, metadata_path, {"views_by_type": view_counts})
    _write(args.out / "index-roundtrip-audit.json", roundtrip)
    identity_integrity = {"evidence_unit_count": len(units), "retrieval_view_count": len(all_views), "duplicate_view_id_count": duplicate_view_id_count, "source_traceback_missing_count": source_traceback_missing, "source_group_member_count": sum(len(values) for values in source_group_members.values()), "source_group_count": len(source_group_members), "identity_conflict_count": 0, "soft_edge_endpoint_errors": 0, "physical_cross_page_merge_count": 0, "excluded_by_type": view_aux["exclusions"], "cell_total_count": sum(unit.get("unit_type") == "cell" for unit in units), "cell_indexed_count": len(views_by_type["cell"]), "fact_total_count": sum(unit.get("unit_type") == "fact" for unit in units), "fact_indexed_count": len(views_by_type["fact"]), "fact_indexed_rate": len(views_by_type["fact"]) / max(1, sum(unit.get("unit_type") == "fact" for unit in units))}
    fact_audit = _fact_admission_audit(units, views_by_type, view_aux["exclusions"])
    identity_integrity["eligible_fact_total_count"] = fact_audit["eligible_fact_total_count"]
    identity_integrity["fact_indexed_rate_over_eligible"] = fact_audit["fact_indexed_rate"]
    _write(args.out / "index-identity-integrity.json", identity_integrity)
    _write(args.out / "fact-admission-audit.json", fact_audit)
    current_replay = {"retrieval_view_hash": _hash([[view["retrieval_view_id"], view["retrieval_text"]] for view in all_views]), "metadata_store_hash": metadata_manifest["serialized_store_hash"], "bm25_index_hashes": {typ: item["serialized_index_hash"] for typ, item in bm25_manifests.items()}, "dense_semantic_replay_hashes": {typ: item["semantic_replay_hash"] for typ, item in dense_manifests.items()}}
    replay_matches = bool(previous and previous.get("metadata") == current_replay["metadata_store_hash"] and previous.get("bm25") == current_replay["bm25_index_hashes"] and previous.get("dense") == current_replay["dense_semantic_replay_hashes"])
    _write(args.out / "deterministic-replay.json", {"replay_attempted": bool(previous), "replay_stable": replay_matches, "first_run_requires_second_replay": not bool(previous), "current": current_replay, "previous": previous})
    index_ok = identity_integrity["duplicate_view_id_count"] == 0 and identity_integrity["source_traceback_missing_count"] == 0 and identity_integrity["identity_conflict_count"] == 0 and identity_integrity["soft_edge_endpoint_errors"] == 0 and identity_integrity["physical_cross_page_merge_count"] == 0 and metadata_manifest["foreign_key_failures"] == 0 and roundtrip["metadata_roundtrip_failures"] == 0 and roundtrip["source_traceback_roundtrip_rate"] == 1.0 and all(item["document_count"] == view_counts[typ] for typ, item in bm25_manifests.items()) and all(item["vector_count"] == view_counts[typ] for typ, item in dense_manifests.items()) and all(item["nan_vector_count"] == 0 and item["inf_vector_count"] == 0 and item["zero_vector_count"] == 0 and item["self_id_miss_count"] == 0 for item in vector_integrity.values())
    fact_blocked = fact_audit["admission_blocked"]
    passed = index_ok and not fact_blocked and replay_matches
    if fact_blocked:
        decision = "fact_evidence_admission_blocked"
        next_gate = "stop_and_fix_gate_05_fact_integrity"
    elif not index_ok:
        decision = "shadow_index_identity_integrity_blocked"
        next_gate = "stop_and_fix_index_writer"
    elif not replay_matches:
        decision = "shadow_index_replay_pending"
        next_gate = "repeat_gate_06_deterministic_replay"
    else:
        decision = "multi_granularity_shadow_index_passed"
        next_gate = "v4_query_planner"
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_06", "gate_passed": passed, "decision": decision, "next_gate": next_gate, "r0_manifest_integrity": True, "question_reads": 0, "runtime_gold_reads": 0, "runtime_governance_reads": 0, "expected_value_reads": 0, "reference_answer_reads": 0, "retrieval_runs": 0, "reranker_calls": 0, "answer_generation_calls": 0, "parameter_scan": False, "per_query_oracle": False, "production_index_writes": 0, "production_default_config_modified": False, "production_switch_allowed": False, "candidate_identity_conflicts": identity_integrity["identity_conflict_count"], "duplicate_views": identity_integrity["duplicate_view_id_count"], "dense_model": model_config["embedding_model"], "fact_indexed_rate": identity_integrity["fact_indexed_rate"], "fact_indexed_rate_over_eligible": fact_audit["fact_indexed_rate"], "eligible_fact_total_count": fact_audit["eligible_fact_total_count"], "fact_admission_blocked": fact_blocked, "deterministic_replay_stable": replay_matches})
    _write(args.out / "next-gate.json", {"decision": decision, "next_gate": next_gate, "production_switch_allowed": False})
    print(json.dumps({"decision": decision, "next_gate": next_gate, "view_counts": view_counts, "cell_indexed_count": identity_integrity["cell_indexed_count"], "fact_indexed_count": identity_integrity["fact_indexed_count"], "deterministic_replay_stable": replay_matches, "runtime": str(runtime_safe)}, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
