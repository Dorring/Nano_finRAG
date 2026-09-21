#!/usr/bin/env python3
"""Gate 08 R3-R0: Build Aligned 628 Control Index.

This is the ONLY new artifact needed before formal prediction.

Extracts three key sets:
  K_legacy  = Gate 08 R2 old Structured Candidate Key Set (~628)
  K_new     = Gate 06 R4 Grade-A Structured Key Set (19,500)
  K_common  = K_legacy ∩ K_new

Builds three Structured Controls:
  S-Legacy  = 628 candidates, old R2 serializer, old index (already exists)
  S-Control = K_common + gate06-r4-v1 serializer + same Dense/BM25 (NEW)
  S-Expanded= 19,500 + gate06-r4-v1 serializer (already exists in Gate 06 R4)

Outputs:
  control-keyset-audit.json
  aligned-control-index-manifest.json
  control-indexes/ (S-Control BM25 + Dense)

Usage:
    python3 scripts/evaluation/build_pdf_v4_gate_08_r3_control_index.py
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.candidate_aligned_view import (  # noqa: E402
    CandidateAlignedView,
    make_structured_view_id,
)

R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
CONTROL_INDEX_DIR = R3_DIR / "control-indexes"

GATE08_R2_VIEWS = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-views/view-pairs.jsonl"
)
GATE08_R2_INDEX_DIR = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-indexes"
)
R5_STRUCTURED_VIEWS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/structured-views.jsonl"
)
GATE06_R4_INDEX_DIR = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4/candidate-indexes"
)

STRUCTURED_TEXT_VERSION = "gate06-r4-v1"
ENCODER_MODEL = "all-MiniLM-L6-v2"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


# ---------------------------------------------------------------------------
# gate06-r4-v1 serializer (copied from build_pdf_v4_gate_06_r4_indexes.py)
# ---------------------------------------------------------------------------


def format_r5_structured_text(view: dict[str, Any]) -> str:
    """Serialize an R5 structured view into deterministic retrieval text.

    Field order is fixed. List items are sorted. Empty fields are omitted.
    """
    parts: list[str] = []
    document_id = str(view.get("document_id") or "")
    pdf_page = view.get("pdf_page")
    if document_id:
        parts.append(f"Document: {document_id}")
    if pdf_page is not None:
        parts.append(f"Page: {pdf_page}")
    section_path = sorted(s for s in (view.get("section_path") or []) if s)
    if section_path:
        parts.append("")
        parts.append("Section:")
        parts.append("\n".join(f"  {s}" for s in section_path))
    table_title = view.get("table_title")
    if table_title:
        parts.append(f"Table: {table_title}")
    metric_paths = sorted(m for m in (view.get("metric_paths") or []) if m)
    if metric_paths:
        parts.append("")
        parts.append("Metric Paths:")
        parts.append("\n".join(f"  {m}" for m in metric_paths))
    periods = sorted(p for p in (view.get("periods") or []) if p)
    if periods:
        parts.append("")
        parts.append("Periods:")
        parts.append("\n".join(f"  {p}" for p in periods))
    facts = view.get("facts") or []
    fact_lines: list[str] = []
    for fact in facts:
        if not isinstance(fact, dict):
            continue
        ftype = str(fact.get("type") or "")
        metric = str(fact.get("metric") or "")
        if ftype == "narrative":
            text_val = str(fact.get("text") or "")
            if text_val:
                fact_lines.append(f"narrative | {text_val}")
        elif ftype == "row_matrix":
            rm_periods = fact.get("periods") or []
            rm_values = fact.get("values") or []
            for p, v in zip(rm_periods, rm_values):
                fact_lines.append(f"{metric} | {p} | {v}")
        else:
            period = str(fact.get("period") or "")
            value = str(fact.get("value") or "")
            scale = str(fact.get("scale") or "")
            fact_lines.append(f"{metric} | {period} | {value} | {scale}")
    if fact_lines:
        parts.append("")
        parts.append("Facts:")
        parts.append("\n".join(f"  {f}" for f in sorted(fact_lines)))
    segments = sorted(s for s in (view.get("segments") or []) if s)
    if segments:
        parts.append("")
        parts.append("Segments:")
        parts.append(", ".join(segments))
    buckets = sorted(b for b in (view.get("buckets") or []) if b)
    if buckets:
        parts.append("")
        parts.append("Buckets:")
        parts.append(", ".join(buckets))
    raw_content = str(view.get("raw_content") or "")
    parts.append("")
    parts.append("Source:")
    parts.append(raw_content)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Step 1: Extract key sets
# ---------------------------------------------------------------------------


def extract_keysets() -> dict[str, Any]:
    """Extract K_legacy, K_new, K_common and compute set differences."""
    print("Extracting key sets...")

    # K_legacy: Gate 08 R2 structured candidate keys
    k_legacy: set[str] = set()
    for rec in _read_jsonl(GATE08_R2_VIEWS):
        if rec.get("structured_view") is not None:
            k_legacy.add(rec["candidate_key"])
    print(f"  K_legacy (Gate 08 R2 structured): {len(k_legacy)}")

    # K_new: Gate 06 R4 / R5 Grade-A structured keys
    k_new: set[str] = set()
    for rec in _read_jsonl(R5_STRUCTURED_VIEWS):
        k_new.add(rec["candidate_key"])
    print(f"  K_new (R5 Grade-A structured): {len(k_new)}")

    k_common = k_legacy & k_new
    legacy_removed = k_legacy - k_new
    new_added = k_new - k_legacy

    print(f"  K_common (intersection): {len(k_common)}")
    print(f"  legacy_removed (in legacy, not in new): {len(legacy_removed)}")
    print(f"  new_added (in new, not in legacy): {len(new_added)}")

    return {
        "legacy_count": len(k_legacy),
        "new_count": len(k_new),
        "common_count": len(k_common),
        "removed_count": len(legacy_removed),
        "added_count": len(new_added),
        "k_legacy": sorted(k_legacy),
        "k_new": sorted(k_new),
        "k_common": sorted(k_common),
        "legacy_removed": sorted(legacy_removed),
        "new_added_sample": sorted(new_added)[:20],
    }


# ---------------------------------------------------------------------------
# Step 2: Build S-Control index
# ---------------------------------------------------------------------------


def load_raw_views_for_keys(keys: set[str]) -> dict[str, dict[str, Any]]:
    """Load raw views from Gate 08 R2 for the given keys."""
    raw_views: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(GATE08_R2_VIEWS):
        ck = rec["candidate_key"]
        if ck in keys:
            raw_views[ck] = rec["raw_view"]
    return raw_views


def load_r5_structured_for_keys(keys: set[str]) -> dict[str, dict[str, Any]]:
    """Load R5 structured views for the given keys."""
    views: dict[str, dict[str, Any]] = {}
    for rec in _read_jsonl(R5_STRUCTURED_VIEWS):
        ck = rec["candidate_key"]
        if ck in keys:
            views[ck] = rec
    return views


def build_control_structured_views(
    k_common: set[str],
    r5_structured: dict[str, dict[str, Any]],
) -> list[CandidateAlignedView]:
    """Build CandidateAlignedView list for K_common using gate06-r4-v1 serializer."""
    views: list[CandidateAlignedView] = []
    for ck in sorted(k_common):
        r5_data = r5_structured.get(ck)
        if r5_data is None:
            print(f"  WARNING: K_common key not in R5: {ck}")
            continue
        retrieval_text = format_r5_structured_text(r5_data)
        view = CandidateAlignedView(
            candidate_key=ck,
            view_type="structured",
            view_id=make_structured_view_id(ck),
            retrieval_text=retrieval_text,
            document_id=str(r5_data.get("document_id") or ""),
            pdf_page=r5_data.get("pdf_page"),
            logical_table_ids=(),
            row_ids=tuple(r5_data.get("row_ids") or []),
            fact_ids=tuple(r5_data.get("semantic_evidence_ids") or []),
            metric_paths=tuple(sorted(r5_data.get("metric_paths") or [])),
            periods=tuple(sorted(r5_data.get("periods") or [])),
            temporal_types=(),
            bridge_grade=str(r5_data.get("bridge_grade") or "raw_only"),
        )
        views.append(view)
    return views


def _tokenize_for_index(text: str) -> str:
    """Tokenize text for FTS5 indexing using jieba_fast."""
    try:
        import jieba_fast  # type: ignore

        tokens = jieba_fast.cut_for_search(text)
    except ImportError:
        import jieba  # type: ignore

        tokens = jieba.cut_for_search(text)
    return " ".join(t for t in tokens if t.strip())


def build_structured_bm25(views: list[CandidateAlignedView], out_dir: Path) -> dict:
    """Build BM25 index (SQLite FTS5) for structured views."""
    bm25_dir = out_dir / "candidate_structured_bm25" / "bm25"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    db_path = bm25_dir / "index.sqlite"
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE documents ("
            "retrieval_view_id TEXT PRIMARY KEY, "
            "candidate_key TEXT NOT NULL, "
            "retrieval_text TEXT NOT NULL, "
            "metadata_json TEXT NOT NULL)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE fts_index USING fts5("
            "content, retrieval_view_id UNINDEXED, tokenize='unicode61')"
        )
        for view in views:
            tokenized = _tokenize_for_index(view.retrieval_text)
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?)",
                (
                    view.view_id,
                    view.candidate_key,
                    view.retrieval_text,
                    json.dumps(view.to_dict(), ensure_ascii=False, sort_keys=True),
                ),
            )
            conn.execute(
                "INSERT INTO fts_index(content, retrieval_view_id) VALUES (?, ?)",
                (tokenized, view.view_id),
            )
        conn.commit()
    # Verify
    with sqlite3.connect(str(db_path)) as conn:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        fts_count = conn.execute("SELECT COUNT(*) FROM fts_index").fetchone()[0]
    return {
        "lane": "candidate_structured_bm25",
        "view_count": len(views),
        "doc_count": doc_count,
        "fts_count": fts_count,
        "ok": doc_count == fts_count == len(views),
        "db_path": str(db_path),
    }


def build_structured_dense(views: list[CandidateAlignedView], out_dir: Path) -> dict:
    """Build Dense index (all-MiniLM-L6-v2) for structured views."""
    import numpy as np  # type: ignore
    from sentence_transformers import SentenceTransformer  # type: ignore

    dense_dir = out_dir / "candidate_structured_dense" / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    ids = [view.view_id for view in views]
    texts = [view.retrieval_text for view in views]

    encoder = SentenceTransformer(ENCODER_MODEL, device="cpu")
    batch_size = 256 if len(texts) > 512 else 32
    vectors = np.asarray(
        encoder.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 1000,
        ),
        dtype=np.float32,
    )

    ids_path = dense_dir / "ids.json"
    vector_path = dense_dir / "vectors.npy"
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    np.save(vector_path, vectors)

    # Verify
    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    zero_rows = int(np.all(vectors == 0, axis=1).sum())
    return {
        "lane": "candidate_structured_dense",
        "view_count": len(views),
        "vector_count": vectors.shape[0],
        "vector_dim": vectors.shape[1],
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_rows,
        "ok": (
            vectors.shape[0] == len(views)
            and nan_count == 0
            and inf_count == 0
            and zero_rows == 0
        ),
        "ids_path": str(ids_path),
        "vector_path": str(vector_path),
    }


def build_control_metadata(views: list[CandidateAlignedView], out_dir: Path) -> dict:
    """Build metadata SQLite for control index (structured-only)."""
    meta_path = out_dir / "candidate-metadata.sqlite"
    if meta_path.exists():
        meta_path.unlink()
    lane_counts: dict[str, int] = {
        "candidate_structured_bm25": 0,
        "candidate_structured_dense": 0,
    }
    with sqlite3.connect(str(meta_path)) as conn:
        conn.execute(
            "CREATE TABLE view_metadata ("
            "lane TEXT NOT NULL, "
            "view_id TEXT NOT NULL, "
            "candidate_key TEXT NOT NULL, "
            "view_type TEXT NOT NULL, "
            "retrieval_text TEXT NOT NULL, "
            "document_id TEXT, "
            "metadata_json TEXT NOT NULL, "
            "PRIMARY KEY (lane, view_id))"
        )
        conn.execute("CREATE INDEX idx_candidate_key ON view_metadata(candidate_key)")
        for view in views:
            meta = json.dumps(view.to_dict(), ensure_ascii=False, sort_keys=True)
            for lane in ("candidate_structured_bm25", "candidate_structured_dense"):
                conn.execute(
                    "INSERT INTO view_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        lane,
                        view.view_id,
                        view.candidate_key,
                        view.view_type,
                        view.retrieval_text,
                        view.document_id,
                        meta,
                    ),
                )
                lane_counts[lane] += 1
        conn.commit()
    return {"metadata_path": str(meta_path), "lane_counts": lane_counts}


def compute_structured_index_hash(index_dir: Path) -> dict[str, str]:
    """Compute hash of a structured index (view_ids + retrieval_text)."""
    meta_path = index_dir / "candidate-metadata.sqlite"
    if not meta_path.exists():
        return {"error": "metadata_not_found", "view_id_hash": "", "text_hash": ""}
    view_ids: list[str] = []
    texts: list[str] = []
    with sqlite3.connect(str(meta_path)) as conn:
        cur = conn.execute(
            "SELECT candidate_key, retrieval_text FROM view_metadata "
            "WHERE lane = 'candidate_structured_bm25' ORDER BY candidate_key"
        )
        for row in cur:
            view_ids.append(row[0])
            texts.append(row[1])
    return {
        "view_id_hash": _sha256_text("\n".join(view_ids))[:16],
        "text_hash": _sha256_text("\n".join(texts))[:16],
        "view_count": len(view_ids),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print("=" * 70)
    print("Gate 08 R3-R0: Aligned 628 Control Index")
    print("=" * 70)

    R3_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract key sets
    print("\n--- Step 1: Extract Key Sets ---")
    keysets = extract_keysets()

    k_common_set = set(keysets["k_common"])
    k_new_set = set(keysets["k_new"])

    # Output control-keyset-audit.json
    audit_path = R3_DIR / "control-keyset-audit.json"
    audit = {
        "gate": "gate-08-r3-r0",
        "description": "Key set audit for Aligned Control experiment",
        "k_legacy_source": "Gate 08 R2 view-pairs.jsonl (structured_view present)",
        "k_new_source": "Gate 05 R5 structured-views.jsonl (Grade-A)",
        "legacy_count": keysets["legacy_count"],
        "new_count": keysets["new_count"],
        "common_count": keysets["common_count"],
        "removed_count": keysets["removed_count"],
        "added_count": keysets["added_count"],
        "k_legacy": keysets["k_legacy"],
        "k_common": keysets["k_common"],
        "legacy_removed": keysets["legacy_removed"],
        "new_added_sample": keysets["new_added_sample"],
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Audit: {audit_path}")

    # Step 2: Build S-Control index
    print("\n--- Step 2: Build S-Control Index ---")
    print(f"  K_common keys: {len(k_common_set)}")
    print(f"  Serializer: {STRUCTURED_TEXT_VERSION}")
    print(f"  Encoder: {ENCODER_MODEL}")

    r5_structured = load_r5_structured_for_keys(k_common_set)
    print(f"  R5 structured views loaded: {len(r5_structured)}")

    control_views = build_control_structured_views(k_common_set, r5_structured)
    print(f"  Control structured views: {len(control_views)}")

    CONTROL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("\n  Building BM25 lane...")
    bm25_stat = build_structured_bm25(control_views, CONTROL_INDEX_DIR)
    print(
        f"    view_count={bm25_stat['view_count']} doc_count={bm25_stat['doc_count']} ok={bm25_stat['ok']}"
    )

    print("  Building Dense lane...")
    dense_stat = build_structured_dense(control_views, CONTROL_INDEX_DIR)
    print(
        f"    view_count={dense_stat['view_count']} vector_count={dense_stat['vector_count']} ok={dense_stat['ok']}"
    )

    print("  Building metadata...")
    meta_stat = build_control_metadata(control_views, CONTROL_INDEX_DIR)
    print(f"    lane_counts={meta_stat['lane_counts']}")

    # Step 3: Verify S-Legacy and S-Expanded exist
    print("\n--- Step 3: Verify S-Legacy and S-Expanded ---")

    s_legacy_hash = compute_structured_index_hash(GATE08_R2_INDEX_DIR)
    print(
        f"  S-Legacy (Gate 08 R2): view_count={s_legacy_hash.get('view_count', 'N/A')} hash={s_legacy_hash.get('view_id_hash', 'N/A')}"
    )

    s_expanded_hash = compute_structured_index_hash(GATE06_R4_INDEX_DIR)
    print(
        f"  S-Expanded (Gate 06 R4): view_count={s_expanded_hash.get('view_count', 'N/A')} hash={s_expanded_hash.get('view_id_hash', 'N/A')}"
    )

    s_control_hash = compute_structured_index_hash(CONTROL_INDEX_DIR)
    print(
        f"  S-Control (NEW): view_count={s_control_hash.get('view_count', 'N/A')} hash={s_control_hash.get('view_id_hash', 'N/A')}"
    )

    # Step 4: R0 integrity gates
    print("\n--- Step 4: R0 Integrity Gates ---")
    gates: dict[str, bool] = {}

    # Control key set = K_common exact
    control_keys_in_index = {v.candidate_key for v in control_views}
    gates["control_keyset_is_k_common"] = control_keys_in_index == k_common_set
    print(f"  control_keyset_is_k_common: {gates['control_keyset_is_k_common']}")

    # Expanded key set = 19,500 exact
    gates["expanded_keyset_is_19500"] = len(k_new_set) == 19500
    print(f"  expanded_keyset_is_19500: {gates['expanded_keyset_is_19500']}")

    # Control count
    gates["control_count_is_common_count"] = (
        len(control_views) == keysets["common_count"]
    )
    print(f"  control_count_is_common_count: {gates['control_count_is_common_count']}")

    # BM25 integrity
    gates["bm25_ok"] = bm25_stat["ok"]
    print(f"  bm25_ok: {gates['bm25_ok']}")

    # Dense integrity
    gates["dense_ok"] = dense_stat["ok"]
    print(f"  dense_ok: {gates['dense_ok']}")

    # S-Legacy exists
    gates["s_legacy_exists"] = s_legacy_hash.get("view_count", 0) == 628
    print(f"  s_legacy_exists (628): {gates['s_legacy_exists']}")

    # S-Expanded exists
    gates["s_expanded_exists"] = s_expanded_hash.get("view_count", 0) == 19500
    print(f"  s_expanded_exists (19500): {gates['s_expanded_exists']}")

    # Dense model same
    gates["dense_model_same"] = dense_stat.get("vector_dim") == 384
    print(f"  dense_model_same (dim=384): {gates['dense_model_same']}")

    # Security
    gates["question_reads"] = True  # No questions read in this script
    gates["gold_reads"] = True  # No gold read in this script
    gates["retrieval_runs"] = True  # No retrieval performed
    print("  question_reads=0, gold_reads=0, retrieval_runs=0: True")

    all_passed = all(gates.values())

    # Step 5: Output aligned-control-index-manifest.json
    print("\n--- Step 5: Output Manifest ---")
    manifest = {
        "gate": "gate-08-r3-r0",
        "structured_text_version": STRUCTURED_TEXT_VERSION,
        "encoder_model": ENCODER_MODEL,
        "gates": gates,
        "all_gates_passed": all_passed,
        "keysets": {
            "legacy_count": keysets["legacy_count"],
            "new_count": keysets["new_count"],
            "common_count": keysets["common_count"],
            "removed_count": keysets["removed_count"],
            "added_count": keysets["added_count"],
        },
        "s_legacy": {
            "source": "Gate 08 R2 candidate-indexes",
            "index_dir": str(GATE08_R2_INDEX_DIR),
            "view_count": s_legacy_hash.get("view_count"),
            "view_id_hash": s_legacy_hash.get("view_id_hash"),
            "text_hash": s_legacy_hash.get("text_hash"),
            "serializer": "gate08-r2-v1 (old R2 template)",
        },
        "s_control": {
            "source": "NEW (K_common + gate06-r4-v1 serializer)",
            "index_dir": str(CONTROL_INDEX_DIR),
            "view_count": len(control_views),
            "view_id_hash": s_control_hash.get("view_id_hash"),
            "text_hash": s_control_hash.get("text_hash"),
            "serializer": STRUCTURED_TEXT_VERSION,
            "bm25": bm25_stat,
            "dense": dense_stat,
        },
        "s_expanded": {
            "source": "Gate 06 R4 candidate-indexes",
            "index_dir": str(GATE06_R4_INDEX_DIR),
            "view_count": s_expanded_hash.get("view_count"),
            "view_id_hash": s_expanded_hash.get("view_id_hash"),
            "text_hash": s_expanded_hash.get("text_hash"),
            "serializer": STRUCTURED_TEXT_VERSION,
        },
        "security": {
            "question_reads": 0,
            "gold_reads": 0,
            "governance_reads": 0,
            "retrieval_runs": 0,
            "reranker_calls": 0,
            "production_index_writes": 0,
            "production_switch_allowed": False,
        },
    }

    manifest_path = R3_DIR / "aligned-control-index-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Manifest: {manifest_path}")

    # Index build report
    report = {
        "total_views": len(control_views),
        "lanes": {
            "candidate_structured_bm25": bm25_stat,
            "candidate_structured_dense": dense_stat,
        },
        "metadata": meta_stat,
        "integrity": {
            "bm25_ok": bm25_stat["ok"],
            "dense_ok": dense_stat["ok"],
            "ok": bm25_stat["ok"] and dense_stat["ok"],
        },
    }
    report_path = CONTROL_INDEX_DIR / "index-build-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    if all_passed:
        print("ALL R0 GATES PASSED")
        print("decision = aligned_control_index_ready")
        print("next_step = gate_08_r3_a_config_parity")
    else:
        print("R0 GATES FAILED")
        for gate_name, gate_ok in gates.items():
            if not gate_ok:
                print(f"  FAIL: {gate_name}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
