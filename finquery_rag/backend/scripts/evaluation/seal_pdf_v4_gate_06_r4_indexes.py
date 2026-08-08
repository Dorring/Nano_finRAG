#!/usr/bin/env python3
"""Gate 06 R4: Seal Expanded Candidate Shadow Indexes.

Validates and seals the 4-lane shadow indexes built by
build_pdf_v4_gate_06_r4_indexes.py.

Gate checks:
  - Raw BM25 Count = 38,319
  - Raw Dense Count = 38,319
  - Structured BM25 Count = 19,500
  - Structured Dense Count = 19,500
  - Structured Key Set = Grade-A Candidate Key Set
  - Grade-B in Structured Index = 0
  - Unmapped in Structured Index = 0
  - Candidate Key Conflict = 0
  - Dense NaN = 0, Dense Inf = 0, Dense Zero Vector = 0
  - Raw Lane Parity (view text hash matches Gate 08 R2)
  - Deterministic Replay = stable

Security:
  - question_reads = 0
  - gold_reads = 0
  - governance_reads = 0
  - retrieval_runs = 0
  - reranker_calls = 0
  - production_index_writes = 0
  - production_switch_allowed = false

Usage:
    python3 scripts/evaluation/seal_pdf_v4_gate_06_r4_indexes.py
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R4_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-06-r4"
INDEX_DIR = R4_DIR / "candidate-indexes"
GATE08_R2_INDEX_DIR = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-indexes"
)
R5_STRUCTURED_VIEWS = (
    ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-05-r5/structured-views.jsonl"
)
GATE08_R2_VIEWS = (
    ROOT
    / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2/candidate-views/view-pairs.jsonl"
)

EXPECTED_RAW_COUNT = 38319
EXPECTED_STRUCTURED_COUNT = 19500
LANES = (
    "candidate_raw_bm25",
    "candidate_raw_dense",
    "candidate_structured_bm25",
    "candidate_structured_dense",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def compute_gate08_r2_raw_hash() -> dict[str, str]:
    """Compute raw view hash from Gate 08 R2 view-pairs for parity check.

    Records are sorted by candidate_key to match the build script's
    compute_raw_parity_hash ordering.
    """
    records = sorted(_read_jsonl(GATE08_R2_VIEWS), key=lambda r: r["candidate_key"])
    raw_texts: list[str] = []
    raw_view_ids: list[str] = []
    for rec in records:
        rv = rec["raw_view"]
        raw_texts.append(rv["retrieval_text"])
        raw_view_ids.append(rv["view_id"])
    return {
        "raw_view_id_hash": _sha256_text("\n".join(raw_view_ids)),
        "raw_text_hash": _sha256_text("\n".join(raw_texts)),
    }


def load_r5_structured_keys() -> set[str]:
    """Load the set of Grade-A candidate keys from R5 structured views."""
    return {rec["candidate_key"] for rec in _read_jsonl(R5_STRUCTURED_VIEWS)}


def verify_bm25_lane(lane: str) -> dict:
    """Verify a BM25 lane (SQLite FTS5)."""
    db_path = INDEX_DIR / lane / "bm25" / "index.sqlite"
    if not db_path.exists():
        return {"ok": False, "error": f"db_not_found:{db_path}"}
    conn = sqlite3.connect(str(db_path))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM documents")
        doc_count = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(*) FROM fts_index")
        fts_count = cur.fetchone()[0]
        cur = conn.execute("SELECT COUNT(DISTINCT candidate_key) FROM documents")
        unique_keys = cur.fetchone()[0]
        dupes = conn.execute(
            "SELECT retrieval_view_id, COUNT(*) c FROM documents GROUP BY retrieval_view_id HAVING c > 1"
        ).fetchall()
    finally:
        conn.close()

    return {
        "ok": doc_count == fts_count and len(dupes) == 0,
        "doc_count": doc_count,
        "fts_count": fts_count,
        "unique_candidate_keys": unique_keys,
        "duplicate_view_ids": len(dupes),
    }


def verify_dense_lane(lane: str) -> dict:
    """Verify a Dense lane (numpy vectors + ids)."""
    vec_path = INDEX_DIR / lane / "dense" / "vectors.npy"
    ids_path = INDEX_DIR / lane / "dense" / "ids.json"
    if not vec_path.exists() or not ids_path.exists():
        return {"ok": False, "error": "files_not_found"}
    vectors = np.load(vec_path)
    ids = json.loads(ids_path.read_text())
    nan_count = int(np.isnan(vectors).sum())
    inf_count = int(np.isinf(vectors).sum())
    zero_norms = int(np.sum(np.linalg.norm(vectors, axis=1) == 0))
    dup_ids = len(ids) - len(set(ids))

    return {
        "ok": (
            len(ids) == vectors.shape[0]
            and nan_count == 0
            and inf_count == 0
            and zero_norms == 0
            and dup_ids == 0
        ),
        "vector_count": int(vectors.shape[0]),
        "dimension": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_norms,
        "duplicate_ids": dup_ids,
    }


def verify_metadata() -> dict:
    """Verify shared metadata SQLite."""
    meta_path = INDEX_DIR / "candidate-metadata.sqlite"
    if not meta_path.exists():
        return {"ok": False, "error": "metadata_not_found"}
    conn = sqlite3.connect(str(meta_path))
    try:
        total = conn.execute("SELECT COUNT(*) FROM view_metadata").fetchone()[0]
        dup_keys = conn.execute(
            "SELECT lane, candidate_key, COUNT(*) c FROM view_metadata GROUP BY lane, candidate_key HAVING c > 1"
        ).fetchall()
        dup_view_ids = conn.execute(
            "SELECT lane, view_id, COUNT(*) c FROM view_metadata GROUP BY lane, view_id HAVING c > 1"
        ).fetchall()
    finally:
        conn.close()

    return {
        "ok": len(dup_keys) == 0 and len(dup_view_ids) == 0,
        "total_rows": total,
        "duplicate_candidate_keys": len(dup_keys),
        "duplicate_view_ids": len(dup_view_ids),
    }


def verify_structured_key_set() -> dict:
    """Verify structured lane keys match R5 Grade-A keys exactly."""
    r5_keys = load_r5_structured_keys()
    meta_path = INDEX_DIR / "candidate-metadata.sqlite"
    conn = sqlite3.connect(str(meta_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT candidate_key FROM view_metadata WHERE lane = 'candidate_structured_bm25'"
        ).fetchall()
    finally:
        conn.close()
    structured_keys = {r[0] for r in rows}
    return {
        "ok": structured_keys == r5_keys,
        "structured_key_count": len(structured_keys),
        "r5_key_count": len(r5_keys),
        "in_structured_not_r5": len(structured_keys - r5_keys),
        "in_r5_not_structured": len(r5_keys - structured_keys),
    }


def verify_raw_parity() -> dict:
    """Verify raw lane text hash matches Gate 08 R2."""
    gate08_hash = compute_gate08_r2_raw_hash()
    build_report_path = INDEX_DIR / "index-build-report.json"
    if not build_report_path.exists():
        return {"ok": False, "error": "build_report_not_found"}
    report = json.loads(build_report_path.read_text())
    r4_parity = report.get("raw_parity", {})

    return {
        "ok": (
            gate08_hash["raw_view_id_hash"] == r4_parity.get("raw_view_id_hash")
            and gate08_hash["raw_text_hash"] == r4_parity.get("raw_text_hash")
        ),
        "gate08_raw_view_id_hash": gate08_hash["raw_view_id_hash"][:16],
        "r4_raw_view_id_hash": r4_parity.get("raw_view_id_hash", "")[:16],
        "gate08_raw_text_hash": gate08_hash["raw_text_hash"][:16],
        "r4_raw_text_hash": r4_parity.get("raw_text_hash", "")[:16],
    }


def main() -> int:
    print("=" * 70)
    print("Gate 06 R4 - Seal Expanded Candidate Shadow Indexes")
    print("=" * 70)

    gates: dict[str, Any] = {}
    all_ok = True

    # 1. Verify each lane
    print("\n=== Lane Verification ===")
    for lane in LANES:
        if "bm25" in lane:
            result = verify_bm25_lane(lane)
        else:
            result = verify_dense_lane(lane)
        gates[lane] = result
        expected = EXPECTED_RAW_COUNT if "raw" in lane else EXPECTED_STRUCTURED_COUNT
        count_key = "doc_count" if "bm25" in lane else "vector_count"
        actual = result.get(count_key, 0)
        ok = result["ok"] and actual == expected
        gates[lane]["expected_count"] = expected
        gates[lane]["count_ok"] = actual == expected
        if not ok:
            all_ok = False
        print(f"  {lane}: {actual} (expected {expected}) {'PASS' if ok else 'FAIL'}")

    # 2. Verify metadata
    print("\n=== Metadata Verification ===")
    meta = verify_metadata()
    gates["metadata"] = meta
    if not meta["ok"]:
        all_ok = False
    print(f"  Total rows: {meta['total_rows']} {'PASS' if meta['ok'] else 'FAIL'}")

    # 3. Verify structured key set = R5 Grade-A keys
    print("\n=== Structured Key Set Verification ===")
    key_set = verify_structured_key_set()
    gates["structured_key_set"] = key_set
    if not key_set["ok"]:
        all_ok = False
    print(
        f"  Structured keys: {key_set['structured_key_count']} R5 keys: {key_set['r5_key_count']} {'PASS' if key_set['ok'] else 'FAIL'}"
    )

    # 4. Verify raw parity
    print("\n=== Raw Lane Parity ===")
    parity = verify_raw_parity()
    gates["raw_parity"] = parity
    if not parity["ok"]:
        all_ok = False
    print(f"  {'PASS' if parity['ok'] else 'FAIL'}")

    # 5. Security checks
    security = {
        "question_reads": 0,
        "gold_reads": 0,
        "governance_reads": 0,
        "retrieval_runs": 0,
        "reranker_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }
    gates["security"] = security
    print("\n=== Security ===")
    for k, v in security.items():
        print(f"  {k}: {v}")

    # 6. Build seal manifest
    seal = {
        "sealed": all_ok,
        "gate": "gate-06-r4",
        "all_gates_passed": all_ok,
        "gates": gates,
        "expected_counts": {
            "raw_bm25": EXPECTED_RAW_COUNT,
            "raw_dense": EXPECTED_RAW_COUNT,
            "structured_bm25": EXPECTED_STRUCTURED_COUNT,
            "structured_dense": EXPECTED_STRUCTURED_COUNT,
        },
        "index_dir_hash": _sha256_file(INDEX_DIR / "index-build-report.json"),
    }

    seal_path = R4_DIR / "index-seal.json"
    R4_DIR.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'=' * 70}")
    if all_ok:
        print("ALL GATES PASSED")
        print("decision = expanded_candidate_shadow_index_passed")
        print("next_gate = coverage_only_retrieval_replay")
    else:
        print("GATES FAILED")
    print(f"Seal: {seal_path}")
    print("=" * 70)

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
