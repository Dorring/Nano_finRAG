"""Gate 08 R2 Candidate-aligned Direct Retrieval shadow index builder + reader.

Builds 4 isolated shadow lanes for candidate-aligned retrieval:

  - candidate_raw_bm25:         FTS5 over raw view text
  - candidate_raw_dense:        MiniLM vectors over raw view text
  - candidate_structured_bm25:  FTS5 over structured view text
  - candidate_structured_dense: MiniLM vectors over structured view text

Each lane is a separate index directory.  BM25 lanes store ``index.sqlite``
(FTS5 with the same schema as Gate 06 R2: ``documents`` table + virtual
``fts_index`` table).  Dense lanes store ``ids.json`` + ``vectors.npy``
(normalized float32 2D array).

A shared metadata SQLite at ``{out_dir}/candidate-metadata.sqlite`` maps
view_id -> candidate_key for every lane.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

try:  # numpy is only needed when dense indexes are read or built.
    import numpy as np
except ImportError:  # pragma: no cover - lightweight unit-test environments
    np = None  # type: ignore[assignment]

from src.pdf_retrieval_v4.candidate_aligned_view import (
    CandidateAlignedView,
    CandidateViewPair,
)


LANES = (
    "candidate_raw_bm25",
    "candidate_raw_dense",
    "candidate_structured_bm25",
    "candidate_structured_dense",
)

BM25_LANES = ("candidate_raw_bm25", "candidate_structured_bm25")
DENSE_LANES = ("candidate_raw_dense", "candidate_structured_dense")

_LANE_VIEW_TYPE = {
    "candidate_raw_bm25": "raw",
    "candidate_raw_dense": "raw",
    "candidate_structured_bm25": "structured",
    "candidate_structured_dense": "structured",
}

_LANE_INDEX_KIND = {
    "candidate_raw_bm25": "bm25",
    "candidate_raw_dense": "dense",
    "candidate_structured_bm25": "bm25",
    "candidate_structured_dense": "dense",
}

_FTS_VARIABLE_CHUNK = 800  # SQLite default variable limit is 999.


@dataclass(frozen=True)
class CandidateSearchHit:
    """A single retrieval hit from one candidate-aligned lane."""

    candidate_key: str
    view_id: str
    lane: str
    bm25_rank: int | None
    dense_rank: int | None
    bm25_score: float | None
    dense_score: float | None


def _json(value: str | None, default: object) -> object:
    try:
        return json.loads(value or "")
    except (TypeError, json.JSONDecodeError):
        return default


def _tokenize(text: str) -> list[str]:
    """Tokenize text using jieba_fast/jieba, with regex fallback.

    Mirrors the tokenization used by the Gate 06 R2 shadow reader so that
    indexed tokens and query tokens share the same vocabulary.
    """
    try:
        import jieba_fast  # type: ignore

        tokens = jieba_fast.cut_for_search(str(text).lower())
    except ImportError:
        try:
            import jieba  # type: ignore

            tokens = jieba.cut_for_search(str(text).lower())
        except ImportError:
            tokens = re.findall(r"[a-z0-9]+", str(text).lower())
    result: list[str] = []
    for token in tokens:
        token = str(token).strip().replace('"', " ")
        if token and len(token) > 1 and re.search(r"[\w\u4e00-\u9fff]", token):
            result.append(token)
    return list(dict.fromkeys(result))[:32]


def _tokenize_for_index(text: str) -> str:
    """Space-joined tokens for FTS5 content insertion."""
    return " ".join(_tokenize(text))


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class CandidateViewIndexBuilder:
    """Build 4 isolated shadow lanes for candidate-aligned retrieval."""

    def __init__(self, out_dir: Path, encoder_model: str = "all-MiniLM-L6-v2") -> None:
        self.out_dir = Path(out_dir)
        self.encoder_model = encoder_model
        self._encoder = None
        self._stats: dict[str, Any] = {}

    # -- helpers --------------------------------------------------------

    def _encoder_instance(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer(self.encoder_model, device="cpu")
        return self._encoder

    def _lane_dir(self, lane: str) -> Path:
        return self.out_dir / lane / _LANE_INDEX_KIND[lane]

    def _views_for_lane(
        self, view_pairs: list[CandidateViewPair], lane: str
    ) -> list[CandidateAlignedView]:
        view_type = _LANE_VIEW_TYPE[lane]
        views: list[CandidateAlignedView] = []
        for pair in view_pairs:
            if view_type == "raw":
                views.append(pair.raw_view)
            elif pair.structured_view is not None:
                views.append(pair.structured_view)
        return views

    # -- lane builders --------------------------------------------------

    def _build_bm25_lane(
        self, lane: str, views: list[CandidateAlignedView]
    ) -> dict[str, Any]:
        lane_dir = self._lane_dir(lane)
        lane_dir.mkdir(parents=True, exist_ok=True)
        db_path = lane_dir / "index.sqlite"
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
        return {"lane": lane, "view_count": len(views), "db_path": str(db_path)}

    def _build_dense_lane(
        self, lane: str, views: list[CandidateAlignedView]
    ) -> dict[str, Any]:
        if np is None:
            raise RuntimeError("numpy_required_for_dense_index")
        lane_dir = self._lane_dir(lane)
        lane_dir.mkdir(parents=True, exist_ok=True)
        ids = [view.view_id for view in views]
        texts = [view.retrieval_text for view in views]
        if not texts:
            vectors = np.zeros((0, 384), dtype=np.float32)
        else:
            encoder = self._encoder_instance()
            # Use a larger batch size and show progress for long-running encodes.
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
        ids_path = lane_dir / "ids.json"
        vector_path = lane_dir / "vectors.npy"
        ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        np.save(vector_path, vectors)
        return {
            "lane": lane,
            "view_count": len(views),
            "ids_path": str(ids_path),
            "vector_path": str(vector_path),
        }

    def _build_metadata(
        self, view_pairs: list[CandidateViewPair]
    ) -> dict[str, Any]:
        meta_path = self.out_dir / "candidate-metadata.sqlite"
        if meta_path.exists():
            meta_path.unlink()
        lane_counts: dict[str, int] = {lane: 0 for lane in LANES}
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
            conn.execute(
                "CREATE INDEX idx_candidate_key ON view_metadata(candidate_key)"
            )
            for pair in view_pairs:
                raw_view = pair.raw_view
                raw_meta = json.dumps(
                    raw_view.to_dict(), ensure_ascii=False, sort_keys=True
                )
                for lane in ("candidate_raw_bm25", "candidate_raw_dense"):
                    conn.execute(
                        "INSERT INTO view_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (
                            lane,
                            raw_view.view_id,
                            raw_view.candidate_key,
                            raw_view.view_type,
                            raw_view.retrieval_text,
                            raw_view.document_id,
                            raw_meta,
                        ),
                    )
                    lane_counts[lane] += 1
                if pair.structured_view is not None:
                    sv = pair.structured_view
                    sv_meta = json.dumps(
                        sv.to_dict(), ensure_ascii=False, sort_keys=True
                    )
                    for lane in ("candidate_structured_bm25", "candidate_structured_dense"):
                        conn.execute(
                            "INSERT INTO view_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (
                                lane,
                                sv.view_id,
                                sv.candidate_key,
                                sv.view_type,
                                sv.retrieval_text,
                                sv.document_id,
                                sv_meta,
                            ),
                        )
                        lane_counts[lane] += 1
            conn.commit()
        return {"metadata_path": str(meta_path), "lane_counts": lane_counts}

    # -- public API -----------------------------------------------------

    def build(self, view_pairs: list[CandidateViewPair]) -> dict[str, Any]:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        # Verify no candidate_key duplicates.
        seen_keys: set[str] = set()
        for pair in view_pairs:
            if pair.candidate_key in seen_keys:
                raise ValueError(f"duplicate_candidate_key:{pair.candidate_key}")
            seen_keys.add(pair.candidate_key)
        lane_stats: dict[str, Any] = {}
        for lane in LANES:
            views = self._views_for_lane(view_pairs, lane)
            if lane in BM25_LANES:
                lane_stats[lane] = self._build_bm25_lane(lane, views)
            else:
                lane_stats[lane] = self._build_dense_lane(lane, views)
        meta_info = self._build_metadata(view_pairs)
        self._stats = {
            "total_pairs": len(view_pairs),
            "unique_candidate_keys": len(seen_keys),
            "lanes": lane_stats,
            "metadata": meta_info,
        }
        integrity = self.verify_integrity()
        self._stats["integrity"] = integrity
        return self._stats

    def verify_integrity(self) -> dict[str, Any]:
        report: dict[str, Any] = {"lanes": {}, "metadata": {}, "ok": True}
        for lane in LANES:
            lane_report = self._verify_lane(lane)
            report["lanes"][lane] = lane_report
            if not lane_report.get("ok", False):
                report["ok"] = False
        meta_report = self._verify_metadata()
        report["metadata"] = meta_report
        if not meta_report.get("ok", False):
            report["ok"] = False
        return report

    def _verify_lane(self, lane: str) -> dict[str, Any]:
        kind = _LANE_INDEX_KIND[lane]
        lane_dir = self._lane_dir(lane)
        if kind == "bm25":
            return self._verify_bm25_lane(lane, lane_dir)
        return self._verify_dense_lane(lane, lane_dir)

    def _verify_bm25_lane(self, lane: str, lane_dir: Path) -> dict[str, Any]:
        db_path = lane_dir / "index.sqlite"
        if not db_path.is_file():
            return {"ok": False, "error": "missing_index_sqlite"}
        with sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            fts_count = conn.execute("SELECT COUNT(*) FROM fts_index").fetchone()[0]
            dupes = conn.execute(
                "SELECT retrieval_view_id, COUNT(*) c FROM documents "
                "GROUP BY retrieval_view_id HAVING c > 1"
            ).fetchall()
        issues: list[str] = []
        if doc_count != fts_count:
            issues.append("doc_fts_count_mismatch")
        if dupes:
            issues.append("duplicate_view_ids")
        return {
            "ok": not issues,
            "doc_count": doc_count,
            "fts_count": fts_count,
            "duplicate_view_ids": [r[0] for r in dupes],
            "issues": issues,
        }

    def _verify_dense_lane(self, lane: str, lane_dir: Path) -> dict[str, Any]:
        ids_path = lane_dir / "ids.json"
        vector_path = lane_dir / "vectors.npy"
        if not ids_path.is_file() or not vector_path.is_file():
            return {"ok": False, "error": "missing_dense_files"}
        if np is None:
            return {"ok": False, "error": "numpy_unavailable"}
        ids = json.loads(ids_path.read_text(encoding="utf-8"))
        vectors = np.load(vector_path, allow_pickle=False)
        issues: list[str] = []
        if len(ids) != len(vectors):
            issues.append("count_mismatch")
        if vectors.ndim != 2:
            issues.append("not_2d")
        else:
            if not np.isfinite(vectors).all():
                issues.append("nan_or_inf")
            norms = np.linalg.norm(vectors, axis=1)
            zero_count = int((norms == 0.0).sum())
            if zero_count > 0:
                issues.append(f"zero_vectors:{zero_count}")
        dupe_ids = len(ids) - len(set(ids))
        if dupe_ids > 0:
            issues.append(f"duplicate_ids:{dupe_ids}")
        return {
            "ok": not issues,
            "vector_count": len(ids),
            "vector_shape": list(vectors.shape),
            "issues": issues,
        }

    def _verify_metadata(self) -> dict[str, Any]:
        meta_path = self.out_dir / "candidate-metadata.sqlite"
        if not meta_path.is_file():
            return {"ok": False, "error": "missing_metadata_sqlite"}
        with sqlite3.connect(f"file:{meta_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            total = conn.execute("SELECT COUNT(*) FROM view_metadata").fetchone()[0]
            dupe_keys = conn.execute(
                "SELECT lane, candidate_key, COUNT(*) c FROM view_metadata "
                "GROUP BY lane, candidate_key HAVING c > 1"
            ).fetchall()
            dupe_views = conn.execute(
                "SELECT lane, view_id, COUNT(*) c FROM view_metadata "
                "GROUP BY lane, view_id HAVING c > 1"
            ).fetchall()
        return {
            "ok": not dupe_keys and not dupe_views,
            "total_rows": total,
            "duplicate_candidate_keys": [(r[0], r[1]) for r in dupe_keys],
            "duplicate_view_ids": [(r[0], r[1]) for r in dupe_views],
        }


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class CandidateViewIndexReader:
    """Read-only reader for the 4 candidate-aligned shadow lanes."""

    def __init__(self, index_dir: Path, *, rrf_k: int = 60) -> None:
        self.index_dir = Path(index_dir)
        self.rrf_k = int(rrf_k)
        meta_path = self.index_dir / "candidate-metadata.sqlite"
        if not meta_path.is_file():
            raise FileNotFoundError(f"metadata_not_found:{meta_path}")
        self._metadata = sqlite3.connect(
            f"file:{meta_path.resolve().as_posix()}?mode=ro", uri=True
        )
        self._view_to_candidate: dict[tuple[str, str], str] = {}
        self._candidate_to_view: dict[tuple[str, str], dict[str, Any]] = {}
        self._lane_view_ids: dict[str, set[str]] = {lane: set() for lane in LANES}
        self._load_metadata()
        self._dense_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        self._encoder = None
        self._query_vector_cache: dict[str, np.ndarray] = {}

    # -- metadata -------------------------------------------------------

    def _load_metadata(self) -> None:
        rows = self._metadata.execute(
            "SELECT lane, view_id, candidate_key, view_type, retrieval_text, "
            "document_id, metadata_json FROM view_metadata"
        )
        for lane, view_id, candidate_key, view_type, text, doc_id, metadata_json in rows:
            lane_s = str(lane)
            view_id_s = str(view_id)
            key_s = str(candidate_key)
            self._view_to_candidate[(lane_s, view_id_s)] = key_s
            self._candidate_to_view[(lane_s, key_s)] = {
                "view_id": view_id_s,
                "candidate_key": key_s,
                "view_type": str(view_type),
                "retrieval_text": str(text or ""),
                "document_id": str(doc_id or ""),
                "metadata": _json(metadata_json, {}),
            }
            self._lane_view_ids.setdefault(lane_s, set()).add(view_id_s)

    # -- BM25 search ----------------------------------------------------

    def _fts_search(
        self,
        lane: str,
        query: str,
        allowed_view_ids: set[str] | None,
        k: int,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        db_path = self.index_dir / lane / "bm25" / "index.sqlite"
        if not db_path.is_file():
            return []
        rows: list[tuple[str, float]] = []
        with sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            if allowed_view_ids is None:
                query_rows = conn.execute(
                    "SELECT f.retrieval_view_id, bm25(fts_index) FROM fts_index f "
                    "JOIN documents d ON d.retrieval_view_id=f.retrieval_view_id "
                    "WHERE fts_index MATCH ? ORDER BY bm25(fts_index) ASC LIMIT ?",
                    (match, max(k, 1)),
                ).fetchall()
                rows = [(str(view_id), -float(score)) for view_id, score in query_rows]
            else:
                ids = sorted(allowed_view_ids & self._lane_view_ids.get(lane, set()))
                for offset in range(0, len(ids), _FTS_VARIABLE_CHUNK):
                    chunk = ids[offset : offset + _FTS_VARIABLE_CHUNK]
                    placeholders = ",".join("?" for _ in chunk)
                    query_rows = conn.execute(
                        "SELECT f.retrieval_view_id, bm25(fts_index) FROM fts_index f "
                        "JOIN documents d ON d.retrieval_view_id=f.retrieval_view_id "
                        f"WHERE fts_index MATCH ? AND f.retrieval_view_id IN ({placeholders}) "
                        "ORDER BY bm25(fts_index) ASC LIMIT ?",
                        (match, *chunk, max(k, 1)),
                    ).fetchall()
                    rows.extend(
                        (str(view_id), -float(score)) for view_id, score in query_rows
                    )
                rows.sort(key=lambda value: (-value[1], value[0]))
        return rows[:k]

    # -- Dense search ---------------------------------------------------

    def _encoder_instance(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._encoder

    def _query_vector(self, query: str) -> np.ndarray:
        if np is None:
            raise RuntimeError("numpy_required_for_dense_search")
        key = str(query)
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            return cached
        vector = np.asarray(
            self._encoder_instance().encode(
                [key], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )[0],
            dtype=np.float32,
        )
        self._query_vector_cache[key] = vector
        return vector

    def _load_dense(self, lane: str) -> tuple[list[str], np.ndarray] | None:
        if np is None:
            raise RuntimeError("numpy_required_for_dense_search")
        cached = self._dense_cache.get(lane)
        if cached is not None:
            return cached
        ids_path = self.index_dir / lane / "dense" / "ids.json"
        vector_path = self.index_dir / lane / "dense" / "vectors.npy"
        if not ids_path.is_file() or not vector_path.is_file():
            return None
        ids = [str(item) for item in json.loads(ids_path.read_text(encoding="utf-8"))]
        vectors = np.asarray(np.load(vector_path, allow_pickle=False), dtype=np.float32)
        if len(ids) != len(vectors):
            raise ValueError(f"dense_count_mismatch:{lane}")
        if vectors.ndim != 2 or not np.isfinite(vectors).all():
            raise ValueError(f"dense_vector_invalid:{lane}")
        self._dense_cache[lane] = (ids, vectors)
        return ids, vectors

    def _dense_search(
        self,
        lane: str,
        query: str,
        allowed_view_ids: set[str] | None,
        k: int,
    ) -> list[tuple[str, float]]:
        if np is None or k <= 0:
            return []
        loaded = self._load_dense(lane)
        if loaded is None:
            return []
        ids, vectors = loaded
        allowed = allowed_view_ids if allowed_view_ids is not None else set(ids)
        positions = [index for index, view_id in enumerate(ids) if view_id in allowed]
        if not positions:
            return []
        query_vector = self._query_vector(query)
        scores = vectors[positions] @ query_vector
        ranked = sorted(
            ((ids[position], float(score)) for position, score in zip(positions, scores)),
            key=lambda value: (-value[1], value[0]),
        )
        return ranked[:k]

    # -- public search --------------------------------------------------

    def search(
        self,
        lane: str,
        query: str,
        *,
        allowed_candidate_keys: set[str] | None = None,
        k: int = 50,
    ) -> list[CandidateSearchHit]:
        if lane not in _LANE_INDEX_KIND:
            raise ValueError(f"unknown_lane:{lane}")
        # Translate candidate_keys to view_ids within this lane.
        allowed_view_ids: set[str] | None = None
        if allowed_candidate_keys is not None:
            allowed_view_ids = set()
            for candidate_key in allowed_candidate_keys:
                view = self._candidate_to_view.get((lane, candidate_key))
                if view is not None:
                    allowed_view_ids.add(view["view_id"])
            if not allowed_view_ids:
                return []
        if lane in BM25_LANES:
            results = self._fts_search(lane, query, allowed_view_ids, k)
            return [
                CandidateSearchHit(
                    candidate_key=self._view_to_candidate.get((lane, view_id), ""),
                    view_id=view_id,
                    lane=lane,
                    bm25_rank=rank,
                    dense_rank=None,
                    bm25_score=score,
                    dense_score=None,
                )
                for rank, (view_id, score) in enumerate(results, 1)
            ]
        results = self._dense_search(lane, query, allowed_view_ids, k)
        return [
            CandidateSearchHit(
                candidate_key=self._view_to_candidate.get((lane, view_id), ""),
                view_id=view_id,
                lane=lane,
                bm25_rank=None,
                dense_rank=rank,
                bm25_score=None,
                dense_score=score,
            )
            for rank, (view_id, score) in enumerate(results, 1)
        ]

    # -- lookups --------------------------------------------------------

    def candidate_key_for_view(self, lane: str, view_id: str) -> str | None:
        return self._view_to_candidate.get((lane, view_id))

    def view_for_candidate(self, lane: str, candidate_key: str) -> dict | None:
        return self._candidate_to_view.get((lane, candidate_key))

    def candidate_keys_for_documents(
        self, lane: str, document_ids: set[str]
    ) -> set[str]:
        """Return all candidate_keys in *lane* whose document_id is in *document_ids*."""
        scope = {str(doc_id) for doc_id in document_ids}
        return {
            candidate_key
            for (lane_name, candidate_key), view in self._candidate_to_view.items()
            if lane_name == lane and str(view.get("document_id", "")) in scope
        }

    # -- lifecycle ------------------------------------------------------

    def close(self) -> None:
        self._metadata.close()

    def __enter__(self) -> CandidateViewIndexReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
