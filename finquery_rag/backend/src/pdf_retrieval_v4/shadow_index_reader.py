"""Read-only readers for the V4 Gate 06 typed shadow indexes.

The reader deliberately has no write path.  BM25 searches the sealed FTS5
index and dense searches the sealed vectors with the Gate 06 embedding model.
Local row retrieval always receives an explicit row-id allow-list, so it
cannot accidentally become a global-row search followed by filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
import sqlite3
from pathlib import Path
from typing import Iterable, Sequence

try:  # numpy is only needed when the real Shadow Reader is opened.
    import numpy as np
except ImportError:  # pragma: no cover - lightweight unit-test environments
    np = None  # type: ignore[assignment]


UNIT_TYPES = (
    "section",
    "table",
    "row",
    "cell",
    "atomic_fact",
    "comparison_fact",
    "bucket_fact",
)


@dataclass(frozen=True)
class SearchHit:
    retrieval_view_id: str
    unit_type: str
    bm25_rank: int | None
    dense_rank: int | None
    fused_rank: int
    bm25_score: float | None = None
    dense_score: float | None = None
    rrf_score: float = 0.0


class ShadowIndexReader:
    """Read the sealed Gate 06 R2 SQLite/NumPy indexes."""

    def __init__(self, runtime_dir: Path, *, rrf_k: int = 60) -> None:
        self.runtime_dir = Path(runtime_dir)
        if self.runtime_dir.name != "pdf-retrieval-v4-gate-06-r2":
            raise ValueError("unsafe_shadow_runtime_path")
        self.rrf_k = int(rrf_k)
        self._metadata = sqlite3.connect(
            f"file:{(self.runtime_dir / 'metadata' / 'metadata.sqlite').resolve().as_posix()}?mode=ro",
            uri=True,
        )
        self._views: dict[str, dict[str, dict[str, object]]] = {typ: {} for typ in UNIT_TYPES}
        self._view_to_type: dict[str, str] = {}
        self._load_metadata()
        self._vectors: dict[str, np.ndarray] = {}
        self._vector_ids: dict[str, list[str]] = {}
        self._vector_positions: dict[str, dict[str, int]] = {}
        self._load_vectors()
        self._encoder = None
        self._query_vector_cache: dict[str, np.ndarray] = {}

    def close(self) -> None:
        self._metadata.close()

    def __enter__(self) -> "ShadowIndexReader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _json(value: str | None, default: object) -> object:
        try:
            return json.loads(value or "")
        except (TypeError, json.JSONDecodeError):
            return default

    def _load_metadata(self) -> None:
        rows = self._metadata.execute(
            "SELECT retrieval_view_id, evidence_unit_id, unit_type, retrieval_text, metadata_json "
            "FROM retrieval_views"
        )
        for view_id, evidence_id, unit_type, text, metadata_json in rows:
            typ = str(unit_type)
            if typ not in self._views:
                continue
            item = {
                "retrieval_view_id": str(view_id),
                "evidence_unit_id": str(evidence_id),
                "unit_type": typ,
                "retrieval_text": str(text or ""),
                "metadata": self._json(metadata_json, {}),
            }
            self._views[typ][str(view_id)] = item
            self._view_to_type[str(view_id)] = typ

    def _load_vectors(self) -> None:
        if np is None:
            raise RuntimeError("numpy_required_for_shadow_reader")
        for typ in UNIT_TYPES:
            ids_path = self.runtime_dir / typ / "dense" / "ids.json"
            vector_path = self.runtime_dir / typ / "dense" / "vectors.npy"
            if not ids_path.is_file() or not vector_path.is_file():
                continue
            ids = [str(item) for item in json.loads(ids_path.read_text(encoding="utf-8"))]
            vectors = np.asarray(np.load(vector_path, allow_pickle=False), dtype=np.float32)
            if len(ids) != len(vectors):
                raise ValueError(f"dense_count_mismatch:{typ}")
            if vectors.ndim != 2 or not np.isfinite(vectors).all():
                raise ValueError(f"dense_vector_invalid:{typ}")
            self._vector_ids[typ] = ids
            self._vectors[typ] = vectors
            self._vector_positions[typ] = {view_id: index for index, view_id in enumerate(ids)}

    @property
    def view_counts(self) -> dict[str, int]:
        return {typ: len(self._views[typ]) for typ in UNIT_TYPES}

    def view(self, view_id: str) -> dict[str, object] | None:
        typ = self._view_to_type.get(str(view_id))
        return self._views.get(typ, {}).get(str(view_id)) if typ else None

    def views(self, unit_type: str) -> list[dict[str, object]]:
        return list(self._views[unit_type]) and list(self._views[unit_type].values())

    def view_ids_for_scope(self, unit_type: str, document_scope: Iterable[str]) -> set[str]:
        scope = {str(value) for value in document_scope}
        return {
            view_id
            for view_id, item in self._views[unit_type].items()
            if str((item.get("metadata") or {}).get("document_id", "")) in scope
        }

    def _tokenize(self, query: str) -> list[str]:
        try:
            import jieba_fast  # type: ignore

            tokens = jieba_fast.cut_for_search(str(query).lower())
        except ImportError:
            tokens = re.findall(r"[a-z0-9]+", str(query).lower())
        result: list[str] = []
        for token in tokens:
            token = str(token).strip().replace('"', " ")
            if token and len(token) > 1 and re.search(r"[\w\u4e00-\u9fff]", token):
                result.append(token)
        return list(dict.fromkeys(result))[:32]

    def _fts_search(
        self,
        unit_type: str,
        query: str,
        allowed_ids: set[str] | None,
        k: int,
    ) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        match = " OR ".join(f'"{token}"' for token in tokens)
        db_path = self.runtime_dir / unit_type / "bm25" / "index.sqlite"
        if not db_path.is_file():
            return []
        with sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True) as conn:
            rows: list[tuple[str, float]] = []
            if allowed_ids is None:
                query_rows = conn.execute(
                    "SELECT f.retrieval_view_id, bm25(fts_index) FROM fts_index f "
                    "JOIN documents d ON d.retrieval_view_id=f.retrieval_view_id "
                    "WHERE fts_index MATCH ? ORDER BY bm25(fts_index) ASC LIMIT ?",
                    (match, max(k, 1)),
                ).fetchall()
                rows = [(str(view_id), -float(score)) for view_id, score in query_rows]
            else:
                ids = sorted(set(allowed_ids) & set(self._views[unit_type]))
                # SQLite's default variable limit is 999.  Query chunks are
                # merged deterministically after each bounded FTS call.
                for offset in range(0, len(ids), 800):
                    chunk = ids[offset : offset + 800]
                    placeholders = ",".join("?" for _ in chunk)
                    query_rows = conn.execute(
                        "SELECT f.retrieval_view_id, bm25(fts_index) FROM fts_index f "
                        "JOIN documents d ON d.retrieval_view_id=f.retrieval_view_id "
                        f"WHERE fts_index MATCH ? AND f.retrieval_view_id IN ({placeholders}) "
                        "ORDER BY bm25(fts_index) ASC LIMIT ?",
                        (match, *chunk, max(k, 1)),
                    ).fetchall()
                    rows.extend((str(view_id), -float(score)) for view_id, score in query_rows)
                rows.sort(key=lambda value: (-value[1], value[0]))
                rows = rows[:k]
            return rows[:k]

    def _encoder_model(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._encoder

    def _query_vector(self, query: str) -> np.ndarray:
        if np is None:
            raise RuntimeError("numpy_required_for_shadow_reader")
        key = str(query)
        cached = self._query_vector_cache.get(key)
        if cached is not None:
            return cached
        vector = np.asarray(
            self._encoder_model().encode(
                [key], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )[0],
            dtype=np.float32,
        )
        self._query_vector_cache[key] = vector
        return vector

    def _dense_search(
        self,
        unit_type: str,
        query: str,
        allowed_ids: set[str] | None,
        k: int,
    ) -> list[tuple[str, float]]:
        if np is None:
            raise RuntimeError("numpy_required_for_shadow_reader")
        vectors = self._vectors.get(unit_type)
        ids = self._vector_ids.get(unit_type)
        if vectors is None or ids is None or k <= 0:
            return []
        allowed = allowed_ids if allowed_ids is not None else set(ids)
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

    def search(
        self,
        unit_type: str,
        query: str,
        *,
        allowed_ids: set[str] | None = None,
        bm25_k: int = 20,
        dense_k: int = 20,
        fused_k: int = 10,
    ) -> list[SearchHit]:
        if unit_type not in self._views:
            raise ValueError(f"unknown_unit_type:{unit_type}")
        bm25 = self._fts_search(unit_type, query, allowed_ids, bm25_k)
        dense = self._dense_search(unit_type, query, allowed_ids, dense_k)
        bm25_rank = {view_id: rank for rank, (view_id, _) in enumerate(bm25, 1)}
        dense_rank = {view_id: rank for rank, (view_id, _) in enumerate(dense, 1)}
        bm25_score = dict(bm25)
        dense_score = dict(dense)
        all_ids = sorted(set(bm25_rank) | set(dense_rank))
        fused: list[tuple[str, float]] = []
        for view_id in all_ids:
            score = 0.0
            if view_id in bm25_rank:
                score += 1.0 / (self.rrf_k + bm25_rank[view_id])
            if view_id in dense_rank:
                score += 1.0 / (self.rrf_k + dense_rank[view_id])
            fused.append((view_id, score))
        fused.sort(key=lambda value: (-value[1], value[0]))
        return [
            SearchHit(
                retrieval_view_id=view_id,
                unit_type=unit_type,
                bm25_rank=bm25_rank.get(view_id),
                dense_rank=dense_rank.get(view_id),
                fused_rank=rank,
                bm25_score=bm25_score.get(view_id),
                dense_score=dense_score.get(view_id),
                rrf_score=score,
            )
            for rank, (view_id, score) in enumerate(fused[:fused_k], 1)
        ]

    def table_row_view_ids(self, logical_table_id: str) -> set[str]:
        result: set[str] = set()
        rows = self._metadata.execute(
            "SELECT member_view_ids_json FROM table_rows WHERE logical_table_id=?",
            (str(logical_table_id),),
        )
        for (encoded,) in rows:
            values = self._json(encoded, [])
            if isinstance(values, list):
                result.update(str(value) for value in values)
        return result & set(self._views["row"])

    def local_rows(
        self,
        logical_table_ids: Sequence[str],
        query: str,
        *,
        bm25_k: int = 5,
        dense_k: int = 5,
        total_cap: int = 40,
    ) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for table_rank, table_id in enumerate(logical_table_ids, 1):
            row_ids = self.table_row_view_ids(table_id)
            local = self.search("row", query, allowed_ids=row_ids, bm25_k=bm25_k, dense_k=dense_k, fused_k=max(bm25_k, dense_k))
            hits.extend(
                SearchHit(
                    retrieval_view_id=item.retrieval_view_id,
                    unit_type=item.unit_type,
                    bm25_rank=item.bm25_rank,
                    dense_rank=item.dense_rank,
                    fused_rank=item.fused_rank,
                    bm25_score=item.bm25_score,
                    dense_score=item.dense_score,
                    rrf_score=item.rrf_score,
                )
                for item in local
            )
        table_rank_map = {str(table_id): rank for rank, table_id in enumerate(logical_table_ids, 1)}
        def key(item: SearchHit) -> tuple[int, int, str]:
            view = self.view(item.retrieval_view_id) or {}
            metadata = view.get("metadata") or {}
            table_id = str(metadata.get("logical_table_id", ""))
            return (table_rank_map.get(table_id, 10**6), item.fused_rank, item.retrieval_view_id)
        return sorted(hits, key=key)[:total_cap]
