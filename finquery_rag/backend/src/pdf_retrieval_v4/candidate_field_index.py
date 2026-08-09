"""Isolated FTS5 indexes for Gate 08 R5 candidate fields."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.pdf_retrieval_v4.candidate_view_index import CandidateSearchHit, _tokenize, _tokenize_for_index

from src.pdf_retrieval_v4.candidate_field_view import CandidateFieldView, FIELD_NAMES

LANE_BY_FIELD = {field: f"structured_{field}_bm25" for field in FIELD_NAMES}


class CandidateFieldIndex:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def build(self, views: dict[str, list[CandidateFieldView]]) -> dict:
        self.root.mkdir(parents=True, exist_ok=True)
        report: dict[str, object] = {"lanes": {}}
        for field in FIELD_NAMES:
            lane = LANE_BY_FIELD[field]
            lane_dir = self.root / lane / "bm25"
            lane_dir.mkdir(parents=True, exist_ok=True)
            path = lane_dir / "index.sqlite"
            if path.exists():
                path.unlink()
            records = sorted(views[field], key=lambda item: item.field_view_id)
            with sqlite3.connect(path) as connection:
                connection.execute("CREATE TABLE documents(field_view_id TEXT PRIMARY KEY,candidate_key TEXT NOT NULL,retrieval_text TEXT NOT NULL,document_id TEXT NOT NULL)")
                connection.execute("CREATE VIRTUAL TABLE fts_index USING fts5(content,field_view_id UNINDEXED,tokenize='unicode61')")
                for view in records:
                    connection.execute("INSERT INTO documents VALUES(?,?,?,?)", (view.field_view_id, view.candidate_key, view.retrieval_text, view.document_id))
                    connection.execute("INSERT INTO fts_index(content,field_view_id) VALUES(?,?)", (_tokenize_for_index(view.retrieval_text), view.field_view_id))
                connection.commit()
            report["lanes"][lane] = {"document_count": len(records), "nonempty_count": sum(bool(item.retrieval_text) for item in records), "path": str(path)}
        return report


class CandidateFieldIndexReader:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def candidate_keys_for_documents(self, field: str, document_ids: set[str]) -> set[str]:
        lane = LANE_BY_FIELD[field]
        path = self.root / lane / "bm25" / "index.sqlite"
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as connection:
            placeholders = ",".join("?" for _ in document_ids)
            rows = connection.execute(f"SELECT candidate_key FROM documents WHERE document_id IN ({placeholders})", tuple(sorted(document_ids))).fetchall()
        return {str(row[0]) for row in rows}

    def search(self, field: str, query: str, *, allowed_candidate_keys: set[str] | None, k: int = 50) -> list[CandidateSearchHit]:
        if field not in FIELD_NAMES:
            raise ValueError(f"unknown_field:{field}")
        tokens = _tokenize(query)
        if not tokens:
            return []
        lane = LANE_BY_FIELD[field]
        path = self.root / lane / "bm25" / "index.sqlite"
        match = " OR ".join(f'"{token}"' for token in tokens)
        with sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True) as connection:
            rows = connection.execute("SELECT d.candidate_key,d.field_view_id,bm25(fts_index) FROM fts_index f JOIN documents d ON d.field_view_id=f.field_view_id WHERE fts_index MATCH ? ORDER BY bm25(fts_index),d.candidate_key LIMIT ?", (match, 19500)).fetchall()
        filtered = [row for row in rows if allowed_candidate_keys is None or str(row[0]) in allowed_candidate_keys][:k]
        return [CandidateSearchHit(candidate_key=str(key), view_id=str(view_id), lane=lane, bm25_rank=rank, dense_rank=None, bm25_score=-float(score), dense_score=None) for rank, (key, view_id, score) in enumerate(filtered, 1)]
