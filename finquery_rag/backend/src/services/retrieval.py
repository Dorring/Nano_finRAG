import sqlite3
import json
import re
try:
    import jieba_fast as jieba
except ImportError:
    import jieba
from collections import defaultdict
from typing import List, Dict
from .chunk_id import ensure_scoped_chunk_id, make_chunk_id

DB_PATH = "rag_bm25.db"


class SqliteBM25Retriever:
    """
    基于 SQLite FTS5 + jieba_fast 的轻量化稀疏检索器。

    替代原 rank-bm25 内存方案，优势：
    - 数据持久化到 SQLite，进程重启无需重建索引
    - WAL 模式支持高并发读写不阻塞
    - jieba_fast 中文分词替代空格分词，中文检索精度大幅提升
    - 内存占用极低，适合 2C2G 等资源受限环境
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._init_db()

    SCHEMA_VERSION = 2
    MAX_SEARCH_LIMIT = 100
    MAX_MATCH_TOKENS = 32

    def _normalize_limit(self, k: int) -> int:
        try:
            limit = int(k)
        except (TypeError, ValueError):
            return 0
        if limit <= 0:
            return 0
        return min(limit, self.MAX_SEARCH_LIMIT)

    def _normalize_query(self, query) -> str:
        if not isinstance(query, str):
            return ""
        return query.strip()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS chunk_store (
                    doc_id TEXT PRIMARY KEY,
                    content TEXT,
                    metadata_json TEXT,
                    user_id INTEGER,
                    doc_name TEXT
                );
            """)

            # Migration: add doc_name column if missing (idempotent)
            cols = [row[1] for row in cursor.execute("PRAGMA table_info(chunk_store)").fetchall()]
            if "doc_name" not in cols:
                cursor.execute("ALTER TABLE chunk_store ADD COLUMN doc_name TEXT")
                for row in cursor.execute("SELECT doc_id, metadata_json FROM chunk_store WHERE doc_name IS NULL").fetchall():
                    try:
                        meta = json.loads(row[1])
                        dn = meta.get("doc_name", "")
                        cursor.execute("UPDATE chunk_store SET doc_name = ? WHERE doc_id = ?", (dn, row[0]))
                    except Exception:
                        pass

            # Migrate FTS5: if old content-backed table exists, drop and recreate
            try:
                cfg = cursor.execute(
                    "SELECT sql FROM sqlite_master WHERE type='table' AND name='fts_index'"
                ).fetchone()
                if cfg and "content='chunk_store'" in (cfg[0] or ""):
                    cursor.execute("DROP TABLE fts_index")
            except Exception:
                pass

            cursor.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
                    content,
                    doc_id UNINDEXED,
                    tokenize='unicode61'
                );
            """)
            conn.commit()

    def _clean_query(self, query: str) -> str:
        """Build a bounded, injection-safe disjunctive FTS5 query.

        Query expansion adds useful related terms. Passing them to FTS5 as a
        whitespace-separated expression makes them implicit AND terms, which
        can discard relevant chunks that do not contain every expansion token.
        Quoted OR terms retain BM25 ranking while treating expansion as recall
        assistance rather than a mandatory filter.
        """
        tokenized = " ".join(jieba.cut_for_search(query.lower()))
        tokens = re.findall(r"\w+", tokenized, flags=re.UNICODE)
        unique_tokens = list(dict.fromkeys(token for token in tokens if token))
        return " OR ".join(
            f'"{token}"' for token in unique_tokens[:self.MAX_MATCH_TOKENS]
        )

    def _normalize_chunk(self, chunk, user_id: int):
        if not isinstance(chunk, dict):
            return None
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            return None
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
        raw_id = metadata.get("doc_id") or chunk.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            return None
        doc_name = metadata.get("doc_name", "")
        if not isinstance(doc_name, str):
            doc_name = ""
        try:
            doc_id = ensure_scoped_chunk_id(raw_id.strip(), user_id, doc_name)
        except ValueError:
            return None
        return doc_id, content, metadata, doc_name

    def add_chunks(self, chunks: List[Dict], user_id: int = None):
        if user_id is None:
            raise ValueError("user_id is required for add_chunks")
        if not isinstance(chunks, (list, tuple)) or not chunks:
            return
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            for c in chunks:
                normalized = self._normalize_chunk(c, user_id)
                if normalized is None:
                    continue
                doc_id, content, metadata, doc_name = normalized

                tokenized_content = " ".join(jieba.cut_for_search(content.lower()))

                # Delete old FTS rows before upsert to prevent duplicate/stale entries
                cursor.execute("DELETE FROM fts_index WHERE doc_id = ?;", (doc_id,))

                cursor.execute(
                    "INSERT OR REPLACE INTO chunk_store(doc_id, content, metadata_json, user_id, doc_name) VALUES (?, ?, ?, ?, ?);",
                    (doc_id, content, json.dumps(metadata, ensure_ascii=False), user_id, doc_name)
                )

                cursor.execute(
                    "INSERT INTO fts_index(content, doc_id) VALUES (?, ?);",
                    (tokenized_content, doc_id)
                )
            conn.commit()

    def search(self, query: str, k: int = 10, doc_name: str = None, user_id: int = None) -> List[Dict]:
        """
        SQLite FTS5 稀疏检索。
        修复：增加 doc_name 和 user_id 过滤，防止跨文档/跨用户数据泄露。
        """
        limit = self._normalize_limit(k)
        if limit <= 0:
            return []

        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        clean_query = self._clean_query(normalized_query)
        if not clean_query.strip():
            return []

        try:
            with sqlite3.connect(self.db_path, timeout=10) as conn:
                cursor = conn.cursor()

                # 基础 SQL
                sql = """
                    SELECT c.doc_id, c.content, c.metadata_json, bm25(fts_index) as score
                    FROM fts_index f
                    JOIN chunk_store c ON f.doc_id = c.doc_id
                    WHERE fts_index MATCH ?
                """
                params = [clean_query]

                if user_id is None:
                    return []
                sql += " AND c.user_id = ?"
                params.append(user_id)

                # Use exact doc_name column match (no LIKE injection risk)
                if doc_name:
                    sql += " AND c.doc_name = ?"
                    params.append(doc_name)

                sql += " ORDER BY score ASC LIMIT ?;"
                params.append(limit)

                cursor.execute(sql, tuple(params))
                rows = cursor.fetchall()

        except sqlite3.OperationalError as e:
            print(f"BM25 search error: {e}")
            return []

        results = []
        seen_doc_ids = set()
        for row in rows:
            doc_id = row[0]
            if doc_id in seen_doc_ids:
                continue
            try:
                metadata = json.loads(row[2] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            seen_doc_ids.add(doc_id)
            results.append({
                "doc_id": doc_id,
                "content": row[1],
                "metadata": metadata,
                "score": -float(row[3]) # BM25 score in FTS5 is negative, so we negate it
            })
        return results

    def delete_doc(self, doc_name: str, user_id: int):
        if user_id is None:
            raise ValueError("user_id is required for delete_doc")
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            doc_ids = [r[0] for r in cursor.execute(
                "SELECT doc_id FROM chunk_store WHERE doc_name = ? AND user_id = ?",
                (doc_name, user_id)
            ).fetchall()]
            if doc_ids:
                placeholders = ",".join("?" * len(doc_ids))
                cursor.execute(f"DELETE FROM chunk_store WHERE doc_id IN ({placeholders})", doc_ids)
                cursor.execute(f"DELETE FROM fts_index WHERE doc_id IN ({placeholders})", doc_ids)
            conn.commit()


    def delete_all_for_user(self, user_id: int):
        """删除指定用户的所有 BM25 索引条目。"""
        if user_id is None:
            raise ValueError("user_id is required for delete_all_for_user")
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            doc_ids = [r[0] for r in cursor.execute(
                "SELECT doc_id FROM chunk_store WHERE user_id = ?", (user_id,)
            ).fetchall()]
            if doc_ids:
                placeholders = ",".join("?" * len(doc_ids))
                cursor.execute(f"DELETE FROM chunk_store WHERE doc_id IN ({placeholders})", doc_ids)
                cursor.execute(f"DELETE FROM fts_index WHERE doc_id IN ({placeholders})", doc_ids)
            conn.commit()

    def integrity_report(self, user_id: int = None) -> Dict:
        """Return a lightweight consistency report for chunk_store and FTS rows.

        The report does not expose chunk content. When user_id is provided,
        missing and duplicate checks are scoped to that tenant's chunk IDs.
        Orphan FTS rows cannot be safely attributed to a tenant after their
        chunk_store rows disappear, so orphan checks are global-only.
        """
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            if user_id is None:
                chunk_count = cursor.execute("SELECT COUNT(*) FROM chunk_store").fetchone()[0]
                fts_count = cursor.execute("SELECT COUNT(*) FROM fts_index").fetchone()[0]
                missing_rows = cursor.execute("""
                    SELECT c.doc_id
                    FROM chunk_store c
                    LEFT JOIN fts_index f ON f.doc_id = c.doc_id
                    WHERE f.doc_id IS NULL
                    ORDER BY c.doc_id
                """).fetchall()
                duplicate_rows = cursor.execute("""
                    SELECT doc_id, COUNT(*) AS n
                    FROM fts_index
                    GROUP BY doc_id
                    HAVING n > 1
                    ORDER BY doc_id
                """).fetchall()
                orphan_rows = cursor.execute("""
                    SELECT f.doc_id
                    FROM fts_index f
                    LEFT JOIN chunk_store c ON c.doc_id = f.doc_id
                    WHERE c.doc_id IS NULL
                    ORDER BY f.doc_id
                """).fetchall()
            else:
                chunk_count = cursor.execute(
                    "SELECT COUNT(*) FROM chunk_store WHERE user_id = ?",
                    (user_id,),
                ).fetchone()[0]
                fts_count = cursor.execute("""
                    SELECT COUNT(*)
                    FROM fts_index f
                    JOIN chunk_store c ON c.doc_id = f.doc_id
                    WHERE c.user_id = ?
                """, (user_id,)).fetchone()[0]
                missing_rows = cursor.execute("""
                    SELECT c.doc_id
                    FROM chunk_store c
                    LEFT JOIN fts_index f ON f.doc_id = c.doc_id
                    WHERE c.user_id = ? AND f.doc_id IS NULL
                    ORDER BY c.doc_id
                """, (user_id,)).fetchall()
                duplicate_rows = cursor.execute("""
                    SELECT f.doc_id, COUNT(*) AS n
                    FROM fts_index f
                    JOIN chunk_store c ON c.doc_id = f.doc_id
                    WHERE c.user_id = ?
                    GROUP BY f.doc_id
                    HAVING n > 1
                    ORDER BY f.doc_id
                """, (user_id,)).fetchall()
                orphan_rows = []

        missing_doc_ids = [row[0] for row in missing_rows]
        duplicate_doc_ids = [row[0] for row in duplicate_rows]
        duplicate_rows_count = sum(int(row[1]) - 1 for row in duplicate_rows)
        orphan_doc_ids = [row[0] for row in orphan_rows]

        issue_count = len(missing_doc_ids) + len(duplicate_doc_ids) + len(orphan_doc_ids)
        return {
            "ok": issue_count == 0,
            "scope": "tenant" if user_id is not None else "global",
            "user_id": user_id,
            "global_orphan_check": user_id is None,
            "chunk_store_count": int(chunk_count),
            "fts_count": int(fts_count),
            "missing_fts_count": len(missing_doc_ids),
            "duplicate_doc_id_count": len(duplicate_doc_ids),
            "duplicate_fts_rows": duplicate_rows_count,
            "orphan_fts_count": len(orphan_doc_ids),
            "issue_count": issue_count,
            "missing_doc_ids": missing_doc_ids[:50],
            "duplicate_doc_ids": duplicate_doc_ids[:50],
            "orphan_doc_ids": orphan_doc_ids[:50],
            "missing_doc_ids_truncated": len(missing_doc_ids) > 50,
            "duplicate_doc_ids_truncated": len(duplicate_doc_ids) > 50,
            "orphan_doc_ids_truncated": len(orphan_doc_ids) > 50,
        }

    def rebuild_fts_index(self, user_id: int = None) -> Dict:
        """Rebuild FTS rows from chunk_store and return the post-rebuild report."""
        deleted_fts_rows = 0
        rebuilt_fts_rows = 0
        with sqlite3.connect(self.db_path, timeout=10) as conn:
            cursor = conn.cursor()
            if user_id is None:
                cursor.execute("DELETE FROM fts_index")
                deleted_fts_rows = max(0, int(cursor.rowcount or 0))
                rows = cursor.execute(
                    "SELECT doc_id, content FROM chunk_store ORDER BY doc_id"
                ).fetchall()
            else:
                doc_ids = [row[0] for row in cursor.execute(
                    "SELECT doc_id FROM chunk_store WHERE user_id = ? ORDER BY doc_id",
                    (user_id,),
                ).fetchall()]
                if doc_ids:
                    placeholders = ",".join("?" for _ in doc_ids)
                    cursor.execute(
                        f"DELETE FROM fts_index WHERE doc_id IN ({placeholders})",
                        doc_ids,
                    )
                    deleted_fts_rows = max(0, int(cursor.rowcount or 0))
                rows = cursor.execute(
                    "SELECT doc_id, content FROM chunk_store WHERE user_id = ? ORDER BY doc_id",
                    (user_id,),
                ).fetchall()

            for doc_id, content in rows:
                tokenized_content = " ".join(jieba.cut_for_search((content or "").lower()))
                cursor.execute(
                    "INSERT INTO fts_index(content, doc_id) VALUES (?, ?);",
                    (tokenized_content, doc_id),
                )
                rebuilt_fts_rows += 1
            conn.commit()

        report = self.integrity_report(user_id=user_id)
        report["rebuild"] = {
            "scope": "tenant" if user_id is not None else "global",
            "user_id": user_id,
            "deleted_fts_rows": deleted_fts_rows,
            "rebuilt_fts_rows": rebuilt_fts_rows,
        }
        return report


from src.retrieval.candidate_fusion import rrf as rrf
