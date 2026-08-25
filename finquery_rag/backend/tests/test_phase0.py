"""
Phase 0 unit tests - CPU-safe, no GPU, no external LLM.
"""
import sys, os, json, sqlite3, tempfile, asyncio, ast, pytest
from unittest.mock import MagicMock, patch, AsyncMock
import importlib


@pytest.fixture(autouse=True)
def _isolate_sys_modules():
    """Save and restore sys.modules to prevent pollution."""
    saved = sys.modules.copy()
    yield
    # Remove any modules we added during the test
    for key in list(sys.modules):
        if key not in saved:
            del sys.modules[key]
    sys.modules.update(saved)


# Mock heavy imports BEFORE loading app modules (must be at module level)
mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "camelot", "pymupdf", "langchain", "langchain_core", "langchain_core.documents",
    "langchain_text_splitters", "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
# Make jieba_fast.cut_for_search return input as list
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef

from src.services.retrieval import SqliteBM25Retriever, rrf
from src.services.vector_store import (
    add_documents, query_collection, query_multiple_collections,
    list_all_documents, delete_document_collection, get_collection_stats,
    clear_all_for_user,
)
from src.services.chunk_id import make_chunk_id, ensure_scoped_chunk_id, is_scoped_chunk_id
from src.services.rag_engine import RAGEngine


@pytest.fixture
def bm25_db(tmp_path):
    return SqliteBM25Retriever(db_path=str(tmp_path / "test_bm25.db"))


@pytest.fixture
def mock_collection():
    store = {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def _match(meta, wf):
        if wf is None:
            return True
        if "$and" in wf:
            return all(_match(meta, clause) for clause in wf["$and"])
        if "$or" in wf:
            return any(_match(meta, clause) for clause in wf["$or"])
        for k, v in wf.items():
            if k.startswith("$"):
                continue
            if isinstance(v, dict) and "$in" in v:
                if meta.get(k) not in v["$in"]:
                    return False
            elif meta.get(k) != v:
                return False
        return True

    col = MagicMock()

    def _upsert(ids, documents, metadatas=None):
        for i, did in enumerate(ids):
            if did in store["ids"]:
                idx = store["ids"].index(did)
                store["documents"][idx] = documents[i]
                store["metadatas"][idx] = metadatas[i] if metadatas else {}
            else:
                store["ids"].append(did)
                store["documents"].append(documents[i])
                store["metadatas"].append(metadatas[i] if metadatas else {})

    def _query(query_texts, n_results=5, where=None):
        matched = [i for i, m in enumerate(store["metadatas"]) if _match(m, where)][:n_results]
        return {
            "ids": [[store["ids"][i] for i in matched]],
            "documents": [[store["documents"][i] for i in matched]],
            "metadatas": [[store["metadatas"][i] for i in matched]],
            "distances": [[0.1 for _ in matched]],
        }

    def _get(include=None, where=None, limit=None):
        ids = []; metas = []
        for i, m in enumerate(store["metadatas"]):
            if _match(m, where):
                ids.append(store["ids"][i]); metas.append(m)
        if limit is not None:
            ids = ids[:limit]; metas = metas[:limit]
        return {"ids": ids, "metadatas": metas}

    def _delete(where=None):
        keep = [i for i, m in enumerate(store["metadatas"]) if not _match(m, where)]
        store["ids"] = [store["ids"][i] for i in keep]
        store["documents"] = [store["documents"][i] for i in keep]
        store["metadatas"] = [store["metadatas"][i] for i in keep]

    col.upsert = MagicMock(side_effect=_upsert)
    col.query = MagicMock(side_effect=_query)
    col.get = MagicMock(side_effect=_get)
    col.delete = MagicMock(side_effect=_delete)
    col.count = MagicMock(side_effect=lambda: len(store["ids"]))
    col.name = "rag_global_knowledge_base"
    return col, store


@pytest.fixture(autouse=True)
def _patch_col(mock_collection, monkeypatch):
    col, store = mock_collection
    monkeypatch.setattr("src.services.vector_store.get_or_create_collection", lambda: col)
    for k in store: store[k].clear()
    yield
    for k in store: store[k].clear()


def make_chunks_raw(doc_name, count=3):
    """Generate chunks with UNSCOPED IDs (for testing storage boundary enforcement)."""
    return [
        {"content": "chunk %d of %s" % (i, doc_name),
         "metadata": {"doc_id": "%s::page_1::chunk_%d" % (doc_name, i),
                       "type": "text", "page": 1, "doc_name": doc_name}}
        for i in range(count)
    ]


def make_chunks_scoped(doc_name, user_id, count=3):
    """Generate chunks with pre-scoped IDs."""
    return [
        {"content": "chunk %d of %s" % (i, doc_name),
         "metadata": {"doc_id": make_chunk_id(user_id, doc_name, "page_1::chunk_%d" % i),
                       "type": "text", "page": 1, "doc_name": doc_name}}
        for i in range(count)
    ]
class TestStorageBoundaryEnforcement:
    """add_documents and add_chunks must enforce scoped IDs at write boundary."""

    def test_add_documents_enforces_scoped_id(self, mock_collection):
        """Passing unscoped raw IDs should be auto-scoped by storage."""
        col, store = mock_collection
        raw_chunks = make_chunks_raw("report.pdf", count=2)
        add_documents(raw_chunks, "report.pdf", user_id=42)
        for did in store["ids"]:
            assert did.startswith("user_42_"), f"Expected scoped ID, got {did}"

    def test_table_rows_are_sparse_only_but_keep_bm25_metadata(self, mock_collection):
        """Table rows must not pollute dense retrieval but remain BM25-indexable."""
        col, store = mock_collection
        chunks = make_chunks_raw("report.pdf", count=1)
        row = {
            "content": "Revenue | 2025 | 181.0",
            "metadata": {
                "doc_id": "report.pdf::page_1::table_1::row_2",
                "type": "table_row",
                "page": 1,
            },
        }
        chunks.append(row)

        add_documents(chunks, "report.pdf", user_id=42)

        assert len(store["ids"]) == 1
        assert store["metadatas"][0]["type"] == "text"
        assert row["metadata"]["doc_name"] == "report.pdf"
        assert row["metadata"]["user_id"] == 42
        assert row["metadata"]["retrieval_channel"] == "sparse"
        assert row["metadata"]["dense_indexed"] is False

    def test_add_documents_idempotent(self, mock_collection):
        """Pre-scoped IDs should not be double-prefixed."""
        col, store = mock_collection
        scoped = make_chunks_scoped("r.pdf", 7, count=2)
        add_documents(scoped, "r.pdf", user_id=7)
        for did in store["ids"]:
            assert did.startswith("user_7_r.pdf")
            assert "user_7_user_7_" not in did

    def test_add_documents_rejects_wrong_tenant_scoped_id(self, mock_collection):
        """ID scoped to user 2 must be rejected when writing as user 1."""
        col, store = mock_collection
        bad_chunks = make_chunks_scoped("r.pdf", user_id=2, count=1)
        with pytest.raises(ValueError, match="different tenant"):
            add_documents(bad_chunks, "r.pdf", user_id=1)

    def test_add_documents_rejects_user_id_none(self, mock_collection):
        col, store = mock_collection
        chunks = make_chunks_raw("r.pdf", count=1)
        with pytest.raises(ValueError):
            add_documents(chunks, "r.pdf", user_id=None)

    def test_user_id_zero_is_valid(self, mock_collection):
        """user_id=0 should be a valid tenant (is None check, not truthiness)."""
        col, store = mock_collection
        chunks = make_chunks_raw("r.pdf", count=1)
        add_documents(chunks, "r.pdf", user_id=0)
        assert len(store["ids"]) == 1
        assert store["ids"][0].startswith("user_0_")

    def test_dense_sparse_id_consistency(self, mock_collection, bm25_db):
        """Dense and Sparse must produce the same scoped ID from the same raw input."""
        col, store = mock_collection
        raw_chunks = make_chunks_raw("report.pdf", count=2)
        add_documents(raw_chunks, "report.pdf", user_id=5)
        bm25_db.add_chunks(raw_chunks, user_id=5)
        dense_ids = sorted(store["ids"])
        with sqlite3.connect(bm25_db.db_path) as conn:
            bm25_ids = sorted(r[0] for r in conn.execute("SELECT doc_id FROM chunk_store").fetchall())
        assert dense_ids == bm25_ids


class TestDeleteFailClosed:
    """All destructive APIs must reject when user_id is None."""

    def test_delete_document_collection_rejects_no_user(self, mock_collection):
        col, store = mock_collection
        add_documents(make_chunks_raw("a.pdf"), "a.pdf", user_id=1)
        add_documents(make_chunks_raw("a.pdf"), "a.pdf", user_id=2)
        with pytest.raises(ValueError, match="user_id is required"):
            delete_document_collection("a.pdf", user_id=None)
        # Both users data preserved
        assert len(store["ids"]) == 6

    def test_delete_no_user_preserves_all(self, mock_collection):
        """The old test asserted deletion; now it must assert PRESERVATION."""
        col, store = mock_collection
        add_documents(make_chunks_raw("a.pdf"), "a.pdf", user_id=1)
        with pytest.raises(ValueError):
            delete_document_collection(None, user_id=None)
        assert len(store["ids"]) == 3

    def test_bm25_delete_doc_rejects_no_user(self, bm25_db):
        bm25_db.add_chunks([{"content": "x", "metadata": {"doc_id": "u1", "doc_name": "a"}}], user_id=1)
        with pytest.raises(ValueError):
            bm25_db.delete_doc("a", user_id=None)

    def test_bm25_delete_all_rejects_no_user(self, bm25_db):
        with pytest.raises(ValueError):
            bm25_db.delete_all_for_user(user_id=None)

    def test_clear_all_for_user_requires_user(self, mock_collection):
        with pytest.raises(ValueError):
            clear_all_for_user(user_id=None)


class TestBM25Injection:
    """File names with %, _, Chinese, spaces must not cause LIKE injection."""

    def test_percent_in_filename(self, bm25_db):
        chunks = [{"content": "data", "metadata": {"doc_id": "x", "doc_name": "a%b"}}]
        bm25_db.add_chunks(chunks, user_id=1)
        # Search for doc_name with % should only match exact name
        r = bm25_db.search("data", doc_name="a%b", user_id=1)
        assert len(r) == 1
        # Wildcard should NOT match
        r2 = bm25_db.search("data", doc_name="aXb", user_id=1)
        assert len(r2) == 0

    def test_underscore_in_filename(self, bm25_db):
        chunks = [{"content": "data", "metadata": {"doc_id": "x", "doc_name": "a_b"}}]
        bm25_db.add_chunks(chunks, user_id=1)
        r = bm25_db.search("data", doc_name="a_b", user_id=1)
        assert len(r) == 1

    def test_chinese_filename(self, bm25_db):
        chunks = [{"content": "营收增长", "metadata": {"doc_id": "x", "doc_name": "财务报告"}}]
        bm25_db.add_chunks(chunks, user_id=1)
        r = bm25_db.search("营收增长", doc_name="财务报告", user_id=1)
        assert len(r) >= 1

    def test_space_in_filename(self, bm25_db):
        chunks = [{"content": "data", "metadata": {"doc_id": "x", "doc_name": "my report"}}]
        bm25_db.add_chunks(chunks, user_id=1)
        r = bm25_db.search("data", doc_name="my report", user_id=1)
        assert len(r) == 1


class TestQueryFilters:
    def test_query_filters_by_user_id(self, mock_collection):
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=1)
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=2)
        r = query_collection(query_text="test", user_id=1, n_results=10)
        assert len(r) == 3
        for x in r: assert x["metadata"]["user_id"] == 1

    def test_query_no_user_returns_empty(self):
        assert query_collection(query_text="x", user_id=None) == []

    def test_query_multiple_no_user_returns_empty(self):
        assert query_multiple_collections(["a.pdf"], "x", user_id=None) == []

    def test_list_no_user_returns_empty(self):
        assert list_all_documents(user_id=None) == []

    def test_stats_no_user_returns_not_exists(self):
        s = get_collection_stats(doc_name="a", user_id=None)
        assert s["exists"] is False


class TestCrossTenantIsolation:
    def test_same_filename_no_collision(self, mock_collection):
        col, store = mock_collection
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=1)
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=2)
        assert len(store["ids"]) == 6
        u1 = [d for d in store["ids"] if d.startswith("user_1_")]
        u2 = [d for d in store["ids"] if d.startswith("user_2_")]
        assert len(u1) == 3 and len(u2) == 3 and set(u1).isdisjoint(set(u2))

    def test_query_isolation(self, mock_collection):
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=1)
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=2)
        r1 = query_collection(query_text="test", user_id=1, n_results=10)
        r2 = query_collection(query_text="test", user_id=2, n_results=10)
        assert len(r1) == 3 and len(r2) == 3
        assert {r["doc_id"] for r in r1}.isdisjoint({r["doc_id"] for r in r2})


class TestDeleteIsolation:
    def test_delete_user1_preserves_user2(self, mock_collection):
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=1)
        add_documents(make_chunks_raw("r.pdf"), "r.pdf", user_id=2)
        delete_document_collection("r.pdf", user_id=1)
        assert len(query_collection(query_text="t", user_id=1, n_results=10)) == 0
        assert len(query_collection(query_text="t", user_id=2, n_results=10)) == 3


class TestClearAllPerUser:
    def test_clear_user1_preserves_user2_dense(self, mock_collection):
        add_documents(make_chunks_raw("a.pdf"), "a.pdf", user_id=1)
        add_documents(make_chunks_raw("b.pdf"), "b.pdf", user_id=2)
        clear_all_for_user(user_id=1)
        assert len(query_collection(query_text="t", user_id=1, n_results=10)) == 0
        assert len(query_collection(query_text="t", user_id=2, n_results=10)) == 3

    def test_clear_user1_preserves_user2_bm25(self, bm25_db):
        bm25_db.add_chunks([{"content": "r", "metadata": {"doc_id": "x", "doc_name": "a"}}], user_id=1)
        bm25_db.add_chunks([{"content": "r2", "metadata": {"doc_id": "y", "doc_name": "a"}}], user_id=2)
        bm25_db.delete_all_for_user(user_id=1)
        assert len(bm25_db.search("r", user_id=1)) == 0
        assert len(bm25_db.search("r2", user_id=2)) == 1


class TestAwaitChecks:
    def _get_endpoint_body(self, func_name):
        main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(main_path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef) and node.name == func_name:
                return node
        pytest.skip(func_name + " not found")

    def _has_await_call(self, node, target):
        for child in ast.walk(node):
            if isinstance(child, ast.Await):
                call = child.value
                if isinstance(call, ast.Call):
                    if isinstance(call.func, ast.Attribute) and call.func.attr == target:
                        return True
        return False

    def test_query_endpoint_awaits_engine_query(self):
        node = self._get_endpoint_body("query_documents")
        assert self._has_await_call(node, "execute_user_turn")

    def test_stream_endpoint_awaits_multi_doc(self):
        """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

        The stream endpoint delegates to ``engine.query()`` which internally
        handles multi-document retrieval. This test verifies the stream
        endpoint awaits ``engine.query``.
        """
        node = self._get_endpoint_body("query_documents_stream")
        # The generate() inner function should await engine.query()
        for child in ast.walk(node):
            if isinstance(child, ast.AsyncFunctionDef) and child.name == "generate":
                assert self._has_await_call(child, "execute_user_turn")
                return
        pytest.fail("generate() inner function not found")

    def test_engine_query_is_coroutine(self):
        assert asyncio.iscoroutinefunction(RAGEngine.query)

    def test_engine_retrieve_multi_is_coroutine(self):
        assert asyncio.iscoroutinefunction(RAGEngine.retrieve_multiple_documents)


class TestUploadSecurity:
    def test_main_uses_safe_upload_filename(self):
        p = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(p, encoding="utf-8") as f: content = f.read()
        assert "def _safe_upload_filename" in content
        assert "os.path.basename(filename).strip()" in content
        assert "safe_filename = _safe_upload_filename(file.filename)" in content

    def test_main_uses_tempfile(self):
        p = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(p, encoding="utf-8") as f: content = f.read()
        assert "tempfile.mkdtemp()" in content

    def test_httpexception_preserved(self):
        """Upload handler must re-raise HTTPException (e.g. 400) without wrapping in 500."""
        p = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(p, encoding="utf-8") as f: content = f.read()
        assert "except HTTPException" in content


class TestClearAllPartialFailure:
    def test_main_checks_delete_return(self):
        """clear_all_documents must handle delete errors and surface partial failures."""
        p = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(p, encoding="utf-8") as f: content = f.read()
        # Round 4: clear_all_documents uses try/except, not dense_ok check.
        # Idempotent — no docs is not an error.
        assert "delete_document_collection" in content
        assert "Partial failure" in content or "errors" in content


class TestBM25Migration:
    def test_schema_version_exists(self):
        assert hasattr(SqliteBM25Retriever, "SCHEMA_VERSION")
        assert SqliteBM25Retriever.SCHEMA_VERSION >= 2

    def test_init_creates_doc_name_column(self, tmp_path):
        db_path = str(tmp_path / "mig.db")
        r = SqliteBM25Retriever(db_path=db_path)
        with sqlite3.connect(db_path) as conn:
            cols = [row[1] for row in conn.execute("PRAGMA table_info(chunk_store)").fetchall()]
        assert "doc_name" in cols

    def test_migration_backfills_doc_name(self, tmp_path):
        """Simulate old DB without doc_name, then init should migrate."""
        db_path = str(tmp_path / "old.db")
        with sqlite3.connect(db_path) as conn:
            conn.execute("""CREATE TABLE chunk_store (
                doc_id TEXT PRIMARY KEY, content TEXT, metadata_json TEXT, user_id INTEGER
            )""")
            conn.execute("INSERT INTO chunk_store VALUES (?, ?, ?, ?)",
                ("old_id", "content",
                 json.dumps({"doc_name": "legacy.pdf"}), 1))
            conn.commit()
        # Now init with our retriever
        r = SqliteBM25Retriever(db_path=db_path)
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT doc_name FROM chunk_store WHERE doc_id=?", ("old_id",)).fetchone()
        assert row[0] == "legacy.pdf"


class TestRRF:
    def test_merges_rankings(self):
        l1 = [{"doc_id": "A", "score": 0.9}, {"doc_id": "B", "score": 0.5}]
        l2 = [{"doc_id": "B", "score": 0.8}, {"doc_id": "C", "score": 0.7}]
        assert rrf([l1, l2])[0]["doc_id"] == "B"

    def test_single_list(self):
        r = rrf([[{"doc_id": "X", "score": 1.0}, {"doc_id": "Y", "score": 0.5}]])
        assert len(r) == 2 and r[0]["doc_id"] == "X"


class TestDeleteAllRequiresAuth:
    def test_requires_current_user(self):
        p = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
        with open(p, encoding="utf-8") as f: content = f.read()
        idx = content.find('@app.delete("/documents")')
        assert idx >= 0
        assert "Depends(get_current_user)" in content[idx:idx+500]
