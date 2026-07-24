"""Phase 11B tests: BM25 index integrity and rebuild tools."""
import json
import os
import sqlite3
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

if "jieba_fast" not in sys.modules:
    jieba = types.SimpleNamespace(cut_for_search=lambda text: text.split())
    sys.modules["jieba_fast"] = jieba

from services.retrieval import SqliteBM25Retriever
from src.evaluation.eval_cli import main as eval_cli_main


def _chunk(doc_id, content="alpha revenue", doc_name="r.pdf"):
    return {
        "content": content,
        "metadata": {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "page": 1,
            "type": "text",
        },
    }


def test_integrity_report_ok_after_add_chunks(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)

    report = retriever.integrity_report()

    assert report["ok"] is True
    assert report["chunk_store_count"] == 1
    assert report["fts_count"] == 1
    assert report["missing_fts_count"] == 0
    assert report["duplicate_doc_id_count"] == 0
    assert report["orphan_fts_count"] == 0


def test_integrity_report_detects_missing_duplicate_and_orphan_rows(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1"), _chunk("r.pdf::2")], user_id=1)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute("DELETE FROM fts_index WHERE doc_id = ?", ("user_1_r.pdf::1",))
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("alpha", "user_1_r.pdf::2"))
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("orphan", "orphan::1"))
        conn.commit()

    report = retriever.integrity_report()

    assert report["ok"] is False
    assert report["missing_doc_ids"] == ["user_1_r.pdf::1"]
    assert report["duplicate_doc_ids"] == ["user_1_r.pdf::2"]
    assert report["duplicate_fts_rows"] == 1
    assert report["orphan_doc_ids"] == ["orphan::1"]


def test_rebuild_fts_index_repairs_global_index(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1"), _chunk("r.pdf::2")], user_id=1)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute("DELETE FROM fts_index WHERE doc_id = ?", ("user_1_r.pdf::1",))
        conn.execute("INSERT INTO fts_index(content, doc_id) VALUES (?, ?)", ("orphan", "orphan::1"))
        conn.commit()

    report = retriever.rebuild_fts_index()

    assert report["ok"] is True
    assert report["chunk_store_count"] == 2
    assert report["fts_count"] == 2
    assert retriever.search("revenue", user_id=1)


def test_tenant_scoped_rebuild_preserves_other_tenant_and_repairs_target(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("a.pdf::1", doc_name="a.pdf")], user_id=1)
    retriever.add_chunks([_chunk("b.pdf::1", doc_name="b.pdf")], user_id=2)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute("DELETE FROM fts_index WHERE doc_id = ?", ("user_1_a.pdf::1",))
        conn.commit()

    report = retriever.rebuild_fts_index(user_id=1)

    assert report["ok"] is True
    assert report["user_id"] == 1
    assert retriever.search("revenue", user_id=1)
    assert retriever.search("revenue", user_id=2)


def test_eval_cli_bm25_check_and_rebuild(tmp_path, capsys):
    db_path = str(tmp_path / "bm25.db")
    retriever = SqliteBM25Retriever(db_path=db_path)
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)
    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM fts_index")
        conn.commit()

    assert eval_cli_main(["bm25-check", "--db", db_path]) == 1
    check_output = json.loads(capsys.readouterr().out)
    assert check_output["missing_fts_count"] == 1

    assert eval_cli_main(["bm25-rebuild", "--db", db_path]) == 0
    rebuild_output = json.loads(capsys.readouterr().out)
    assert rebuild_output["ok"] is True



def test_search_deduplicates_duplicate_fts_rows(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute(
            "INSERT INTO fts_index(content, doc_id) VALUES (?, ?)",
            ("alpha revenue", "user_1_r.pdf::1"),
        )
        conn.commit()

    results = retriever.search("revenue", user_id=1)

    assert [row["doc_id"] for row in results] == ["user_1_r.pdf::1"]


def test_search_skips_rows_with_corrupt_metadata_json(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1"), _chunk("r.pdf::2")], user_id=1)
    with sqlite3.connect(retriever.db_path) as conn:
        conn.execute(
            "UPDATE chunk_store SET metadata_json = ? WHERE doc_id = ?",
            ("{bad-json", "user_1_r.pdf::1"),
        )
        conn.commit()

    results = retriever.search("revenue", user_id=1)

    assert [row["doc_id"] for row in results] == ["user_1_r.pdf::2"]



def test_search_rejects_non_positive_and_invalid_limits(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)

    assert retriever.search("revenue", k=0, user_id=1) == []
    assert retriever.search("revenue", k=-1, user_id=1) == []
    assert retriever.search("revenue", k="bad", user_id=1) == []


def test_search_caps_large_limits(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    chunks = [
        _chunk("r.pdf::%s" % i, content="alpha revenue %s" % i)
        for i in range(retriever.MAX_SEARCH_LIMIT + 5)
    ]
    retriever.add_chunks(chunks, user_id=1)

    results = retriever.search("revenue", k=1000, user_id=1)

    assert len(results) == retriever.MAX_SEARCH_LIMIT



def test_search_rejects_non_string_and_blank_queries(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([_chunk("r.pdf::1")], user_id=1)

    assert retriever.search(None, user_id=1) == []
    assert retriever.search(123, user_id=1) == []
    assert retriever.search("   ", user_id=1) == []


def test_search_expansion_terms_are_disjunctive_not_mandatory(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    retriever.add_chunks([
        _chunk("r.pdf::revenue", content="platform revenue was reported"),
        _chunk("r.pdf::period", content="2025 annual disclosure"),
    ], user_id=1)

    results = retriever.search(
        "platform revenue 2025 unrelated expansion token",
        user_id=1,
        k=10,
    )

    assert {row["doc_id"] for row in results} == {
        "user_1_r.pdf::revenue",
        "user_1_r.pdf::period",
    }



def test_add_chunks_ignores_empty_and_invalid_batches(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))

    retriever.add_chunks([], user_id=1)
    retriever.add_chunks(None, user_id=1)
    retriever.add_chunks("bad", user_id=1)

    report = retriever.integrity_report(user_id=1)
    assert report["ok"] is True
    assert report["chunk_store_count"] == 0
    assert report["fts_count"] == 0


def test_add_chunks_skips_invalid_chunks_and_indexes_valid_rows(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    chunks = [
        None,
        {},
        {"content": "", "metadata": {"doc_id": "empty::1", "doc_name": "r.pdf"}},
        {"content": "no id", "metadata": {"doc_name": "r.pdf"}},
        {"content": 123, "metadata": {"doc_id": "bad-content::1", "doc_name": "r.pdf"}},
        {"content": "wrong tenant", "metadata": {"doc_id": "user_2_r.pdf::1", "doc_name": "r.pdf"}},
        {"id": "fallback::1", "content": "alpha revenue fallback", "metadata": "bad"},
        _chunk("r.pdf::1", content="alpha revenue valid"),
    ]

    retriever.add_chunks(chunks, user_id=1)

    results = retriever.search("revenue", user_id=1, k=10)
    assert sorted(row["doc_id"] for row in results) == [
        "user_1_fallback::1",
        "user_1_r.pdf::1",
    ]


def test_add_chunks_still_requires_user_id(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))

    try:
        retriever.add_chunks([_chunk("r.pdf::1")], user_id=None)
    except ValueError as exc:
        assert "user_id is required" in str(exc)
    else:
        raise AssertionError("add_chunks should require user_id")
