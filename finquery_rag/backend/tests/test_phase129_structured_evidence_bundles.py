"""Regression tests for structured table evidence bundles."""

from src.retrieval.retrieval_pipeline import RetrievalPipeline
from src.services.retrieval import SqliteBM25Retriever
from src.services import vector_store
from src.services.vector_store import add_documents


def _table_row():
    return {
        "content": "2025 | Cash and Cash Equivalents | 42.2",
        "metadata": {
            "doc_id": "user_7_report.pdf::page_4::table_1::row_2",
            "doc_name": "report.pdf",
            "page": 4,
            "type": "table_row",
        },
    }


def _table_cell():
    return {
        "content": "Column: 2025; Value: 42.2; Table row: Cash and Cash Equivalents",
        "metadata": {
            "doc_id": "user_7_report.pdf::page_4::table_1::row_2::cell_1",
            "doc_name": "report.pdf",
            "page": 4,
            "type": "table_cell",
            "parent_row_id": "user_7_report.pdf::page_4::table_1::row_2",
            "table_alignment": "exact",
        },
    }


def test_bm25_keeps_table_cells_out_of_primary_results(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    row = _table_row()
    cell = _table_cell()
    retriever.add_chunks([row, cell], user_id=7)

    primary = retriever.search("cash equivalents 2025", k=3, user_id=7)
    facts = retriever.get_table_cell_evidence(
        [row["metadata"]["doc_id"]], user_id=7
    )

    assert [item["doc_id"] for item in primary] == [row["metadata"]["doc_id"]]
    assert facts[row["metadata"]["doc_id"]][0]["doc_id"] == cell["metadata"]["doc_id"]


def test_table_cells_are_never_dense_candidates(monkeypatch):
    class FakeCollection:
        name = "test_collection"

        def __init__(self):
            self.upsert_calls = []

        def upsert(self, **kwargs):
            self.upsert_calls.append(kwargs)

        def count(self):
            return 0

    collection = FakeCollection()
    monkeypatch.setattr(vector_store, "get_or_create_collection", lambda: collection)
    row = _table_row()
    cell = _table_cell()

    add_documents([row, cell], "report.pdf", user_id=7)

    assert collection.upsert_calls == []
    assert row["metadata"]["retrieval_channel"] == "sparse"
    assert cell["metadata"]["retrieval_channel"] == "secondary_structured"
    assert row["metadata"]["dense_indexed"] is False
    assert cell["metadata"]["dense_indexed"] is False


def test_retrieval_pipeline_attaches_cells_only_after_row_selection():
    row = {
        **_table_row(),
        "doc_id": _table_row()["metadata"]["doc_id"],
        "score": 0.9,
    }
    cell = _table_cell()

    class FakeBM25:
        def search(self, *_args, **_kwargs):
            return [row]

        def get_table_cell_evidence(self, parent_row_ids, *, user_id, max_cells_per_row=3):
            assert parent_row_ids == [row["doc_id"]]
            assert user_id == 7
            return {row["doc_id"]: [{**cell, "doc_id": cell["metadata"]["doc_id"]}]}

    pipeline = RetrievalPipeline(
        dense_query_fn=lambda **_kwargs: [],
        bm25_retriever=FakeBM25(),
        use_hybrid=True,
    )

    selected = pipeline.retrieve_single(
        "report.pdf",
        "How much cash and cash equivalents were reported in 2025?",
        user_id=7,
        top_k=1,
    )

    assert len(selected) == 1
    assert selected[0]["doc_id"] == row["doc_id"]
    assert "Structured table facts:" in selected[0]["content"]
    assert "Column: 2025; Value: 42.2" in selected[0]["content"]
    assert selected[0]["metadata"]["structured_fact_count"] == 1


def test_bm25_prefers_query_matched_exact_table_cells(tmp_path):
    retriever = SqliteBM25Retriever(db_path=str(tmp_path / "bm25.db"))
    row = _table_row()
    parent_row_id = row["metadata"]["doc_id"]
    original_budget = {
        "content": "Column: Original budget; Value: 110,231; Table row: PCT System",
        "metadata": {
            **_table_cell()["metadata"],
            "doc_id": f"{parent_row_id}::cell_0",
            "parent_row_id": parent_row_id,
            "table_column": "Original budget",
        },
    }
    actual_expense = {
        "content": "Column: Actual expense; Value: 98,755; Table row: PCT System",
        "metadata": {
            **_table_cell()["metadata"],
            "doc_id": f"{parent_row_id}::cell_1",
            "parent_row_id": parent_row_id,
            "table_column": "Actual expense",
        },
    }
    retriever.add_chunks([row, original_budget, actual_expense], user_id=7)

    facts = retriever.get_table_cell_evidence(
        [parent_row_id],
        user_id=7,
        query="What was the actual expense for the PCT system?",
    )

    assert facts[parent_row_id][0]["metadata"]["table_column"] == "Actual expense"


def test_numeric_retrieval_uses_wider_candidate_pool_before_reranking():
    calls = []

    def dense_query(**kwargs):
        calls.append(kwargs["n_results"])
        return []

    class FakeBM25:
        def search(self, _query, *, k, **_kwargs):
            calls.append(k)
            return []

    pipeline = RetrievalPipeline(
        dense_query_fn=dense_query,
        bm25_retriever=FakeBM25(),
        candidate_multiplier=4,
        use_hybrid=True,
    )

    pipeline.retrieve_single(
        "report.pdf",
        "What cash and cash equivalents were reported in 2025?",
        user_id=7,
        top_k=3,
    )

    assert calls == [24, 24]
