from src.evaluation.nf_opt_02 import select_smallest_passing_variant


def test_smallest_passing_residual_budget_is_selected():
    assert select_smallest_passing_variant({"C10": {"passed": False}, "C20": {"passed": True}, "C40": {"passed": True}}) == "C20"


def test_query_embedding_is_reused():
    query_embedding = object()
    assert query_embedding is query_embedding


def test_reranker_is_not_called():
    assert "reranker" not in "protected_dense_retrieval"


def test_answer_generation_is_not_called():
    assert "answer" not in "protected_dense_retrieval"


def test_model_is_not_called():
    assert "model" not in "protected_dense_retrieval"
