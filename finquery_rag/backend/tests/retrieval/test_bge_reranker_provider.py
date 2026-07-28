import pytest
from src.services.reranker import BgeV2M3Reranker

class FakeModel:
    def compute_score(self, pairs, **kwargs): return [0.2, 0.9]

def test_equal_interface_keeps_stable_score_order():
    reranker=BgeV2M3Reranker(model=FakeModel(),device="cpu")
    rows=reranker.rerank("q",[{"doc_id":"b","content":"one","metadata":{}},{"doc_id":"a","content":"two","metadata":{}}])
    assert [row["doc_id"] for row in rows] == ["a","b"]

def test_provider_output_count_must_match_input():
    class BadModel:
        def compute_score(self, pairs, **kwargs): return [0.5]
    with pytest.raises(RuntimeError, match="output count"):
        BgeV2M3Reranker(model=BadModel(),device="cpu").rerank("q",[{"doc_id":"a","content":"x","metadata":{}},{"doc_id":"b","content":"y","metadata":{}}])
