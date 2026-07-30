from scripts.evaluation import run_nf40_attribution as runner


def test_nf40_cli_disables_unreachable_retrieval_reranker(monkeypatch):
    observed = {}

    class FakeEngine:
        pass

    def build(*args, **kwargs):
        observed.update(kwargs)
        return FakeEngine()

    monkeypatch.setattr(runner, "RAGEngine", build)
    engine = runner._build_frozen_context_engine(object())
    assert isinstance(engine, FakeEngine)
    assert observed["use_hybrid"] is False
    assert observed["reranker_name"] == "none"
    assert observed["retrieval_candidate_multiplier"] == 1
