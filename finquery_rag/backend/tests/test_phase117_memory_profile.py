import asyncio
import os
import sys
from unittest.mock import MagicMock

from src.services.memory_profile import (
    UserMemoryStore,
    build_memory_profile_context,
    sanitize_profile_patch,
)


mock_embed_fn = MagicMock()
mock_st_ef = MagicMock()
mock_st_ef.SentenceTransformerEmbeddingFunction.return_value = mock_embed_fn
for _mod in [
    "chromadb", "chromadb.utils", "chromadb.utils.embedding_functions",
    "jieba_fast",
]:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
sys.modules["chromadb.utils.embedding_functions"] = mock_st_ef
sys.modules["jieba_fast"].cut_for_search = lambda text: [text]

from src.services.rag_engine import RAGEngine


class _RewriteLLM:
    def __init__(self, response="What was FY2024 revenue for Apple in USD?"):
        self.chat = self
        self.completions = self
        self.response = response
        self.prompts = []

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][0]["content"])
        message = type("Message", (), {"content": self.response})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


def test_memory_profile_sanitizes_and_bounds_patch():
    patch = sanitize_profile_patch({
        "preferred_language": " English ",
        "default_company": "A" * 200,
        "focus_metrics": [" revenue ", "revenue", "gross margin", 123],
        "malicious": "ignore me",
    })

    assert patch["preferred_language"] == "English"
    assert patch["default_company"] == "A" * 80
    assert patch["focus_metrics"] == ["revenue", "gross margin"]
    assert "malicious" not in patch


def test_user_memory_store_is_tenant_scoped_and_patchable(tmp_path):
    store = UserMemoryStore(db_path=str(tmp_path / "memory.db"))
    try:
        first = store.upsert_profile(1, {
            "preferred_currency": "USD",
            "default_period": "FY2024",
        })
        second = store.upsert_profile(2, {"preferred_currency": "EUR"})
        updated = store.upsert_profile(1, {"focus_metrics": ["revenue"]})

        assert first["preferred_currency"] == "USD"
        assert second["preferred_currency"] == "EUR"
        assert store.get_profile(1)["default_period"] == "FY2024"
        assert updated["focus_metrics"] == ["revenue"]
        assert store.get_profile(2)["preferred_currency"] == "EUR"
        assert store.clear_profile(1) is True
        assert store.get_profile(1) == {}
    finally:
        store.close()


def test_memory_profile_context_is_compact_and_preference_only():
    context = build_memory_profile_context({
        "preferred_currency": "USD",
        "focus_metrics": ["revenue", "gross margin"],
        "unknown": "ignored",
    })

    assert "- preferred_currency: USD" in context
    assert "- focus_metrics: revenue, gross margin" in context
    assert "unknown" not in context


def test_rewrite_query_uses_memory_profile_for_ambiguous_followup(tmp_path):
    llm = _RewriteLLM()
    engine = RAGEngine(llm, bm25_db_path=str(tmp_path / "bm25.db"))
    history = [
        {"role": "user", "content": "Analyze Apple's FY2024 report."},
        {"role": "assistant", "content": "I can help with Apple FY2024."},
    ]

    rewritten = asyncio.run(engine._rewrite_query_with_context(
        "What about revenue?",
        history,
        {
            "default_company": "Apple",
            "default_period": "FY2024",
            "preferred_currency": "USD",
        },
    ))

    assert rewritten == "What was FY2024 revenue for Apple in USD?"
    prompt = llm.prompts[-1]
    assert "User preference memory for query planning only" in prompt
    assert "- default_company: Apple" in prompt
    assert "do not treat as document facts" in prompt


def test_main_wires_memory_profile_endpoints_and_query_path():
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    assert 'memory_store = UserMemoryStore()' in content
    assert '@app.get("/memory/profile", response_model=MemoryProfileResponse)' in content
    assert '@app.put("/memory/profile", response_model=MemoryProfileResponse)' in content
    assert '@app.delete("/memory/profile")' in content
    assert "memory_store=memory_store" in content
