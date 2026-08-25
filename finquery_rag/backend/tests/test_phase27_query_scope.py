"""Phase 27 tests: query document filters are constrained to ready docs."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from services.query_scope import resolve_query_document_names


def test_resolve_query_uses_all_ready_docs_when_no_filter():
    resolved, invalid = resolve_query_document_names(
        requested_doc_names=None,
        ready_doc_names=["a.pdf", "b.pdf"],
        fallback_doc_names=["legacy.pdf"],
    )

    assert resolved == ["a.pdf", "b.pdf"]
    assert invalid == []


def test_resolve_query_rejects_unready_or_unknown_requested_docs():
    resolved, invalid = resolve_query_document_names(
        requested_doc_names=["ready.pdf", "failed.pdf"],
        ready_doc_names=["ready.pdf"],
        fallback_doc_names=["failed.pdf"],
    )

    assert resolved == []
    assert invalid == ["failed.pdf"]


def test_resolve_query_falls_back_for_legacy_indexes_without_registry():
    resolved, invalid = resolve_query_document_names(
        requested_doc_names=None,
        ready_doc_names=[],
        fallback_doc_names=["legacy.pdf"],
    )

    assert resolved == ["legacy.pdf"]
    assert invalid == []


def test_resolve_query_deduplicates_preserving_order():
    resolved, invalid = resolve_query_document_names(
        requested_doc_names=["b.pdf", "a.pdf", "b.pdf"],
        ready_doc_names=["a.pdf", "b.pdf"],
    )

    assert resolved == ["b.pdf", "a.pdf"]
    assert invalid == []


def test_main_query_endpoints_use_ready_document_resolver_static():
    """Phase 3 hotfix: /query/stream now calls engine.query() uniformly.

    Both /query and /query/stream resolve document names via
    ``_resolve_query_document_names_for_user`` and pass the resolved
    names to ``engine.query(doc_names=...)``. The legacy
    ``doc_names = resolved_doc_names`` assignment was removed when
    /query/stream was unified.
    """
    main_path = os.path.join(os.path.dirname(__file__), "..", "src", "main.py")
    content = open(main_path, encoding="utf-8").read()

    assert "def _resolve_query_document_names_for_user" in content
    assert "document_registry.list_documents(user_id)" in content
    assert "Documents are not ready or not found" in content
    assert "document_names=_resolve_query_document_names_for_user(" in content
    assert "request.document_names" in content
