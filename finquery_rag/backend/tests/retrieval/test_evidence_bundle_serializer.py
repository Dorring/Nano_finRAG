from src.retrieval.evidence_bundle_serializer import build_evidence_bundle, build_token_budgeted_text, serialize_for_reranking

def test_table_row_and_headers_are_preserved():
    bundle = build_evidence_bundle({"doc_id":"row-1","content":"Cash | 42.2","metadata":{"type":"table_row","doc_name":"a.pdf","page":2,"table_headers":["Metric","2025"],"row_label":"Cash","table_title":"Liquidity"}})
    text = serialize_for_reranking(bundle)
    assert "HEADERS: Metric | 2025" in text
    assert "ROW: Cash | 42.2" in text

def test_optional_narrative_cannot_remove_row_header():
    bundle = build_evidence_bundle({"doc_id":"row-1","content":"Cash | 42.2","metadata":{"type":"table_row","row_label":"Cash","table_headers":["Metric","2025"]}})
    assert "HEADERS:" in build_token_budgeted_text(bundle, max_length=1)
