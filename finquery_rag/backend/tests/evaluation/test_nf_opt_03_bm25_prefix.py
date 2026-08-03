from src.evaluation.nf_opt_03 import identity_integrity, prefix_integrity


def _candidate(key, document_id="doc-a"):
    return {"candidate_key": key, "document_id": document_id, "content_hash": f"h-{key}"}


def test_expanded_bm25_preserves_current_prefix():
    base = [_candidate("a"), _candidate("b")]
    expanded = base + [_candidate("c")]
    result = prefix_integrity(base, expanded)
    assert result["passed"] is True
    assert result["order_changed"] is False


def test_bm25_candidate_identity_is_stable():
    result = identity_integrity([_candidate("a")], allowed_document_ids={"doc-a"})
    assert result["passed"] is True
    assert result["missing_identity_count"] == 0


def test_expanded_prefix_change_fails_closed():
    result = prefix_integrity([_candidate("a"), _candidate("b")], [_candidate("b"), _candidate("a")])
    assert result["passed"] is False
