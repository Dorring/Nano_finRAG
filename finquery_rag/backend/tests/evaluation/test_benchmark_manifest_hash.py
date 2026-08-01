from scripts.evaluation.benchmark_foundation import corpus_hash, document_identity_hash


def _doc(document_id="doc-1"):
    return {"document_id": document_id, "filename": f"{document_id}.pdf", "file_sha256": "a" * 64, "page_count": 10, "chunk_count": 20}


def test_corpus_hash_is_deterministic_and_order_independent():
    assert corpus_hash([_doc("b"), _doc("a")]) == corpus_hash([_doc("a"), _doc("b")])


def test_document_identity_hash_changes_when_file_changes():
    first = _doc()
    second = {**first, "file_sha256": "b" * 64}
    assert document_identity_hash(first) != document_identity_hash(second)
