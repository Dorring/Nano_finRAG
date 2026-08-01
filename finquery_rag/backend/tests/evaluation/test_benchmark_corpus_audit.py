from scripts.evaluation.benchmark_foundation import document_identity_payload


def test_corpus_identity_payload_excludes_runtime_paths_and_text():
    payload = document_identity_payload({"document_id": "doc", "filename": "doc.pdf", "file_sha256": "a" * 64, "page_count": 1, "chunk_count": 2, "runtime_path": "/secret", "content": "private"})
    assert set(payload) == {"document_id", "filename", "file_sha256", "page_count", "chunk_count"}
