import os
import tempfile

from src.services.document_registry import DocumentRegistry


def _ready(registry, document_id, file_hash, *, parser, splitter, embedding):
    registry.register(
        document_id,
        1,
        "report.pdf",
        file_hash,
        status="pending",
        parser_version=parser,
        splitter_version=splitter,
        embedding_version=embedding,
    )
    registry.transition(document_id, "parsing")
    registry.mark_indexing(document_id)
    registry.mark_ready(document_id, 1, f"content-{document_id}")


def test_duplicate_detection_is_scoped_to_processing_lineage():
    handle = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    handle.close()
    try:
        registry = DocumentRegistry(db_path=handle.name)
        file_hash = DocumentRegistry.file_hash(b"same PDF bytes")
        _ready(
            registry,
            "native-v1",
            file_hash,
            parser="native-layout-v2",
            splitter="page-boundary-section-v2",
            embedding="all-MiniLM-L6-v2",
        )

        assert registry.find_by_file_hash(
            1,
            file_hash,
            parser_version="native-layout-v2",
            splitter_version="page-boundary-section-v2",
            embedding_version="all-MiniLM-L6-v2",
        )["document_id"] == "native-v1"
        assert registry.find_by_file_hash(
            1,
            file_hash,
            parser_version="mineru-content-list-v1",
            splitter_version="page-boundary-section-v2",
            embedding_version="all-MiniLM-L6-v2",
        ) is None
        # Legacy callers without lineage retain the original duplicate lookup.
        assert registry.find_by_file_hash(1, file_hash)["document_id"] == "native-v1"
    finally:
        os.unlink(handle.name)
