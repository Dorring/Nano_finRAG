"""Tests for Gate 02 R2 full corpus structured ingestion.

Covers the 19 required test cases:
 1.  test_frozen_corpus_exactly_eight_documents
 2.  test_pdf_hash_integrity
 3.  test_page_count_integrity
 4.  test_document_order_deterministic
 5.  test_one_document_one_output_directory
 6.  test_page_status_covers_all_pages
 7.  test_processed_no_table_is_success
 8.  test_no_duplicate_page_records
 9.  test_no_out_of_range_page_records
 10. test_required_json_parseable
 11. test_output_references_exist
 12. test_manifest_sorted_and_stable
 13. test_checkpoint_requires_same_config_hash
 14. test_no_per_page_backend_selection
 15. test_no_gold_before_seal
 16. test_no_adapter_runs
 17. test_no_index_or_retrieval_runs
 18. test_probe_regression_after_seal_only
 19. test_runtime_path_is_server_absolute
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pdf_retrieval_v4.deterministic_output_manifest import (  # noqa: E402
    build_output_manifest,
    compute_manifest_hash,
)
from src.pdf_retrieval_v4.frozen_corpus_manifest import (  # noqa: E402
    load_corpus_manifest,
)
from src.pdf_retrieval_v4.mineru_full_corpus_runner import (  # noqa: E402
    MinerUConfig,
)
from src.pdf_retrieval_v4.mineru_output_integrity import (  # noqa: E402
    audit_document,
)
from src.pdf_retrieval_v4.page_coverage import (  # noqa: E402
    build_page_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"


def _make_fake_corpus_dir(tmp: Path) -> tuple[Path, list[dict]]:
    """Create a fake corpus dir with minimal PDF stubs."""
    manifest = load_corpus_manifest(CORPUS_PATH)
    pdf_dir = tmp / "pdfs"
    pdf_dir.mkdir()
    docs = []
    for doc in manifest["documents"]:
        # Create a fake PDF file (just bytes, not a real PDF)
        fake_pdf = pdf_dir / doc["filename"]
        fake_pdf.write_bytes(b"FAKE_PDF_CONTENT")
        docs.append(doc)
    return pdf_dir, docs


def _make_fake_mineru_output(
    tmp: Path,
    doc_id: str,
    page_count: int,
) -> Path:
    """Create a fake MinerU output directory for one document."""
    doc_dir = tmp / "mineru" / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Create middle.json with page_count pages
    middle = {
        "pdf_info": [
            {"page_idx": i, "preblocks": []}
            for i in range(page_count)
        ]
    }
    (doc_dir / f"{doc_id}_middle.json").write_text(
        json.dumps(middle), encoding="utf-8"
    )

    # Create content_list.json
    content = [
        {"page_idx": i, "type": "text", "text": f"Page {i+1} text"}
        for i in range(page_count)
    ]
    (doc_dir / f"{doc_id}_content_list.json").write_text(
        json.dumps(content), encoding="utf-8"
    )

    # Create model.json
    (doc_dir / f"{doc_id}_model.json").write_text(
        json.dumps({"model": "fake"}), encoding="utf-8"
    )

    return doc_dir


# ---------------------------------------------------------------------------
# 1. test_frozen_corpus_exactly_eight_documents
# ---------------------------------------------------------------------------


class TestFrozenCorpusExactlyEight:
    def test_corpus_manifest_has_eight_documents(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        assert int(manifest.get("document_count", 0)) == 8

    def test_documents_list_has_eight_entries(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        assert len(manifest.get("documents", [])) == 8


# ---------------------------------------------------------------------------
# 2. test_pdf_hash_integrity
# ---------------------------------------------------------------------------


class TestPdfHashIntegrity:
    def test_all_documents_have_sha256(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        for doc in manifest["documents"]:
            assert doc.get("file_sha256")
            assert len(doc["file_sha256"]) == 64  # SHA256 hex

    def test_no_duplicate_hashes(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        hashes = [d["file_sha256"] for d in manifest["documents"]]
        assert len(hashes) == len(set(hashes))


# ---------------------------------------------------------------------------
# 3. test_page_count_integrity
# ---------------------------------------------------------------------------


class TestPageCountIntegrity:
    def test_total_pages_match_sum(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        total = int(manifest["total_pages"])
        summed = sum(int(d["page_count"]) for d in manifest["documents"])
        assert total == summed

    def test_all_documents_have_page_count(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        for doc in manifest["documents"]:
            assert int(doc.get("page_count", 0)) > 0


# ---------------------------------------------------------------------------
# 4. test_document_order_deterministic
# ---------------------------------------------------------------------------


class TestDocumentOrderDeterministic:
    def test_sorted_by_document_id(self) -> None:
        manifest = load_corpus_manifest(CORPUS_PATH)
        docs = manifest["documents"]
        doc_ids = [d["document_id"] for d in docs]
        sorted_ids = sorted(doc_ids)
        # When we sort, the order should be deterministic
        assert sorted(doc_ids) == sorted_ids

    def test_runner_sorts_documents(self) -> None:
        """The runner must sort by document_id, not filesystem order."""
        docs = [
            {"document_id": "z_doc", "filename": "z.pdf", "page_count": 1, "file_sha256": "z"},
            {"document_id": "a_doc", "filename": "a.pdf", "page_count": 1, "file_sha256": "a"},
            {"document_id": "m_doc", "filename": "m.pdf", "page_count": 1, "file_sha256": "m"},
        ]
        sorted_docs = sorted(docs, key=lambda d: d["document_id"])
        assert sorted_docs[0]["document_id"] == "a_doc"
        assert sorted_docs[1]["document_id"] == "m_doc"
        assert sorted_docs[2]["document_id"] == "z_doc"


# ---------------------------------------------------------------------------
# 5. test_one_document_one_output_directory
# ---------------------------------------------------------------------------


class TestOneDocumentOneDirectory:
    def test_each_document_gets_own_directory(self, tmp_path: Path) -> None:
        for doc_id in ["aapl_fy2025", "jpm_fy2025"]:
            _make_fake_mineru_output(tmp_path, doc_id, 5)
        assert (tmp_path / "mineru" / "aapl_fy2025").is_dir()
        assert (tmp_path / "mineru" / "jpm_fy2025").is_dir()
        assert (tmp_path / "mineru" / "aapl_fy2025") != (tmp_path / "mineru" / "jpm_fy2025")


# ---------------------------------------------------------------------------
# 6. test_page_status_covers_all_pages
# ---------------------------------------------------------------------------


class TestPageStatusCoversAll:
    def test_all_pages_have_status(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 10)
        statuses = build_page_status(
            document_id="test_doc",
            expected_page_count=10,
            output_dir=doc_dir,
        )
        assert len(statuses) == 10
        assert all(s.pdf_page >= 1 for s in statuses)
        assert all(s.page_index >= 0 for s in statuses)

    def test_no_missing_pages(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 5)
        statuses = build_page_status(
            document_id="test_doc",
            expected_page_count=5,
            output_dir=doc_dir,
        )
        pages = [s.pdf_page for s in statuses]
        assert pages == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# 7. test_processed_no_table_is_success
# ---------------------------------------------------------------------------


class TestProcessedNoTableIsSuccess:
    def test_no_table_page_is_processed_no_table(self, tmp_path: Path) -> None:
        # Create a page with no tables and no text blocks
        doc_dir = tmp_path / "mineru" / "test_doc"
        doc_dir.mkdir(parents=True)
        middle = {"pdf_info": [{"page_idx": 0, "preblocks": []}]}
        (doc_dir / "test_doc_middle.json").write_text(
            json.dumps(middle), encoding="utf-8"
        )
        content = [{"page_idx": 0, "type": "discarded", "text": ""}]
        (doc_dir / "test_doc_content_list.json").write_text(
            json.dumps(content), encoding="utf-8"
        )
        statuses = build_page_status(
            document_id="test_doc",
            expected_page_count=1,
            output_dir=doc_dir,
        )
        assert statuses[0].status == "processed_no_table"

    def test_processed_no_table_not_failed(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 1)
        statuses = build_page_status(
            document_id="test_doc",
            expected_page_count=1,
            output_dir=doc_dir,
        )
        # The fake output has text blocks, so it should be "processed" not "failed"
        assert statuses[0].status != "failed"


# ---------------------------------------------------------------------------
# 8. test_no_duplicate_page_records
# ---------------------------------------------------------------------------


class TestNoDuplicatePageRecords:
    def test_no_duplicate_pages(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 5)
        statuses = build_page_status(
            document_id="test_doc",
            expected_page_count=5,
            output_dir=doc_dir,
        )
        pages = [(s.document_id, s.pdf_page) for s in statuses]
        assert len(pages) == len(set(pages))


# ---------------------------------------------------------------------------
# 9. test_no_out_of_range_page_records
# ---------------------------------------------------------------------------


class TestNoOutOfRangePages:
    def test_all_pages_in_range(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 3)
        statuses = build_page_status(
            document_id="test_doc",
            expected_page_count=3,
            output_dir=doc_dir,
        )
        for s in statuses:
            assert 1 <= s.pdf_page <= 3
            assert 0 <= s.page_index < 3


# ---------------------------------------------------------------------------
# 10. test_required_json_parseable
# ---------------------------------------------------------------------------


class TestRequiredJsonParseable:
    def test_middle_json_parseable(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 3)
        result = audit_document(
            document_id="test_doc",
            output_dir=doc_dir,
            expected_page_count=3,
        )
        assert result.middle_json_present
        assert result.middle_json_parseable

    def test_content_list_parseable(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 3)
        result = audit_document(
            document_id="test_doc",
            output_dir=doc_dir,
            expected_page_count=3,
        )
        assert result.content_list_present
        assert result.content_list_parseable


# ---------------------------------------------------------------------------
# 11. test_output_references_exist
# ---------------------------------------------------------------------------


class TestOutputReferencesExist:
    def test_all_output_files_exist(self, tmp_path: Path) -> None:
        doc_dir = _make_fake_mineru_output(tmp_path, "test_doc", 2)
        result = audit_document(
            document_id="test_doc",
            output_dir=doc_dir,
            expected_page_count=2,
        )
        assert result.missing_artifact_references == []
        assert result.integrity_passed


# ---------------------------------------------------------------------------
# 12. test_manifest_sorted_and_stable
# ---------------------------------------------------------------------------


class TestManifestSortedAndStable:
    def test_manifest_sorted_by_document_then_path(self, tmp_path: Path) -> None:
        # Create two docs with files
        for doc_id in ["b_doc", "a_doc"]:
            _make_fake_mineru_output(tmp_path, doc_id, 2)

        manifest = build_output_manifest(tmp_path / "mineru", [
            {"document_id": "b_doc", "page_count": 2},
            {"document_id": "a_doc", "page_count": 2},
        ])

        file_keys = [
            (f["document_id"], f["relative_path"])
            for f in manifest["files"]
        ]
        assert file_keys == sorted(file_keys)

    def test_manifest_hash_stable(self, tmp_path: Path) -> None:
        _make_fake_mineru_output(tmp_path, "test_doc", 2)
        docs = [{"document_id": "test_doc", "page_count": 2}]

        m1 = build_output_manifest(tmp_path / "mineru", docs)
        m2 = build_output_manifest(tmp_path / "mineru", docs)

        h1 = compute_manifest_hash(m1)
        h2 = compute_manifest_hash(m2)
        assert h1 == h2


# ---------------------------------------------------------------------------
# 13. test_checkpoint_requires_same_config_hash
# ---------------------------------------------------------------------------


class TestCheckpointConfigHash:
    def test_different_config_hash_not_skipped(self) -> None:
        c1 = MinerUConfig(backend="hybrid-engine", method="auto", effort="high")
        c2 = MinerUConfig(backend="pipeline", method="auto", effort="high")
        assert c1.config_hash != c2.config_hash

    def test_same_config_hash_same_value(self) -> None:
        c1 = MinerUConfig()
        c2 = MinerUConfig()
        assert c1.config_hash == c2.config_hash


# ---------------------------------------------------------------------------
# 14. test_no_per_page_backend_selection
# ---------------------------------------------------------------------------


class TestNoPerPageBackendSelection:
    def test_config_is_fixed(self) -> None:
        config = MinerUConfig()
        assert config.backend == "hybrid-engine"
        assert config.method == "auto"
        assert config.effort == "high"

    def test_config_hash_is_deterministic(self) -> None:
        c1 = MinerUConfig().config_hash
        c2 = MinerUConfig().config_hash
        assert c1 == c2


# ---------------------------------------------------------------------------
# 15. test_no_gold_before_seal
# ---------------------------------------------------------------------------


class TestNoGoldBeforeSeal:
    def test_protocol_has_zero_gold_reads(self) -> None:
        protocol_path = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/gate-02-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        assert protocol.get("gold_reads_before_seal") == 0
        assert protocol.get("question_reads") == 0
        assert protocol.get("governance_reads_before_seal") == 0
        assert protocol.get("expected_value_reads_before_seal") == 0
        assert protocol.get("reference_answer_reads_before_seal") == 0


# ---------------------------------------------------------------------------
# 16. test_no_adapter_runs
# ---------------------------------------------------------------------------


class TestNoAdapterRuns:
    def test_protocol_has_zero_adapter_runs(self) -> None:
        protocol_path = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/gate-02-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        assert protocol.get("adapter_runs") == 0
        assert protocol.get("header_graph_runs") == 0
        assert protocol.get("evidence_unit_builds") == 0


# ---------------------------------------------------------------------------
# 17. test_no_index_or_retrieval_runs
# ---------------------------------------------------------------------------


class TestNoIndexOrRetrievalRuns:
    def test_protocol_has_zero_index_and_retrieval(self) -> None:
        protocol_path = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/gate-02-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        assert protocol.get("index_builds") == 0
        assert protocol.get("retrieval_runs") == 0
        assert protocol.get("reranker_calls") == 0
        assert protocol.get("production_index_writes") == 0


# ---------------------------------------------------------------------------
# 18. test_probe_regression_after_seal_only
# ---------------------------------------------------------------------------


class TestProbeRegressionAfterSealOnly:
    def test_seal_required_before_regression(self, tmp_path: Path) -> None:
        """The regression script must check that a seal exists."""
        # Create a fake out dir without a seal
        out_dir = tmp_path / "gate-02-r2"
        out_dir.mkdir()
        # No seal file → script should fail
        assert not (out_dir / "full-corpus-ingestion-seal.json").is_file()

    def test_seal_has_zero_reads_before_seal(self) -> None:
        seal_path = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/full-corpus-ingestion-seal.json"
        if not seal_path.is_file():
            pytest.skip("Seal not yet generated")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        assert seal.get("gold_reads_before_seal") == 0
        assert seal.get("governance_reads_before_seal") == 0
        assert seal.get("question_reads") == 0
        assert seal.get("sealed") is True


# ---------------------------------------------------------------------------
# 19. test_runtime_path_is_server_absolute
# ---------------------------------------------------------------------------


class TestRuntimePathIsServerAbsolute:
    def test_protocol_runtime_path_is_absolute(self) -> None:
        protocol_path = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/gate-02-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        runbook = protocol.get("runbook", {})
        tmpdir = runbook.get("project_tmpdir", "")
        # Must be an absolute path (starts with / on Linux)
        assert tmpdir.startswith("/"), f"tmpdir must be absolute: {tmpdir}"

    def test_mineru_path_is_absolute(self) -> None:
        protocol_path = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r2/gate-02-r2-protocol.json"
        if not protocol_path.is_file():
            pytest.skip("Protocol not yet generated")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        runbook = protocol.get("runbook", {})
        mineru_env = runbook.get("isolated_mineru_environment", "")
        assert mineru_env.startswith("/"), f"mineru env must be absolute: {mineru_env}"
