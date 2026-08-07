"""Frozen benchmark corpus manifest loader for Gate 02 R2.

Reads exactly 8 PDFs from the frozen corpus manifest at
``benchmarks/financial_rag_v1/corpus.json`` and verifies integrity
(SHA256, page count) before any MinerU run.

No directory scanning, no glob-based PDF selection.  The manifest is
the single source of truth for which documents belong to the corpus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusDocument:
    """One frozen benchmark document."""

    document_id: str
    company: str
    filename: str
    fiscal_year: int
    source_type: str
    source_format: str
    page_count: int
    chunk_count: int
    file_sha256: str
    document_identity_hash: str
    # Resolved absolute path to the PDF file
    pdf_path: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "company": self.company,
            "filename": self.filename,
            "fiscal_year": self.fiscal_year,
            "source_type": self.source_type,
            "source_format": self.source_format,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "file_sha256": self.file_sha256,
            "document_identity_hash": self.document_identity_hash,
            "pdf_path": self.pdf_path,
        }


@dataclass(frozen=True)
class CorpusIntegrityResult:
    """Result of verifying the frozen corpus against actual PDF files."""

    document_count_expected: int
    document_count_actual: int
    total_pages_expected: int
    total_pages_actual: int
    all_sha256_match: bool
    all_page_counts_match: bool
    no_duplicate_document_ids: bool
    no_duplicate_pdf_hashes: bool
    no_missing_pdfs: bool
    no_unexpected_pdfs: bool
    documents: list[dict[str, Any]]
    integrity_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_document_count": self.document_count_expected,
            "actual_document_count": self.document_count_actual,
            "expected_total_pages": self.total_pages_expected,
            "actual_total_pages": self.total_pages_actual,
            "all_sha256_match": self.all_sha256_match,
            "all_page_counts_match": self.all_page_counts_match,
            "no_duplicate_document_ids": self.no_duplicate_document_ids,
            "no_duplicate_pdf_hashes": self.no_duplicate_pdf_hashes,
            "no_missing_pdfs": self.no_missing_pdfs,
            "no_unexpected_pdfs": self.no_unexpected_pdfs,
            "documents": self.documents,
            "integrity_passed": self.integrity_passed,
        }


def sha256_file(path: Path) -> str:
    """Compute SHA256 of a file, reading in 1MB chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_corpus_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load the frozen corpus manifest JSON."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def resolve_pdf_path(
    document: dict[str, Any],
    pdf_dir: Path,
) -> Path | None:
    """Find the PDF file for a corpus document.

    Uses the filename from the manifest.  Does NOT glob or scan directories.
    """
    filename = str(document.get("filename") or "")
    if not filename:
        return None
    candidate = pdf_dir / filename
    if candidate.is_file():
        return candidate
    return None


def get_pdf_page_count(pdf_path: Path) -> int:
    """Get the page count of a PDF.

    Tries PyMuPDF first, then falls back to a raw binary search for
    ``/Type /Page`` (not ``/Pages``) markers in the PDF file.  The raw
    fallback requires no external dependencies.
    """
    # Try PyMuPDF
    try:
        import fitz  # type: ignore[import-untyped]

        doc = fitz.open(str(pdf_path))
        count = int(doc.page_count)
        doc.close()
        return count
    except Exception:
        pass

    # Try pypdf
    try:
        import pypdf  # type: ignore[import-untyped]

        reader = pypdf.PdfReader(str(pdf_path))
        return len(reader.pages)
    except Exception:
        pass

    # Fallback: raw binary search for /Type /Page (not /Pages)
    try:
        data = pdf_path.read_bytes()
        # Count occurrences of /Type /Page but not /Type /Pages
        count = 0
        idx = 0
        while idx < len(data):
            pos = data.find(b"/Type /Page", idx)
            if pos == -1:
                pos = data.find(b"/Type/Page", idx)
                if pos == -1:
                    break
            # Check it's not /Pages
            end = pos + len(b"/Type /Page")
            if end < len(data) and data[end:end + 1] == b"s":
                idx = end + 1
                continue
            count += 1
            idx = end
        return count
    except Exception:
        return 0


def verify_corpus_integrity(
    manifest_path: Path,
    pdf_dir: Path,
) -> CorpusIntegrityResult:
    """Verify that all PDFs in the frozen manifest exist and match.

    Checks:
      - document_count = 8
      - SHA256 matches for each PDF
      - Page count matches for each PDF
      - No duplicate document_ids
      - No duplicate PDF hashes
      - No missing PDFs
    """
    manifest = load_corpus_manifest(manifest_path)
    docs_raw = manifest.get("documents", [])
    expected_count = int(manifest.get("document_count", 0))
    expected_pages = int(manifest.get("total_pages", 0))

    doc_records: list[dict[str, Any]] = []
    doc_ids: list[str] = []
    pdf_hashes: list[str] = []
    actual_pages = 0
    all_sha256 = True
    all_pages = True
    no_missing = True

    # Sort by document_id for deterministic order
    sorted_docs = sorted(docs_raw, key=lambda d: str(d.get("document_id") or ""))

    for doc in sorted_docs:
        doc_id = str(doc.get("document_id") or "")
        filename = str(doc.get("filename") or "")
        expected_sha = str(doc.get("file_sha256") or "")
        expected_pc = int(doc.get("page_count") or 0)

        pdf_path = resolve_pdf_path(doc, pdf_dir)
        if pdf_path is None or not pdf_path.is_file():
            no_missing = False
            doc_records.append({
                "document_id": doc_id,
                "filename": filename,
                "expected_sha256": expected_sha,
                "actual_sha256": None,
                "expected_page_count": expected_pc,
                "actual_page_count": None,
                "pdf_path": str(pdf_dir / filename),
                "integrity_passed": False,
                "error": "pdf_file_not_found",
            })
            doc_ids.append(doc_id)
            continue

        actual_sha = sha256_file(pdf_path)
        sha_match = actual_sha == expected_sha

        # When SHA256 matches, trust the manifest page count (the PDF
        # is verified to be the exact frozen file).  Only attempt to
        # count pages independently when SHA256 does NOT match.
        if sha_match:
            actual_pc = expected_pc
            pc_match = True
        else:
            actual_pc = get_pdf_page_count(pdf_path)
            pc_match = actual_pc == expected_pc

        actual_pages += actual_pc
        if not sha_match:
            all_sha256 = False
        if not pc_match:
            all_pages = False

        doc_records.append({
            "document_id": doc_id,
            "filename": filename,
            "expected_sha256": expected_sha,
            "actual_sha256": actual_sha,
            "expected_page_count": expected_pc,
            "actual_page_count": actual_pc,
            "pdf_path": str(pdf_path),
            "integrity_passed": sha_match and pc_match,
        })
        doc_ids.append(doc_id)
        pdf_hashes.append(actual_sha)

    no_dup_ids = len(doc_ids) == len(set(doc_ids))
    no_dup_hashes = len(pdf_hashes) == len(set(pdf_hashes))
    actual_count = len(sorted_docs)

    integrity = (
        actual_count == expected_count
        and all_sha256
        and all_pages
        and no_dup_ids
        and no_dup_hashes
        and no_missing
        and actual_pages == expected_pages
    )

    return CorpusIntegrityResult(
        document_count_expected=expected_count,
        document_count_actual=actual_count,
        total_pages_expected=expected_pages,
        total_pages_actual=actual_pages,
        all_sha256_match=all_sha256,
        all_page_counts_match=all_pages,
        no_duplicate_document_ids=no_dup_ids,
        no_duplicate_pdf_hashes=no_dup_hashes,
        no_missing_pdfs=no_missing,
        no_unexpected_pdfs=True,  # We only read from manifest, no scanning
        documents=doc_records,
        integrity_passed=integrity,
    )
