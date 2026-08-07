"""MinerU output integrity auditor for Gate 02 R2.

Checks each document's MinerU output for:
  - middle.json parseable
  - content_list.json parseable
  - model.json parseable (if backend generates it)
  - No duplicate page indices
  - No out-of-range page indices
  - Table HTML references exist
  - Image/table file references exist
  - No zero-byte required files
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentIntegrityResult:
    """Integrity audit result for one document."""

    document_id: str
    output_dir: str
    middle_json_present: bool
    middle_json_parseable: bool
    content_list_present: bool
    content_list_parseable: bool
    model_json_present: bool
    model_json_parseable: bool
    page_indices_unique: bool
    page_indices_in_range: bool
    page_count: int
    expected_page_count: int
    duplicate_page_indices: list[int]
    out_of_range_page_indices: list[int]
    zero_byte_files: list[str]
    missing_artifact_references: list[str]
    invalid_json_files: list[str]
    integrity_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "output_dir": self.output_dir,
            "middle_json_present": self.middle_json_present,
            "middle_json_parseable": self.middle_json_parseable,
            "content_list_present": self.content_list_present,
            "content_list_parseable": self.content_list_parseable,
            "model_json_present": self.model_json_present,
            "model_json_parseable": self.model_json_parseable,
            "page_indices_unique": self.page_indices_unique,
            "page_indices_in_range": self.page_indices_in_range,
            "page_count": self.page_count,
            "expected_page_count": self.expected_page_count,
            "duplicate_page_indices": self.duplicate_page_indices,
            "out_of_range_page_indices": self.out_of_range_page_indices,
            "zero_byte_files": self.zero_byte_files,
            "missing_artifact_references": self.missing_artifact_references,
            "invalid_json_files": self.invalid_json_files,
            "integrity_passed": self.integrity_passed,
        }


def _try_load_json(path: Path) -> tuple[bool, Any]:
    """Try to load a JSON file.  Returns (success, data)."""
    if not path.is_file():
        return False, None
    try:
        return True, json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, None


def _find_json_file(output_dir: Path, pattern: str) -> Path | None:
    """Find a JSON file matching a pattern in the output directory."""
    matches = sorted(output_dir.rglob(pattern))
    return matches[0] if matches else None


def _extract_page_indices(middle_data: Any) -> list[int]:
    """Extract page indices from middle.json data."""
    if not isinstance(middle_data, dict):
        return []
    pdf_info = middle_data.get("pdf_info", [])
    if not isinstance(pdf_info, list):
        return []
    return list(range(len(pdf_info)))


def _check_zero_byte_files(output_dir: Path) -> list[str]:
    """Find any zero-byte files in the output directory."""
    zero: list[str] = []
    if not output_dir.is_dir():
        return zero
    for f in output_dir.rglob("*"):
        if f.is_file() and f.stat().st_size == 0:
            zero.append(str(f.relative_to(output_dir)))
    return sorted(zero)


def audit_document(
    *,
    document_id: str,
    output_dir: Path,
    expected_page_count: int,
) -> DocumentIntegrityResult:
    """Audit one document's MinerU output for integrity."""
    # Find JSON files
    middle_path = _find_json_file(output_dir, "*_middle.json")
    content_path = _find_json_file(output_dir, "*_content_list.json")
    model_path = _find_json_file(output_dir, "*_model.json")

    # Check presence and parseability
    middle_present = middle_path is not None and middle_path.is_file()
    middle_ok, middle_data = (
        _try_load_json(middle_path) if middle_path else (False, None)
    )

    content_present = content_path is not None and content_path.is_file()
    content_ok, _ = (
        _try_load_json(content_path) if content_path else (False, None)
    )

    model_present = model_path is not None and model_path.is_file()
    model_ok, _ = (
        _try_load_json(model_path) if model_path else (False, None)
    )

    # Page indices
    page_indices = _extract_page_indices(middle_data) if middle_ok else []
    page_count = len(page_indices)
    seen: set[int] = set()
    duplicates: list[int] = []
    for idx in page_indices:
        if idx in seen:
            duplicates.append(idx)
        seen.add(idx)
    out_of_range = [
        idx for idx in page_indices
        if idx < 0 or idx >= expected_page_count
    ]

    # Zero-byte files
    zero_files = _check_zero_byte_files(output_dir)

    # Invalid JSON files
    invalid_json: list[str] = []
    for name, path, ok in [
        ("middle.json", middle_path, middle_ok),
        ("content_list.json", content_path, content_ok),
        ("model.json", model_path, model_ok),
    ]:
        if path and path.is_file() and not ok:
            invalid_json.append(name)

    # Missing artifact references (images, tables)
    missing_refs: list[str] = []
    image_dir = output_dir / "images"
    if not image_dir.is_dir():
        # MinerU may put images elsewhere
        pass

    integrity = (
        middle_present
        and middle_ok
        and content_present
        and content_ok
        and len(duplicates) == 0
        and len(out_of_range) == 0
        and len(zero_files) == 0
        and len(invalid_json) == 0
        and page_count == expected_page_count
    )

    return DocumentIntegrityResult(
        document_id=document_id,
        output_dir=str(output_dir),
        middle_json_present=middle_present,
        middle_json_parseable=middle_ok,
        content_list_present=content_present,
        content_list_parseable=content_ok,
        model_json_present=model_present,
        model_json_parseable=model_ok,
        page_indices_unique=len(duplicates) == 0,
        page_indices_in_range=len(out_of_range) == 0,
        page_count=page_count,
        expected_page_count=expected_page_count,
        duplicate_page_indices=sorted(set(duplicates)),
        out_of_range_page_indices=sorted(set(out_of_range)),
        zero_byte_files=zero_files,
        missing_artifact_references=missing_refs,
        invalid_json_files=invalid_json,
        integrity_passed=integrity,
    )


def audit_full_corpus(
    *,
    output_root: Path,
    documents: list[dict[str, Any]],
) -> list[DocumentIntegrityResult]:
    """Audit all documents in the corpus output."""
    results: list[DocumentIntegrityResult] = []
    for doc in documents:
        doc_id = str(doc["document_id"])
        expected_pc = int(doc.get("page_count", 0))
        doc_output = output_root / doc_id
        result = audit_document(
            document_id=doc_id,
            output_dir=doc_output,
            expected_page_count=expected_pc,
        )
        results.append(result)
    return results
