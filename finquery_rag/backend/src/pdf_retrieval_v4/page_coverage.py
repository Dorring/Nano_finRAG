"""Page coverage tracker for Gate 02 R2.

Generates per-page processing status records for all pages in the
corpus, even pages with no tables or text.  Each page gets exactly one
record (no duplicates, no out-of-range).
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PageStatus:
    """Processing status for one page."""

    document_id: str
    pdf_page: int  # 1-based user-visible page number
    page_index: int  # 0-based internal index
    status: str  # "processed" | "processed_no_table" | "failed"
    mineru_page_present: bool
    content_list_present: bool
    middle_json_present: bool
    table_count: int
    text_block_count: int
    image_count: int
    error_type: str | None
    error_message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "pdf_page": self.pdf_page,
            "page_index": self.page_index,
            "status": self.status,
            "mineru_page_present": self.mineru_page_present,
            "content_list_present": self.content_list_present,
            "middle_json_present": self.middle_json_present,
            "table_count": self.table_count,
            "text_block_count": self.text_block_count,
            "image_count": self.image_count,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


def _find_middle_json(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.rglob("*_middle.json"))
    return matches[0] if matches else None


def _find_content_list(output_dir: Path) -> Path | None:
    matches = sorted(output_dir.rglob("*_content_list.json"))
    return matches[0] if matches else None


def _count_page_blocks(content_data: Any, page_index: int) -> tuple[int, int]:
    """Count text blocks and tables for a page from content_list.json."""
    if not isinstance(content_data, list):
        return 0, 0
    text_blocks = 0
    tables = 0
    for block in content_data:
        if not isinstance(block, dict):
            continue
        if block.get("page_idx") != page_index:
            continue
        block_type = str(block.get("type") or "")
        if block_type in ("text", "title", "discarded") and block.get("text"):
            text_blocks += 1
        elif block_type == "table":
            tables += 1
    return text_blocks, tables


def _count_page_tables(middle_data: Any, page_index: int) -> int:
    """Count tables for a page from middle.json."""
    if not isinstance(middle_data, dict):
        return 0
    pdf_info = middle_data.get("pdf_info", [])
    if not isinstance(pdf_info, list) or page_index >= len(pdf_info):
        return 0
    page_data = pdf_info[page_index]
    if not isinstance(page_data, dict):
        return 0
    # Count preblocks that are tables
    count = 0
    preblocks = page_data.get("preblocks", [])
    if isinstance(preblocks, list):
        for block in preblocks:
            if isinstance(block, dict) and block.get("type") == "table":
                count += 1
    return count


def _count_images(output_dir: Path) -> int:
    """Count image files in the output directory."""
    image_dir = output_dir / "images"
    if not image_dir.is_dir():
        return 0
    return sum(
        1 for f in image_dir.iterdir()
        if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png", ".svg")
    )


def build_page_status(
    *,
    document_id: str,
    expected_page_count: int,
    output_dir: Path,
) -> list[PageStatus]:
    """Build page status records for all pages in a document.

    Generates one record per page (1..expected_page_count), even if
    the page has no tables or text.
    """
    middle_path = _find_middle_json(output_dir)
    content_path = _find_content_list(output_dir)

    middle_data: Any = None
    content_data: Any = None

    if middle_path and middle_path.is_file():
        try:
            middle_data = json.loads(middle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            middle_data = None

    if content_path and content_path.is_file():
        try:
            content_data = json.loads(content_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            content_data = None

    image_count = _count_images(output_dir)
    middle_present = middle_data is not None
    content_present = content_data is not None

    # Determine actual pages from middle.json
    actual_pages = 0
    if isinstance(middle_data, dict):
        pdf_info = middle_data.get("pdf_info", [])
        if isinstance(pdf_info, list):
            actual_pages = len(pdf_info)

    statuses: list[PageStatus] = []
    for page_idx in range(expected_page_count):
        pdf_page = page_idx + 1  # 1-based

        mineru_page_present = page_idx < actual_pages
        text_blocks, content_tables = _count_page_blocks(content_data, page_idx)
        middle_tables = _count_page_tables(middle_data, page_idx)
        table_count = max(content_tables, middle_tables)

        if not mineru_page_present:
            status = "failed"
            error_type = "page_missing_from_output"
            error_msg = f"Page {pdf_page} not found in MinerU output"
        elif table_count == 0 and text_blocks == 0:
            status = "processed_no_table"
            error_type = None
            error_msg = None
        else:
            status = "processed"
            error_type = None
            error_msg = None

        statuses.append(PageStatus(
            document_id=document_id,
            pdf_page=pdf_page,
            page_index=page_idx,
            status=status,
            mineru_page_present=mineru_page_present,
            content_list_present=content_present,
            middle_json_present=middle_present,
            table_count=table_count,
            text_block_count=text_blocks,
            image_count=image_count,
            error_type=error_type,
            error_message=error_msg,
        ))

    return statuses


def write_page_status_gzip(
    path: Path,
    all_statuses: list[PageStatus],
) -> None:
    """Write all page statuses to a gzipped JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as handle:
            for status in all_statuses:
                handle.write(
                    (
                        json.dumps(
                            status.to_dict(),
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        )
                        + "\n"
                    ).encode("utf-8")
                )


@dataclass(frozen=True)
class PageCoverageSummary:
    """Summary of page coverage across the corpus."""

    total_pages: int
    processed: int
    processed_no_table: int
    failed: int
    missing_pages: int
    duplicate_pages: int
    out_of_range_pages: int
    all_pages_covered: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_pages": self.total_pages,
            "processed": self.processed,
            "processed_no_table": self.processed_no_table,
            "failed": self.failed,
            "missing_pages": self.missing_pages,
            "duplicate_pages": self.duplicate_pages,
            "out_of_range_pages": self.out_of_range_pages,
            "all_pages_covered": self.all_pages_covered,
        }


def summarize_page_coverage(
    all_statuses: list[PageStatus],
    documents: list[dict[str, Any]],
) -> PageCoverageSummary:
    """Summarize page coverage across all documents."""
    expected_total = sum(int(d.get("page_count", 0)) for d in documents)
    processed = sum(1 for s in all_statuses if s.status == "processed")
    no_table = sum(1 for s in all_statuses if s.status == "processed_no_table")
    failed = sum(1 for s in all_statuses if s.status == "failed")

    # Check for duplicates and out-of-range
    seen: set[tuple[str, int]] = set()
    duplicates = 0
    out_of_range = 0
    for s in all_statuses:
        key = (s.document_id, s.pdf_page)
        if key in seen:
            duplicates += 1
        seen.add(key)
        doc = next(
            (d for d in documents if d.get("document_id") == s.document_id),
            None,
        )
        if doc and s.pdf_page > int(doc.get("page_count", 0)):
            out_of_range += 1

    missing = expected_total - len(all_statuses)

    return PageCoverageSummary(
        total_pages=expected_total,
        processed=processed,
        processed_no_table=no_table,
        failed=failed,
        missing_pages=max(0, missing),
        duplicate_pages=duplicates,
        out_of_range_pages=out_of_range,
        all_pages_covered=(
            missing == 0
            and duplicates == 0
            and out_of_range == 0
            and failed == 0
        ),
    )
