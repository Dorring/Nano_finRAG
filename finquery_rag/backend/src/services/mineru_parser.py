"""Optional MinerU CLI adapter with a stable FinQuery chunk contract.

MinerU is intentionally an opt-in dependency.  This adapter consumes its
flat ``*_content_list.json`` artifact rather than its internal model output,
which keeps the RAG ingestion schema stable across parser implementations.
"""

from __future__ import annotations

import json
import os
import html
import re
import shlex
import subprocess
import tempfile
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

from .chunk_id import make_chunk_id


class MinerUParseError(RuntimeError):
    """Raised when an explicitly selected MinerU parser cannot produce output."""


def _mineru_subprocess_env() -> dict[str, str]:
    """Return a child environment without changing backend GPU visibility."""
    env = os.environ.copy()
    force_cpu = (
        os.getenv("MINERU_FORCE_CPU", "").strip().lower() in {"1", "true", "yes"}
    )
    if force_cpu:
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        visible_devices = os.getenv("MINERU_CUDA_VISIBLE_DEVICES", "").strip()
        if visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = visible_devices
    return env


def _as_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(_as_text(item) for item in value if _as_text(item)).strip()
    if isinstance(value, dict):
        return _as_text(value.get("content", ""))
    return ""


def _normalize_table_fragment(value: str) -> str:
    """Turn MinerU table HTML into readable row-oriented evidence."""
    value = html.unescape(value or "")
    value = re.sub(r"<\s*/\s*tr\s*>", "\n", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*(?:td|th)\b[^>]*>", " | ", value, flags=re.IGNORECASE)
    value = re.sub(r"<\s*/\s*(?:td|th)\s*>", " ", value, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", " ", value)
    rows = []
    for row in value.splitlines():
        row = re.sub(r"[\t\r\f\v ]+", " ", row).strip(" |")
        if row:
            rows.append(row)
    return "\n".join(rows)


def _block_text(block: dict) -> str:
    """Extract searchable text without erasing table cell boundaries."""
    block_type = str(block.get("type", ""))
    if block_type == "table":
        parts = []
        for label, value in (
            ("", block.get("table_caption")),
            ("", block.get("table_body")),
            ("Notes: ", block.get("table_footnote")),
        ):
            raw = _as_text(value)
            if not raw:
                continue
            normalized = _normalize_table_fragment(raw)
            if normalized:
                parts.append(f"{label}{normalized}")
        return "\n".join(parts).strip()

    parts = [_as_text(block.get("text")), _as_text(block.get("content"))]
    text = "\n".join(part for part in parts if part)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _front_matter_title(records: list[dict]) -> str:
    """Infer a cover title from early text blocks, without document-specific rules."""
    lines = []
    stop_markers = ("abstract", "keywords", "paper id", "anonymous")
    for record in records:
        if record["page"] != 1 or record["block_type"] != "text":
            continue
        candidate = re.sub(r"\s+", " ", record["text"]).strip()
        if not candidate:
            continue
        if any(marker in candidate.lower() for marker in stop_markers):
            break
        if len(candidate) > 180:
            break
        lines.append(candidate)
        if len(lines) >= 5:
            break
    title = " ".join(lines).strip()
    return title if len(title) >= 8 else ""


def _window_records(records: list[dict], *, max_chars: int) -> list[dict]:
    """Merge adjacent text records within a page/section into useful evidence windows."""
    windows: list[dict] = []
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current and current["text"].strip():
            windows.append(current)
        current = None

    for record in records:
        if record["block_type"] == "table":
            flush()
            windows.append(record)
            continue
        compatible = (
            current is not None
            and current["page"] == record["page"]
            and current["section_path"] == record["section_path"]
        )
        candidate = f'{current["text"]}\n{record["text"]}' if compatible else record["text"]
        if not compatible or len(candidate) > max_chars:
            flush()
            current = dict(record)
        else:
            current["text"] = candidate
    flush()
    return windows

def _load_content_list(output_dir: Path) -> list[dict]:
    candidates = sorted(
        path for path in output_dir.rglob("*_content_list.json")
        if not path.name.endswith("_content_list_v2.json")
    )
    if not candidates:
        raise MinerUParseError("MinerU did not produce a content_list.json artifact")
    try:
        payload = json.loads(candidates[0].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MinerUParseError(f"Cannot read MinerU content list: {exc}") from exc
    if not isinstance(payload, list):
        raise MinerUParseError("MinerU content list has an unsupported shape")
    return [block for block in payload if isinstance(block, dict)]


def _run_mineru(pdf_path: str, output_dir: Path) -> list[dict]:
    command = shlex.split(os.getenv("MINERU_COMMAND", "mineru"))
    if not command:
        raise MinerUParseError("MINERU_COMMAND is empty")
    backend = os.getenv("MINERU_BACKEND", "pipeline")
    method = os.getenv("MINERU_METHOD", "auto").strip().lower()
    if method not in {"auto", "txt", "ocr"}:
        raise MinerUParseError("MINERU_METHOD must be one of: auto, txt, ocr")
    args = command + ["-p", pdf_path, "-o", str(output_dir), "-b", backend, "-m", method]
    api_url = os.getenv("MINERU_API_URL", "").strip()
    if api_url:
        args.extend(["--api-url", api_url])
    timeout = max(1, int(os.getenv("MINERU_TIMEOUT_SECONDS", "600")))
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout,
            check=False, env=_mineru_subprocess_env(),
        )
    except FileNotFoundError as exc:
        raise MinerUParseError("MinerU CLI is unavailable; install/configure MINERU_COMMAND") from exc
    except subprocess.TimeoutExpired as exc:
        raise MinerUParseError(f"MinerU timed out after {timeout}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()[-600:]
        raise MinerUParseError(f"MinerU exited with {result.returncode}: {detail}")
    return _load_content_list(output_dir)


def append_table_row_children(chunks: list[dict]) -> list[dict]:
    """Add retrievable table rows and column-labelled numeric cell evidence."""
    expanded = list(chunks or [])
    for parent in chunks or []:
        metadata = parent.get("metadata") or {}
        if metadata.get("type") != "table":
            continue
        parent_id = metadata.get("doc_id")
        if not isinstance(parent_id, str) or not parent_id.strip():
            continue

        lines = [
            line.strip()
            for line in str(parent.get("content") or "").splitlines()
            if line.strip()
        ]
        header_lines = []
        for line in lines:
            if _is_table_data_row(line):
                break
            if not _is_table_separator(line):
                header_lines.append(line)
        # Preserve nearby table title, units, and column labels. They carry
        # the scale and period needed to interpret a row value correctly.
        header_context = header_lines[-3:]
        header_cells = _table_header_cells(header_context)

        row_count = 0
        cell_count = 0
        for line_index, line in enumerate(lines):
            if not _is_table_data_row(line):
                continue
            row_count += 1
            if row_count > 80:
                break
            content = "\n".join([*header_context, line])[:900]
            row_metadata = {
                **metadata,
                "type": "table_row",
                "subtype": "numeric_row",
                "parent_table_id": parent_id,
                "parent_id": parent_id,
                "table_row_index": line_index,
                "table_header_context": "\n".join(header_context)[:600],
                "table_row_child": True,
                "doc_id": f"{parent_id}::row_{line_index}",
            }
            row_chunk = {"content": content, "metadata": row_metadata}
            expanded.append(row_chunk)
            if cell_count >= 250:
                continue
            for cell_chunk in _table_cell_children(
                row_chunk=row_chunk,
                raw_row=line,
                header_cells=header_cells,
                unit_context=header_context[:-1],
            ):
                if cell_count >= 250:
                    break
                expanded.append(cell_chunk)
                cell_count += 1
    return expanded


def _table_header_cells(header_context: list[str]) -> list[str]:
    """Return the richest pipe-delimited header row, preserving column order.

    PDF extraction frequently wraps a multi-line header.  The closest line can
    be a descriptor such as ``Program | Program Title`` while the preceding
    line carries the actual value columns.  Prefer the row with the most
    non-empty cells; ties keep the later row closest to the data.
    """
    candidates = []
    for position, line in enumerate(header_context or []):
        cells = _split_table_cells(line)
        non_empty = [cell for cell in cells if cell]
        if len(non_empty) >= 2:
            candidates.append((len(non_empty), position, cells))
    if not candidates:
        return []
    return max(candidates, key=lambda item: (item[0], item[1]))[2]


def _split_table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in (line or "").strip().strip("|").split("|")]


def _table_cell_children(
    *,
    row_chunk: dict,
    raw_row: str,
    header_cells: list[str],
    unit_context: list[str],
) -> list[dict]:
    """Emit narrow, column-labelled numeric evidence from a structured table row.

    Rows remain the primary representation, while cells make values from
    multi-column financial statements independently retrievable. Alignment is
    positional and document-agnostic: a leading descriptor cell is treated as
    the row label when the remaining cells match the header count.
    """
    if not header_cells:
        return []
    row_cells = _split_table_cells(raw_row)
    if len(row_cells) < len(header_cells):
        return []

    if len(row_cells) == len(header_cells) + 1:
        row_label = row_cells[0]
        values = row_cells[1:]
    elif len(row_cells) == len(header_cells):
        row_label = ""
        values = row_cells
    else:
        values = row_cells[-len(header_cells):]
        row_label = " ".join(cell for cell in row_cells[:-len(header_cells)] if cell)

    if len(values) != len(header_cells):
        return []

    row_metadata = dict(row_chunk.get("metadata") or {})
    row_doc_id = str(row_metadata.get("doc_id") or "").strip()
    if not row_doc_id:
        return []

    context = " ".join(item.strip() for item in unit_context if item.strip())[:360]
    children = []
    for column_index, (header, value) in enumerate(zip(header_cells, values)):
        header = header.strip()
        value = value.strip()
        if not header or not value or not re.search(r"\d", value):
            continue
        # Put the semantic column and its value first. This keeps dates in a
        # row label as supporting context rather than the leading numeric
        # candidate for a numeric extractor.
        parts = [f"Column: {header}", f"Value: {value}"]
        if row_label:
            parts.append(f"Table row: {row_label}")
        if context:
            parts.append(f"Table context: {context}")
        cell_metadata = {
            **row_metadata,
            "type": "table_cell",
            "subtype": "numeric_cell",
            "parent_row_id": row_doc_id,
            "table_column": header,
            "table_column_index": column_index,
            "table_cell_child": True,
            "doc_id": f"{row_doc_id}::cell_{column_index}",
        }
        children.append({"content": "; ".join(parts)[:900], "metadata": cell_metadata})
    return children


def _is_table_separator(line: str) -> bool:
    return bool(re.fullmatch(r"[|:\- ]+", line or ""))


def _is_table_data_row(line: str) -> bool:
    if "|" not in (line or "") or _is_table_separator(line):
        return False
    if _is_table_header_row(line):
        return False
    return bool(re.search(r"\d", line))


def _is_table_header_row(line: str) -> bool:
    """Treat descriptor-plus-period columns as headers, not financial values.

    Extracted financial tables often render column years as ``2020 (1)`` or
    spread labels across multiple lines.  A line with two or more year labels
    and no non-year financial amount is therefore a header even when the year
    cells include footnote markers.
    """
    cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
    if not cells:
        return False
    normalized = [re.sub(r"[^a-z]", "", cell.lower()) for cell in cells]
    header_terms = {
        "metric", "metrics", "activity", "activities", "description", "item",
        "year", "years", "program", "programtitle", "particulars", "amount",
        "budget", "actual", "difference", "note",
    }
    if any(cell in header_terms for cell in normalized):
        return True

    numeric_tokens = re.findall(r"(?<![A-Za-z])\d[\d,]*(?:\.\d+)?", line or "")
    years = [token for token in numeric_tokens if re.fullmatch(r"(?:19|20)\d{2}", token)]
    non_year_amounts = [
        token for token in numeric_tokens
        if not re.fullmatch(r"(?:19|20)\d{2}", token)
        and ("," in token or len(token.replace(",", "")) >= 3)
    ]
    return len(years) >= 2 and not non_year_amounts


def process_pdf_with_mineru(
    pdf_path: str,
    *,
    user_id: int | None,
    recursive_splitter,
    long_chunk_threshold: int,
    hierarchy_metadata_fn,
    chunk_content_with_section_fn,
) -> tuple[list[dict], int]:
    """Run MinerU and emit stable, section-aware evidence chunks."""
    doc_name = os.path.basename(pdf_path)
    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)
    with tempfile.TemporaryDirectory(prefix="finquery_mineru_") as output_root:
        blocks = _run_mineru(pdf_path, Path(output_root))

    headings: dict[int, str] = {}
    ignored_types = {"header", "footer", "page_number", "aside_text", "page_footnote"}
    records: list[dict] = []
    for block in blocks:
        original_type = str(block.get("type", "text"))
        if original_type in ignored_types:
            continue
        text = _block_text(block)
        if not text:
            continue
        try:
            page = int(block.get("page_idx", 0)) + 1
        except (TypeError, ValueError):
            page = 1
        page = min(max(page, 1), max(page_count, 1))
        level = block.get("text_level")
        if original_type == "text" and isinstance(level, int) and level > 0:
            headings[level] = text
            headings = {key: value for key, value in headings.items() if key <= level}
        section_path = " > ".join(headings[key] for key in sorted(headings))
        records.append({
            "page": page,
            "block_type": "table" if original_type == "table" else "text",
            "section_path": section_path,
            "text": text,
        })

    chunks: list[dict] = []
    title = _front_matter_title(records)
    if title:
        chunks.append({
            "content": f"Title: {title}",
            "metadata": {
                "mineru_type": "front_matter",
                "parser_backend": "mineru",
                "type": "front_matter",
                "subtype": "title",
                "page": 1,
                "source": pdf_path,
                "doc_id": make_chunk_id(user_id, doc_name, "page_1::front_matter_title"),
            },
        })

    chunk_idx = 0
    table_numbers: dict[int, int] = {}
    for record in _window_records(records, max_chars=max(1, long_chunk_threshold)):
        page = record["page"]
        block_type = record["block_type"]
        section_path = record["section_path"]
        text = record["text"]
        metadata = {"mineru_type": block_type, "parser_backend": "mineru"}
        hierarchy = hierarchy_metadata_fn(
            metadata,
            user_id=user_id,
            doc_name=doc_name,
            page=page,
            chunk_idx=chunk_idx,
            parent_content=text,
        )
        if section_path:
            hierarchy["section_path"] = section_path
            hierarchy["section_title"] = section_path.split(" > ")[-1]

        if block_type == "text" and len(text) > long_chunk_threshold:
            split_contents = [
                item.page_content
                for item in recursive_splitter.split_documents(
                    [Document(page_content=text, metadata=metadata)]
                )
            ]
        else:
            split_contents = [text]

        for sub_idx, split_content in enumerate(split_contents):
            if block_type == "table":
                table_numbers[page] = table_numbers.get(page, 0) + 1
                suffix = f"page_{page}::table_{table_numbers[page]}"
            elif len(split_contents) > 1:
                suffix = f"page_{page}::chunk_{chunk_idx}_{sub_idx}"
            else:
                suffix = f"page_{page}::chunk_{chunk_idx}"
            chunks.append({
                "content": chunk_content_with_section_fn(split_content, section_path),
                "metadata": {
                    **metadata,
                    **hierarchy,
                    "type": "table" if block_type == "table" else "text",
                    "page": page,
                    "source": pdf_path,
                    "doc_id": make_chunk_id(user_id, doc_name, suffix),
                },
            })
        chunk_idx += 1
    chunks = append_table_row_children(chunks)
    if not chunks:
        raise MinerUParseError("MinerU returned no readable text or table blocks")
    return chunks, page_count
