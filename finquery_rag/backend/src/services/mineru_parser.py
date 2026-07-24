"""Optional MinerU CLI adapter with a stable FinQuery chunk contract.

MinerU is intentionally an opt-in dependency.  This adapter consumes its
flat ``*_content_list.json`` artifact rather than its internal model output,
which keeps the RAG ingestion schema stable across parser implementations.
"""

from __future__ import annotations

import json
import os
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


def _as_text(value) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return " ".join(_as_text(item) for item in value if _as_text(item)).strip()
    if isinstance(value, dict):
        return _as_text(value.get("content", ""))
    return ""


def _block_text(block: dict) -> str:
    block_type = str(block.get("type", ""))
    if block_type == "table":
        parts = [
            _as_text(block.get("table_caption")),
            _as_text(block.get("table_body")),
            _as_text(block.get("table_footnote")),
        ]
    else:
        parts = [_as_text(block.get("text")), _as_text(block.get("content"))]
    text = "\n".join(part for part in parts if part)
    # Preserve cell words and values while keeping HTML-only table output
    # useful to lexical and dense retrieval.
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


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
    args = command + ["-p", pdf_path, "-o", str(output_dir), "-b", backend, "-m", "auto"]
    api_url = os.getenv("MINERU_API_URL", "").strip()
    if api_url:
        args.extend(["--api-url", api_url])
    timeout = max(1, int(os.getenv("MINERU_TIMEOUT_SECONDS", "600")))
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise MinerUParseError("MinerU CLI is unavailable; install/configure MINERU_COMMAND") from exc
    except subprocess.TimeoutExpired as exc:
        raise MinerUParseError(f"MinerU timed out after {timeout}s") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown error").strip()[-600:]
        raise MinerUParseError(f"MinerU exited with {result.returncode}: {detail}")
    return _load_content_list(output_dir)


def process_pdf_with_mineru(
    pdf_path: str,
    *,
    user_id: int | None,
    recursive_splitter,
    long_chunk_threshold: int,
    hierarchy_metadata_fn,
    chunk_content_with_section_fn,
) -> tuple[list[dict], int]:
    """Run optional MinerU parsing and emit the same chunk schema as native ingest."""
    doc_name = os.path.basename(pdf_path)
    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)
    with tempfile.TemporaryDirectory(prefix="finquery_mineru_") as output_root:
        blocks = _run_mineru(pdf_path, Path(output_root))

    chunks: list[dict] = []
    headings: dict[int, str] = {}
    chunk_idx = 0
    ignored_types = {"header", "footer", "page_number", "aside_text", "page_footnote"}
    for block in blocks:
        block_type = str(block.get("type", "text"))
        if block_type in ignored_types:
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
        if block_type == "text" and isinstance(level, int) and level > 0:
            headings[level] = text
            headings = {key: value for key, value in headings.items() if key <= level}
        section_path = " > ".join(headings[key] for key in sorted(headings))
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
        # Avoid constructing a LangChain Document for an already-small
        # MinerU block.  Besides avoiding needless work, this keeps the
        # parser boundary independent of optional LangChain test doubles.
        if len(text) > long_chunk_threshold:
            split_contents = [
                item.page_content
                for item in recursive_splitter.split_documents(
                    [Document(page_content=text, metadata=metadata)]
                )
            ]
        else:
            split_contents = [text]
        for sub_idx, split_content in enumerate(split_contents):
            suffix = f"page_{page}::chunk_{chunk_idx}_{sub_idx}" if len(split_contents) > 1 else f"page_{page}::chunk_{chunk_idx}"
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
    if not chunks:
        raise MinerUParseError("MinerU returned no readable text or table blocks")
    return chunks, page_count
