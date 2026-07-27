import pymupdf
import re
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from .process_tables import enhance_table_with_context, extract_tables_with_camelot
from .chunk_id import make_chunk_id
from .mineru_parser import (
    MinerUParseError,
    append_table_row_children,
    process_pdf_with_mineru,
)

# 1. 用于长章节内部二次切分的备选方案
# 适配 2048 上下文：chunk_size 从 1000 降至 350，overlap 从 200 降至 50
RECURSIVE_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=350,
    chunk_overlap=50,
    length_function=len,
    separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""]
)

# 2. 核心：Markdown 逻辑结构切分器配置
MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ],
    strip_headers=False
)

# 长章节二次切分阈值：从 1500 降至 500，适配短上下文
LONG_CHUNK_THRESHOLD = 500
NATIVE_PARSER_VERSION = "native-layout-v2"
MINERU_PARSER_VERSION = "mineru-content-list-v2"
SPLITTER_VERSION = "page-boundary-section-v2"


def _resolve_parser_backend(pdf_path: str) -> str:
    """Resolve an explicit parser choice, with a conservative auto policy.

    ``auto`` remains native for born-digital PDFs. It can route low-text
    documents (usually scans) to a separately deployed MinerU service only
    when the operator explicitly enables that behaviour.
    """
    requested = os.getenv("PARSER_BACKEND", "native").strip().lower()
    if requested not in {"native", "mineru", "auto"}:
        raise ValueError("PARSER_BACKEND must be one of: native, mineru, auto")
    if requested != "auto":
        return requested
    auto_enabled = os.getenv("MINERU_AUTO_ENABLED", "false").strip().lower() in {"1", "true", "yes"}
    if not auto_enabled or not os.getenv("MINERU_API_URL", "").strip():
        return "native"
    sample_pages = max(1, int(os.getenv("MINERU_AUTO_SAMPLE_PAGES", "3")))
    min_chars = max(1, int(os.getenv("MINERU_AUTO_MIN_TEXT_CHARS", "80")))
    try:
        with pymupdf.open(pdf_path) as candidate:
            observed = sum(
                len(candidate[index].get_text("text").strip())
                for index in range(min(len(candidate), sample_pages))
            )
        return "mineru" if observed < min_chars else "native"
    except Exception as exc:
        print(f"Parser auto-detection failed: {exc}; using native parser.")
        return "native"


def get_ingest_lineage(pdf_path: str) -> tuple[str, str]:
    """Return parser and splitter versions used for this concrete upload."""
    backend = _resolve_parser_backend(pdf_path)
    parser_version = MINERU_PARSER_VERSION if backend == "mineru" else NATIVE_PARSER_VERSION
    return parser_version, SPLITTER_VERSION


def _analyze_font_hierarchy(page: pymupdf.Page) -> dict:
    """
    动态分析当前页的字体层级。
    返回一个 {字体大小: markdown层级} 的映射字典。
    """
    spans_info = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["text"].strip():
                    spans_info.append({"size": span["size"], "text": span["text"]})

    if not spans_info:
        return {}

    unique_sizes = sorted(list(set([s["size"] for s in spans_info])), reverse=True)
    sizes_list = [s["size"] for s in spans_info]
    most_common_size = max(set(sizes_list), key=sizes_list.count)

    hierarchy_map = {}
    level = 1
    for size in unique_sizes:
        if size > most_common_size and level <= 3:
            hierarchy_map[size] = level
            level += 1

    return hierarchy_map


def _reconstruct_page_to_markdown(page: pymupdf.Page, tab_bboxes: list, hierarchy_map: dict) -> str:
    """
    将 PyMuPDF 提取的纯文本，逆向重构为 Markdown 格式。
    剔除表格区域，并根据字体层级添加 # 符号。
    """
    md_lines = []
    blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)["blocks"]

    for block in blocks:
        if block["type"] != 0:
            continue

        block_rect = pymupdf.Rect(block["bbox"])
        is_in_table = any(block_rect.intersects(pymupdf.Rect(tab)) for tab in tab_bboxes)
        if is_in_table:
            continue

        block_text_parts = []
        block_main_size = 0
        for line in block["lines"]:
            for span in line["spans"]:
                block_text_parts.append(span["text"])
                if span["size"] > block_main_size:
                    block_main_size = span["size"]

        full_text = "".join(block_text_parts).strip()
        if not full_text:
            continue

        if block_main_size in hierarchy_map:
            level = hierarchy_map[block_main_size]
            md_lines.append(f"{'#' * level} {full_text}")
        else:
            md_lines.append(full_text)

    return "\n\n".join(md_lines)




def _safe_find_pymupdf_tables(page: pymupdf.Page) -> list:
    """Return PyMuPDF table objects without allowing detection to block ingest."""
    try:
        finder = page.find_tables()
        return list(getattr(finder, "tables", []) or [])
    except Exception as exc:
        print(f"PyMuPDF table detection failed on page {page.number + 1}: {exc}; continuing without table bboxes.")
        return []


def _safe_find_table_bboxes(page: pymupdf.Page, tables: list | None = None) -> list:
    """Return table bboxes from already detected tables when available."""
    tables = _safe_find_pymupdf_tables(page) if tables is None else tables
    bboxes = []
    for table in tables:
        try:
            bbox = getattr(table, "bbox", None)
            if bbox:
                bboxes.append(tuple(bbox))
        except Exception as exc:
            print(f"Skipping PyMuPDF table bbox on page {page.number + 1}: {exc}")
    return bboxes


def _markdown_escape_table_cell(value) -> str:
    text = re.sub(r"\\s+", " ", str(value or "")).strip()
    return text.replace("|", "\\|")


def _native_table_to_markdown(table) -> str:
    """Convert one PyMuPDF table to small, dependency-free Markdown."""
    try:
        raw_rows = table.extract()
    except Exception as exc:
        print(f"Skipping PyMuPDF table extraction: {exc}")
        return ""
    rows = [
        [_markdown_escape_table_cell(cell) for cell in row]
        for row in (raw_rows or [])
        if any(_markdown_escape_table_cell(cell) for cell in row)
    ]
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized_rows = [row + [""] * (width - len(row)) for row in rows]
    header = normalized_rows[0]
    body = normalized_rows[1:] or [[""] * width]
    separator = ["---"] * width
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [header, separator, *body]
    )


def _extract_pymupdf_table_entries(page: pymupdf.Page, tables: list) -> list[dict]:
    """Create usable table entries for pages where Camelot has no result."""
    entries = []
    for table in tables:
        markdown = _native_table_to_markdown(table)
        if not markdown:
            continue
        try:
            bbox = tuple(getattr(table, "bbox", None) or ()) or None
        except Exception:
            bbox = None
        entries.append({"md": markdown, "bbox": bbox})
    return entries


def _extract_layout_table_row_entries(page: pymupdf.Page, tables: list) -> list[dict]:
    """Rebuild numeric table rows from native PDF word coordinates.

    A visual table may survive detection while Camelot or ``Table.extract``
    still separates its labels and values onto different Markdown lines.  PDF
    word coordinates preserve the original horizontal reading order, so this
    creates a narrow sparse-evidence row for words sharing a baseline inside
    a detected table box.  It deliberately adds no inferred headers or values.
    """
    boxes = []
    for table_index, table in enumerate(tables or [], start=1):
        try:
            bbox = tuple(getattr(table, "bbox", None) or ())
        except Exception:
            bbox = ()
        if len(bbox) != 4:
            continue
        boxes.append((table_index, bbox))
    if not boxes:
        return []

    try:
        raw_words = page.get_text("words") or []
    except Exception as exc:
        print(f"Skipping layout table rows on page {page.number + 1}: {exc}")
        return []

    entries = []
    for table_index, (left, top, right, bottom) in boxes:
        words = []
        for word in raw_words:
            if len(word) < 5:
                continue
            x0, y0, x1, y1, text = word[:5]
            text = str(text or "").strip()
            if not text:
                continue
            center_y = (float(y0) + float(y1)) / 2
            center_x = (float(x0) + float(x1)) / 2
            if (
                float(left) - 2 <= center_x <= float(right) + 2
                and float(top) - 2 <= center_y <= float(bottom) + 2
            ):
                words.append((float(x0), float(y0), float(x1), text))

        rows: list[dict] = []
        for x0, y0, x1, text in sorted(words, key=lambda item: (item[1], item[0])):
            if not rows or abs(y0 - rows[-1]["y"]) > 2.5:
                rows.append({"y": y0, "words": [(x0, x1, text)]})
            else:
                rows[-1]["words"].append((x0, x1, text))

        for row_index, row in enumerate(rows):
            ordered = sorted(row["words"], key=lambda item: item[0])
            if len(ordered) < 2:
                continue
            parts = []
            previous_x1 = None
            for x0, x1, text in ordered:
                # In common financial PDFs, adjacent numeric columns are
                # separated by roughly 16+ points, while normal words within
                # a label have much smaller gaps.  Keep the threshold low
                # enough to retain distinct compact amount columns.
                if previous_x1 is not None and x0 - previous_x1 > 16:
                    parts.append("|")
                parts.append(text)
                previous_x1 = x1
            content = " ".join(parts).strip()
            if not re.search(r"\d", content):
                continue
            entries.append({
                "content": content[:900],
                "table_index": table_index,
                "row_index": row_index,
            })
            if len(entries) >= 300:
                return entries
    return entries

def _clean_front_matter_line(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    text = re.sub(r"^\d{1,3}\s+", "", text)
    text = re.sub(r"\s+\d{1,3}$", "", text)
    return text.strip()


def _normalize_cover_title(text: str) -> str:
    """Normalize common PDF cover typography without changing its meaning."""
    cleaned = re.sub(r"\s+", " ", text or "").strip(" -")
    cleaned = re.sub(r"\bannualreport\b", "Annual Report", cleaned, flags=re.IGNORECASE)
    return cleaned


def _is_usable_cover_title(text: str) -> bool:
    """Reject partial cover labels such as ANNUAL or a page number."""
    cleaned = _normalize_cover_title(text)
    words = re.findall(r"[A-Za-z][A-Za-z0-9&'/-]*", cleaned)
    if len(cleaned) < 12 or len(words) < 2:
        return False
    return cleaned.lower() not in {
        "annual report",
        "annual",
        "report",
        "financial statements",
    }


def _fallback_cover_title(lines: list[dict], page_height: float) -> str | None:
    """Build a title from adjacent cover lines when font ranking is incomplete.

    Marketing-style covers often use a large year plus a smaller tagline and
    report label. Treating the three lines as one local run is more robust
    than accepting a single oversized word as the title.
    """
    title_lines = []
    pending_year = None
    for item in sorted(lines, key=lambda entry: entry["y0"]):
        text = _normalize_cover_title(item["text"])
        lowered = text.lower()
        if item["y0"] > page_height * 0.45:
            break
        if lowered.startswith(("abstract", "keywords", "paper id", "anonymous")):
            break
        if "www." in lowered or "http://" in lowered or "https://" in lowered:
            break
        if re.fullmatch(r"(?:19|20)\d{2}", text):
            pending_year = text
            continue
        if text.isdigit():
            continue
        if pending_year and not title_lines:
            title_lines.append(pending_year)
        pending_year = None
        title_lines.append(text)
        if len(title_lines) >= 4:
            break

    candidate = _normalize_cover_title(" ".join(title_lines))
    return candidate if _is_usable_cover_title(candidate) else None


def _extract_title_from_first_page(page: pymupdf.Page) -> str | None:
    """Extract a likely title from page 1 using font-size/layout signals."""
    try:
        blocks = page.get_text("dict").get("blocks", [])
    except Exception:
        return None

    lines = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
            if not spans:
                continue
            raw_text = " ".join(span.get("text", "").strip() for span in spans)
            clean_text = _clean_front_matter_line(raw_text)
            if not clean_text:
                continue
            max_size = max(float(span.get("size", 0) or 0) for span in spans)
            y0 = min(float(span.get("bbox", [0, 0, 0, 0])[1]) for span in spans)
            lines.append({"text": clean_text, "size": max_size, "y0": y0})

    if not lines:
        return None

    max_size = max(line["size"] for line in lines)
    min_title_size = max(12.0, max_size * 0.75)
    continuation_size = max(10.0, max_size * 0.45)
    page_height = float(getattr(page.rect, "height", 1000) or 1000)
    title_lines = []

    for line in sorted(lines, key=lambda item: item["y0"]):
        lower = line["text"].lower()
        if lower.startswith(("abstract", "keywords", "paper id", "anonymous")):
            if title_lines:
                break
            continue
        if line["y0"] > page_height * 0.45 and title_lines:
            break
        if line["size"] >= min_title_size or line["size"] >= continuation_size:
            title_lines.append(line["text"])
        elif title_lines:
            break

    if not title_lines:
        return None
    title = _normalize_cover_title(" ".join(title_lines[:5]))
    if _is_usable_cover_title(title):
        return title
    return _fallback_cover_title(lines, page_height)


def _section_path_from_metadata(metadata: dict) -> str:
    """Return a stable section path from MarkdownHeaderTextSplitter metadata."""
    if not isinstance(metadata, dict):
        return ""
    headers = []
    for key in ("Header 1", "Header 2", "Header 3"):
        value = metadata.get(key)
        if isinstance(value, str):
            cleaned = re.sub(r"\s+", " ", value).strip(" #")
            if cleaned and "PAGE_MARKER_" not in cleaned:
                headers.append(cleaned)
    return " > ".join(headers)


def _compact_parent_excerpt(content: str, *, max_chars: int = 1400) -> str:
    """Keep a bounded parent section excerpt for context expansion."""
    text = re.sub(r"\s+", " ", content or "").strip()
    if len(text) <= max_chars:
        return text
    head_budget = max_chars // 2
    tail_budget = max_chars - head_budget - 16
    return f"{text[:head_budget].rstrip()} [...] {text[-tail_budget:].lstrip()}"


def _hierarchy_metadata(
    metadata: dict,
    *,
    user_id: int | None,
    doc_name: str,
    page: int,
    chunk_idx: int,
    parent_content: str,
) -> dict:
    """Attach section-aware parent-child fields to a child chunk."""
    section_path = _section_path_from_metadata(metadata)
    parent_id = make_chunk_id(user_id, doc_name, f"page_{page}::parent_{chunk_idx}")
    enriched = {
        "parent_id": parent_id,
        "parent_page": page,
        "parent_excerpt": _compact_parent_excerpt(parent_content),
        "parent_child": True,
    }
    if section_path:
        enriched["section_path"] = section_path
        enriched["section_title"] = section_path.split(" > ")[-1]
    return enriched


def _chunk_content_with_section(content: str, section_path: str) -> str:
    """Prepend short section context to child text for lexical/vector retrieval."""
    body = (content or "").strip()
    if not section_path:
        return body
    prefix = f"Section: {section_path}"
    if body.startswith(prefix):
        return body
    return f"{prefix}\n{body}"


def _split_page_markdown(page_markdown: str, source: str) -> list[Document]:
    """Split one page while keeping the caller-owned page identity intact.

    Splitting a document-wide Markdown stream allows heading metadata from one
    page to overwrite the synthetic page marker of later chunks.  Page
    attribution is a retrieval and citation contract, so page boundaries are
    deliberately preserved here; section headers still work within each page.
    """
    try:
        return MARKDOWN_SPLITTER.split_text(page_markdown)
    except Exception as exc:
        print(f"Markdown split failed, fallback to recursive. Error: {exc}")
        fallback = Document(page_content=page_markdown, metadata={"source": source})
        return RECURSIVE_SPLITTER.split_documents([fallback])


def _iter_page_markdown_splits(
    page_markdowns: list[tuple[int, str]], source: str
):
    """Yield Markdown splits with their immutable PDF page number.

    ``MarkdownHeaderTextSplitter`` only preserves headings in its metadata; it
    has no intrinsic knowledge of PDF pages.  Keeping the page in this small
    iterator makes the ingestion invariant explicit and independently testable.
    """
    for page_number, page_markdown in page_markdowns:
        for split_doc in _split_page_markdown(page_markdown, source):
            yield page_number, split_doc

def process_pdf(pdf_path: str, user_id: int = None) -> tuple[list[dict], int]:
    """
    经济型结构化切分管线：基于规则重构Markdown，实现树状逻辑切分。
    适配 2048 上下文：更小的 chunk_size，更紧凑的切分策略。

    :param pdf_path: 待处理的PDF文件路径
    :return: (切分后的文本块列表, PDF总页数)
    """
    backend = _resolve_parser_backend(pdf_path)
    if backend == "mineru":
        print("[Parser] Using optional MinerU backend")
        try:
            return process_pdf_with_mineru(
                pdf_path,
                user_id=user_id,
                recursive_splitter=RECURSIVE_SPLITTER,
                long_chunk_threshold=LONG_CHUNK_THRESHOLD,
                hierarchy_metadata_fn=_hierarchy_metadata,
                chunk_content_with_section_fn=_chunk_content_with_section,
            )
        except MinerUParseError:
            # An explicit MinerU choice must fail visibly; silently mixing two
            # parsers would make document lineage and evaluation unreliable.
            raise

    chunks = []
    doc_name = os.path.basename(pdf_path)

    doc = pymupdf.open(pdf_path)
    pages = len(doc)

    print(f"\n{'=' * 60}")
    print(f"[Economic Structured Mode] Processing: {doc_name}")
    print(f"{'=' * 60}")

    tables_by_page = extract_tables_with_camelot(pdf_path)

    title = _extract_title_from_first_page(doc[0]) if pages else None
    if title:
        title_doc_id = make_chunk_id(user_id, doc_name, "page_1::front_matter_title")
        chunks.append({
            "content": f"Title: {title}",
            "metadata": {
                "type": "front_matter",
                "subtype": "title",
                "page": 1,
                "source": pdf_path,
                "doc_id": title_doc_id,
            },
        })

    page_markdowns: list[tuple[int, str]] = []
    layout_table_rows: list[tuple[int, dict]] = []

    for page_num in range(pages):
        page = doc[page_num]
        actual_page_num = page_num + 1

        native_tables = _safe_find_pymupdf_tables(page)
        # Camelot is preferred when it succeeds.  Native PyMuPDF extraction is
        # a local fallback, which keeps numeric evidence available offline.
        if not tables_by_page.get(actual_page_num):
            native_entries = _extract_pymupdf_table_entries(page, native_tables)
            if native_entries:
                tables_by_page[actual_page_num] = native_entries

        layout_table_rows.extend(
            (actual_page_num, entry)
            for entry in _extract_layout_table_row_entries(page, native_tables)
        )

        # Do not remove a detected table from regular text unless the page has
        # a structured table chunk to retain that evidence. Duplication is
        # safer than silently losing a financial value on parser failure.
        tab_bboxes = (
            _safe_find_table_bboxes(page, native_tables)
            if tables_by_page.get(actual_page_num)
            else []
        )
        hierarchy_map = _analyze_font_hierarchy(page)
        page_md = _reconstruct_page_to_markdown(page, tab_bboxes, hierarchy_map)

        if not page_md.strip():
            continue

        page_markdowns.append((actual_page_num, page_md))

    # Split pages independently so every child chunk inherits the known PDF
    # page rather than relying on fragile synthetic Markdown headers.
    chunk_idx = 0
    for actual_page, split_doc in _iter_page_markdown_splits(page_markdowns, pdf_path):
        content = split_doc.page_content
        metadata = split_doc.metadata.copy()

        if not content:
            continue

        hierarchy_meta = _hierarchy_metadata(
            metadata,
            user_id=user_id,
            doc_name=doc_name,
            page=actual_page,
            chunk_idx=chunk_idx,
            parent_content=content,
        )
        section_path = hierarchy_meta.get("section_path", "")

        # Long sections are split again for the short generation context.
        if len(content) > LONG_CHUNK_THRESHOLD:
            sub_docs = RECURSIVE_SPLITTER.split_documents(
                [Document(page_content=content, metadata=metadata)]
            )
            for sub_idx, sub_doc in enumerate(sub_docs):
                doc_id = make_chunk_id(
                    user_id, doc_name, f"page_{actual_page}::chunk_{chunk_idx}_{sub_idx}"
                )
                chunks.append({
                    "content": _chunk_content_with_section(sub_doc.page_content, section_path),
                    "metadata": {
                        **metadata,
                        **hierarchy_meta,
                        "type": "text",
                        "page": actual_page,
                        "source": pdf_path,
                        "doc_id": doc_id,
                    },
                })
        else:
            doc_id = make_chunk_id(user_id, doc_name, f"page_{actual_page}::chunk_{chunk_idx}")
            chunks.append({
                "content": _chunk_content_with_section(content, section_path),
                "metadata": {
                    **metadata,
                    **hierarchy_meta,
                    "type": "text",
                    "page": actual_page,
                    "source": pdf_path,
                    "doc_id": doc_id,
                },
            })
        chunk_idx += 1

    # 处理表格块（使用 NVIDIA API 增强表格上下文）
    for actual_page_num, table_list in tables_by_page.items():
        page = doc[actual_page_num - 1]
        page_text = page.get_text("text")

        for table_idx, table_md_dict in enumerate(table_list):
            # 使用 NVIDIA API 增强表格（不再需要 llm_client 参数）
            enhanced_table = enhance_table_with_context(table_md_dict, page_text, actual_page_num)
            doc_id = make_chunk_id(user_id, doc_name, f"page_{actual_page_num}::table_{table_idx + 1}")

            # 将增强结果拼接为字符串，与文本块的 content 类型保持一致
            table_content = enhanced_table["content"]
            if enhanced_table["summary"]:
                table_content = f"Summary: {enhanced_table['summary']}\n\n{table_content}"
            table_parent_id = make_chunk_id(user_id, doc_name, f"page_{actual_page_num}::table_parent_{table_idx + 1}")

            chunks.append({
                "content": table_content,
                "metadata": {
                    "type": "table",
                    "page": actual_page_num,
                    "source": pdf_path,
                    "doc_id": doc_id,
                    "table_num": table_idx + 1,
                    "parent_id": table_parent_id,
                    "parent_page": actual_page_num,
                    "parent_excerpt": _compact_parent_excerpt(table_content),
                    "parent_child": True,
                }
            })

    # Keep coordinate-reconstructed rows as primary sparse evidence. They are
    # complementary to parser-produced Markdown rows and retain only observed
    # words, so they cannot manufacture a financial value.
    for actual_page_num, entry in layout_table_rows:
        table_index = entry["table_index"]
        row_index = entry["row_index"]
        parent_id = make_chunk_id(
            user_id, doc_name, f"page_{actual_page_num}::layout_table_parent_{table_index}"
        )
        doc_id = make_chunk_id(
            user_id,
            doc_name,
            f"page_{actual_page_num}::layout_table_{table_index}::row_{row_index}",
        )
        chunks.append({
            "content": entry["content"],
            "metadata": {
                "type": "table_row",
                "subtype": "layout_coordinate_row",
                "page": actual_page_num,
                "source": pdf_path,
                "doc_id": doc_id,
                "parent_id": parent_id,
                "parent_table_id": parent_id,
                "table_num": table_index,
                "table_row_index": row_index,
                "table_layout_extracted": True,
            },
        })

    doc.close()

    chunks = append_table_row_children(chunks)
    table_count = sum(1 for c in chunks if c["metadata"]["type"] == "table")
    table_row_count = sum(1 for c in chunks if c["metadata"]["type"] == "table_row")
    text_count = len(chunks) - table_count - table_row_count

    print(
        f"Extracted {len(chunks)} structured chunks: "
        f"({text_count} text, {table_count} tables, {table_row_count} table rows)"
    )
    print(f"{'=' * 60}\n")

    return chunks, pages
