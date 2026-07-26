import camelot
import os
import re
from typing import Any

import requests


def format_table(table: Any) -> str:
    """Format a Camelot table as Markdown."""
    if table.df.empty or len(table.df) < 1:
        return ""

    # Camelot may embed newlines, carriage returns, or non-breaking spaces
    # inside individual dataframe cells.  Leaving them in place turns one
    # visual table row into several Markdown lines, disconnecting labels from
    # their numeric values downstream.  Normalise all whitespace within each
    # cell before serialising, while preserving the dataframe's columns.
    formatted_table = table.df.apply(
        lambda column: column.map(
            lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
        )
    )

    try:
        final_table = formatted_table.rename(columns=formatted_table.iloc[0]).drop(formatted_table.index[0]).reset_index(drop=True)
    except Exception:
        final_table = formatted_table

    return final_table.to_markdown(index=False)


def is_usable_table_markdown(markdown: str) -> bool:
    """Reject parser output that looks like a flattened page, not a table.

    Camelot can return a non-empty dataframe after merging most of a visual
    table into one or two very long cells.  Treating that artifact as a valid
    table prevents the native PyMuPDF fallback from running and creates
    misleading row/cell evidence.  The check intentionally validates only
    structure and length; it does not depend on document names, page numbers,
    or financial terms.
    """
    rows = []
    for line in (markdown or "").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 2 or all(not cell for cell in cells):
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)

    if len(rows) < 2:
        return False
    widths = {len(row) for row in rows}
    if len(widths) != 1 or next(iter(widths)) < 2:
        return False

    non_empty = [cell for row in rows for cell in row if cell]
    if not non_empty:
        return False
    # One giant flattened cell is not a reliable row/column structure. Keep
    # the threshold generous for narrative labels while failing malformed PDF
    # extraction that packs a whole table into a single field.
    if max(len(cell) for cell in non_empty) > 320:
        return False
    return True


def _read_tables(pdf_path: str, pages: str) -> list[Any]:
    """Best-effort Camelot extraction. Failures must not block PDF ingest."""
    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream", edge_tol=50, row_tol=10)
        if len(tables) > 0:
            return list(tables)
        print("Stream mode found 0 tables, falling back to lattice.")
    except Exception as exc:
        print(f"Stream mode failed: {exc}, falling back to lattice.")

    try:
        return list(camelot.read_pdf(pdf_path, pages=pages, flavor="lattice"))
    except Exception as exc:
        print(f"Lattice mode also failed: {exc}; continuing without table chunks.")
        return []


def _safe_table_bbox(table: Any) -> tuple | None:
    try:
        bbox = getattr(table, "bbox", None)
        if not bbox:
            return None
        return tuple(bbox)
    except Exception as exc:
        print(f"Skipping table bbox: {exc}")
        return None


def extract_tables_with_camelot(pdf_path: str, pages: str = "1-end") -> dict[int, list[dict]]:
    """
    Extract PDF tables with Camelot as a non-critical enhancement.

    Returns: {page_num: [{"md": table_markdown, "bbox": optional_bbox}, ...]}.
    A failure to detect tables should never fail document upload; text extraction
    in ingest.py remains the primary ingestion path.
    """
    tables_by_page: dict[int, list[dict]] = {}
    tables = _read_tables(pdf_path, pages)

    for table in tables:
        try:
            page_num = int(table.page)
            table_markdown = format_table(table=table)
        except Exception as exc:
            print(f"Skipping malformed table: {exc}")
            continue

        if not table_markdown or not is_usable_table_markdown(table_markdown):
            if table_markdown:
                print(f"Skipping structurally malformed table on page {page_num}.")
            continue

        tables_by_page.setdefault(page_num, []).append({
            "md": table_markdown,
            "bbox": _safe_table_bbox(table),
        })

    print(f"Extracted {sum(len(v) for v in tables_by_page.values())} usable tables in total")
    return tables_by_page


def enhance_table_with_context(table_md: dict, page_text: str, page_num: int) -> dict:
    """
    Use the optional NVIDIA API to clean and summarize an extracted table.
    Falls back to the original Markdown when the API is not configured or fails.
    """
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_model = os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")

    if not nvidia_api_key:
        print("NVIDIA_API_KEY not set, skipping table enhancement")
        return {"summary": "", "content": table_md["md"]}

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    prompt = f"""You are a data preprocessing assistant for a retrieval system operating on financial documents.

Rules:
1. Write a concise 2-3 sentence description of what the table represents
2. Clean the table: remove stray text, preserve headers and alignment
3. Do NOT change numeric values, dates, currencies, or units
4. Do NOT infer missing values or recompute totals

Page number: {page_num}

Table (markdown):
{table_md["md"]}

Surrounding text from the same page:
{page_text}

Return the result in this format:
TABLE SUMMARY:
<summary text>

CLEANED TABLE:
<markdown table>"""

    payload = {
        "model": nvidia_model,
        "temperature": 0.2,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1024,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {nvidia_api_key}",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response_json = response.json()

        enhanced = response_json["choices"][0]["message"]["content"].strip()

        summary = ""
        cleaned_table = table_md["md"]
        summary_match = re.search(r"TABLE SUMMARY:\s*(.*?)(?=\nCLEANED TABLE:)", enhanced, re.DOTALL)
        table_match = re.search(r"CLEANED TABLE:\s*(.*)", enhanced, re.DOTALL)

        if summary_match:
            summary = summary_match.group(1).strip()
        if table_match:
            parsed_table = table_match.group(1).strip()
            if parsed_table:
                cleaned_table = parsed_table

        return {"summary": summary, "content": cleaned_table}

    except Exception as exc:
        print(f"NVIDIA API table enhancement failed: {exc}")
        return {"summary": "", "content": table_md["md"]}
