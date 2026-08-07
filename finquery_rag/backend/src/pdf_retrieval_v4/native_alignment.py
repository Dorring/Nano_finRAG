from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.evaluation.run_pdf_v4_gate_01_r1 import extract_bbox, iter_dicts
from src.pdf_retrieval_v4.table_html_parser import compact_text, parse_table_html, tokenize_text


def page_words(pdf_path: Path, page_number: int) -> list[dict[str, Any]]:
    import fitz

    document = fitz.open(pdf_path)
    page = document[page_number - 1]
    return [
        {
            "index": i,
            "text": item[4],
            "bbox": [float(x) for x in item[:4]],
            "line": item[6],
            "block": item[5],
        }
        for i, item in enumerate(page.get_text("words"))
    ]


def inside(word: dict[str, Any], bbox: list[float], margin: float = 0.0) -> bool:
    x0, y0, x1, y1 = bbox
    wx0, wy0, wx1, wy1 = word["bbox"]
    return wx1 >= x0 - margin and wx0 <= x1 + margin and wy1 >= y0 - margin and wy0 <= y1 + margin


def union_bbox(words: list[dict[str, Any]]) -> list[float] | None:
    if not words:
        return None
    return [
        min(w["bbox"][0] for w in words),
        min(w["bbox"][1] for w in words),
        max(w["bbox"][2] for w in words),
        max(w["bbox"][3] for w in words),
    ]


def column_bands(bbox: list[float], width: int) -> list[list[float]]:
    if width <= 0:
        return []
    x0, y0, x1, y1 = bbox
    step = (x1 - x0) / width
    return [[x0 + index * step, y0, x0 + (index + 1) * step, y1] for index in range(width)]


def match_native_words(
    cell_text: str,
    words: list[dict[str, Any]],
    row_index: int,
    row_count: int,
    table_bbox: list[float],
    used: set[int],
) -> list[dict[str, Any]]:
    target = [compact_text(token) for token in tokenize_text(cell_text) if compact_text(token)]
    if not target:
        return []
    y0, y1 = table_bbox[1], table_bbox[3]
    expected_y = y0 + (row_index + 0.5) * (y1 - y0) / max(row_count, 1)
    candidates = [
        w
        for w in words
        if w["index"] not in used
        and abs((w["bbox"][1] + w["bbox"][3]) / 2 - expected_y)
        <= max(18.0, (y1 - y0) / max(row_count, 1) * 1.5)
    ]
    selected: list[dict[str, Any]] = []
    last_x = -1.0

    def token_matches(target_token: str, word_text: str) -> bool:
        word_token = compact_text(word_text)
        if not target_token or not word_token:
            return False
        if target_token == word_token:
            return True
        return (len(target_token) >= 4 and target_token in word_token) or (
            len(word_token) >= 4 and word_token in target_token
        )

    for token in target:
        matching = [
            w
            for w in candidates
            if w["index"] not in {x["index"] for x in selected}
            and w["bbox"][0] >= last_x
            and token_matches(token, w["text"])
        ]
        if not matching:
            matching = [
                w
                for w in words
                if w["index"] not in used
                and w["index"] not in {x["index"] for x in selected}
                and token_matches(token, w["text"])
            ]
        if not matching:
            continue
        chosen = min(matching, key=lambda w: (abs((w["bbox"][1] + w["bbox"][3]) / 2 - expected_y), w["bbox"][0]))
        selected.append(chosen)
        last_x = chosen["bbox"][2]

    # Numeric continuation rejoining (geometric only, never infers digits)
    for anchor in list(selected):
        if not re.fullmatch(r"[$€£¥₹₽₩]?[-+−()\d,\.\s]+", anchor["text"].strip()):
            continue
        current = anchor
        while True:
            ax0, ay0, ax1, ay1 = current["bbox"]
            continuations = [
                word
                for word in words
                if word["index"] not in used
                and word["index"] not in {item["index"] for item in selected}
                and re.fullmatch(r"[-+−()\d,\.]+", word["text"].strip())
                and 0.0 <= word["bbox"][1] - ay1 <= 3.0
                and (abs(word["bbox"][0] - ax0) <= 3.0 or abs(word["bbox"][2] - ax1) <= 3.0)
                and word["bbox"][2] - word["bbox"][0] <= max(8.0, (ax1 - ax0) * 0.5)
            ]
            if not continuations:
                break
            continuation = min(continuations, key=lambda word: (word["bbox"][1], word["bbox"][0]))
            continuation["_numeric_continuation"] = True
            selected.append(continuation)
            current = continuation
    for word in selected:
        used.add(word["index"])
    return selected


def extract_middle_tables(middle_path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(middle_path.read_text(encoding="utf-8"))
    pages: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    for page_index, page in enumerate(payload.get("pdf_info", [])):
        seen: set[str] = set()
        for block in iter_dicts(page):
            raw = block.get("html")
            if not isinstance(raw, str) or "<table" not in raw.lower():
                continue
            bbox = extract_bbox(block)
            key = hashlib.sha256(
                json.dumps([raw, bbox], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            pages[page_index].append({"html": raw, "bbox": bbox, "parsed": parse_table_html(raw)})
    return dict(pages)


def resolve_pdf(source_path: str, pdf_root: Path, shared_root: Path) -> Path | None:
    direct = Path(source_path)
    if direct.is_file():
        return direct
    for root in (
        pdf_root,
        shared_root,
        shared_root / "finquery_rag/backend/runtime/benchmark/financial_rag_v1/review-package/pdfs",
    ):
        candidate = root / direct.name
        if candidate.is_file():
            return candidate
        if root.is_dir():
            matches = sorted(root.rglob(direct.name))
            if matches:
                return matches[0]
    return None
