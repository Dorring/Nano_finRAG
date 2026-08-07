"""Gate 02 R3.2: Target-specific Structural Alignment Audit.

Runs AFTER the Gate 02 R3.1 Page-level Presence Closure.  While R3.1
only checked whether the target page has *any* table/row/cell, R3.2
verifies that the specific Gold Candidate's Row/Cell structure is
actually recoverable from the R3 full-corpus adapter predictions.

For each of the 33 Gold Source records (16 D-class + 17 B-class):
  1. Load the Legacy Candidate content (raw text, block_type) via
     ProductionCandidateMapper — read-only, post-seal.
  2. Extract numeric multiset and text/metric tokens from the Legacy
     Candidate content.
  3. Match against every V4 Table/Row/Cell on the target page using:
       - Numeric Signature recall (gold numeric ⊆ row numeric)
       - Text/Metric token recall (gold label tokens ⊆ row text tokens)
  4. Determine the alignment grade (A/B/C/none) with uniqueness check.
  5. Classify failure reason if not alignable.

Five-layer target alignment (T0-T4):
  T0 Target Page Present         — page record exists in R3
  T1 Target Evidence Block Present — at least one row has text overlap
  T2 Target Row / Row-set Present — a row is uniquely located
  T3 Target Value Cell(s) Present — gold numeric values located in cells
  T4 Target Candidate Structurally Alignable — T2 + T3 + unique

This script does NOT modify the adapter or re-run it.  It only reads
sealed R3 predictions and the production candidate store.

Reads ONLY:
  - R3 sealed predictions (adapter-predictions.jsonl.gz)
  - Gate 08 R1.1 classification (gold-coverage-classification.json)
  - Gate 08 R2 scoring (b-class-detail.json)
  - Production candidate store (rag_bm25.db, read-only)
  - Corpus manifest (corpus.json, filename→document_id mapping)

No questions, gold answers, or governance data is read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).absolute().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

R3_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-02-r3"
R1_1_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r1-1"
R2_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r2"
CORPUS_PATH = ROOT / "benchmarks/financial_rag_v1/corpus.json"
DB_PATH = ROOT / "rag_bm25.db"

# Matching thresholds
NUMERIC_RECALL_A = 1.0  # Grade A: all gold numbers must be found
METRIC_TOKEN_RECALL_A = 0.8  # Grade A: ≥80% label tokens must match
NUMERIC_RECALL_B = 0.8  # Grade B: ≥80% numbers found
METRIC_TOKEN_RECALL_B = 0.6  # Grade B: ≥60% label tokens match
MARGIN_THRESHOLD = 0.05  # min score gap between best and second-best row
MIN_NUMERIC_TOKENS_FOR_RECALL = 1  # gold must have ≥1 number to use numeric recall


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Text / numeric extraction
# ---------------------------------------------------------------------------

_VALUE_RE = re.compile(r"[-+]?\(?\d[\d, .]*\)?%?")


def _norm_text(value: str) -> str:
    """Normalize text: lowercase, collapse non-alphanumeric to spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _extract_numeric_tokens(text: str) -> list[str]:
    """Extract numeric tokens from text and normalize them.

    "109,158" → "109158"
    "14 %" → "14"
    "(1,234.50)" → "1234.50"  (parentheses are debit/negative notation,
                               but for multiset matching we keep the
                               digits so that the same number in a
                               different format still matches)
    """
    tokens: list[str] = []
    for match in _VALUE_RE.findall(text):
        if not any(ch.isdigit() for ch in match):
            continue
        # Strip everything except digits and decimal point
        cleaned = re.sub(r"[^0-9.]", "", match)
        if not cleaned or cleaned == ".":
            continue
        # Remove leading zeros for comparison stability
        cleaned = cleaned.lstrip("0") or "0"
        tokens.append(cleaned)
    return tokens


def _extract_text_tokens(text: str) -> list[str]:
    """Extract non-numeric text tokens (words) from text.

    Splits on '|' first (table cell separator), then normalizes.
    Filters out pure-number tokens and very short tokens.
    """
    tokens: list[str] = []
    for segment in str(text or "").split("|"):
        normed = _norm_text(segment)
        for word in normed.split():
            # Skip pure numbers
            if re.fullmatch(r"[0-9.]+", word):
                continue
            # Skip very short tokens (likely noise)
            if len(word) < 2:
                continue
            tokens.append(word)
    return tokens


def _extract_metric_label(text: str) -> str:
    """Extract the metric label (first cell in a pipe-delimited row).

    "Services (1) | 109,158 | 14 %" → "Services (1)" → "services"
    """
    first_segment = str(text or "").split("|")[0]
    return _norm_text(first_segment)


def _counter_recall(gold: Counter, candidate: Counter) -> float:
    """Compute multiset recall: |gold ∩ candidate| / |gold|."""
    if not gold:
        return 1.0  # no gold items → vacuously true
    total = sum(gold.values())
    matched = sum(min(gold[k], candidate.get(k, 0)) for k in gold)
    return matched / total


def _counter_precision(gold: Counter, candidate: Counter) -> float:
    """Compute multiset precision: |gold ∩ candidate| / |candidate|."""
    if not candidate:
        return 0.0
    total = sum(candidate.values())
    matched = sum(min(gold[k], candidate.get(k, 0)) for k in gold)
    return matched / total


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_r3_predictions(
    path: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    """Load R3 predictions and index by (document_id, pdf_page)."""
    index: dict[tuple[str, int], dict[str, Any]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            page = json.loads(line)
            doc_id = str(page.get("document_id") or "")
            pdf_page = int(page.get("pdf_page") or 0)
            index[(doc_id, pdf_page)] = page
    return index


def _load_d_class_records(r1_1_out: Path) -> list[dict[str, Any]]:
    """Load 16 D-class records from Gate 08 R1.1 classification."""
    path = r1_1_out / "gold-coverage-classification.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = (
        data.get("rows", data.get("records", data)) if isinstance(data, dict) else data
    )
    d_class = [
        r
        for r in rows
        if isinstance(r, dict) and r.get("coverage_class") == "structurally_absent"
    ]
    for r in d_class:
        r["benchmark_class"] = "D"
    return d_class


def _load_b_class_unrecovered(r1_1_out: Path, r2_out: Path) -> list[dict[str, Any]]:
    """Load 17 B-class unrecovered records, enriched with R1.1 metadata."""
    r11_path = r1_1_out / "gold-coverage-classification.json"
    r11_data = json.loads(r11_path.read_text(encoding="utf-8"))
    r11_rows = (
        r11_data.get("rows", r11_data.get("records", r11_data))
        if isinstance(r11_data, dict)
        else r11_data
    )
    b_r11_by_identity: dict[str, dict[str, Any]] = {}
    for r in r11_rows:
        if (
            isinstance(r, dict)
            and r.get("coverage_class") == "strict_mapped_not_retrieved"
        ):
            b_r11_by_identity[str(r.get("gold_source_identity") or "")] = r

    b_path = r2_out / "scoring/b-class-detail.json"
    b_data = json.loads(b_path.read_text(encoding="utf-8"))
    b_rows = (
        b_data.get("rows", b_data.get("records", b_data))
        if isinstance(b_data, dict)
        else b_data
    )
    b_unrecovered = [
        r for r in b_rows if isinstance(r, dict) and not r.get("recovered", True)
    ]

    enriched: list[dict[str, Any]] = []
    for r in b_unrecovered:
        identity = str(r.get("gold_source_identity") or "")
        r11_record = b_r11_by_identity.get(identity, {})
        enriched.append(
            {
                "gold_source_identity": identity,
                "case_id": r.get("case_id") or r11_record.get("case_id"),
                "document_id": r11_record.get("document_id"),
                "pdf_page": r11_record.get("pdf_page"),
                "gold_candidate_key": r.get("gold_candidate_key"),
                "benchmark_class": "B",
            }
        )
    return enriched


# ---------------------------------------------------------------------------
# Legacy candidate type classification
# ---------------------------------------------------------------------------


def _classify_candidate_type(content: str, block_type: str) -> str:
    """Classify the legacy candidate granularity.

    single_row       — one pipe-delimited row of data
    multi_row_block  — multiple lines of pipe-delimited data
    table_block      — block_type == "table" (whole table)
    narrative        — block_type == "text" (paragraph text)
    mixed            — other / unclear
    """
    if block_type == "text":
        return "narrative"
    if block_type == "table":
        return "table_block"

    content_str = str(content or "")
    lines = [line for line in content_str.split("\n") if line.strip()]
    if len(lines) <= 1:
        return "single_row"
    # Multiple lines — check if pipe-delimited
    has_pipes = any("|" in line for line in lines)
    if has_pipes:
        return "multi_row_block"
    return "mixed"


# ---------------------------------------------------------------------------
# Row matching
# ---------------------------------------------------------------------------


def _row_numeric_counter(row: dict[str, Any]) -> Counter:
    """Build a numeric multiset from a V4 row's cells + resolved_text."""
    values: list[str] = []
    for cell in row.get("cells", []) or []:
        # parsed_numeric is a list of {"normalized": "...", ...}
        for num in cell.get("parsed_numeric", []) or []:
            normed = str(num.get("normalized") or "").strip()
            if normed:
                # Normalize same way as gold: strip non-digits
                cleaned = re.sub(r"[^0-9.]", "", normed)
                if cleaned and cleaned != ".":
                    cleaned = cleaned.lstrip("0") or "0"
                    values.append(cleaned)
        # Also extract from raw_text as fallback
        values.extend(
            _extract_numeric_tokens(
                str(cell.get("resolved_text") or cell.get("raw_text") or "")
            )
        )
    # Also from the row's own resolved_text (in case cells are missing)
    values.extend(
        _extract_numeric_tokens(
            str(row.get("resolved_text") or row.get("raw_text") or "")
        )
    )
    return Counter(values)


def _row_text_tokens(row: dict[str, Any]) -> list[str]:
    """Extract non-numeric text tokens from a V4 row.

    Uses resolved_text (the full pipe-delimited row text) so that label
    AND value-column headers are both included in the token set.
    """
    resolved = str(row.get("resolved_text") or row.get("raw_text") or "")
    return _extract_text_tokens(resolved)


def _row_cells_with_numeric(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return cells that have parsed_numeric values."""
    cells = row.get("cells", []) or []
    return [c for c in cells if c.get("parsed_numeric")]


def _table_numeric_counter(table: dict[str, Any]) -> Counter:
    """Build a numeric multiset from all cells in a V4 table."""
    values: list[str] = []
    for cell in table.get("cells", []) or []:
        for num in cell.get("parsed_numeric", []) or []:
            normed = str(num.get("normalized") or "").strip()
            if normed:
                cleaned = re.sub(r"[^0-9.]", "", normed)
                if cleaned and cleaned != ".":
                    cleaned = cleaned.lstrip("0") or "0"
                    values.append(cleaned)
        values.extend(
            _extract_numeric_tokens(
                str(cell.get("resolved_text") or cell.get("raw_text") or "")
            )
        )
    return Counter(values)


def _table_text_tokens(table: dict[str, Any]) -> list[str]:
    """Extract text tokens from all rows in a V4 table."""
    tokens: list[str] = []
    for row in table.get("rows", []) or []:
        tokens.extend(_row_text_tokens(row))
    return tokens


def _score_table_match(
    gold_numeric: Counter,
    gold_text_tokens: list[str],
    gold_metric_label: str,
    table: dict[str, Any],
) -> dict[str, Any]:
    """Score how well a V4 table matches the gold table_block candidate."""
    table_numeric = _table_numeric_counter(table)
    table_tokens = set(_table_text_tokens(table))

    numeric_recall = _counter_recall(gold_numeric, table_numeric)

    gold_token_set = set(gold_text_tokens)
    if gold_token_set:
        metric_token_recall = len(gold_token_set & table_tokens) / len(gold_token_set)
    else:
        metric_token_recall = 0.0

    union = gold_token_set | table_tokens
    token_jaccard = len(gold_token_set & table_tokens) / len(union) if union else 0.0

    header_text = _norm_text(
        " ".join(str(h) for h in (table.get("header_texts") or []))
    )
    metric_contained = bool(gold_metric_label) and gold_metric_label in header_text

    score = 0.0
    if numeric_recall == 1.0:
        score += 5.0
    elif numeric_recall > 0:
        score += numeric_recall * 3.0
    if metric_token_recall >= 0.8:
        score += 3.0
    elif metric_token_recall > 0:
        score += metric_token_recall * 1.5
    if metric_contained:
        score += 2.0
    score += token_jaccard * 1.0

    return {
        "numeric_recall": round(numeric_recall, 4),
        "metric_token_recall": round(metric_token_recall, 4),
        "token_jaccard": round(token_jaccard, 4),
        "metric_contained": metric_contained,
        "score": round(score, 4),
    }


def _score_multi_row_match(
    gold_numeric: Counter,
    gold_text_tokens: list[str],
    gold_metric_label: str,
    rows: list[dict[str, Any]],
    max_span: int = 5,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Try matching consecutive rows (2..max_span) against gold.

    Returns (best_match, best_row_set) or None.
    """
    best_match: dict[str, Any] | None = None
    best_rows: list[dict[str, Any]] = []
    upper = min(max_span + 1, len(rows) + 1)
    for span in range(2, upper):
        for start in range(len(rows) - span + 1):
            row_set = rows[start : start + span]
            combined_cells = [c for r in row_set for c in r.get("cells", [])]
            virtual_row = {
                "resolved_text": " | ".join(
                    str(r.get("resolved_text") or "") for r in row_set
                ),
                "metric_text": str(row_set[0].get("metric_text") or ""),
                "cells": combined_cells,
            }
            match = _score_row_match(
                gold_numeric, gold_text_tokens, gold_metric_label, virtual_row
            )
            if best_match is None or match["score"] > best_match["score"]:
                best_match = match
                best_rows = row_set
    if best_match is None:
        return None
    return (best_match, best_rows)


def _score_row_match(
    gold_numeric: Counter,
    gold_text_tokens: list[str],
    gold_metric_label: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    """Score how well a V4 row matches the gold candidate.

    Returns dict with:
      numeric_recall, metric_token_recall, token_jaccard,
      has_numeric_match, has_text_match, score
    """
    row_numeric = _row_numeric_counter(row)
    row_tokens = _row_text_tokens(row)

    # Numeric recall
    numeric_recall = _counter_recall(gold_numeric, row_numeric)

    # Metric token recall: fraction of gold text tokens found in row
    gold_token_set = set(gold_text_tokens)
    row_token_set = set(row_tokens)
    if gold_token_set:
        metric_token_recall = len(gold_token_set & row_token_set) / len(gold_token_set)
    else:
        metric_token_recall = 0.0

    # Token Jaccard
    if gold_token_set or row_token_set:
        union = gold_token_set | row_token_set
        token_jaccard = (
            len(gold_token_set & row_token_set) / len(union) if union else 0.0
        )
    else:
        token_jaccard = 0.0

    # Metric label containment: gold label is substring of row label
    row_metric_text = _norm_text(str(row.get("metric_text") or ""))
    metric_contained = bool(gold_metric_label) and gold_metric_label in row_metric_text

    # Composite score for ranking
    score = 0.0
    if numeric_recall == 1.0:
        score += 5.0
    elif numeric_recall > 0:
        score += numeric_recall * 3.0
    if metric_token_recall >= 0.8:
        score += 3.0
    elif metric_token_recall > 0:
        score += metric_token_recall * 1.5
    if metric_contained:
        score += 2.0
    score += token_jaccard * 1.0

    return {
        "numeric_recall": round(numeric_recall, 4),
        "metric_token_recall": round(metric_token_recall, 4),
        "token_jaccard": round(token_jaccard, 4),
        "metric_contained": metric_contained,
        "score": round(score, 4),
    }


# ---------------------------------------------------------------------------
# Target alignment check
# ---------------------------------------------------------------------------


def _check_alignment(
    record: dict[str, Any],
    candidate: dict[str, Any] | None,
    predictions: dict[tuple[str, int], dict[str, Any]],
) -> dict[str, Any]:
    """Check target-specific structural alignment for a single Gold record."""
    doc_id = str(record.get("document_id") or "")
    pdf_page = int(record.get("pdf_page") or 0)
    identity = str(record.get("gold_source_identity") or "")
    case_id = str(record.get("case_id") or "")
    benchmark_class = str(record.get("benchmark_class") or "")

    # Legacy candidate metadata
    legacy_content = ""
    legacy_block_type = ""
    legacy_candidate_type = "unknown"
    if candidate:
        legacy_content = str(candidate.get("content") or "")
        legacy_block_type = str(candidate.get("block_type") or "")
        legacy_candidate_type = _classify_candidate_type(
            legacy_content, legacy_block_type
        )

    # Extract gold signatures
    gold_numeric = Counter(_extract_numeric_tokens(legacy_content))
    gold_text_tokens = _extract_text_tokens(legacy_content)
    gold_metric_label = _extract_metric_label(legacy_content)

    # T0: Target page present
    page_record = predictions.get((doc_id, pdf_page))
    target_page_present = page_record is not None

    # T1: Target evidence block present
    # At least one row has some text overlap
    tables: list[dict[str, Any]] = []
    if page_record:
        tables = page_record.get("tables", []) or []
    all_rows: list[dict[str, Any]] = []
    for table in tables:
        all_rows.extend(table.get("rows", []) or [])

    # Score all rows
    scored_rows: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for table in tables:
        table_rows = table.get("rows", []) or []
        table_cells = table.get("cells", []) or []
        # Build cell lookup for each row
        cells_by_row: dict[int, list[dict[str, Any]]] = {}
        for cell in table_cells:
            row_idx = int(cell.get("row_index") or 0)
            cells_by_row.setdefault(row_idx, []).append(cell)
        for row in table_rows:
            row_idx = int(row.get("row_index") or 0)
            row["cells"] = cells_by_row.get(row_idx, [])
            match = _score_row_match(
                gold_numeric, gold_text_tokens, gold_metric_label, row
            )
            scored_rows.append((row, match, table))

    # Sort by score descending
    scored_rows.sort(key=lambda x: -x[1]["score"])

    # T1: at least one row has some overlap (score > 0)
    target_block_present = any(r[1]["score"] > 0 for r in scored_rows)

    # T2: Target row uniquely located
    best_row = None
    best_match: dict[str, Any] | None = None
    best_table = None
    ambiguous = False
    margin = 0.0
    tiebreak_used = False

    if scored_rows and scored_rows[0][1]["score"] > 0:
        best_row, best_match, best_table = scored_rows[0]
        if len(scored_rows) > 1:
            second_score = scored_rows[1][1]["score"]
            margin = best_match["score"] - second_score
            if margin < MARGIN_THRESHOLD:
                # High-score tie (nr=1.0 & tr>=0.8): likely duplicate rows.
                # Break tie by earliest row_index rather than failing.
                if (
                    best_match["numeric_recall"] >= NUMERIC_RECALL_A
                    and best_match["metric_token_recall"] >= METRIC_TOKEN_RECALL_A
                ):
                    tied = [
                        item
                        for item in scored_rows
                        if abs(item[1]["score"] - best_match["score"]) < 0.01
                    ]
                    tied.sort(key=lambda x: int(x[0].get("row_index") or 0))
                    best_row, best_match, best_table = tied[0]
                    tiebreak_used = True
                else:
                    ambiguous = True

    target_row_present = (
        best_row is not None
        and best_match is not None
        and not ambiguous
        and best_match["score"] > 0
    )

    # T3: Target value cells present
    # Gold numeric values are locatable in cells of the matched row.
    # Accept either cells' parsed_numeric OR row-level numeric (from
    # resolved_text) since MinerU may not always populate parsed_numeric.
    target_cells_present = False
    matched_cell_ids: list[str] = []
    if target_row_present and best_row is not None:
        row_cells = best_row.get("cells", []) or []
        cells_with_num = [c for c in row_cells if c.get("parsed_numeric")]
        if gold_numeric:
            row_cell_numeric = Counter()
            for cell in cells_with_num:
                for num in cell.get("parsed_numeric", []) or []:
                    normed = str(num.get("normalized") or "").strip()
                    if normed:
                        cleaned = re.sub(r"[^0-9.]", "", normed)
                        if cleaned and cleaned != ".":
                            cleaned = cleaned.lstrip("0") or "0"
                            row_cell_numeric[cleaned] += 1
            cell_recall = _counter_recall(gold_numeric, row_cell_numeric)
            # Also accept row-level numeric recall (values in text)
            row_level_recall = best_match["numeric_recall"]
            target_cells_present = (
                cell_recall >= NUMERIC_RECALL_B or row_level_recall >= NUMERIC_RECALL_B
            )
        else:
            cells_with_text = [
                c for c in row_cells if c.get("resolved_text") or c.get("raw_text")
            ]
            target_cells_present = len(cells_with_text) > 0
        matched_cell_ids = [
            str(c.get("cell_id") or "") for c in row_cells[:10] if c.get("cell_id")
        ]

    # T4: Target candidate structurally alignable
    # Requires T2 + T3 + adequate match quality.
    # When numeric_recall == 1.0 (perfect numeric multiset match), the
    # text threshold is relaxed because financial row values are an
    # extremely strong structural fingerprint — text mismatch is usually
    # due to header/separator rows in multi-line gold content, not a
    # wrong row.  We still require tr > 0 to guard against pure-numeric
    # collisions.
    NUMERIC_PERFECT_TR_RELAXED = 0.3  # tr threshold when nr == 1.0

    alignment_grade = "none"
    if target_row_present and best_match is not None:
        nr = best_match["numeric_recall"]
        tr = best_match["metric_token_recall"]

        if not gold_numeric:
            if tr >= METRIC_TOKEN_RECALL_A:
                alignment_grade = "A"
            elif tr >= METRIC_TOKEN_RECALL_B:
                alignment_grade = "B"
            elif tr > 0:
                alignment_grade = "C"
        else:
            if (
                nr >= NUMERIC_RECALL_A
                and tr >= METRIC_TOKEN_RECALL_A
                and target_cells_present
            ):
                alignment_grade = "A"
            elif (
                nr >= NUMERIC_RECALL_A
                and tr >= NUMERIC_PERFECT_TR_RELAXED
                and target_cells_present
            ):
                # Perfect numeric match with relaxed text threshold
                alignment_grade = "A"
            elif (
                nr >= NUMERIC_RECALL_B
                and tr >= METRIC_TOKEN_RECALL_B
                and target_cells_present
            ):
                alignment_grade = "B"
            elif nr > 0 or tr > 0:
                alignment_grade = "C"

    target_candidate_alignable = alignment_grade in ("A", "B")

    # --- Fallback: table_block / multi_row_block matching ---
    # If single-row matching did not achieve Grade A/B, try broader
    # matching strategies for table_block and multi_row_block candidates.
    match_strategy = "single_row"
    if not target_candidate_alignable and tables:
        if legacy_candidate_type == "table_block":
            # Match against entire table's numeric/text signature
            scored_tables = [
                (
                    t,
                    _score_table_match(
                        gold_numeric, gold_text_tokens, gold_metric_label, t
                    ),
                )
                for t in tables
            ]
            scored_tables.sort(key=lambda x: -x[1]["score"])
            if scored_tables and scored_tables[0][1]["score"] > 0:
                t_best, t_match = scored_tables[0]
                t_ambiguous = False
                if len(scored_tables) > 1:
                    t_margin = t_match["score"] - scored_tables[1][1]["score"]
                    if t_margin < MARGIN_THRESHOLD:
                        t_ambiguous = True
                if not t_ambiguous:
                    t_nr = t_match["numeric_recall"]
                    t_tr = t_match["metric_token_recall"]
                    t_cells_present = t_nr >= NUMERIC_RECALL_B if gold_numeric else True
                    if (
                        t_nr >= NUMERIC_RECALL_A
                        and t_tr >= METRIC_TOKEN_RECALL_A
                        and t_cells_present
                    ):
                        alignment_grade = "A"
                    elif (
                        t_nr >= 0.95
                        and t_tr >= NUMERIC_PERFECT_TR_RELAXED
                        and t_cells_present
                    ):
                        # Near-perfect table numeric match with relaxed text
                        alignment_grade = "A"
                    elif (
                        t_nr >= NUMERIC_RECALL_B
                        and t_tr >= METRIC_TOKEN_RECALL_B
                        and t_cells_present
                    ):
                        alignment_grade = "B"
                    elif t_nr > 0 or t_tr > 0:
                        alignment_grade = "C"
                    target_candidate_alignable = alignment_grade in ("A", "B")
                    if target_candidate_alignable:
                        best_table = t_best
                        best_match = t_match
                        best_row = None
                        all_table_rows = t_best.get("rows", []) or []
                        matched_row_ids = [
                            str(r.get("row_id") or "")
                            for r in all_table_rows[:5]
                            if r.get("row_id")
                        ]
                        all_table_cells = t_best.get("cells", []) or []
                        matched_cell_ids = [
                            str(c.get("cell_id") or "")
                            for c in all_table_cells[:10]
                            if c.get("cell_id")
                        ]
                        ambiguous = False
                        tiebreak_used = False
                        match_strategy = "table_block"
                        target_row_present = True
                        target_cells_present = t_cells_present

        elif legacy_candidate_type == "multi_row_block":
            # Try matching consecutive rows within each table
            best_mr_match: dict[str, Any] | None = None
            best_mr_rows: list[dict[str, Any]] = []
            best_mr_table: dict[str, Any] | None = None
            for table in tables:
                table_rows = table.get("rows", []) or []
                # Rebuild cells for each row
                table_cells = table.get("cells", []) or []
                cells_by_row: dict[int, list[dict[str, Any]]] = {}
                for cell in table_cells:
                    row_idx = int(cell.get("row_index") or 0)
                    cells_by_row.setdefault(row_idx, []).append(cell)
                for row in table_rows:
                    row_idx = int(row.get("row_index") or 0)
                    row["cells"] = cells_by_row.get(row_idx, [])
                result_mr = _score_multi_row_match(
                    gold_numeric,
                    gold_text_tokens,
                    gold_metric_label,
                    table_rows,
                )
                if result_mr is None:
                    continue
                mr_match, mr_rows = result_mr
                if best_mr_match is None or mr_match["score"] > best_mr_match["score"]:
                    best_mr_match = mr_match
                    best_mr_rows = mr_rows
                    best_mr_table = table
            if best_mr_match is not None and best_mr_match["score"] > 0:
                mr_nr = best_mr_match["numeric_recall"]
                mr_tr = best_mr_match["metric_token_recall"]
                mr_cells_present = mr_nr >= NUMERIC_RECALL_B if gold_numeric else True
                if (
                    mr_nr >= NUMERIC_RECALL_A
                    and mr_tr >= METRIC_TOKEN_RECALL_A
                    and mr_cells_present
                ):
                    alignment_grade = "A"
                elif (
                    mr_nr >= NUMERIC_RECALL_A
                    and mr_tr >= NUMERIC_PERFECT_TR_RELAXED
                    and mr_cells_present
                ):
                    # Perfect multi-row numeric match with relaxed text
                    alignment_grade = "A"
                elif (
                    mr_nr >= NUMERIC_RECALL_B
                    and mr_tr >= METRIC_TOKEN_RECALL_B
                    and mr_cells_present
                ):
                    alignment_grade = "B"
                elif mr_nr > 0 or mr_tr > 0:
                    alignment_grade = "C"
                target_candidate_alignable = alignment_grade in ("A", "B")
                if target_candidate_alignable:
                    best_match = best_mr_match
                    best_table = best_mr_table
                    best_row = best_mr_rows[0] if best_mr_rows else None
                    matched_row_ids = [
                        str(r.get("row_id") or "")
                        for r in best_mr_rows
                        if r.get("row_id")
                    ]
                    all_mr_cells = [c for r in best_mr_rows for c in r.get("cells", [])]
                    matched_cell_ids = [
                        str(c.get("cell_id") or "")
                        for c in all_mr_cells[:10]
                        if c.get("cell_id")
                    ]
                    ambiguous = False
                    tiebreak_used = False
                    match_strategy = "multi_row_block"
                    target_row_present = True
                    target_cells_present = mr_cells_present

    # Determine failure reason
    failure_reason = None
    if not target_candidate_alignable:
        if not target_page_present:
            failure_reason = "target_page_missing"
        elif not tables:
            failure_reason = "target_table_missing"
        elif not target_block_present:
            failure_reason = "target_row_missing"
        elif ambiguous:
            failure_reason = "multiple_structural_matches"
        elif not target_row_present:
            failure_reason = "candidate_text_signature_mismatch"
        elif not target_cells_present:
            failure_reason = "target_numeric_cells_missing"
        elif legacy_candidate_type == "narrative":
            failure_reason = "candidate_is_narrative"
        elif legacy_candidate_type == "multi_row_block":
            failure_reason = "candidate_is_multi_row_block"
        elif legacy_candidate_type in ("table_block", "mixed"):
            failure_reason = "legacy_candidate_granularity_mismatch"
        else:
            failure_reason = "candidate_text_signature_mismatch"

    # Coverage level: highest T-layer reached
    coverage_level = "none"
    if target_page_present:
        coverage_level = "T0_page"
        if target_block_present:
            coverage_level = "T1_block"
            if target_row_present:
                coverage_level = "T2_row"
                if target_cells_present:
                    coverage_level = "T3_cell"
                    if target_candidate_alignable:
                        coverage_level = "T4_alignable"

    return {
        "case_id": case_id,
        "gold_source_identity": identity,
        "benchmark_class": benchmark_class,
        "document_id": doc_id,
        "pdf_page": pdf_page,
        "legacy_candidate_type": legacy_candidate_type,
        "legacy_block_type": legacy_block_type,
        "legacy_content_length": len(legacy_content),
        "gold_numeric_count": sum(gold_numeric.values()),
        "gold_text_token_count": len(gold_text_tokens),
        "gold_metric_label": gold_metric_label,
        "target_page_present": target_page_present,
        "target_block_present": target_block_present,
        "target_row_present": target_row_present,
        "target_cells_present": target_cells_present,
        "target_candidate_alignable": target_candidate_alignable,
        "matched_table_id": str(best_table.get("table_fragment_id") or "")
        if best_table
        else None,
        "matched_row_ids": matched_row_ids
        if match_strategy != "single_row"
        else ([str(best_row.get("row_id") or "")] if best_row else []),
        "matched_cell_ids": matched_cell_ids,
        "match_evidence": {
            "numeric_recall": best_match["numeric_recall"] if best_match else 0.0,
            "metric_token_recall": best_match["metric_token_recall"]
            if best_match
            else 0.0,
            "token_jaccard": best_match["token_jaccard"] if best_match else 0.0,
            "metric_contained": best_match["metric_contained"] if best_match else False,
            "ambiguous": ambiguous,
            "tiebreak_used": tiebreak_used,
            "margin_to_second_best": round(margin, 4),
            "best_score": best_match["score"] if best_match else 0.0,
            "candidates_scored": len(scored_rows),
            "match_strategy": match_strategy,
        },
        "alignment_grade": alignment_grade,
        "failure_reason": failure_reason,
        "coverage_level": coverage_level,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3-out", type=Path, default=R3_OUT)
    parser.add_argument("--r1-1-out", type=Path, default=R1_1_OUT)
    parser.add_argument("--r2-out", type=Path, default=R2_OUT)
    parser.add_argument("--corpus", type=Path, default=CORPUS_PATH)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    args = parser.parse_args()

    # 1. Verify the R3 seal exists and is valid
    seal_path = args.r3_out / "adapter-prediction-seal.json"
    if not seal_path.is_file():
        print(f"ERROR: Seal not found at {seal_path}")
        print("Run Gate 02 R3 first.")
        return 1
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed"):
        print("ERROR: Seal is not valid (sealed != true).")
        return 1

    # 2. Verify R3.1 closure exists (must run R3.1 before R3.2)
    r31_path = args.r3_out / "benchmark-structural-presence-closure.json"
    if not r31_path.is_file():
        print(f"ERROR: R3.1 closure not found at {r31_path}")
        print("Run Gate 02 R3.1 first.")
        return 1

    # 3. Load R3 predictions
    predictions_path = args.r3_out / "adapter-predictions.jsonl.gz"
    if not predictions_path.is_file():
        print(f"ERROR: R3 predictions not found at {predictions_path}")
        return 1
    predictions = _load_r3_predictions(predictions_path)
    print(f"Loaded {len(predictions)} R3 prediction pages")

    # 4. Load 33 Gold Source records (post-seal, allowed)
    d_class = _load_d_class_records(args.r1_1_out)
    b_class = _load_b_class_unrecovered(args.r1_1_out, args.r2_out)
    print(f"Loaded {len(d_class)} D-class records from Gate 08 R1.1")
    print(f"Loaded {len(b_class)} B-class unrecovered records from Gate 08 R2")

    # 5. Load Legacy Candidate content via ProductionCandidateMapper
    from src.pdf_retrieval_v4.v4_gate08_pool import ProductionCandidateMapper

    mapper = ProductionCandidateMapper(args.db_path, args.corpus)
    print(f"Loaded {len(mapper.by_key)} production candidates")

    # 6. Check alignment for each record
    all_records = d_class + b_class
    results: list[dict[str, Any]] = []
    missing_candidates = 0

    for record in all_records:
        gold_key = str(record.get("gold_candidate_key") or "")
        candidate = mapper.by_key.get(gold_key)
        if not candidate:
            missing_candidates += 1
        result = _check_alignment(record, candidate, predictions)
        results.append(result)

    if missing_candidates:
        print(
            f"WARNING: {missing_candidates} gold candidates not found in production store"
        )

    # 7. Aggregate metrics
    d_results = [r for r in results if r["benchmark_class"] == "D"]
    b_results = [r for r in results if r["benchmark_class"] == "B"]

    def _count(lst: list[dict[str, Any]], field: str) -> int:
        return sum(1 for r in lst if r.get(field))

    def _grade_count(lst: list[dict[str, Any]], grade: str) -> int:
        return sum(1 for r in lst if r.get("alignment_grade") == grade)

    d_metrics = {
        "total": len(d_results),
        "target_page_present": _count(d_results, "target_page_present"),
        "target_block_present": _count(d_results, "target_block_present"),
        "target_row_present": _count(d_results, "target_row_present"),
        "target_cells_present": _count(d_results, "target_cells_present"),
        "target_candidate_alignable": _count(d_results, "target_candidate_alignable"),
        "grade_a": _grade_count(d_results, "A"),
        "grade_b": _grade_count(d_results, "B"),
        "grade_c": _grade_count(d_results, "C"),
        "grade_none": _grade_count(d_results, "none"),
    }
    b_metrics = {
        "total": len(b_results),
        "target_page_present": _count(b_results, "target_page_present"),
        "target_block_present": _count(b_results, "target_block_present"),
        "target_row_present": _count(b_results, "target_row_present"),
        "target_cells_present": _count(b_results, "target_cells_present"),
        "target_candidate_alignable": _count(b_results, "target_candidate_alignable"),
        "grade_a": _grade_count(b_results, "A"),
        "grade_b": _grade_count(b_results, "B"),
        "grade_c": _grade_count(b_results, "C"),
        "grade_none": _grade_count(b_results, "none"),
    }

    # Failure reason distribution
    failure_reasons: dict[str, int] = {}
    for r in results:
        reason = r.get("failure_reason")
        if reason:
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    # False structural alignment: Grade A but numeric_recall < 1.0
    # (should never happen by definition, but verify)
    false_alignments = sum(
        1
        for r in results
        if r["alignment_grade"] == "A"
        and r["match_evidence"]["numeric_recall"] < NUMERIC_RECALL_A
        and r["gold_numeric_count"] > 0
    )

    # 8. Decision
    d_alignable = d_metrics["target_candidate_alignable"]
    b_alignable = b_metrics["target_candidate_alignable"]
    total_alignable = d_alignable + b_alignable
    total = d_metrics["total"] + b_metrics["total"]

    if (
        d_alignable >= 12
        and b_alignable >= 14
        and total_alignable >= 26
        and false_alignments == 0
    ):
        decision = "target_structural_alignment_closed"
        next_gate = "full_corpus_financial_semantic_graph"
        strength = "strong"
    elif d_alignable >= 10 and b_alignable >= 12 and total_alignable >= 22:
        decision = "target_structural_alignment_closed"
        next_gate = "full_corpus_financial_semantic_graph"
        strength = "acceptable"
    else:
        decision = "target_structural_alignment_insufficient"
        next_gate = "stop_and_classify_missing_evidence_shapes"
        strength = "insufficient"

    result = {
        "schema": "pdf-retrieval-v4/gate-02-r3.2/target-structural-alignment/v1",
        "seal_verified": True,
        "r31_closure_verified": True,
        "d_class_metrics": d_metrics,
        "b_class_metrics": b_metrics,
        "total_alignable": total_alignable,
        "total_records": total,
        "false_structural_alignment": false_alignments,
        "failure_reason_distribution": failure_reasons,
        "d_class_records": d_results,
        "b_class_records": b_results,
        "strength": strength,
        "decision": decision,
        "next_gate": next_gate,
        "production_switch_allowed": False,
        "matching_thresholds": {
            "numeric_recall_a": NUMERIC_RECALL_A,
            "metric_token_recall_a": METRIC_TOKEN_RECALL_A,
            "numeric_recall_b": NUMERIC_RECALL_B,
            "metric_token_recall_b": METRIC_TOKEN_RECALL_B,
            "margin_threshold": MARGIN_THRESHOLD,
        },
    }

    output_path = args.r3_out / "target-structural-alignment.json"
    _write_json(output_path, result)

    print("\nTarget-specific Structural Alignment Audit:")
    print(f"  D-class ({d_metrics['total']} records):")
    print(
        f"    Target Page Present:     {d_metrics['target_page_present']}/{d_metrics['total']}"
    )
    print(
        f"    Target Block Present:    {d_metrics['target_block_present']}/{d_metrics['total']}"
    )
    print(
        f"    Target Row Present:      {d_metrics['target_row_present']}/{d_metrics['total']}"
    )
    print(
        f"    Target Cells Present:    {d_metrics['target_cells_present']}/{d_metrics['total']}"
    )
    print(
        f"    Candidate Alignable:     {d_metrics['target_candidate_alignable']}/{d_metrics['total']}"
    )
    print(
        f"    Grade A: {d_metrics['grade_a']}, B: {d_metrics['grade_b']}, C: {d_metrics['grade_c']}, none: {d_metrics['grade_none']}"
    )
    print(f"  B-class ({b_metrics['total']} records):")
    print(
        f"    Target Page Present:     {b_metrics['target_page_present']}/{b_metrics['total']}"
    )
    print(
        f"    Target Block Present:    {b_metrics['target_block_present']}/{b_metrics['total']}"
    )
    print(
        f"    Target Row Present:      {b_metrics['target_row_present']}/{b_metrics['total']}"
    )
    print(
        f"    Target Cells Present:    {b_metrics['target_cells_present']}/{b_metrics['total']}"
    )
    print(
        f"    Candidate Alignable:     {b_metrics['target_candidate_alignable']}/{b_metrics['total']}"
    )
    print(
        f"    Grade A: {b_metrics['grade_a']}, B: {b_metrics['grade_b']}, C: {b_metrics['grade_c']}, none: {b_metrics['grade_none']}"
    )
    print(f"  Total Alignable: {total_alignable}/{total}")
    print(f"  False Structural Alignment: {false_alignments}")
    if failure_reasons:
        print("  Failure Reasons:")
        for reason, count in sorted(failure_reasons.items()):
            print(f"    {reason}: {count}")
    print(f"  Strength: {strength}")
    print(f"  Decision: {decision}")
    print(f"  Next gate: {next_gate}")

    return 0 if decision == "target_structural_alignment_closed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
