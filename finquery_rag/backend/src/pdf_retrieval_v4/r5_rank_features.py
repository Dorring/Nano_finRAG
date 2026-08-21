"""Generic deterministic feature extraction for R5 Candidate Ranking Recovery.

All features are extracted strictly from candidate records and query/runtime context.
No Gold information, question IDs, reference answers, or manual maps are used.
"""

from __future__ import annotations

import re
from typing import Any

WORD_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.%/-]*|[\u4e00-\u9fff]+")
DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}(?:[-/]\d{1,2}(?:[-/]\d{1,2})?)?\b")
TOKEN_CACHE: dict[str, list[str]] = {}


def tokenize(text: Any) -> list[str]:
    s = str(text or "").lower()
    if s not in TOKEN_CACHE:
        tokens = list(dict.fromkeys(WORD_PATTERN.findall(s)))
        TOKEN_CACHE[s] = tokens
    return TOKEN_CACHE[s]


def normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def extract_quoted_phrases(text: Any) -> list[str]:
    return [
        m.group(1).strip()
        for m in re.finditer(r"['\"]([^'\"]+)['\"]", str(text or ""))
        if m.group(1).strip()
    ]


def extract_temporal_target(item: dict[str, Any]) -> tuple[list[str], str]:
    ts = item.get("temporal_scope") or {}
    dates = []
    for key in ("period_end", "document_end"):
        if ts.get(key):
            dates.append(str(ts[key]))
    for key in ("periods", "period_end_values"):
        for val in ts.get(key) or []:
            if val:
                dates.append(str(val))
    semantics = str(ts.get("fact_semantics") or "UNKNOWN").upper()
    return list(dict.fromkeys(dates)), semantics


def compute_overlap(source_tokens: set[str], target_tokens: set[str]) -> float:
    if not source_tokens or not target_tokens:
        return 0.0
    return len(source_tokens & target_tokens) / len(source_tokens)


def extract_candidate_features(
    candidate: dict[str, Any],
    item: dict[str, Any],
    query_text: str | None = None,
    phrase_text: str | None = None,
) -> dict[str, float]:
    """Extract generic numeric features for a single candidate under an item context."""
    q_str = str(query_text or item.get("question") or "")
    q_norm = normalize_text(q_str)
    q_tokens = set(tokenize(q_str))
    quoted_phrases = extract_quoted_phrases(q_str)

    p_str = str(phrase_text or "")
    p_norm = normalize_text(p_str)
    p_tokens = set(tokenize(p_str)) if p_str else q_tokens
    active_tokens = p_tokens if p_tokens else q_tokens

    if p_str:
        quoted_phrases.extend(extract_quoted_phrases(p_str))
        if p_norm and p_norm not in [normalize_text(x) for x in quoted_phrases]:
            quoted_phrases.append(p_str)

    # A4 baseline rank and score features
    a4_rank = float(candidate.get("rank") or 200)
    rrf_score = float(candidate.get("rrf_score") or candidate.get("a4_score") or 0.0)
    bm25_score = float(candidate.get("bm25_score") or 0.0)
    dense_score = float(candidate.get("dense_score") or 0.0)
    child_overlap = float(candidate.get("child_overlap") or 0.0)

    # Evidence type
    etype = str(candidate.get("evidence_type") or "UNKNOWN").upper()
    is_table_row = 1.0 if etype == "TABLE_ROW" else 0.0
    is_text = 1.0 if etype == "TEXT" else 0.0
    is_table = 1.0 if etype == "TABLE" else 0.0

    # Row label
    row_label = str(candidate.get("row_label") or "")
    row_norm = normalize_text(row_label)
    row_tokens = set(tokenize(row_label))
    row_exact = 1.0 if row_norm and (row_norm in q_norm or (p_norm and row_norm in p_norm)) else 0.0
    row_overlap = compute_overlap(active_tokens, row_tokens)

    # Header path & cells
    headers = list(candidate.get("column_headers") or [])
    header_str = " ".join(str(h) for h in headers if h)
    header_norm = normalize_text(header_str)
    header_tokens = set(tokenize(header_str))
    header_overlap = compute_overlap(active_tokens, header_tokens)

    # Table title
    table_title = str(candidate.get("table_title") or "")
    title_norm = normalize_text(table_title)
    title_tokens = set(tokenize(table_title))
    title_exact = 1.0 if title_norm and (title_norm in q_norm or (p_norm and title_norm in p_norm)) else 0.0
    title_overlap = compute_overlap(active_tokens, title_tokens)

    # Metric path
    rec = candidate.get("record") or {}
    mp = rec.get("metric_path") or candidate.get("metric_path") or {}
    if isinstance(mp, dict):
        mp_str = " ".join(str(mp.get(k) or "") for k in ("metric_path", "root_metric", "leaf_metric") if mp.get(k))
    else:
        mp_str = str(mp or "")
    mp_norm = normalize_text(mp_str)
    mp_tokens = set(tokenize(mp_str))
    mp_exact = 1.0 if mp_norm and (mp_norm in q_norm or (p_norm and mp_norm in p_norm)) else 0.0
    mp_overlap = compute_overlap(active_tokens, mp_tokens)

    # Quoted phrase matching across row label, headers, and body
    phrase_in_row_or_header = 0.0
    phrase_in_title = 0.0
    for qp in quoted_phrases:
        qp_norm = normalize_text(qp)
        if not qp_norm:
            continue
        if qp_norm in row_norm or qp_norm in header_norm or qp_norm in mp_norm:
            phrase_in_row_or_header = 1.0
        if qp_norm in title_norm:
            phrase_in_title = 1.0

    # Content / text
    body = str(candidate.get("retrieval_text_v2") or candidate.get("content") or "")
    body_tokens = set(tokenize(body))
    body_overlap = compute_overlap(active_tokens, body_tokens)

    # Period features
    target_dates, target_semantics = extract_temporal_target(item)
    cand_end = str(candidate.get("period_end") or "")
    cand_sem = str(candidate.get("period_semantics") or "UNKNOWN").upper()

    target_date_match = 0.0
    period_incompatible = 0.0
    period_unknown = 1.0 if (not cand_end and cand_sem == "UNKNOWN") else 0.0

    if target_dates:
        if cand_end and cand_end in target_dates:
            target_date_match = 1.0
        elif cand_end and cand_end not in target_dates:
            period_incompatible = 1.0

    semantic_match = 0.0
    if target_semantics != "UNKNOWN":
        if cand_sem == target_semantics:
            semantic_match = 1.0
        elif cand_sem != "UNKNOWN" and cand_sem != target_semantics:
            period_incompatible = 1.0

    # Units / completeness
    has_unit = 1.0 if candidate.get("unit") or candidate.get("currency") or candidate.get("scale") else 0.0
    has_period = 1.0 if cand_end or cand_sem != "UNKNOWN" else 0.0
    completeness = (is_table_row * 0.4) + (has_period * 0.3) + (has_unit * 0.3)

    return {
        "a4_rank": a4_rank,
        "a4_rrf_score": rrf_score,
        "a4_reciprocal_rank": 1.0 / (60.0 + a4_rank),
        "bm25_score": bm25_score,
        "dense_score": dense_score,
        "child_overlap": child_overlap,
        "is_table_row": is_table_row,
        "is_text": is_text,
        "is_table": is_table,
        "row_exact": row_exact,
        "row_overlap": row_overlap,
        "title_exact": title_exact,
        "title_overlap": title_overlap,
        "mp_exact": mp_exact,
        "mp_overlap": mp_overlap,
        "header_overlap": header_overlap,
        "phrase_in_row_or_header": phrase_in_row_or_header,
        "phrase_in_title": phrase_in_title,
        "body_overlap": body_overlap,
        "target_date_match": target_date_match,
        "semantic_match": semantic_match,
        "period_incompatible": period_incompatible,
        "period_unknown": period_unknown,
        "has_unit": has_unit,
        "has_period": has_period,
        "completeness": completeness,
    }


def score_candidate(
    candidate: dict[str, Any],
    item: dict[str, Any],
    stage: str = "R4",
    features: dict[str, float] | None = None,
    qwen_score: float | None = None,
) -> float:
    """Compute deterministic blended ranking score for a candidate.

    Ablation stages:
    - R0: Pure A4 original order (RRF score)
    - R1: A4 + deterministic period compatibility
    - R2: A4 + metric / row structural relevance
    - R3: A4 + structural + table / header context
    - R4: A4 + all deterministic structured features
    - Q1 / Q2: Deterministic + Qwen score feature
    """
    f = features or extract_candidate_features(candidate, item)
    a4_score = f["a4_rrf_score"]

    if stage == "R0":
        return a4_score

    # Period gate / score component
    period_score = 0.0
    if f["period_incompatible"] > 0.5:
        period_score = -0.15
    else:
        if f["target_date_match"] > 0.5:
            period_score += 0.20
        if f["semantic_match"] > 0.5:
            period_score += 0.10

    if stage == "R1":
        return a4_score + 0.05 * period_score

    # Structural metric / row score
    struct_score = (
        0.30 * f["phrase_in_row_or_header"]
        + 0.25 * f["row_exact"]
        + 0.15 * f["row_overlap"]
        + 0.20 * f["mp_exact"]
        + 0.10 * f["mp_overlap"]
    )

    if stage == "R2":
        return a4_score + 0.05 * period_score + 0.12 * struct_score

    # Context score (table title, header path)
    context_score = (
        0.15 * f["phrase_in_title"]
        + 0.10 * f["title_exact"]
        + 0.08 * f["title_overlap"]
        + 0.12 * f["header_overlap"]
    )

    if stage == "R3":
        return a4_score + 0.05 * period_score + 0.12 * struct_score + 0.06 * context_score

    # Full R4: all deterministic features + table row preference over table chunk
    r4_score = (
        a4_score
        + 0.06 * period_score
        + 0.14 * struct_score
        + 0.06 * context_score
        + 0.03 * f["completeness"]
        + 0.03 * f["is_table_row"] * (f["row_overlap"] + f["phrase_in_row_or_header"])
        - 0.02 * f["is_table"]
    )

    if stage == "R4":
        return r4_score

    # Qwen integration (if qwen_score is provided)
    if qwen_score is not None:
        if stage == "Q0":
            return qwen_score
        if stage == "Q1":
            return a4_score + 0.15 * qwen_score
        if stage == "Q2":
            return r4_score + 0.10 * qwen_score

    return r4_score
