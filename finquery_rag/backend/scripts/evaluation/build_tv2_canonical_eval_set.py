"""TV2-FINAL-01: Build canonical evaluation dataset from frozen Fact Store.

This script reads the production Fact Store JSONL and deterministically
constructs a stratified 120-question evaluation set.  It never calls any
model, never reads historical questions for reuse, and performs a
contamination check against the Category C question blacklist.

Usage:
    python build_tv2_canonical_eval_set.py \
        --fact-store /path/to/financial-facts.jsonl \
        --category-c /path/to/category_c_questions.json \
        --out-dir /path/to/output-dir \
        [--seed 42] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CURRENCY_STRIP = re.compile(r"^[\s$€£¥]+|[\s$€£¥]+$")
_PAREN_NEG = re.compile(r"^\((.+)\)$")


def _parse_number(raw: str | None) -> Decimal | None:
    """Parse a financial string value into Decimal."""
    if raw is None:
        return None
    text = _CURRENCY_STRIP.sub("", str(raw).strip())
    m = _PAREN_NEG.match(text)
    if m:
        text = "-" + m.group(1)
    text = text.replace(",", "").replace("%", "").strip()
    if not text:
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question_fingerprint(q: str) -> str:
    """Normalise a question for contamination matching."""
    return re.sub(r"\s+", " ", q.strip().lower())


# ---------------------------------------------------------------------------
# Fact Store loading
# ---------------------------------------------------------------------------

def load_facts(path: Path) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            facts.append(json.loads(line))
    return facts


def build_indices(facts: list[dict[str, Any]]):
    """Build lookup structures over the fact store."""
    # Facts with valid metric + period + parseable numeric value
    usable: list[dict[str, Any]] = []
    by_doc: dict[str, list[dict]] = defaultdict(list)
    by_doc_metric_period: dict[tuple[str, str, str], dict] = {}
    metric_to_docs: dict[str, set[str]] = defaultdict(set)
    doc_metric_periods: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for f in facts:
        metric = f.get("metric")
        period = f.get("period")
        value = f.get("value")
        doc_id = f.get("document_id", "")
        if not metric or not period or _parse_number(value) is None:
            continue
        usable.append(f)
        by_doc[doc_id].append(f)
        key = (doc_id, metric, period)
        if key not in by_doc_metric_period:
            by_doc_metric_period[key] = f
        metric_to_docs[metric].add(doc_id)
        doc_metric_periods[doc_id][metric].add(period)

    return usable, by_doc, by_doc_metric_period, metric_to_docs, doc_metric_periods


ENTITY_MAP = {
    "aapl_fy2025": "Apple",
    "jpm_fy2025": "JPMorganChase",
    "ko_fy2025": "The Coca-Cola Company",
    "msft_fy2025": "Microsoft",
    "nvda_fy2025": "NVIDIA",
    "pfe_fy2024": "Pfizer",
    "tsla_fy2025": "Tesla",
    "v_fy2025": "Visa",
}

DOC_PERIODS = {
    "aapl_fy2025": ("FY2024", "FY2025"),
    "jpm_fy2025": ("FY2024", "FY2025"),
    "ko_fy2025": ("FY2024", "FY2025"),
    "msft_fy2025": ("FY2024", "FY2025"),
    "nvda_fy2025": ("FY2024", "FY2025"),
    "pfe_fy2024": ("FY2023", "FY2024"),
    "tsla_fy2025": ("FY2024", "FY2025"),
    "v_fy2025": ("FY2024", "FY2025"),
}


# ---------------------------------------------------------------------------
# Question templates
# ---------------------------------------------------------------------------

LOOKUP_TEMPLATES = [
    "What was {entity}'s {metric} in {period}?",
    "According to {entity}'s {fy_label} 10-K filing, what was the reported {metric}?",
    "How much did {entity} report for {metric} as of fiscal year {year}?",
    "In the {entity} annual report, what is the figure for {metric} in {period}?",
    "What is the {metric} value reported by {entity} for {period}?",
]

GROWTH_TEMPLATES = [
    "What was {entity}'s year-over-year growth rate in {metric} from {period_prev} to {period_cur}?",
    "Calculate the percentage change in {entity}'s {metric} between {period_prev} and {period_cur}.",
]

PCTSHARE_TEMPLATES = [
    "What percentage of {entity}'s {denom_metric} was {numer_metric} in {period}?",
    "For {entity} in {period}, what is {numer_metric} as a share of {denom_metric}?",
]

DIFF_TEMPLATES = [
    "What is the difference between {entity}'s {metric} in {period_cur} and {period_prev}?",
    "By how much did {entity}'s {metric} change from {period_prev} to {period_cur}?",
]

COMPARE_TEMPLATES = [
    "Which company had a higher {metric} in {period}, {entity_a} or {entity_b}?",
    "Compare {entity_a} and {entity_b}: which reported a larger {metric} in {period}?",
]

DIFF_COMPARE_TEMPLATES = [
    "What is the difference in {metric} between {entity_a} and {entity_b} in {period}?",
]

RANK_TEMPLATES = [
    "Rank the following companies by {metric} in {period} from highest to lowest: {entity_list}.",
]


# ---------------------------------------------------------------------------
# Stratum builders
# ---------------------------------------------------------------------------

def _pick_clean_metrics(
    doc_facts: list[dict],
    period: str,
    rng: random.Random,
    count: int,
    *,
    exclude_metrics: set[str] | None = None,
    max_metric_len: int = 80,
) -> list[dict]:
    """Pick 'count' facts for a doc+period with clean, short metric names."""
    # Blacklist overly generic / ambiguous metric names
    _GENERIC_METRICS = {
        "Total", "Other", "Worldwide", "Net change", "Balance at beginning of year",
        "Balance at end of year", "Beginning balances", "Ending balances",
        "2025", "2024", "2023", "2022", "2021",
    }
    candidates = [
        f for f in doc_facts
        if f["period"] == period
        and len(f["metric"]) <= max_metric_len
        and "/" not in f["metric"]  # skip hierarchical path metrics
        and ":" not in f["metric"]
        and not f["metric"].startswith("ASSETS")
        and not f["metric"].startswith("Item ")
        and f["metric"] not in _GENERIC_METRICS
        and _parse_number(f["value"]) is not None
        and (exclude_metrics is None or f["metric"] not in exclude_metrics)
    ]
    rng.shuffle(candidates)
    seen_metrics: set[str] = set()
    picked: list[dict] = []
    for c in candidates:
        if c["metric"] not in seen_metrics:
            seen_metrics.add(c["metric"])
            picked.append(c)
        if len(picked) >= count:
            break
    return picked


def build_stratum1(
    by_doc: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Stratum 1: Single-slot factual lookup (40 questions, 5 per company)."""
    questions: list[dict[str, Any]] = []
    idx = 0
    for doc_id in sorted(ENTITY_MAP.keys()):
        entity = ENTITY_MAP[doc_id]
        _, primary_period = DOC_PERIODS[doc_id]
        facts_for_doc = by_doc.get(doc_id, [])
        picked = _pick_clean_metrics(facts_for_doc, primary_period, rng, 5)
        for fact in picked:
            idx += 1
            tmpl = LOOKUP_TEMPLATES[idx % len(LOOKUP_TEMPLATES)]
            year = fact["period"].replace("FY", "")
            q_text = tmpl.format(
                entity=entity,
                metric=fact["metric"],
                period=fact["period"],
                fy_label=fact["period"],
                year=year,
            )
            questions.append({
                "id": "tv2f01-s1-%s-%03d" % (doc_id.split("_")[0], idx),
                "stratum": "factual_lookup",
                "question": q_text,
                "document_id": doc_id,
                "entity": entity,
                "expected_intent": "financial_lookup",
                "gold": {
                    "fact_ids": [fact["candidate_key"]],
                    "expected_value": fact["value"],
                    "metric": fact["metric"],
                    "period": fact["period"],
                    "operation": None,
                    "operands": None,
                    "expected_outcome": "ANSWER",
                    "tolerance": None,
                },
                "tags": ["canonical", "stratum1", "factual_lookup", doc_id.split("_")[0]],
            })
    return questions


def build_stratum2(
    by_doc: dict[str, list[dict]],
    by_doc_metric_period: dict[tuple[str, str, str], dict],
    doc_metric_periods: dict[str, dict[str, set[str]]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Stratum 2: Arithmetic calculation (35 questions)."""
    questions: list[dict[str, Any]] = []
    idx = 0

    # --- growth_rate (10 questions) ---
    _GENERIC = {"Total", "Other", "Worldwide", "Net change", "Balance at beginning of year",
                 "Balance at end of year", "Beginning balances", "Ending balances",
                 "2025", "2024", "2023", "2022", "2021"}
    growth_candidates: list[tuple[str, str]] = []  # (doc_id, metric)
    for doc_id in sorted(ENTITY_MAP.keys()):
        prev_period, cur_period = DOC_PERIODS[doc_id]
        for metric, periods in doc_metric_periods[doc_id].items():
            if prev_period in periods and cur_period in periods:
                if len(metric) <= 80 and "/" not in metric and ":" not in metric and metric not in _GENERIC:
                    f_prev = by_doc_metric_period.get((doc_id, metric, prev_period))
                    f_cur = by_doc_metric_period.get((doc_id, metric, cur_period))
                    if f_prev and f_cur:
                        v_prev = _parse_number(f_prev["value"])
                        v_cur = _parse_number(f_cur["value"])
                        if v_prev and v_cur and v_prev != 0:
                            growth_candidates.append((doc_id, metric))
    rng.shuffle(growth_candidates)
    # Pick up to 10, spread across companies
    doc_growth_count: dict[str, int] = defaultdict(int)
    for doc_id, metric in growth_candidates:
        if len(questions) - idx >= 10:
            break
        if doc_growth_count[doc_id] >= 2:
            continue
        prev_period, cur_period = DOC_PERIODS[doc_id]
        f_prev = by_doc_metric_period[(doc_id, metric, prev_period)]
        f_cur = by_doc_metric_period[(doc_id, metric, cur_period)]
        v_prev = _parse_number(f_prev["value"])
        v_cur = _parse_number(f_cur["value"])
        assert v_prev and v_cur and v_prev != 0
        growth = (v_cur - v_prev) / v_prev
        growth_str = str(growth.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        entity = ENTITY_MAP[doc_id]
        tmpl = GROWTH_TEMPLATES[len(questions) % len(GROWTH_TEMPLATES)]
        q_text = tmpl.format(
            entity=entity, metric=metric,
            period_prev=prev_period, period_cur=cur_period,
        )
        questions.append({
            "id": "tv2f01-s2-growth-%03d" % (len(questions) + 1),
            "stratum": "arithmetic_calculation",
            "question": q_text,
            "document_id": doc_id,
            "entity": entity,
            "expected_intent": "financial_calculation",
            "gold": {
                "fact_ids": [f_prev["candidate_key"], f_cur["candidate_key"]],
                "expected_value": growth_str,
                "metric": metric,
                "period_current": cur_period,
                "period_previous": prev_period,
                "operation": "growth_rate",
                "operands": {"current": f_cur["value"], "previous": f_prev["value"]},
                "expected_outcome": "ANSWER",
                "tolerance": 0.0001,
            },
            "tags": ["canonical", "stratum2", "growth_rate", doc_id.split("_")[0]],
        })
        doc_growth_count[doc_id] += 1

    # --- difference (7 questions) ---
    diff_count = 0
    doc_diff_count: dict[str, int] = defaultdict(int)
    rng.shuffle(growth_candidates)  # reuse the YoY-capable pool
    for doc_id, metric in growth_candidates:
        if diff_count >= 7:
            break
        if doc_diff_count[doc_id] >= 2:
            continue
        # Don't reuse metrics already used in growth
        used_metrics = {q["gold"]["metric"] for q in questions if q["gold"].get("operation") == "growth_rate" and q["document_id"] == doc_id}
        if metric in used_metrics:
            continue
        prev_period, cur_period = DOC_PERIODS[doc_id]
        f_prev = by_doc_metric_period[(doc_id, metric, prev_period)]
        f_cur = by_doc_metric_period[(doc_id, metric, cur_period)]
        v_prev = _parse_number(f_prev["value"])
        v_cur = _parse_number(f_cur["value"])
        if v_prev is None or v_cur is None:
            continue
        diff_val = v_cur - v_prev
        diff_str = str(diff_val)
        entity = ENTITY_MAP[doc_id]
        tmpl = DIFF_TEMPLATES[diff_count % len(DIFF_TEMPLATES)]
        q_text = tmpl.format(
            entity=entity, metric=metric,
            period_prev=prev_period, period_cur=cur_period,
        )
        questions.append({
            "id": "tv2f01-s2-diff-%03d" % (diff_count + 1),
            "stratum": "arithmetic_calculation",
            "question": q_text,
            "document_id": doc_id,
            "entity": entity,
            "expected_intent": "financial_calculation",
            "gold": {
                "fact_ids": [f_prev["candidate_key"], f_cur["candidate_key"]],
                "expected_value": diff_str,
                "metric": metric,
                "period_current": cur_period,
                "period_previous": prev_period,
                "operation": "difference",
                "operands": {"current": f_cur["value"], "previous": f_prev["value"]},
                "expected_outcome": "ANSWER",
                "tolerance": 0.0001,
            },
            "tags": ["canonical", "stratum2", "difference", doc_id.split("_")[0]],
        })
        diff_count += 1
        doc_diff_count[doc_id] += 1

    # --- percentage_share (8 questions) ---
    pct_count = 0
    for doc_id in sorted(ENTITY_MAP.keys()):
        if pct_count >= 8:
            break
        entity = ENTITY_MAP[doc_id]
        _, primary_period = DOC_PERIODS[doc_id]
        doc_facts = by_doc.get(doc_id, [])
        clean_facts = _pick_clean_metrics(doc_facts, primary_period, rng, 20)
        # Find pairs where one is plausibly a sub-component
        for i, f_numer in enumerate(clean_facts):
            if pct_count >= 8:
                break
            v_numer = _parse_number(f_numer["value"])
            if v_numer is None or v_numer == 0:
                continue
            for f_denom in clean_facts[i + 1:]:
                if pct_count >= 8:
                    break
                v_denom = _parse_number(f_denom["value"])
                if v_denom is None or v_denom == 0:
                    continue
                if abs(v_numer) < abs(v_denom):
                    ratio = v_numer / v_denom
                    ratio_str = str(ratio.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
                    tmpl = PCTSHARE_TEMPLATES[pct_count % len(PCTSHARE_TEMPLATES)]
                    q_text = tmpl.format(
                        entity=entity,
                        numer_metric=f_numer["metric"],
                        denom_metric=f_denom["metric"],
                        period=primary_period,
                    )
                    questions.append({
                        "id": "tv2f01-s2-pctshare-%03d" % (pct_count + 1),
                        "stratum": "arithmetic_calculation",
                        "question": q_text,
                        "document_id": doc_id,
                        "entity": entity,
                        "expected_intent": "financial_calculation",
                        "gold": {
                            "fact_ids": [f_numer["candidate_key"], f_denom["candidate_key"]],
                            "expected_value": ratio_str,
                            "metric_numerator": f_numer["metric"],
                            "metric_denominator": f_denom["metric"],
                            "period": primary_period,
                            "operation": "percentage_share",
                            "operands": {"part": f_numer["value"], "total": f_denom["value"]},
                            "expected_outcome": "ANSWER",
                            "tolerance": 0.0001,
                        },
                        "tags": ["canonical", "stratum2", "percentage_share", doc_id.split("_")[0]],
                    })
                    pct_count += 1
                    break  # one per numerator

    # --- gross_margin / net_margin / debt_ratio / sum / average (10 remaining) ---
    # Fill with additional growth/diff/pctshare to hit 35 total
    remaining = 35 - len(questions)
    extra_count = 0
    rng.shuffle(growth_candidates)
    used_keys = {(q["document_id"], q["gold"].get("metric", "")) for q in questions}
    for doc_id, metric in growth_candidates:
        if extra_count >= remaining:
            break
        if (doc_id, metric) in used_keys:
            continue
        prev_period, cur_period = DOC_PERIODS[doc_id]
        f_prev = by_doc_metric_period.get((doc_id, metric, prev_period))
        f_cur = by_doc_metric_period.get((doc_id, metric, cur_period))
        if not f_prev or not f_cur:
            continue
        v_prev = _parse_number(f_prev["value"])
        v_cur = _parse_number(f_cur["value"])
        if v_prev is None or v_cur is None:
            continue
        # Alternate between sum and average
        entity = ENTITY_MAP[doc_id]
        if extra_count % 2 == 0:
            op = "sum"
            result = v_cur + v_prev
            q_text = "What is the sum of %s's %s across %s and %s?" % (entity, metric, prev_period, cur_period)
        else:
            op = "average"
            result = (v_cur + v_prev) / 2
            q_text = "What is the average of %s's %s between %s and %s?" % (entity, metric, prev_period, cur_period)
        result_str = str(result.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))
        questions.append({
            "id": "tv2f01-s2-%s-%03d" % (op, extra_count + 1),
            "stratum": "arithmetic_calculation",
            "question": q_text,
            "document_id": doc_id,
            "entity": entity,
            "expected_intent": "financial_calculation",
            "gold": {
                "fact_ids": [f_prev["candidate_key"], f_cur["candidate_key"]],
                "expected_value": result_str,
                "metric": metric,
                "operation": op,
                "operands": {"a": f_prev["value"], "b": f_cur["value"]},
                "expected_outcome": "ANSWER",
                "tolerance": 0.0001,
            },
            "tags": ["canonical", "stratum2", op, doc_id.split("_")[0]],
        })
        extra_count += 1
        used_keys.add((doc_id, metric))

    return questions


def build_stratum3(
    by_doc_metric_period: dict[tuple[str, str, str], dict],
    metric_to_docs: dict[str, set[str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Stratum 3: Cross-entity comparison (20 questions)."""
    questions: list[dict[str, Any]] = []

    # Find metrics in >=3 docs with clean names
    cross_metrics: list[tuple[str, list[str]]] = []
    for metric, docs in metric_to_docs.items():
        if len(docs) >= 3 and len(metric) <= 60 and "/" not in metric and ":" not in metric:
            cross_metrics.append((metric, sorted(docs)))
    rng.shuffle(cross_metrics)

    # --- "Which had higher" comparisons (10) ---
    compare_count = 0
    for metric, docs in cross_metrics:
        if compare_count >= 10:
            break
        period = "FY2025" if any(by_doc_metric_period.get((d, metric, "FY2025")) for d in docs) else "FY2024"
        available = [(d, by_doc_metric_period.get((d, metric, period))) for d in docs]
        available = [(d, f) for d, f in available if f is not None and _parse_number(f["value"]) is not None]
        if len(available) < 2:
            continue
        rng.shuffle(available)
        d_a, f_a = available[0]
        d_b, f_b = available[1]
        v_a = _parse_number(f_a["value"])
        v_b = _parse_number(f_b["value"])
        entity_a = ENTITY_MAP[d_a]
        entity_b = ENTITY_MAP[d_b]
        higher = entity_a if v_a > v_b else entity_b
        tmpl = COMPARE_TEMPLATES[compare_count % len(COMPARE_TEMPLATES)]
        q_text = tmpl.format(
            entity_a=entity_a, entity_b=entity_b,
            metric=metric, period=period,
        )
        questions.append({
            "id": "tv2f01-s3-compare-%03d" % (compare_count + 1),
            "stratum": "cross_entity_comparison",
            "question": q_text,
            "document_id": None,
            "entity": None,
            "expected_intent": "financial_comparison",
            "gold": {
                "fact_ids": [f_a["candidate_key"], f_b["candidate_key"]],
                "expected_value": None,
                "expected_higher": higher,
                "values": {entity_a: f_a["value"], entity_b: f_b["value"]},
                "metric": metric,
                "period": period,
                "operation": "comparison",
                "expected_outcome": "ANSWER",
                "tolerance": None,
            },
            "tags": ["canonical", "stratum3", "comparison", d_a.split("_")[0], d_b.split("_")[0]],
        })
        compare_count += 1

    # --- "Difference between two companies" (5) ---
    diff_count = 0
    for metric, docs in cross_metrics:
        if diff_count >= 5:
            break
        # Skip metrics already used in comparisons
        used = {q["gold"]["metric"] for q in questions}
        if metric in used:
            continue
        period = "FY2025" if any(by_doc_metric_period.get((d, metric, "FY2025")) for d in docs) else "FY2024"
        available = [(d, by_doc_metric_period.get((d, metric, period))) for d in docs]
        available = [(d, f) for d, f in available if f is not None and _parse_number(f["value"]) is not None]
        if len(available) < 2:
            continue
        rng.shuffle(available)
        d_a, f_a = available[0]
        d_b, f_b = available[1]
        v_a = _parse_number(f_a["value"])
        v_b = _parse_number(f_b["value"])
        diff_val = v_a - v_b
        entity_a = ENTITY_MAP[d_a]
        entity_b = ENTITY_MAP[d_b]
        q_text = DIFF_COMPARE_TEMPLATES[0].format(
            entity_a=entity_a, entity_b=entity_b,
            metric=metric, period=period,
        )
        questions.append({
            "id": "tv2f01-s3-crossdiff-%03d" % (diff_count + 1),
            "stratum": "cross_entity_comparison",
            "question": q_text,
            "document_id": None,
            "entity": None,
            "expected_intent": "financial_calculation",
            "gold": {
                "fact_ids": [f_a["candidate_key"], f_b["candidate_key"]],
                "expected_value": str(diff_val),
                "values": {entity_a: f_a["value"], entity_b: f_b["value"]},
                "metric": metric,
                "period": period,
                "operation": "cross_entity_difference",
                "expected_outcome": "ANSWER",
                "tolerance": 0.0001,
            },
            "tags": ["canonical", "stratum3", "cross_diff", d_a.split("_")[0], d_b.split("_")[0]],
        })
        diff_count += 1

    # --- "Rank N companies" (5) ---
    rank_count = 0
    for metric, docs in cross_metrics:
        if rank_count >= 5:
            break
        used = {q["gold"]["metric"] for q in questions}
        if metric in used:
            continue
        period = "FY2025" if any(by_doc_metric_period.get((d, metric, "FY2025")) for d in docs) else "FY2024"
        available = [(d, by_doc_metric_period.get((d, metric, period))) for d in docs]
        available = [(d, f) for d, f in available if f is not None and _parse_number(f["value"]) is not None]
        if len(available) < 3:
            continue
        rng.shuffle(available)
        subset = available[:min(4, len(available))]
        ranked = sorted(subset, key=lambda x: _parse_number(x[1]["value"]), reverse=True)
        entity_list = ", ".join(ENTITY_MAP[d] for d, _ in subset)
        ranking = [ENTITY_MAP[d] for d, _ in ranked]
        fact_ids = [f["candidate_key"] for _, f in subset]
        values = {ENTITY_MAP[d]: f["value"] for d, f in subset}
        q_text = RANK_TEMPLATES[0].format(
            entity_list=entity_list, metric=metric, period=period,
        )
        questions.append({
            "id": "tv2f01-s3-rank-%03d" % (rank_count + 1),
            "stratum": "cross_entity_comparison",
            "question": q_text,
            "document_id": None,
            "entity": None,
            "expected_intent": "financial_comparison",
            "gold": {
                "fact_ids": fact_ids,
                "expected_value": None,
                "expected_ranking": ranking,
                "values": values,
                "metric": metric,
                "period": period,
                "operation": "ranking",
                "expected_outcome": "ANSWER",
                "tolerance": None,
            },
            "tags": ["canonical", "stratum3", "ranking"],
        })
        rank_count += 1

    return questions


def build_stratum4(
    by_doc: dict[str, list[dict]],
    metric_to_docs: dict[str, set[str]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Stratum 4: Adversarial abstention / negative (25 questions)."""
    questions: list[dict[str, Any]] = []

    # --- Out-of-corpus entity (5) ---
    oof_entities = [
        ("Amazon", "AMZN"), ("Alphabet", "GOOG"), ("Meta Platforms", "META"),
        ("Netflix", "NFLX"), ("Walmart", "WMT"),
    ]
    common_metrics = ["Total revenue", "Net income", "Total assets", "Operating income", "Earnings per share"]
    for i, ((entity, ticker), metric) in enumerate(zip(oof_entities, common_metrics)):
        q_text = "What was %s's %s in FY2025?" % (entity, metric)
        questions.append({
            "id": "tv2f01-s4-ooc-%03d" % (i + 1),
            "stratum": "adversarial_abstention",
            "question": q_text,
            "document_id": None,
            "entity": entity,
            "expected_intent": "financial_lookup",
            "gold": {
                "fact_ids": [],
                "expected_value": None,
                "expected_outcome": "ABSTENTION",
                "expected_failure_reason": "ENTITY_NOT_FOUND",
            },
            "tags": ["canonical", "stratum4", "abstention", "out_of_corpus"],
        })

    # --- Out-of-period (5) ---
    oop_entities = [
        ("aapl_fy2025", "Apple", "FY2020"),
        ("msft_fy2025", "Microsoft", "FY2019"),
        ("nvda_fy2025", "NVIDIA", "FY2021"),
        ("tsla_fy2025", "Tesla", "FY2018"),
        ("v_fy2025", "Visa", "FY2017"),
    ]
    for i, (doc_id, entity, bad_period) in enumerate(oop_entities):
        doc_facts = by_doc.get(doc_id, [])
        if doc_facts:
            metric = _pick_clean_metrics(doc_facts, DOC_PERIODS[doc_id][1], rng, 1)
            metric_name = metric[0]["metric"] if metric else "Net income"
        else:
            metric_name = "Net income"
        q_text = "What was %s's %s in %s?" % (entity, metric_name, bad_period)
        questions.append({
            "id": "tv2f01-s4-oop-%03d" % (i + 1),
            "stratum": "adversarial_abstention",
            "question": q_text,
            "document_id": doc_id,
            "entity": entity,
            "expected_intent": "financial_lookup",
            "gold": {
                "fact_ids": [],
                "expected_value": None,
                "expected_outcome": "ABSTENTION",
                "expected_failure_reason": "OUT_OF_PERIOD",
            },
            "tags": ["canonical", "stratum4", "abstention", "out_of_period"],
        })

    # --- Non-existent metric (5) ---
    fake_metrics = [
        ("aapl_fy2025", "Apple", "Number of Employees"),
        ("jpm_fy2025", "JPMorganChase", "Subscriber Count"),
        ("ko_fy2025", "The Coca-Cola Company", "Data Center Capacity"),
        ("nvda_fy2025", "NVIDIA", "Store Count"),
        ("tsla_fy2025", "Tesla", "Advertising Revenue"),
    ]
    for i, (doc_id, entity, fake_metric) in enumerate(fake_metrics):
        _, period = DOC_PERIODS[doc_id]
        q_text = "What was %s's %s in %s?" % (entity, fake_metric, period)
        questions.append({
            "id": "tv2f01-s4-fakemetric-%03d" % (i + 1),
            "stratum": "adversarial_abstention",
            "question": q_text,
            "document_id": doc_id,
            "entity": entity,
            "expected_intent": "financial_lookup",
            "gold": {
                "fact_ids": [],
                "expected_value": None,
                "expected_outcome": "ABSTENTION",
                "expected_failure_reason": "METRIC_NOT_FOUND",
            },
            "tags": ["canonical", "stratum4", "abstention", "nonexistent_metric"],
        })

    # --- Cross-entity metric mismatch (5) ---
    # Find metrics only in 1-2 docs
    rare = {m: docs for m, docs in metric_to_docs.items()
            if len(docs) <= 2 and len(m) <= 60 and "/" not in m and ":" not in m
            and not m.startswith("Item")}
    rare_list = list(rare.items())
    rng.shuffle(rare_list)
    mismatch_count = 0
    for metric, source_docs in rare_list:
        if mismatch_count >= 5:
            break
        source_doc = list(source_docs)[0]
        # Pick a different entity to ask about
        other_docs = [d for d in ENTITY_MAP if d not in source_docs]
        if not other_docs:
            continue
        target_doc = rng.choice(other_docs)
        target_entity = ENTITY_MAP[target_doc]
        _, period = DOC_PERIODS[target_doc]
        q_text = "What was %s's %s in %s?" % (target_entity, metric, period)
        questions.append({
            "id": "tv2f01-s4-mismatch-%03d" % (mismatch_count + 1),
            "stratum": "adversarial_abstention",
            "question": q_text,
            "document_id": target_doc,
            "entity": target_entity,
            "expected_intent": "financial_lookup",
            "gold": {
                "fact_ids": [],
                "expected_value": None,
                "expected_outcome": "ABSTENTION",
                "expected_failure_reason": "METRIC_ENTITY_MISMATCH",
                "metric_exists_in": list(source_docs),
            },
            "tags": ["canonical", "stratum4", "abstention", "metric_mismatch"],
        })
        mismatch_count += 1

    # --- Impossible calculation (5) ---
    impossible_calcs = [
        ("aapl_fy2025", "Apple", "iPhone", "FY2022", "FY2023",
         "What was Apple's year-over-year growth rate in iPhone revenue from FY2022 to FY2023?"),
        ("jpm_fy2025", "JPMorganChase", "Net interest income", "FY2021", "FY2022",
         "Calculate JPMorganChase's growth rate in Net interest income from FY2021 to FY2022."),
        ("ko_fy2025", "The Coca-Cola Company", "Total net operating revenues", "FY2020", "FY2021",
         "What was The Coca-Cola Company's percentage change in Total net operating revenues from FY2020 to FY2021?"),
        ("nvda_fy2025", "NVIDIA", "Cost of revenue", "FY2019", "FY2020",
         "Calculate the growth rate in NVIDIA's Cost of revenue between FY2019 and FY2020."),
        ("tsla_fy2025", "Tesla", "Automotive sales", "FY2018", "FY2019",
         "What was Tesla's year-over-year change in Automotive sales from FY2018 to FY2019?"),
    ]
    for i, (doc_id, entity, metric, p1, p2, q_text) in enumerate(impossible_calcs):
        questions.append({
            "id": "tv2f01-s4-impossible-%03d" % (i + 1),
            "stratum": "adversarial_abstention",
            "question": q_text,
            "document_id": doc_id,
            "entity": entity,
            "expected_intent": "financial_calculation",
            "gold": {
                "fact_ids": [],
                "expected_value": None,
                "expected_outcome": "ABSTENTION",
                "expected_failure_reason": "MISSING_OPERAND",
                "operation": "growth_rate",
                "period_current": p2,
                "period_previous": p1,
            },
            "tags": ["canonical", "stratum4", "abstention", "impossible_calculation"],
        })

    return questions


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------

def check_contamination(
    questions: list[dict[str, Any]],
    category_c: list[str],
) -> list[dict[str, Any]]:
    """Flag any questions that match Category C text."""
    c_fingerprints = {_question_fingerprint(q) for q in category_c}
    contaminated: list[dict[str, Any]] = []
    for q in questions:
        fp = _question_fingerprint(q["question"])
        if fp in c_fingerprints:
            contaminated.append({"id": q["id"], "question": q["question"], "match_type": "exact_text"})
    return contaminated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fact-store", type=Path, required=True)
    parser.add_argument("--category-c", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    print("Loading fact store from %s ..." % args.fact_store)
    facts = load_facts(args.fact_store)
    print("  Loaded %d facts" % len(facts))

    usable, by_doc, by_doc_metric_period, metric_to_docs, doc_metric_periods = build_indices(facts)
    print("  Usable facts (metric+period+numeric): %d" % len(usable))

    # Build strata
    print("Building Stratum 1 (factual lookup) ...")
    s1 = build_stratum1(by_doc, rng)
    print("  Generated %d questions" % len(s1))

    print("Building Stratum 2 (arithmetic calculation) ...")
    s2 = build_stratum2(by_doc, by_doc_metric_period, doc_metric_periods, rng)
    print("  Generated %d questions" % len(s2))

    print("Building Stratum 3 (cross-entity comparison) ...")
    s3 = build_stratum3(by_doc_metric_period, metric_to_docs, rng)
    print("  Generated %d questions" % len(s3))

    print("Building Stratum 4 (adversarial abstention) ...")
    s4 = build_stratum4(by_doc, metric_to_docs, rng)
    print("  Generated %d questions" % len(s4))

    all_questions = s1 + s2 + s3 + s4
    print()
    print("Total questions: %d" % len(all_questions))
    print("  Stratum 1 (factual_lookup): %d" % len(s1))
    print("  Stratum 2 (arithmetic_calculation): %d" % len(s2))
    print("  Stratum 3 (cross_entity_comparison): %d" % len(s3))
    print("  Stratum 4 (adversarial_abstention): %d" % len(s4))

    # Contamination check
    contaminated: list[dict] = []
    if args.category_c and args.category_c.exists():
        print()
        print("Running contamination check against %s ..." % args.category_c)
        category_c_questions = json.loads(args.category_c.read_text(encoding="utf-8"))
        contaminated = check_contamination(all_questions, category_c_questions)
        print("  Contaminated: %d" % len(contaminated))
        for c in contaminated:
            print("    CONTAMINATED: %s -> %s" % (c["id"], c["question"][:80]))
    else:
        print("  (No Category C file provided, skipping contamination check)")

    # Build manifest
    stratum_dist = {
        "factual_lookup": len(s1),
        "arithmetic_calculation": len(s2),
        "cross_entity_comparison": len(s3),
        "adversarial_abstention": len(s4),
    }
    entity_dist = {}
    for q in all_questions:
        e = q.get("entity") or "cross_entity"
        entity_dist[e] = entity_dist.get(e, 0) + 1

    manifest = {
        "stage": "TV2-FINAL-01",
        "dataset_name": "canonical-eval-v1",
        "total_questions": len(all_questions),
        "stratum_distribution": stratum_dist,
        "entity_distribution": entity_dist,
        "contamination_audit": {
            "category_c_count": len(json.loads(args.category_c.read_text(encoding="utf-8"))) if args.category_c and args.category_c.exists() else 0,
            "contaminated_count": len(contaminated),
            "contaminated_ids": [c["id"] for c in contaminated],
            "status": "CLEAN" if len(contaminated) == 0 else "CONTAMINATED",
        },
        "seed": args.seed,
        "fact_store_path": str(args.fact_store),
        "execution_inputs": {
            "questions_read": False,
            "gold_read": False,
            "answer_text_read": False,
            "conversation_history_read": False,
            "model_called": False,
        },
    }

    if args.dry_run:
        print()
        print("DRY RUN — manifest:")
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    # Write outputs
    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = args.out_dir / "canonical-eval-v1.jsonl"
    gold_path = args.out_dir / "gold-evidence-v1.jsonl"
    manifest_path = args.out_dir / "dataset-manifest.json"

    with eval_path.open("w", encoding="utf-8", newline="\n") as f:
        for q in all_questions:
            # Write question-only record (without gold for blind execution)
            q_only = {k: v for k, v in q.items() if k != "gold"}
            f.write(json.dumps(q_only, ensure_ascii=False, sort_keys=True) + "\n")

    with gold_path.open("w", encoding="utf-8", newline="\n") as f:
        for q in all_questions:
            gold_rec = {"id": q["id"], **q["gold"]}
            f.write(json.dumps(gold_rec, ensure_ascii=False, sort_keys=True) + "\n")

    # Add hashes to manifest
    manifest["output_files"] = {
        "eval_jsonl": str(eval_path),
        "eval_jsonl_sha256": _sha256_file(eval_path),
        "gold_jsonl": str(gold_path),
        "gold_jsonl_sha256": _sha256_file(gold_path),
    }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Write SHA-256 sidecar
    sha_path = eval_path.with_suffix(eval_path.suffix + ".sha256")
    sha_path.write_text(manifest["output_files"]["eval_jsonl_sha256"] + "\n", encoding="utf-8")

    print()
    print("Written:")
    print("  %s (%d questions)" % (eval_path, len(all_questions)))
    print("  %s" % gold_path)
    print("  %s" % manifest_path)
    print("  %s" % sha_path)
    print()
    print("Contamination status: %s" % manifest["contamination_audit"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
