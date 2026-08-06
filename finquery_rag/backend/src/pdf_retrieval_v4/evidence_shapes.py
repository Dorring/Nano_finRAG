from __future__ import annotations

import re

from .query_plan_models import EvidenceShape


_BUCKET_PATTERNS = (
    r"less than\s+one\s+year",
    r"one\s*(?:-|–|to)\s*three\s+years?",
    r"three\s*(?:-|–|to)\s*five\s+years?",
    r"thereafter",
    r"past\s+due\s+\d+\s*(?:-|–|to)\s*\d+\s+days?",
    r"investment\s+grade",
    r"non[- ]investment\s+grade",
)


def detect_bucket_label(question: str) -> str | None:
    text = " ".join((question or "").split())
    for pattern in _BUCKET_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)
    return None


def classify_evidence_shapes(
    task_type: str,
    operation: str | None,
    *,
    bucket_label: str | None = None,
    metric_count: int = 0,
    period_count: int = 0,
) -> tuple[EvidenceShape, ...]:
    """Return a fixed, ordered evidence-shape contract for a Query Profile."""
    if task_type == "unsupported":
        return ("raw_fallback",)
    if task_type == "narrative_or_note":
        return ("narrative_section", "raw_fallback")
    if bucket_label:
        if task_type == "table_single_fact":
            return ("table_context", "row_matrix", "bucket_fact", "raw_fallback")
        return ("table_context", "row_matrix", "bucket_fact", "multi_operand_set", "raw_fallback")
    if task_type in {"single_metric_multi_period", "multi_metric_comparison", "calculation_multi_operand"}:
        shapes: list[EvidenceShape] = ["table_context", "row_matrix", "atomic_fact", "multi_operand_set"]
        if operation in {"difference", "growth_rate"}:
            shapes.insert(3, "comparison_fact")
        shapes.append("raw_fallback")
        return tuple(shapes)
    if task_type == "table_single_fact":
        return ("table_context", "row_matrix", "atomic_fact", "raw_fallback")
    if task_type == "general_single_fact":
        shapes: list[EvidenceShape] = ["narrative_section", "table_context", "row_matrix", "raw_fallback"]
        if metric_count == 1 and period_count == 1:
            shapes.insert(3, "atomic_fact")
        return tuple(shapes)
    return ("raw_fallback",)
