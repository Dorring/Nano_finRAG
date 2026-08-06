from __future__ import annotations

import re
from collections.abc import Callable

from .evidence_shapes import detect_bucket_label
from .query_plan_models import OperandSlot


_ISSUER_WORDS = re.compile(
    r"\b(?:apple|microsoft|nvidia|jpmorgan(?:\s+chase)?|tesla|visa|pfizer|"
    r"the\s+coca[- ]cola\s+company)\b(?:'s)?",
    re.IGNORECASE,
)
_PERIOD_WORDS = re.compile(r"\b(?:FY\s*|fiscal\s+|year ended\s+|during\s+)?(?:19|20)\d{2}\b", re.IGNORECASE)


def _clean_phrase(value: str) -> str:
    value = _ISSUER_WORDS.sub(" ", value)
    value = _PERIOD_WORDS.sub(" ", value)
    value = re.sub(r"\b(?:reported|report|according to|the company|company's|company)\b", " ", value, flags=re.I)
    value = re.sub(r"\b(?:what|was|were|is|are|how much|how many|percentage|percent)\b", " ", value, flags=re.I)
    value = re.sub(r"\s+", " ", value).strip(" ,:;.-")
    return value


def _question_metric_candidates(question: str, operation: str | None, fallback: list[str]) -> list[str]:
    text = " ".join((question or "").split())
    lowered = text.lower()
    if operation == "growth_rate":
        match = re.search(r"\bof\s+(.+?)(?:\s+reported|\s+from\s+FY|\s+between\s+FY|\?|$)", text, re.I)
        if match:
            phrase = _clean_phrase(match.group(1).lstrip("of "))
            if phrase:
                return [phrase]
    if operation == "percentage_share":
        match = re.search(r"percentage\s+of\s+(.+?)\s+came\s+from\s+(.+?)(?:\?|$)", text, re.I)
        if match:
            denominator = _clean_phrase(match.group(1))
            numerator = _clean_phrase(match.group(2))
            return [x for x in (numerator, denominator) if x]
    if operation == "difference" and len(fallback) < 2:
        match = re.search(r"(?:difference\s+between|compare)\s+(.+?)\s+(?:and|with)\s+(.+?)(?:\?|$)", text, re.I)
        if match:
            return [_clean_phrase(match.group(1)), _clean_phrase(match.group(2))]
    if operation in {"gross_margin", "net_margin", "debt_ratio"} and len(fallback) < 2:
        defaults = {
            "gross_margin": ["gross profit", "revenue"],
            "net_margin": ["net income", "revenue"],
            "debt_ratio": ["debt", "total assets"],
        }
        return defaults[operation]
    if operation in {"sum", "average"} and len(fallback) < 2:
        marker = "sum of" if operation == "sum" else "average of"
        if marker in lowered:
            tail = text[lowered.index(marker) + len(marker):]
            pieces = re.split(r"\s*(?:,|\band\b|\+|&)\s*", tail, flags=re.I)
            cleaned = [_clean_phrase(piece) for piece in pieces if _clean_phrase(piece)]
            if len(cleaned) >= 2:
                return cleaned
    cleaned_fallback = [_clean_phrase(x) for x in fallback if _clean_phrase(x)]
    return cleaned_fallback


def _temporal_kind(question: str, *, bucket_label: str | None, period: str | None) -> str:
    if bucket_label:
        return "bucket"
    lowered = (question or "").lower()
    if any(x in lowered for x in ("change from", "increase from", "decrease from", "growth", "compared with", "year-over-year", "yoy")):
        return "comparison"
    if any(x in lowered for x in ("as of", "balance at", " at december", "at june", "at september")):
        return "point"
    if any(x in lowered for x in ("year ended", "three months ended", "nine months ended", "during fy", "for fy")):
        return "duration"
    return "unspecified" if period else "unspecified"


def _concepts(phrase: str, resolver: Callable[[str], tuple[str, ...]] | None) -> tuple[str, ...]:
    return resolver(phrase) if resolver is not None and phrase else ()


def build_operand_slots(profile, question: str, resolver: Callable[[str], tuple[str, ...]] | None = None) -> tuple[OperandSlot, ...]:
    fallback = [item.raw_text for item in profile.metric_phrases]
    metrics = _question_metric_candidates(question, profile.operation, fallback)
    periods = [item.normalized_period for item in profile.periods if item.normalized_period]
    bucket = detect_bucket_label(question)
    operation = profile.operation
    slots: list[OperandSlot] = []

    if profile.task_type == "calculation_multi_operand":
        if operation == "growth_rate":
            ordered = sorted(set(periods), key=lambda x: int(x[-4:]))
            phrase = metrics[0] if metrics else (fallback[0] if fallback else "")
            if ordered:
                slots = [
                    OperandSlot("current_period", "current_period", phrase, _concepts(phrase, resolver), ordered[-1], "comparison", None, None, "atomic_fact"),
                    OperandSlot("base_period", "base_period", phrase, _concepts(phrase, resolver), ordered[0], "comparison", None, None, "atomic_fact"),
                ]
        elif operation == "percentage_share":
            period = periods[0] if periods else None
            roles = ("numerator", "denominator")
            for index, phrase in enumerate(metrics[:2]):
                slots.append(OperandSlot(roles[index], roles[index], phrase, _concepts(phrase, resolver), period, _temporal_kind(question, bucket_label=bucket, period=period), None, None, "atomic_fact"))
        else:
            roles_by_operation = {
                "difference": ("minuend", "subtrahend"),
                "gross_margin": ("gross_profit", "revenue"),
                "net_margin": ("net_income", "revenue"),
                "debt_ratio": ("debt", "total_assets"),
                "sum": tuple(f"item_{i}" for i in range(1, len(metrics) + 1)),
                "average": tuple(f"item_{i}" for i in range(1, len(metrics) + 1)),
                "scale_conversion": ("source_value",),
            }
            roles = roles_by_operation.get(operation or "", tuple(f"operand_{i}" for i in range(1, len(metrics) + 1)))
            period = periods[0] if periods else None
            for index, phrase in enumerate(metrics):
                role = roles[index] if index < len(roles) else f"operand_{index + 1}"
                shape = "bucket_fact" if bucket else "atomic_fact"
                slots.append(OperandSlot(f"{role}", role, phrase, _concepts(phrase, resolver), period, _temporal_kind(question, bucket_label=bucket, period=period), bucket, None, shape))
    elif profile.task_type == "single_metric_multi_period":
        phrase = metrics[0] if metrics else ""
        for index, period in enumerate(periods, start=1):
            slots.append(OperandSlot(f"period_{index}", f"period_{index}", phrase, _concepts(phrase, resolver), period, _temporal_kind(question, bucket_label=bucket, period=period), bucket, None, "atomic_fact"))
    elif profile.task_type == "multi_metric_comparison":
        period = periods[0] if periods else None
        for index, phrase in enumerate(metrics, start=1):
            role = "left" if index == 1 else "right" if index == 2 else f"metric_{index}"
            slots.append(OperandSlot(f"metric_{role}", role, phrase, _concepts(phrase, resolver), period, _temporal_kind(question, bucket_label=bucket, period=period), bucket, None, "atomic_fact"))
    elif profile.task_type in {"table_single_fact", "general_single_fact"}:
        phrase = metrics[0] if metrics else ""
        period = periods[0] if periods else None
        shape = "bucket_fact" if bucket else "atomic_fact"
        slots.append(OperandSlot("fact", "value", phrase, _concepts(phrase, resolver), period, _temporal_kind(question, bucket_label=bucket, period=period), bucket, None, shape))
    return tuple(slots)
