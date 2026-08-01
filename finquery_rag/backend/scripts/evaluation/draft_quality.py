"""Semantic and annotation-quality gates for benchmark Draft records.

These checks operate only on benchmark authoring data.  They deliberately do
not infer an answer from the index and do not promote a Draft to Golden.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


GENERIC_SECTIONS = {
    "single_document",
    "table_fact",
    "multi_source",
    "calculation",
    "unit_scale_period_trap",
    "draft_pending_review",
}
GENERIC_TABLE_TITLES = GENERIC_SECTIONS | {"table", "table title"}
PERIOD_RE = re.compile(r"\bFY20\d{2}\b", re.IGNORECASE)


def normalize_period(value: Any) -> str | None:
    if value is None:
        return None
    match = re.search(r"(?:FY)?(20\d{2})", str(value), flags=re.IGNORECASE)
    return f"FY{match.group(1)}" if match else None


def requested_periods(question: Mapping[str, Any]) -> list[str]:
    explicit = question.get("requested_periods")
    if isinstance(explicit, list) and explicit:
        return [normalize_period(item) for item in explicit if normalize_period(item)]
    requested = question.get("requested_period")
    if requested:
        normalized = normalize_period(requested)
        if normalized:
            return [normalized]
    return [normalize_period(item) for item in PERIOD_RE.findall(str(question.get("question", ""))) if normalize_period(item)]


def semantic_group(question: Mapping[str, Any]) -> str:
    explicit = question.get("semantic_group")
    if explicit:
        return str(explicit)
    case_id = str(question.get("case_id", ""))
    company = str(question.get("company", ""))
    suffix = case_id.rsplit("_", 1)[-1]
    if suffix in {"001", "002", "003"}:
        return f"{company}:generated_total_metric"
    if suffix in {"004", "005"}:
        return f"{company}:generated_table_metric"
    return re.sub(r"\W+", " ", str(question.get("question", "")).casefold()).strip()


def _source_periods(label: Mapping[str, Any]) -> list[str]:
    periods: list[str] = []
    for source in label.get("expected_sources", []):
        if not isinstance(source, Mapping):
            continue
        for key in ("period", "column_header"):
            normalized = normalize_period(source.get(key))
            if normalized and normalized not in periods:
                periods.append(normalized)
    return periods


def _answer_periods(label: Mapping[str, Any]) -> list[str]:
    answer = label.get("expected_answer", {})
    values = answer.get("periods", []) if isinstance(answer, Mapping) else []
    periods = [normalize_period(value) for value in values if normalize_period(value)]
    if not periods and isinstance(answer, Mapping):
        normalized = normalize_period(answer.get("period"))
        if normalized:
            periods.append(normalized)
    return periods


def _ambiguous_metric(question: Mapping[str, Any]) -> bool:
    text = str(question.get("question", "")).casefold()
    if "segment revenue" in text and not any(
        term in text
        for term in (
            "productivity and business processes",
            "intelligent cloud",
            "more personal computing",
            "data center",
            "automotive",
            "energy generation",
            "emea",
        )
    ):
        return True
    if "product revenue" in text and not any(
        product in text
        for product in ("comirnaty", "paxlovid", "eliquis", "prevnar")
    ):
        return True
    if "europe segment" in text or "payment volume" in text:
        return True
    if "gross margin" in text and not any(
        term in text for term in ("gaap", "non-gaap", "percentage", "amount")
    ):
        return True
    return False


def _comparison_contract_error(question: Mapping[str, Any]) -> bool:
    if not question.get("requires_multiple_sources"):
        return False
    contract = question.get("output_contract")
    return contract not in {
        "report_both",
        "difference",
        "percentage_share",
        "growth_rate_comparison",
        "higher_and_difference",
    }


def quality_audit(
    questions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    questions_list = list(questions)
    labels_by_id = {str(label.get("case_id")): label for label in labels}

    groups: defaultdict[str, list[str]] = defaultdict(list)
    for question in questions_list:
        groups[semantic_group(question)].append(str(question.get("case_id")))
    duplicate_groups = {
        key: values for key, values in groups.items() if len(values) > 1
    }
    semantic_duplicate_count = sum(len(values) - 1 for values in duplicate_groups.values())

    ambiguous_cases = [
        str(question.get("case_id"))
        for question in questions_list
        if _ambiguous_metric(question)
    ]
    undefined_comparison_cases = [
        str(question.get("case_id"))
        for question in questions_list
        if _comparison_contract_error(question)
    ]
    period_mismatch_cases: list[str] = []
    placeholder_sections: list[str] = []
    placeholder_tables: list[str] = []
    missing_review_plan: list[str] = []
    for question in questions_list:
        case_id = str(question.get("case_id"))
        label = labels_by_id.get(case_id, {})
        requested = set(requested_periods(question))
        answer_periods = set(_answer_periods(label))
        source_periods = set(_source_periods(label))
        if not question.get("answerable", True):
            requested = set()
        elif requested and (
            not requested.issubset(answer_periods)
            or not requested.issubset(source_periods)
        ):
            period_mismatch_cases.append(case_id)
        for source in label.get("expected_sources", []):
            if not isinstance(source, Mapping):
                continue
            if str(source.get("section", "")).casefold() in GENERIC_SECTIONS:
                placeholder_sections.append(case_id)
            if str(source.get("table_title", "")).casefold() in GENERIC_TABLE_TITLES:
                placeholder_tables.append(case_id)
        if not isinstance(label.get("review_plan"), Mapping):
            missing_review_plan.append(case_id)

    no_answer_questions = [
        question for question in questions_list if not question.get("answerable", True)
    ]
    no_answer_types = Counter(
        str(question.get("no_answer_type", "missing"))
        for question in no_answer_questions
    )
    no_answer_template_count = sum(
        count - 1 for count in no_answer_types.values() if count > 1
    )
    answer_key_missing_cases: list[str] = []
    answer_key_status_missing_cases: list[str] = []
    source_unverified_count = 0
    for question in questions_list:
        if not question.get("answerable", True):
            continue
        case_id = str(question.get("case_id"))
        label = labels_by_id.get(case_id, {})
        answer = label.get("expected_answer", {})
        has_scalar = isinstance(answer, Mapping) and answer.get("canonical_value") is not None
        has_components = isinstance(answer, Mapping) and bool(answer.get("component_values"))
        if not (has_scalar or has_components):
            answer_key_missing_cases.append(case_id)
        if answer.get("answer_key_status") != "entered_unverified":
            answer_key_status_missing_cases.append(case_id)
        source_unverified_count += sum(
            int(not bool(source.get("source_verified")))
            for source in label.get("expected_sources", [])
            if isinstance(source, Mapping)
        )
    golden_case_count = sum(
        int(labels_by_id.get(str(question.get("case_id")), {}).get("label_status") in {"golden", "sealed"})
        for question in questions_list
    )
    return {
        "question_count": len(questions_list),
        "semantic_duplicate_count": semantic_duplicate_count,
        "semantic_duplicate_groups": duplicate_groups,
        "ambiguous_metric_count": len(ambiguous_cases),
        "ambiguous_metric_cases": ambiguous_cases,
        "undefined_comparison_count": len(undefined_comparison_cases),
        "undefined_comparison_cases": undefined_comparison_cases,
        "question_label_period_mismatch": len(period_mismatch_cases),
        "period_mismatch_cases": period_mismatch_cases,
        "placeholder_section_count": len(placeholder_sections),
        "placeholder_table_title_count": len(placeholder_tables),
        "missing_review_plan_count": len(missing_review_plan),
        "no_answer_template_count": no_answer_template_count,
        "no_answer_type_counts": dict(no_answer_types),
        "no_answer_count": len(no_answer_questions),
        "answer_key_missing_count": len(answer_key_missing_cases),
        "answer_key_missing_cases": answer_key_missing_cases,
        "answer_key_status_missing_count": len(answer_key_status_missing_cases),
        "answer_key_status_missing_cases": answer_key_status_missing_cases,
        "source_unverified_count": source_unverified_count,
        "golden_case_count": golden_case_count,
        "quality_valid": not any(
            (
                semantic_duplicate_count,
                ambiguous_cases,
                undefined_comparison_cases,
                period_mismatch_cases,
                placeholder_sections,
                placeholder_tables,
                missing_review_plan,
                no_answer_template_count,
                answer_key_missing_cases,
                answer_key_status_missing_cases,
                golden_case_count,
            )
        ),
    }
