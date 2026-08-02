"""Annotation workflow contract for the Financial RAG Draft set.

This module only manages human-review metadata.  It deliberately does not
inspect the production index, resolve PDF pages, or promote a Draft to Golden.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Mapping


LEGACY_ACTIONS = {"keep", "rewrite", "replace", "manual_source_review"}
ANSWER_REVIEW_ACTION = "manual_answer_source_review"
NEGATIVE_REVIEW_ACTION = "manual_negative_evidence_review"
READY_QUESTION_STATUS = "ready_for_human_verification"
PERCENTAGE_REPRESENTATION = "percentage_points"


def _is_percentage(question: Mapping[str, Any], answer: Mapping[str, Any]) -> bool:
    return question.get("answer_type") == "percentage" or answer.get("unit") == "percentage"


def _has_candidate_identity(source: Mapping[str, Any]) -> bool:
    candidate_key = source.get("candidate_key")
    if not isinstance(candidate_key, str) or not candidate_key.strip():
        return False
    granularity = source.get("identity_granularity", "candidate_key")
    if granularity not in {"candidate_key", "evidence_id", "row", "chunk"}:
        return False
    if granularity == "chunk" and source.get("identity_limitation") != "row_identity_not_available":
        return False
    return True


def _source_counts(label: Mapping[str, Any]) -> tuple[int, int]:
    sources = label.get("expected_sources", [])
    return len(sources), sum(int(_has_candidate_identity(source)) for source in sources if isinstance(source, Mapping))


def _pdf_source_verified(source: Mapping[str, Any]) -> bool:
    return bool(source.get("pdf_page_verified")) and bool(source.get("pdf_content_verified"))


def ready_for_golden(case: Mapping[str, Any]) -> bool:
    """Return whether one case satisfies every human-review gate."""
    source_count = int(case.get("expected_source_count", 0))
    verified_source_count = int(case.get("verified_source_count", 0))
    return all(
        (
            case.get("question_review_status") == "reviewed",
            case.get("answer_review_status") == "reviewed",
            case.get("source_review_status") == "reviewed",
            case.get("calculation_review_status") in {"reviewed", "not_applicable"},
            case.get("negative_evidence_review_status") in {"reviewed", "not_applicable"},
            source_count == verified_source_count,
            bool(case.get("all_sources_have_candidate_identity")),
        )
    )


def close_annotation_contract(
    questions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
    *,
    revision: int = 2,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Make the current Drafts ready for manual verification.

    Existing `replace`/`rewrite` values are retained only in
    `superseded_review_action`; they are not re-applied to question text.
    """
    labels_by_id = {str(item["case_id"]): item for item in labels}
    reviews_by_id = {str(item["case_id"]): item for item in reviews}
    closed_questions: list[dict[str, Any]] = []
    closed_labels: list[dict[str, Any]] = []
    closed_reviews: list[dict[str, Any]] = []

    for original in questions:
        case_id = str(original["case_id"])
        question = deepcopy(dict(original))
        label = deepcopy(dict(labels_by_id[case_id]))
        review = deepcopy(dict(reviews_by_id[case_id]))
        answerable = bool(question.get("answerable")) and not bool(label.get("expected_no_answer"))
        old_action = (
            question.get("superseded_review_action")
            or label.get("superseded_review_action")
            or question.get("review_action")
            or label.get("review_action")
        )
        if old_action not in LEGACY_ACTIONS:
            old_action = None
        action = ANSWER_REVIEW_ACTION if answerable else NEGATIVE_REVIEW_ACTION
        question.update(
            {
                "review_action": action,
                "question_revision": revision,
                "question_revision_status": READY_QUESTION_STATUS,
                "superseded_review_action": old_action,
            }
        )
        label.update(
            {
                "review_action": action,
                "question_revision": revision,
                "superseded_review_action": old_action,
            }
        )
        plan = dict(label.get("review_plan") or {})
        plan.update(
            {
                "action": action,
                "question_revision": revision,
                "question_revision_status": READY_QUESTION_STATUS,
                "source_review_required": answerable,
            }
        )
        label["review_plan"] = plan

        answer = dict(label.get("expected_answer") or {})
        if answerable:
            if question.get("output_contract") == "report_both":
                answer.update(
                    {
                        "canonical_value": None,
                        "currency": None,
                        "unit": None,
                        "value_type": "composite",
                    }
                )
            if _is_percentage(question, answer):
                answer["percentage_representation"] = PERCENTAGE_REPRESENTATION
                answer["tolerance"] = "0.01" if label.get("calculation") else "0.001"
            elif answer.get("value_type") not in {"composite", None}:
                answer["tolerance"] = "0"
            label["expected_answer"] = answer
        else:
            answer["answer_key_status"] = "pending_negative_evidence"
            label["expected_answer"] = answer
        label["label_status"] = "draft"
        label["review_status"] = "unreviewed"
        label["ready_for_golden"] = False
        review.update(
            {
                "review_action": action,
                "question_revision": revision,
                "question_revision_status": READY_QUESTION_STATUS,
                "superseded_review_action": old_action,
                "question_review_status": "pending",
                "answer_review_status": "pending",
                "source_review_status": "pending" if answerable else "not_applicable",
                "calculation_review_status": "pending" if question.get("requires_calculation") else "not_applicable",
                "negative_evidence_review_status": "not_applicable" if answerable else "pending",
                "expected_source_count": len(label.get("expected_sources", [])),
                "verified_source_count": sum(int(bool(source.get("source_verified"))) for source in label.get("expected_sources", [])),
                "all_sources_have_candidate_identity": all(_has_candidate_identity(source) for source in label.get("expected_sources", [])),
                "reviewer": None,
                "reviewed_at": None,
                "review_notes": None,
                "ready_for_golden": False,
            }
        )
        if not answerable:
            review.update(
                {
                    "searched_terms": list((label.get("no_answer_review") or {}).get("searched_terms", [])),
                    "searched_synonyms": [],
                    "searched_sections": list((label.get("no_answer_review") or {}).get("searched_sections", [])),
                    "full_document_search_completed": False,
                    "positive_match_count": 0,
                    "negative_evidence_reviewed": False,
                }
            )
        closed_questions.append(question)
        closed_labels.append(label)
        closed_reviews.append(review)
    return closed_questions, closed_labels, closed_reviews


def build_annotation_worklist(
    questions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    labels_by_id = {str(item["case_id"]): item for item in labels}
    reviews_by_id = {str(item["case_id"]): item for item in reviews}
    worklist: list[dict[str, Any]] = []
    for question in questions:
        case_id = str(question["case_id"])
        label = labels_by_id[case_id]
        review = reviews_by_id[case_id]
        source_count, verified_count = _source_counts(label)
        pdf_verified_count = sum(
            int(_pdf_source_verified(source))
            for source in label.get("expected_sources", [])
            if isinstance(source, Mapping)
        )
        item = {
            "case_id": case_id,
            "company": question.get("company"),
            "question": question.get("question"),
            "question_review_status": review.get("question_review_status", "pending"),
            "answer_review_status": review.get("answer_review_status", "pending"),
            "source_review_status": review.get("source_review_status", "not_applicable"),
            "calculation_review_status": review.get("calculation_review_status", "not_applicable"),
            "negative_evidence_review_status": review.get("negative_evidence_review_status", "not_applicable"),
            "expected_source_count": source_count,
            "verified_source_count": verified_count,
            "pdf_verified_source_count": pdf_verified_count,
            "all_sources_have_candidate_identity": all(_has_candidate_identity(source) for source in label.get("expected_sources", [])),
            "reviewer": review.get("reviewer"),
            "reviewed_at": review.get("reviewed_at"),
            "review_notes": review.get("review_notes"),
        }
        item["ready_for_golden"] = ready_for_golden(item)
        worklist.append(item)
    return worklist


def annotation_contract_report(
    questions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    questions_list = list(questions)
    labels_list = list(labels)
    reviews_list = list(reviews)
    labels_by_id = {str(item["case_id"]): item for item in labels_list}
    reviews_by_id = {str(item["case_id"]): item for item in reviews_list}
    answerable = [item for item in questions_list if item.get("answerable")]
    no_answer = [item for item in questions_list if not item.get("answerable")]
    source_count = sum(len(labels_by_id[str(item["case_id"])].get("expected_sources", [])) for item in answerable)
    verified_source_count = sum(
        int(bool(source.get("source_verified")))
        for item in answerable
        for source in labels_by_id[str(item["case_id"])].get("expected_sources", [])
    )
    pdf_verified_source_count = sum(
        int(_pdf_source_verified(source))
        for item in answerable
        for source in labels_by_id[str(item["case_id"])].get("expected_sources", [])
        if isinstance(source, Mapping)
    )
    composite_count = 0
    invalid_composite = 0
    percentage_count = 0
    percentage_missing = 0
    tolerance_missing = 0
    stale_actions = 0
    candidate_identity_count = 0
    calculation_pending = 0
    negative_pending = 0
    for question in questions_list:
        case_id = str(question["case_id"])
        label = labels_by_id[case_id]
        answer = label.get("expected_answer", {})
        review = reviews_by_id[case_id]
        if question.get("review_action") in LEGACY_ACTIONS or label.get("review_action") in LEGACY_ACTIONS:
            stale_actions += 1
        candidate_identity_count += sum(int(_has_candidate_identity(source)) for source in label.get("expected_sources", []))
        if question.get("output_contract") == "report_both":
            composite_count += 1
            if not (
                answer.get("value_type") == "composite"
                and answer.get("canonical_value") is None
                and answer.get("currency") is None
                and answer.get("unit") is None
                and len(answer.get("component_values", [])) >= 2
            ):
                invalid_composite += 1
        if question.get("answerable") and _is_percentage(question, answer):
            percentage_count += 1
            if answer.get("percentage_representation") != PERCENTAGE_REPRESENTATION:
                percentage_missing += 1
        if question.get("answerable") and answer.get("value_type") not in {"composite", None} and answer.get("tolerance") is None:
            tolerance_missing += 1
        if review.get("calculation_review_status") == "pending":
            calculation_pending += 1
        if review.get("negative_evidence_review_status") == "pending":
            negative_pending += 1
    ready_count = sum(int(ready_for_golden(item)) for item in build_annotation_worklist(questions_list, labels_list, reviews_list))
    return {
        "case_count": len(questions_list),
        "answerable_case_count": len(answerable),
        "no_answer_case_count": len(no_answer),
        "expected_source_record_count": source_count,
        "verified_source_record_count": verified_source_count,
        "pdf_verified_source_record_count": pdf_verified_source_count,
        "composite_answer_count": composite_count,
        "invalid_composite_contract_count": invalid_composite,
        "percentage_answer_count": percentage_count,
        "percentage_representation_missing_count": percentage_missing,
        "numeric_tolerance_missing_count": tolerance_missing,
        "stale_review_action_count": stale_actions,
        "ready_for_human_verification_count": sum(int(item.get("question_revision_status") == READY_QUESTION_STATUS) for item in questions_list),
        "candidate_identity_record_count": candidate_identity_count,
        "candidate_identity_completeness_rate": candidate_identity_count / source_count if source_count else 1.0,
        "calculation_pending_count": calculation_pending,
        "negative_evidence_pending_count": negative_pending,
        "golden_case_count": sum(int(item.get("label_status") in {"golden", "sealed"}) for item in labels_list),
        "ready_for_golden_count": ready_count,
        "contract_valid": not any((invalid_composite, percentage_missing, tolerance_missing, stale_actions)),
    }


def validate_annotation_contract(
    questions: Iterable[Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    reviews: Iterable[Mapping[str, Any]],
) -> list[str]:
    """Return actionable contract errors without performing source verification."""
    report = annotation_contract_report(questions, labels, reviews)
    errors: list[str] = []
    if report["invalid_composite_contract_count"]:
        errors.append("report_both answers must use the composite contract")
    if report["percentage_representation_missing_count"]:
        errors.append("percentage answers must declare percentage_representation")
    if report["numeric_tolerance_missing_count"]:
        errors.append("numeric/currency/percentage answers must declare tolerance")
    if report["stale_review_action_count"]:
        errors.append("stale review actions remain")
    return errors
