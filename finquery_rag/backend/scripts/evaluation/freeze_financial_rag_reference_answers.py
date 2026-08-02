"""Freeze reviewed answer values as a reference-only benchmark layer.

This command updates review metadata only. It does not bind indexed
candidate identities, mark sources verified, or promote any case to Golden.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from scripts.evaluation.annotation_contract import build_annotation_worklist
from scripts.evaluation.benchmark_foundation import (
    corpus_hash,
    load_json,
    load_jsonl,
    stable_json_hash,
    write_jsonl,
)


ROOT = Path(__file__).parents[2]
BENCHMARK = ROOT / "benchmarks" / "financial_rag_v1"
DATA = BENCHMARK / "data"


def _question_hash_payload(question: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": question["case_id"],
        "question": question["question"],
        "document_scope": sorted(question.get("document_scope", [])),
    }


def _answer_hash_payload(label: dict[str, Any]) -> dict[str, Any]:
    answer = dict(label.get("expected_answer") or {})
    answer.pop("answer_key_status", None)
    return {
        "case_id": label["case_id"],
        "expected_answer": answer,
        "calculation": label.get("calculation"),
        "expected_no_answer": bool(label.get("expected_no_answer")),
    }


def _set_review_state(
    question: dict[str, Any],
    label: dict[str, Any],
    review: dict[str, Any],
) -> None:
    answerable = bool(question.get("answerable")) and not bool(label.get("expected_no_answer"))
    answer = dict(label.get("expected_answer") or {})
    answer["answer_key_status"] = (
        "verified_reference" if answerable else "verified_reference_no_answer"
    )
    label["expected_answer"] = answer
    label["label_status"] = "draft"
    label["review_status"] = "unreviewed"
    label["ready_for_golden"] = False
    label["review_notes"] = (
        "Reference answer frozen after PDF recheck; indexed candidate identity and "
        "source verification remain pending. This record is not Golden."
    )

    plan = dict(label.get("review_plan") or {})
    plan["question_reviewed"] = True
    plan["answer_reviewed"] = answerable
    plan["calculation_reviewed"] = (
        True if question.get("requires_calculation") else "not_applicable"
    )
    plan["source_reviewed"] = False
    plan["ready_for_golden"] = False
    label["review_plan"] = plan

    review["question_review_status"] = "reviewed"
    review["answer_review_status"] = "reviewed" if answerable else "pending"
    review["source_review_status"] = "pending" if answerable else "not_applicable"
    review["calculation_review_status"] = (
        "reviewed" if question.get("requires_calculation") else "not_applicable"
    )
    review["negative_evidence_review_status"] = (
        "not_applicable" if answerable else "pending"
    )
    review["verified_source_count"] = 0
    review["all_sources_have_candidate_identity"] = False if answerable else True
    review["ready_for_golden"] = False
    review["reviewer"] = None
    review["reviewed_at"] = None
    review["review_notes"] = (
        "Question/reference-answer state reviewed; source or negative-evidence "
        "Golden gates remain open."
    )


def freeze_reference_answers(
    *,
    corpus_path: Path = BENCHMARK / "corpus.json",
    questions_path: Path = DATA / "questions.draft.jsonl",
    labels_path: Path = DATA / "labels.draft.jsonl",
    review_path: Path = DATA / "review-status.jsonl",
    output_dir: Path = DATA,
) -> dict[str, Any]:
    questions = load_jsonl(questions_path)
    labels = load_jsonl(labels_path)
    reviews = load_jsonl(review_path)
    if not (len(questions) == len(labels) == len(reviews) == 72):
        raise ValueError("reference freeze requires exactly 72 aligned records")

    labels_by_id = {item["case_id"]: item for item in labels}
    reviews_by_id = {item["case_id"]: item for item in reviews}
    if len(labels_by_id) != len(labels) or len(reviews_by_id) != len(reviews):
        raise ValueError("duplicate case_id in reference freeze inputs")

    reference_questions: list[dict[str, Any]] = []
    reference_labels: list[dict[str, Any]] = []
    updated_labels: list[dict[str, Any]] = []
    updated_reviews: list[dict[str, Any]] = []
    for question in questions:
        case_id = question["case_id"]
        label = deepcopy(labels_by_id[case_id])
        review = deepcopy(reviews_by_id[case_id])
        _set_review_state(question, label, review)
        reference_questions.append(deepcopy(question))
        reference_labels.append(deepcopy(label))
        updated_labels.append(label)
        updated_reviews.append(review)

    write_jsonl(output_dir / "questions.reference.jsonl", reference_questions)
    write_jsonl(output_dir / "labels.reference.jsonl", reference_labels)
    write_jsonl(output_dir / "review-status.jsonl", updated_reviews)
    write_jsonl(output_dir / "labels.draft.jsonl", updated_labels)
    write_jsonl(
        output_dir / "annotation-worklist.jsonl",
        build_annotation_worklist(questions, updated_labels, updated_reviews),
    )

    corpus = load_json(corpus_path)
    question_payload = sorted(
        (_question_hash_payload(item) for item in reference_questions),
        key=lambda item: item["case_id"],
    )
    answer_payload = sorted(
        (_answer_hash_payload(item) for item in reference_labels),
        key=lambda item: item["case_id"],
    )
    answerable_count = sum(
        not item["expected_no_answer"] for item in reference_labels
    )
    calculation_count = sum(
        bool(item.get("calculation"))
        for item in reference_labels
        if not item["expected_no_answer"]
    )
    composite_count = sum(
        item.get("expected_answer", {}).get("value_type") == "composite"
        for item in reference_labels
        if not item["expected_no_answer"]
    )
    manifest = {
        "artifact_schema": "financial-rag-v1/reference-answers/v1",
        "benchmark_id": "financial-rag-v1",
        "status": "reference_only",
        "question_count": len(reference_questions),
        "answerable_count": answerable_count,
        "no_answer_count": len(reference_questions) - answerable_count,
        "calculation_count": calculation_count,
        "composite_answer_count": composite_count,
        "question_hash": stable_json_hash(question_payload),
        "reference_answer_hash": stable_json_hash(answer_payload),
        "corpus_hash": corpus.get("corpus_hash") or corpus_hash(corpus["documents"]),
        "source_verified_count": 0,
        "candidate_identity_count": 0,
        "golden_case_count": 0,
        "golden_promotion_allowed": False,
        "source_identity_pending": True,
        "negative_evidence_pending_count": len(reference_questions) - answerable_count,
        "files": [
            "data/questions.reference.jsonl",
            "data/labels.reference.jsonl",
        ],
    }
    manifest_path = output_dir / "reference-answer-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=BENCHMARK / "corpus.json")
    parser.add_argument("--questions", type=Path, default=DATA / "questions.draft.jsonl")
    parser.add_argument("--labels", type=Path, default=DATA / "labels.draft.jsonl")
    parser.add_argument("--review", type=Path, default=DATA / "review-status.jsonl")
    parser.add_argument("--out-dir", type=Path, default=DATA)
    args = parser.parse_args()
    print(json.dumps(
        freeze_reference_answers(
            corpus_path=args.corpus,
            questions_path=args.questions,
            labels_path=args.labels,
            review_path=args.review,
            output_dir=args.out_dir,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
