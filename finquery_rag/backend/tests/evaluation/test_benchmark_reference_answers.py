from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluation.benchmark_foundation import load_jsonl


ROOT = Path(__file__).parents[2]
DATA = ROOT / "benchmarks" / "financial_rag_v1" / "data"


def test_reference_answer_files_are_independent_and_complete():
    questions = load_jsonl(DATA / "questions.reference.jsonl")
    labels = load_jsonl(DATA / "labels.reference.jsonl")
    assert len(questions) == len(labels) == 72
    assert sum(not label["expected_no_answer"] for label in labels) == 64
    assert sum(bool(label.get("calculation")) for label in labels) == 11
    assert sum(
        label["expected_answer"].get("value_type") == "composite"
        for label in labels
    ) == 5


def test_reference_status_does_not_open_golden_gate():
    labels = load_jsonl(DATA / "labels.draft.jsonl")
    reviews = load_jsonl(DATA / "review-status.jsonl")
    answerable = [label for label in labels if not label["expected_no_answer"]]
    no_answer = [label for label in labels if label["expected_no_answer"]]
    assert all(
        label["expected_answer"]["answer_key_status"] == "verified_reference"
        for label in answerable
    )
    assert all(
        label["expected_answer"]["answer_key_status"] == "verified_reference_no_answer"
        for label in no_answer
    )
    assert all(label["label_status"] == "draft" for label in labels)
    assert all(not label["ready_for_golden"] for label in labels)
    assert all(not source["source_verified"] for label in labels for source in label["expected_sources"])
    assert all(not item["ready_for_golden"] for item in reviews)


def test_reference_manifest_has_distinct_hashes_and_pending_identity():
    manifest = json.loads(
        (DATA / "reference-answer-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "reference_only"
    assert manifest["question_count"] == 72
    assert manifest["answerable_count"] == 64
    assert manifest["no_answer_count"] == 8
    assert manifest["calculation_count"] == 11
    assert manifest["source_verified_count"] == 0
    assert manifest["candidate_identity_count"] == 0
    assert manifest["golden_promotion_allowed"] is False
    assert len(manifest["question_hash"]) == 64
    assert len(manifest["reference_answer_hash"]) == 64
