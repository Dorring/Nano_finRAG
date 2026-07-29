"""Independent question and label fingerprints for evaluation cases.

NF37 reused the entire cases-file digest for both ``question_hash`` and
``label_hash``. A single file hash cannot prove that questions and labels
remained unchanged independently: a label-only edit would also change the
question hash, and vice versa. This module derives two separate hashes from
normalized payloads so question-only and label-only changes are detectable.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from src.evaluation.evaluation import EvaluationCase


def stable_json_hash(value: Any) -> str:
    """Hash a JSON-serializable value with a stable canonical form."""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_question_payload(cases: Iterable[EvaluationCase]) -> list[dict[str, Any]]:
    """Return only question identity fields, ordered by case_id."""
    return [
        {
            "case_id": case.case_id,
            "question": case.question,
            "document_scope": sorted(case.document_names or []),
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]


def build_label_payload(cases: Iterable[EvaluationCase]) -> list[dict[str, Any]]:
    """Return only expected-answer/label fields, ordered by case_id."""
    return [
        {
            "case_id": case.case_id,
            "expected_answer_contains": list(case.expected_answer_contains),
            "expected_numbers": list(case.expected_numbers),
            "expected_sources": [source.to_stable_dict() for source in case.expected_sources],
            "expected_no_answer": case.expected_no_answer,
            "expected_calculations": [calc.to_dict() for calc in case.expected_calculations],
            "expected_intent": case.expected_intent,
        }
        for case in sorted(cases, key=lambda item: item.case_id)
    ]


def question_fingerprint(cases: Iterable[EvaluationCase]) -> str:
    return stable_json_hash(build_question_payload(cases))


def label_fingerprint(cases: Iterable[EvaluationCase]) -> str:
    return stable_json_hash(build_label_payload(cases))
