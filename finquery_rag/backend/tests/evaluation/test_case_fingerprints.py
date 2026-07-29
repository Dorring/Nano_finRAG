"""Tests for independent question and label fingerprints.

NF37 previously reused the entire cases-file digest for both question_hash
and label_hash. These tests verify that the two fingerprints are now derived
from independent normalized payloads, so question-only and label-only edits
are detectable separately.
"""
from dataclasses import replace

from src.evaluation.case_fingerprints import (
    label_fingerprint,
    question_fingerprint,
    stable_json_hash,
)
from src.evaluation.evaluation import EvaluationCase, ExpectedSource


def _base_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="c1",
            question="What is the revenue?",
            document_names=("report_a.pdf",),
            expected_sources=(
                ExpectedSource(filename="report_a.pdf", page=3),
                ExpectedSource(filename="report_a.pdf", page=4),
            ),
            expected_numbers=("100",),
            expected_answer_contains=("100",),
        ),
        EvaluationCase(
            case_id="c2",
            question="How much cash?",
            document_names=("report_b.pdf",),
            expected_sources=(ExpectedSource(filename="report_b.pdf", page=1),),
            expected_numbers=("42",),
            expected_no_answer=False,
        ),
    ]


def test_question_and_label_fingerprints_are_distinct():
    cases = _base_cases()
    assert question_fingerprint(cases) != label_fingerprint(cases)


def test_question_hash_changes_when_question_changes():
    cases = _base_cases()
    original = question_fingerprint(cases)
    modified = [replace(cases[0], question="What is the net revenue?")] + cases[1:]
    assert question_fingerprint(modified) != original


def test_label_hash_does_not_change_for_question_only_change():
    cases = _base_cases()
    original_label = label_fingerprint(cases)
    modified = [replace(cases[0], question="Completely different question")] + cases[1:]
    assert label_fingerprint(modified) == original_label


def test_label_hash_changes_when_label_changes():
    cases = _base_cases()
    original = label_fingerprint(cases)
    modified = [
        replace(cases[0], expected_numbers=("999",)),
    ] + cases[1:]
    assert label_fingerprint(modified) != original


def test_question_hash_does_not_change_for_label_only_change():
    cases = _base_cases()
    original_question = question_fingerprint(cases)
    modified = [replace(cases[0], expected_numbers=("999",), expected_answer_contains=("999",))] + cases[1:]
    assert question_fingerprint(modified) == original_question


def test_label_hash_changes_when_source_page_changes():
    cases = _base_cases()
    original = label_fingerprint(cases)
    modified = [
        replace(
            cases[0],
            expected_sources=(ExpectedSource(filename="report_a.pdf", page=99),),
        )
    ] + cases[1:]
    assert label_fingerprint(modified) != original


def test_label_hash_changes_when_no_answer_flag_flips():
    cases = _base_cases()
    original = label_fingerprint(cases)
    modified = [cases[0], replace(cases[1], expected_no_answer=True)]
    assert label_fingerprint(modified) != original


def test_fingerprint_is_order_independent():
    cases = _base_cases()
    reversed_cases = list(reversed(cases))
    assert question_fingerprint(reversed_cases) == question_fingerprint(cases)
    assert label_fingerprint(reversed_cases) == label_fingerprint(cases)


def test_page_int_and_string_produce_same_fingerprint():
    """page=1 and page="1" are semantically the same citation target."""
    int_page = [EvaluationCase(case_id="c", question="q", expected_sources=(ExpectedSource(filename="a.pdf", page=1),))]
    str_page = [EvaluationCase(case_id="c", question="q", expected_sources=(ExpectedSource(filename="a.pdf", page="1"),))]
    assert label_fingerprint(int_page) == label_fingerprint(str_page)


def test_stable_json_hash_is_deterministic():
    payload = {"b": 1, "a": 2, "c": [3, 2, 1]}
    assert stable_json_hash(payload) == stable_json_hash(payload)


def test_stable_json_hash_key_order_independent():
    assert stable_json_hash({"a": 1, "b": 2}) == stable_json_hash({"b": 2, "a": 1})
