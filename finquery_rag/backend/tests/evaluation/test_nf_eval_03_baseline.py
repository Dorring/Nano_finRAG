from __future__ import annotations

from pathlib import Path

from scripts.evaluation.run_nf_eval_03_baseline import (
    BENCHMARK,
    _answer_correct,
    _load_inputs,
    _numeric_matches,
    _stage_metrics,
)


def test_nf_eval_03_loads_the_frozen_72_case_golden_set() -> None:
    inputs = _load_inputs()

    assert len(inputs.questions) == 72
    assert len(inputs.labels_by_id) == 72
    assert inputs.manifest["golden_ready"] is True
    assert inputs.manifest["question_hash"] == (
        "3b1433ea7546020ec7ab90a9cb700bb58d90aa040f65e6610d6bc5b1330e97b7"
    )
    assert inputs.manifest["reference_answer_hash"] == (
        "181c9ed45ae904353431ef6d427b139e1d5f0b8f437c2fdd9f25e2f9b7abb7a7"
    )


def test_nf_eval_03_scope_has_exactly_eight_documents() -> None:
    inputs = _load_inputs()
    document_ids = {item["document_id"] for item in inputs.corpus["documents"]}

    assert inputs.corpus["document_count"] == 8
    assert len(document_ids) == 8
    assert all(
        set(question["document_scope"]).issubset(document_ids)
        for question in inputs.questions
    )


def test_numeric_matching_normalizes_display_scale_and_percentage_points() -> None:
    assert _numeric_matches(
        "Apple reported $416.161 billion in FY2025.",
        {
            "canonical_value": "416161000000",
            "currency": "USD",
            "unit": "currency",
            "tolerance": "0",
        },
    )
    assert _numeric_matches(
        "The growth rate was 6.43%.",
        {
            "canonical_value": "6.4255",
            "unit": "percentage",
            "percentage_representation": "percentage_points",
            "tolerance": "0.01",
        },
    )


def test_composite_matching_requires_every_component() -> None:
    answer = {
        "value_type": "composite",
        "canonical_value": None,
        "tolerance": "0",
        "component_values": [
            {
                "canonical_value": "115186000000",
                "display_value": "$115.186 billion",
                "currency": "USD",
                "unit": "currency",
            },
            {
                "canonical_value": "75.0",
                "display_value": "75.0%",
                "unit": "percentage",
            },
        ],
    }

    assert _numeric_matches("Data Center was $115.186 billion and gross margin was 75.0%.", answer)
    assert not _numeric_matches("Data Center was $115.186 billion.", answer)


def test_no_answer_requires_refusal_without_numeric_claims() -> None:
    question = {"answerable": False}
    label = {"expected_no_answer": True, "expected_answer": {"text": "not disclosed"}}

    assert _answer_correct("The report does not disclose this metric.", question, label)
    assert not _answer_correct("The metric was 42%.", question, label)


def test_stage_metrics_use_answerable_source_denominator() -> None:
    labels = [
        {
            "case_id": "answerable",
            "expected_no_answer": False,
            "expected_sources": [{"filename": "a.pdf", "page": 2, "evidence_id": "e2"}],
        },
        {
            "case_id": "no-answer",
            "expected_no_answer": True,
            "expected_sources": [],
        },
    ]
    rankings = {
        "answerable": [
            {"filename": "a.pdf", "page": 1, "evidence_id": "other"},
            {"filename": "a.pdf", "page": 2, "evidence_id": "e2"},
        ],
        "no-answer": [],
    }

    metrics = _stage_metrics(labels, rankings, 5)

    assert metrics["case_hit"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert metrics["source_recall"] == {"count": 1, "denominator": 1, "rate": 1.0}
    assert metrics["mrr"] == 0.5


def test_benchmark_paths_are_inside_backend() -> None:
    assert Path(BENCHMARK, "data", "questions.golden.jsonl").is_file()
