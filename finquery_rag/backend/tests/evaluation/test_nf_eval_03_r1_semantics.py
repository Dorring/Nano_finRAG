import json
import shutil

import pytest

from scripts.evaluation.run_nf_eval_03_r1 import (
    BENCHMARK,
    DATA,
    DEFAULT_NEGATIVE,
    BaselineConfigurationError,
    _load_inputs,
    _metric,
    _repair_flags,
    score_answer_contract,
    source_identity_matches,
)


def _label(answer, *, no_answer=False):
    return {
        "case_id": "case-1",
        "expected_no_answer": no_answer,
        "expected_sources": []
        if no_answer
        else [
            {
                "candidate_key": "candidate:v1:gold",
                "evidence_id": "gold-evidence",
                "filename": "report.pdf",
                "page": 10,
            }
        ],
        "expected_answer": answer,
    }


def _currency_answer():
    return {
        "canonical_value": "416161000000",
        "currency": "USD",
        "display_value": "$416.161 billion",
        "period": "FY2025",
        "scale": "1",
        "tolerance": "0",
        "unit": "currency",
        "value_type": "currency",
    }


def _metric_row(
    *,
    raw_available=True,
    expected_source_count=1,
    matched=0,
    emitted=1,
    correct_emitted=0,
    raw_value=False,
    released_value=False,
):
    recall = matched / expected_source_count if expected_source_count else 1.0
    precision = correct_emitted / emitted if emitted else 0.0
    row = {
        "expected_no_answer": False,
        "expected_source_count": expected_source_count,
        "matched_expected_source_count": matched,
        "emitted_citation_count": emitted,
        "correct_emitted_citation_count": correct_emitted,
        "raw_available": raw_available,
    }
    for prefix, value in (("raw", raw_value), ("released", released_value)):
        row[f"{prefix}_value_correct"] = value
        row[f"{prefix}_currency_correct"] = value
        row[f"{prefix}_unit_correct"] = value
        row[f"{prefix}_scale_correct"] = value
        row[f"{prefix}_period_correct"] = value
        row[f"{prefix}_component_count_correct"] = value
        row[f"{prefix}_component_assignment_correct"] = value
        row[f"{prefix}_text_contract_correct"] = value
        row[f"{prefix}_answer_contract_correct"] = value
        row[f"{prefix}_citation_recall"] = recall
        row[f"{prefix}_citation_precision"] = precision
        row[f"{prefix}_citation_full_recall"] = recall == 1.0
        row[f"{prefix}_citation_perfect_precision"] = precision == 1.0
        row[f"{prefix}_grounded_pass"] = value and recall == 1.0
    return row


def test_bound_source_requires_candidate_identity():
    expected = {
        "candidate_key": "candidate:v1:gold",
        "evidence_id": "gold-evidence",
        "filename": "report.pdf",
        "page": 10,
    }
    assert not source_identity_matches(
        expected,
        {"evidence_id": "gold-evidence", "filename": "report.pdf", "page": 10},
    )


def test_evidence_id_mismatch_does_not_fall_back_to_page():
    expected = {"evidence_id": "gold-evidence", "filename": "report.pdf", "page": 10}
    candidate = {"evidence_id": "other-evidence", "filename": "report.pdf", "page": 10}
    assert not source_identity_matches(expected, candidate)


def test_candidate_key_has_priority():
    expected = {
        "candidate_key": "candidate:v1:gold",
        "evidence_id": "gold-evidence",
        "filename": "report.pdf",
        "page": 10,
    }
    assert source_identity_matches(
        expected,
        {"candidate_key": "candidate:v1:gold", "evidence_id": "other-evidence"},
    )


def test_bound_golden_uses_zero_page_fallback():
    from scripts.evaluation.run_nf_eval_03_r1 import _stage_metrics

    label = _label(_currency_answer())
    ranking = {"case-1": [{"candidate_key": "candidate:v1:other", "filename": "report.pdf", "page": 10}]}
    metrics = _stage_metrics([label], ranking, 5)
    assert metrics["source_recall"]["count"] == 0
    assert metrics["page_fallback_count"] == 0


def test_naked_number_does_not_match_billion_answer():
    score = score_answer_contract(
        "416.161",
        {"question": "What was revenue in FY2025?"},
        _label(_currency_answer()),
    )
    assert not score["value_correct"]
    assert not score["answer_contract_correct"]


def test_wrong_currency_fails_contract():
    score = score_answer_contract(
        "€416.161 billion in FY2025",
        {"question": "What was revenue in FY2025?"},
        _label(_currency_answer()),
    )
    assert score["value_correct"]
    assert not score["currency_correct"]
    assert not score["answer_contract_correct"]


def test_wrong_period_fails_contract():
    score = score_answer_contract(
        "$416.161 billion in FY2024",
        {"question": "What was revenue in FY2025?"},
        _label(_currency_answer()),
    )
    assert score["value_correct"]
    assert not score["period_correct"]
    assert not score["answer_contract_correct"]


def test_wrong_scale_fails_contract():
    score = score_answer_contract(
        "USD 416.161 in FY2025",
        {"question": "What was revenue in FY2025?"},
        _label(_currency_answer()),
    )
    assert not score["value_correct"]
    assert not score["scale_correct"]


def test_composite_components_require_metric_assignment():
    label = _label(
        {
            "canonical_value": None,
            "currency": None,
            "unit": None,
            "value_type": "composite",
            "component_values": [
                {
                    "metric": "Data Center revenue",
                    "canonical_value": "115186000000",
                    "display_value": "$115.186 billion",
                    "currency": "USD",
                    "unit": "currency",
                    "period": "FY2025",
                    "tolerance": "0",
                },
                {
                    "metric": "GAAP gross margin",
                    "canonical_value": "75.0",
                    "display_value": "75.0%",
                    "currency": None,
                    "unit": "percentage",
                    "value_type": "percentage",
                    "period": "FY2025",
                    "tolerance": "0.001",
                },
            ],
        }
    )
    score = score_answer_contract(
        "$115.186 billion and 75.0% in FY2025",
        {"question": "Report both metrics."},
        label,
    )
    assert not score["component_assignment_correct"]
    assert not score["answer_contract_correct"]


def test_raw_and_released_numeric_metrics_are_separate():
    records = [_metric_row(raw_available=True, raw_value=False, released_value=True)]
    raw = _metric(records, released=False)
    released = _metric(records, released=True)
    assert raw["answer_value_pass"]["count"] == 0
    assert released["answer_value_pass"]["count"] == 1


def test_raw_unavailable_is_not_filled_from_released():
    records = [_metric_row(raw_available=False, raw_value=False, released_value=True)]
    raw = _metric(records, released=False)
    released = _metric(records, released=True)
    assert raw["case_denominator"] == 0
    assert released["answer_value_pass"]["count"] == 1


def test_grounded_pass_requires_answer_and_source():
    row = _metric_row(
        raw_available=True,
        expected_source_count=1,
        matched=0,
        emitted=1,
        correct_emitted=0,
        raw_value=True,
        released_value=True,
    )
    metrics = _metric([row], released=True)
    assert metrics["answer_contract_pass"]["count"] == 1
    assert metrics["grounded_pass"]["count"] == 0


def test_macro_and_micro_citation_metrics_are_distinct():
    records = [
        _metric_row(expected_source_count=2, matched=1, emitted=1, correct_emitted=1, released_value=True),
        _metric_row(expected_source_count=1, matched=1, emitted=3, correct_emitted=1, released_value=True),
    ]
    metrics = _metric(records, released=True)
    assert metrics["micro_source_recall"] == pytest.approx(2 / 3)
    assert metrics["micro_citation_precision"] == pytest.approx(2 / 4)
    assert metrics["macro_citation_recall"] == pytest.approx(0.75)
    assert metrics["macro_citation_precision"] == pytest.approx((1.0 + 1 / 3) / 2)


def test_repair_attempt_uses_explicit_flag():
    assert _repair_flags({"repair": {"was_repaired": False, "fallback_used": False}}) == {
        "repair_attempted": False,
        "repair_applied": False,
        "repair_succeeded": False,
        "repair_failed": False,
    }
    assert _repair_flags({"repair": {"was_repaired": False, "fallback_used": True}})["repair_attempted"]


def test_all_golden_hashes_are_recomputed():
    inputs = _load_inputs()
    for key in (
        "question_hash",
        "reference_answer_hash",
        "source_identity_hash",
        "negative_evidence_hash",
        "review_status_hash",
        "corpus_hash",
        "golden_manifest_sha256",
    ):
        assert inputs.hash_report["actual"][key]
    assert all(inputs.hash_report["matches"].values())


def test_source_identity_hash_mismatch_fails_closed(tmp_path):
    manifest_path = tmp_path / "golden-manifest.json"
    manifest = json.loads((DATA / "golden-manifest.json").read_text(encoding="utf-8"))
    manifest["source_identity_hash"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    paths = {}
    for name, source in (
        ("corpus.json", BENCHMARK / "corpus.json"),
        ("questions.jsonl", DATA / "questions.golden.jsonl"),
        ("labels.jsonl", DATA / "labels.golden.jsonl"),
        ("reviews.jsonl", DATA / "review-status.golden.jsonl"),
        ("negative.json", DEFAULT_NEGATIVE),
    ):
        target = tmp_path / name
        shutil.copyfile(source, target)
        paths[name] = target
    with pytest.raises(BaselineConfigurationError):
        _load_inputs(
            corpus_path=paths["corpus.json"],
            manifest_path=manifest_path,
            questions_path=paths["questions.jsonl"],
            labels_path=paths["labels.jsonl"],
            review_status_path=paths["reviews.jsonl"],
            negative_report_path=paths["negative.json"],
        )
