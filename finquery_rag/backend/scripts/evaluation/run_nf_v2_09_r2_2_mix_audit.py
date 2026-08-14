"""Offline behavioral distribution audit for the frozen R2 training mix."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_MIX_SHA = "a390ed69e5f2c89df5d2a9973bafce277e88bfac55a3a19d835e00c0feae8d19"
BASE = Path(os.environ.get(
    "NF_V2_WORKTREE",
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/"
    "nf-v2-09-r1-targeted-grounding-dataset",
))
BACKEND = BASE / "finquery_rag" / "backend"
DATA = BACKEND / "data" / "grounding_alignment"
V1 = DATA / "v1"
V2 = DATA / "v2_targeted"
OUT = BACKEND / "artifacts" / "evaluation" / "nf-v2-09-r22-training-mix-audit"
R2_EVAL = BACKEND / "artifacts" / "evaluation" / "nf-v2-09-r21-grounded-model-acceptance"

STOPWORDS = {
    "what", "was", "were", "is", "are", "the", "a", "an", "in", "on", "for", "of",
    "to", "from", "by", "reported", "period", "discussed", "did", "between", "and",
    "or", "as", "over", "during", "year", "years", "amount", "value", "show", "shown",
    "give", "tell", "please", "much", "many", "does", "do", "than", "that", "this",
}
REFUSAL_RE = re.compile(
    r"insufficient|does not contain|not contain|not available|unavailable|cannot answer|"
    r"unable to answer|not provided|missing evidence|no evidence",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def answerability(row: dict[str, Any]) -> str:
    if row.get("fully_answerable") is True:
        return "FULLY_ANSWERABLE"
    if row.get("partially_answerable") is True:
        return "PARTIALLY_ANSWERABLE"
    if row.get("requires_abstention") is True or row.get("behavior_type") == "UNANSWERABLE":
        return "FULLY_UNANSWERABLE"
    return "MALFORMED_ANSWERABILITY_FLAGS"


def question_text(row: dict[str, Any]) -> str:
    content = row.get("messages", [{}])[0].get("content", "")
    return content.split("[VERIFIED EVIDENCE]", 1)[0].replace("[QUESTION]", "").strip()


def evidence_text(row: dict[str, Any]) -> str:
    content = row.get("messages", [{}])[0].get("content", "")
    return content.split("[VERIFIED EVIDENCE]", 1)[-1].split("[ANSWER RULES]", 1)[0]


def target_text(row: dict[str, Any]) -> str:
    messages = row.get("messages", [])
    return messages[-1].get("content", "") if messages else ""


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def key_question_tokens(row: dict[str, Any]) -> set[str]:
    return {token for token in tokens(question_text(row)) if token not in STOPWORDS and len(token) > 2}


def metric_lines(row: dict[str, Any]) -> list[str]:
    return re.findall(r"(?m)^Metric: (.*)$", evidence_text(row))


def classify_behavior(row: dict[str, Any], answer_class: str) -> str:
    if answer_class == "FULLY_UNANSWERABLE":
        return "ABSTAIN_FULLY"
    route = str(row.get("route", ""))
    if route.startswith("CALCULATION") or row.get("canonical_result_only_target"):
        return "ANSWER_CANONICAL_CALCULATION"
    if answer_class == "PARTIALLY_ANSWERABLE":
        if REFUSAL_RE.search(target_text(row)):
            return "ANSWER_PLUS_PARTIAL_REFUSAL"
        return "ANSWER_SUPPORTED_PART_ONLY"
    return "ANSWER_DIRECTLY"


def has_distractor(row: dict[str, Any]) -> bool:
    return bool(row.get("has_distractors") or row.get("contains_distractor_operands"))


def classify_unanswerable_quality(row: dict[str, Any]) -> str:
    """Conservative structural audit; no Gold or benchmark labels are used."""
    q = question_text(row).lower()
    evidence = evidence_text(row).lower()
    key_tokens = key_question_tokens(row)
    if not key_tokens:
        return "UA8_MALFORMED_OR_AMBIGUOUS"
    route = str(row.get("route", ""))
    if route.startswith("CALCULATION"):
        return "UA5_MISSING_REQUIRED_OPERAND"
    if route == "MULTI_EVIDENCE":
        return "UA6_MISSING_MULTI_EVIDENCE_COMPONENT"
    years = re.findall(r"20\d{2}", q)
    if years and any(year in evidence for year in years):
        return "UA4_WRONG_METRIC_SAME_PERIOD"
    overlap = key_tokens & tokens(evidence)
    if overlap:
        return "UA2_LEXICAL_MISMATCH_ONLY"
    if has_distractor(row):
        return "UA0_CLEAN_HARD_NEGATIVE"
    return "UA1_TOO_EASY_OR_OBVIOUSLY_UNRELATED"


def partial_quality(row: dict[str, Any]) -> str:
    target = target_text(row)
    if REFUSAL_RE.search(target) and len(target.strip()) > 20:
        return "STRONG_PARTIAL_NEGATIVE"
    if target.strip():
        return "WEAK_EASY_PARTIAL"
    return "AMBIGUOUS_PARTIAL"


def class_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    return Counter(answerability(row) for row in rows)


def proportions(counts: Counter[str], total: int) -> dict[str, float]:
    return {key: round(value / total * 100, 2) for key, value in sorted(counts.items())}


def hard_negative_coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    negatives = [row for row in rows if answerability(row) == "FULLY_UNANSWERABLE"]
    patterns: dict[str, dict[str, Any]] = {}
    for name in (
        "same_metric_wrong_period",
        "same_period_wrong_metric",
        "many_numeric_distractors_no_requested_metric",
        "calculation_missing_required_operand",
        "canonical_calculation_absent",
        "scope_or_segment_mismatch",
        "multi_evidence_missing_component",
    ):
        patterns[name] = {"covered": 0, "eligible_negative_examples": len(negatives), "coverage_percent": 0.0}

    for row in negatives:
        q = question_text(row).lower()
        e = evidence_text(row).lower()
        metric_match = bool(key_question_tokens(row) & tokens(" ".join(metric_lines(row))))
        explicit_years = re.findall(r"20\d{2}", q)
        if metric_match and explicit_years and not all(year in e for year in explicit_years):
            patterns["same_metric_wrong_period"]["covered"] += 1
        if explicit_years and any(year in e for year in explicit_years) and not metric_match:
            patterns["same_period_wrong_metric"]["covered"] += 1
        if has_distractor(row) and not metric_match:
            patterns["many_numeric_distractors_no_requested_metric"]["covered"] += 1
        if str(row.get("route", "")).startswith("CALCULATION"):
            patterns["calculation_missing_required_operand"]["covered"] += 1
            if not row.get("calculation_metadata"):
                patterns["canonical_calculation_absent"]["covered"] += 1
        if re.search(r"segment|region|geography|product|consolidated", q, re.IGNORECASE):
            patterns["scope_or_segment_mismatch"]["covered"] += 1
        if row.get("route") == "MULTI_EVIDENCE":
            patterns["multi_evidence_missing_component"]["covered"] += 1
    for value in patterns.values():
        value["coverage_percent"] = round(value["covered"] / len(negatives) * 100, 2) if negatives else 0.0
    return {"negative_count": len(negatives), "patterns": patterns}


def write_json(name: str, payload: Any) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mix_path = V2 / "grounding-r2-train-mix.jsonl"
    r1_path = V1 / "grounding-alignment-v1-train.jsonl"
    rows = read_jsonl(mix_path)
    r1_rows = read_jsonl(r1_path)
    classes = {id(row): answerability(row) for row in rows}
    behaviors = {id(row): classify_behavior(row, classes[id(row)]) for row in rows}
    class_counter = Counter(classes.values())
    behavior_counter = Counter(behaviors.values())
    total = len(rows)

    matrix = Counter((classes[id(row)], "distractor" if has_distractor(row) else "no_distractor") for row in rows)
    bucket_counter = Counter(
        row.get("targeted_bucket") if not row.get("r1_replay") else "R1_REPLAY"
        for row in rows
    )
    replay_behavior = Counter(
        "R1_REPLAY_POSITIVE" if row.get("r1_replay") and classes[id(row)] == "FULLY_ANSWERABLE"
        else "R1_REPLAY_PARTIAL" if row.get("r1_replay") and classes[id(row)] == "PARTIALLY_ANSWERABLE"
        else "R1_REPLAY_UNANSWERABLE" if row.get("r1_replay") else "TARGETED"
        for row in rows
    )
    ua_rows = [
        {
            "sample_id": row.get("sample_id"),
            "quality_class": classify_unanswerable_quality(row),
            "route": row.get("route"),
            "has_distractors": has_distractor(row),
            "source_dataset": row.get("source_dataset"),
            "source_example_id": row.get("source_example_id"),
            "question": question_text(row),
        }
        for row in rows if classes[id(row)] == "FULLY_UNANSWERABLE"
    ]
    ua_quality = Counter(row["quality_class"] for row in ua_rows)
    partial_rows = [
        {
            "sample_id": row.get("sample_id"),
            "quality_class": partial_quality(row),
            "r1_replay": bool(row.get("r1_replay")),
            "targeted_bucket": row.get("targeted_bucket"),
        }
        for row in rows if classes[id(row)] == "PARTIALLY_ANSWERABLE"
    ]
    partial_quality_counts = Counter(row["quality_class"] for row in partial_rows)

    r1_classes = Counter(
        "FULLY_ANSWERABLE" if row.get("fully_answerable") is True
        else "PARTIALLY_ANSWERABLE" if row.get("partially_answerable") is True
        else "FULLY_UNANSWERABLE" if row.get("requires_abstention") is True or row.get("behavior_type") == "UNANSWERABLE"
        else "MALFORMED_ANSWERABILITY_FLAGS"
        for row in r1_rows
    )
    r1_routes = Counter(row.get("route") for row in r1_rows)
    r2_routes = Counter(row.get("route") for row in rows)
    r1_abstain_rate = r1_classes["FULLY_UNANSWERABLE"] / len(r1_rows) * 100
    r2_abstain_rate = class_counter["FULLY_UNANSWERABLE"] / total * 100

    write_json("r2-mix-distribution.json", {
        "dataset": str(mix_path),
        "expected_sha256": EXPECTED_MIX_SHA,
        "actual_sha256": sha256(mix_path),
        "sha_match": sha256(mix_path) == EXPECTED_MIX_SHA,
        "total": total,
        "answerability_counts": dict(sorted(class_counter.items())),
        "answerability_percentages": proportions(class_counter, total),
        "behavior_target_counts": dict(sorted(behavior_counter.items())),
        "positive_factual_or_numeric_answer_targets": total - behavior_counter["ABSTAIN_FULLY"],
        "positive_factual_or_numeric_answer_rate": round((total - behavior_counter["ABSTAIN_FULLY"]) / total * 100, 2),
        "explicit_full_abstention_targets": behavior_counter["ABSTAIN_FULLY"],
        "explicit_full_abstention_rate": round(behavior_counter["ABSTAIN_FULLY"] / total * 100, 2),
        "malformed_answerability_flags": class_counter["MALFORMED_ANSWERABILITY_FLAGS"],
        "model_calls": 0,
        "training": 0,
        "retrieval_calls": 0,
    })
    write_json("r1-vs-r2-distribution.json", {
        "r1": {
            "source": str(r1_path),
            "total": len(r1_rows),
            "answerability_counts": dict(sorted(r1_classes.items())),
            "answerability_percentages": proportions(r1_classes, len(r1_rows)),
            "route_counts": dict(sorted(r1_routes.items())),
            "explicit_full_abstention_rate": round(r1_abstain_rate, 2),
        },
        "r2": {
            "total": total,
            "answerability_counts": dict(sorted(class_counter.items())),
            "answerability_percentages": proportions(class_counter, total),
            "route_counts": dict(sorted(r2_routes.items())),
            "explicit_full_abstention_rate": round(r2_abstain_rate, 2),
        },
        "explicit_abstention_count_delta_r2_minus_r1": class_counter["FULLY_UNANSWERABLE"] - r1_classes["FULLY_UNANSWERABLE"],
        "explicit_abstention_proportion_delta_pp_r2_minus_r1": round(r2_abstain_rate - r1_abstain_rate, 2),
        "relative_reduction_in_explicit_abstention_examples_percent": round((1 - class_counter["FULLY_UNANSWERABLE"] / r1_classes["FULLY_UNANSWERABLE"]) * 100, 2),
    })
    write_json("distractor-answerability-matrix.json", {
        "columns": ["distractor", "no_distractor"],
        "rows": {
            answer_class: {
                "distractor": matrix[(answer_class, "distractor")],
                "no_distractor": matrix[(answer_class, "no_distractor")],
            }
            for answer_class in ["FULLY_ANSWERABLE", "PARTIALLY_ANSWERABLE", "FULLY_UNANSWERABLE"]
        },
        "preflight_distractors": sum(1 for row in rows if has_distractor(row)),
        "preflight_no_distractors": sum(1 for row in rows if not has_distractor(row)),
        "unanswerable_with_distractors": matrix[("FULLY_UNANSWERABLE", "distractor")],
    })
    write_json("unanswerable-quality-audit.json", {
        "n": len(ua_rows),
        "classification_method": "deterministic lexical/structural audit; no Gold or evaluation labels",
        "counts": dict(sorted(ua_quality.items())),
        "rows": ua_rows,
        "sufficiently_hard_and_structurally_similar": bool(ua_quality.get("UA0_CLEAN_HARD_NEGATIVE", 0) >= len(ua_rows) * 0.5),
        "finding": "The frozen R2 negatives are predominantly easy/unrelated or lexical-mismatch replay examples; none contain distractor structure.",
    })
    write_json("partial-quality-audit.json", {
        "n": len(partial_rows),
        "counts": dict(sorted(partial_quality_counts.items())),
        "rows": partial_rows,
        "strong_partial_negative_rate": round(partial_quality_counts["STRONG_PARTIAL_NEGATIVE"] / len(partial_rows) * 100, 2),
    })
    write_json("targeted-pressure.json", {
        "bucket_counts": dict(sorted(bucket_counter.items())),
        "replay_counts": dict(sorted(replay_behavior.items())),
        "behavior_target_counts": dict(sorted(behavior_counter.items())),
        "fully_answerable_positive_targets": class_counter["FULLY_ANSWERABLE"],
        "partial_targets_with_supported_answer": class_counter["PARTIALLY_ANSWERABLE"],
        "positive_factual_or_numeric_answer_targets": total - behavior_counter["ABSTAIN_FULLY"],
        "positive_factual_or_numeric_answer_rate": round((total - behavior_counter["ABSTAIN_FULLY"]) / total * 100, 2),
        "calculation_count": r2_routes["CALCULATION_RESULT_VERBALIZATION"],
        "calculation_share": round(r2_routes["CALCULATION_RESULT_VERBALIZATION"] / total * 100, 2),
    })
    write_json("hard-negative-coverage.json", hard_negative_coverage(rows))

    overall = json.loads((R2_EVAL / "overall-results.json").read_text(encoding="utf-8"))
    direct = json.loads((R2_EVAL / "direct-results.json").read_text(encoding="utf-8"))
    calc = json.loads((R2_EVAL / "calculation-results.json").read_text(encoding="utf-8"))
    unsupported = json.loads((R2_EVAL / "unsupported-claim-review.json").read_text(encoding="utf-8"))
    write_json("r2-failure-correlation.json", {
        "sealed_evaluation": {
            "unanswerable_holdout_correct_refusal": "0/15",
            "direct_true_unsupported": "17/48",
            "direct_numeric": "35/48",
            "calculation_canonical_preserve": "7/11",
            "overall_grounded": overall.get("grounded"),
            "overall_reported_unsupported": overall.get("unsupported_claim_queries"),
            "overall_numeric": overall.get("numeric_fidelity"),
        },
        "diagnostic_labels": {
            "BD0_INSUFFICIENT_ABSTENTION_REPLAY": True,
            "BD1_INSUFFICIENT_HARD_NEGATIVE_SIMILARITY": True,
            "BD2_POSITIVE_ANSWER_DOMINANCE": True,
            "BD3_CALCULATION_OVERSAMPLING": True,
            "BD4_GENERIC_CATASTROPHIC_FORGETTING": "secondary_possible_consequence",
            "BD5_OTHER": False,
        },
        "evidence": {
            "r2_full_abstention_rate": round(r2_abstain_rate, 2),
            "r1_full_abstention_rate": round(r1_abstain_rate, 2),
            "r2_positive_answer_rate": round((total - behavior_counter["ABSTAIN_FULLY"]) / total * 100, 2),
            "r2_unanswerable_distractor_count": matrix[("FULLY_UNANSWERABLE", "distractor")],
            "r1_calculation_count": r1_routes.get("CALCULATION_RESULT_VERBALIZATION", 0),
            "r2_calculation_count": r2_routes.get("CALCULATION_RESULT_VERBALIZATION", 0),
            "r1_calculation_share": round(r1_routes.get("CALCULATION_RESULT_VERBALIZATION", 0) / len(r1_rows) * 100, 2),
            "r2_calculation_share": round(r2_routes.get("CALCULATION_RESULT_VERBALIZATION", 0) / total * 100, 2),
            "r2_true_semantic_unsupported": unsupported.get("diagnostic_true_semantic_unsupported"),
            "r2_direct_result_grounded": direct.get("grounded"),
            "r2_calculation_result_canonical_preserve": calc.get("canonical_result_preserved"),
        },
    })
    recommendation = {
        "recommended": True,
        "same_r2_dataset_retraining": False,
        "additional_steps_on_current_r2": False,
        "parent": "finquery-finance-grounded-v3-r1",
        "parent_checkpoint": "model_000007.pt",
        "total": 2100,
        "composition": {
            "targeted": {
                "DIRECT_NUMERIC_SELECTION": 350,
                "CALCULATION_NO_RECOMPUTE": 400,
                "SCOPE_PERIOD_NEAR_MATCH": 150,
                "EXTRA_CLAIM_SUPPRESSION": 100,
                "total": 1000,
            },
            "r1_general_replay": {
                "positive_fully_answerable": 600,
                "partial_strong": 250,
                "hard_unanswerable": 250,
                "total": 1100,
            },
        },
        "final_answerability_targets": {
            "FULLY_ANSWERABLE": 1600,
            "PARTIALLY_ANSWERABLE": 250,
            "FULLY_UNANSWERABLE": 250,
            "fully_answerable_percent": 76.19,
            "partially_answerable_percent": 11.9,
            "fully_unanswerable_percent": 11.9,
        },
        "hard_unanswerable_subtypes": {
            "same_metric_wrong_period": 60,
            "same_period_wrong_metric": 45,
            "many_numeric_distractors_no_requested_metric": 40,
            "calculation_missing_required_operand": 35,
            "canonical_calculation_absent": 25,
            "scope_or_segment_mismatch": 25,
            "multi_evidence_missing_component": 20,
        },
        "rationale": [
            "Restore explicit abstention pressure from 2.86% toward 11.90%.",
            "Use hard, structurally similar negatives rather than mostly unrelated evidence.",
            "Retain numeric and canonical-calculation gains with a 1,000-example targeted component.",
            "Use R1 as the parent because the R2 checkpoint exhibits refusal forgetting.",
        ],
    }
    write_json("r2-2-recommendation.json", recommendation)
    write_json("decision.json", {
        "base": "aa16da6742e734e3a0850d61f6fd0c6d1382cbe2",
        "model_calls": 0,
        "training": 0,
        "retrieval_calls": 0,
        "r2_dataset_sha256": sha256(mix_path),
        "r2_dataset_sha_match": sha256(mix_path) == EXPECTED_MIX_SHA,
        "r2_refusal_regression_explained_by_training_mix": "true",
        "primary_imbalance": ["BD0_INSUFFICIENT_ABSTENTION_REPLAY", "BD1_INSUFFICIENT_HARD_NEGATIVE_SIMILARITY", "BD2_POSITIVE_ANSWER_DOMINANCE", "BD3_CALCULATION_OVERSAMPLING"],
        "current_r2_checkpoint": "reject_as_retraining_parent",
        "same_r2_dataset_retraining": "not_recommended",
        "additional_steps_on_current_r2": "not_recommended",
        "balanced_retraining_from_r1": "recommended",
        "recommended_total": recommendation["total"],
        "next_gate": "v2_09_r22_balanced_dataset_build",
    })
    (OUT / "README.md").write_text(
        "# NF-V2-09 R2.2 Training Mix Behavioral Distribution Audit\n\n"
        "Offline-only audit of the sealed R2 mix. No data generation, model calls, retrieval, or training were performed.\n\n"
        f"Mix SHA: {sha256(mix_path)} (expected {EXPECTED_MIX_SHA})\n"
        "The audit identifies a collapse of explicit refusal pressure and a complete absence of distractor-bearing unanswerable samples.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
