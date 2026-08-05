"""Verify the prediction seal, then score the terminal PDF SR-V2 transfer."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.evaluation.run_pdf_retrieval_v2_lite import _write

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-sr-v2-terminal-transfer"
DEFAULT_LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
DEFAULT_QUESTIONS = ROOT / "benchmarks/financial_rag_v1/data/questions.golden.jsonl"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _keys(record: dict[str, Any], stage: str) -> list[str]:
    return [str(item["candidate_key"]) for item in record[stage]]


def _score(predictions: list[dict[str, Any]], labels: dict[str, dict[str, Any]], questions: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], set[tuple[str, str]], dict[str, dict[str, Any]]]:
    stages = {"bm25_200": "bm25_top_200", "dense_200": "dense_top_200", "union_200": "union_top_200", "rrf_40": "rrf_top_40", "reranker_20": "reranker_top_20", "final_5": "final_top_5"}
    counts = Counter()
    strict_hits: set[tuple[str, str]] = set()
    per_case = {}
    reciprocal = 0.0
    answerable_cases = 0
    multi_total = multi_complete = 0
    for prediction in predictions:
        case_id = str(prediction["case_id"])
        label = labels[case_id]
        question = questions[case_id]
        if label.get("expected_no_answer"):
            continue
        answerable_cases += 1
        sources = list(label.get("expected_sources") or [])
        gold = [str(item["candidate_key"]) for item in sources]
        stage_hits = {}
        for metric, field in stages.items():
            ranked = _keys(prediction, field)
            matched = sum(key in ranked for key in gold)
            counts[metric] += matched
            stage_hits[metric] = matched
        final_keys = _keys(prediction, "final_top_5")
        for source_index, key in enumerate(gold):
            if key in final_keys:
                strict_hits.add((case_id, key))
        counts["case_hit"] += int(any(key in final_keys for key in gold))
        complete = all(key in final_keys for key in gold)
        counts["all_gold"] += int(complete)
        if len(gold) > 1:
            multi_total += 1
            multi_complete += int(complete)
        ranks = [final_keys.index(key) + 1 for key in gold if key in final_keys]
        reciprocal += 1 / min(ranks) if ranks else 0
        per_case[case_id] = {"answer_type": question.get("answer_type"), "requires_calculation": question.get("requires_calculation"), "requires_multiple_sources": question.get("requires_multiple_sources"), "gold_source_count": len(gold), "stage_matched_sources": stage_hits, "final_case_hit": bool(ranks), "final_all_gold": complete}
    metrics = {"answerable_case_count": answerable_cases, "gold_source_count": 80, "bm25_source_recall_at_200": counts["bm25_200"] / 80, "dense_source_recall_at_200": counts["dense_200"] / 80, "union_source_recall_at_200": counts["union_200"] / 80, "rrf_source_recall_at_40": counts["rrf_40"] / 80, "reranker_source_recall_at_20": counts["reranker_20"] / 80, "strict_final_source_recall_at_5": counts["final_5"] / 80, "final_case_hit_at_5": counts["case_hit"] / answerable_cases, "final_all_gold_coverage": counts["all_gold"] / answerable_cases, "multi_evidence_complete_coverage": multi_complete / multi_total if multi_total else 0, "mrr_at_5": reciprocal / answerable_cases, "hit_counts": dict(counts), "multi_evidence_case_count": multi_total}
    return metrics, strict_hits, per_case


def run(args: argparse.Namespace) -> int:
    seal_path = args.out_dir / "prediction-seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected = {"protocol_hash": _sha(args.out_dir / "terminal-transfer-protocol.json"), "baseline_index_manifest_hash": _sha(args.out_dir / "baseline-index-manifest.json"), "e1_index_manifest_hash": _sha(args.out_dir / "e1-index-manifest.json"), "baseline_prediction_hash": _sha(args.out_dir / "baseline-predictions.json"), "e1_prediction_hash": _sha(args.out_dir / "e1-predictions.json")}
    if not seal["predictions_sealed"] or seal["labels_read_before_seal"] != 0 or any(seal[key] != value for key, value in expected.items()):
        raise RuntimeError("prediction seal verification failed before labels load")
    labels = {str(item["case_id"]): item for item in _jsonl(args.labels)}
    questions = {str(item["case_id"]): item for item in _jsonl(args.questions)}
    baseline_payload = json.loads((args.out_dir / "baseline-predictions.json").read_text(encoding="utf-8"))
    e1_payload = json.loads((args.out_dir / "e1-predictions.json").read_text(encoding="utf-8"))
    baseline, baseline_hits, baseline_cases = _score(baseline_payload["predictions"], labels, questions)
    e1, e1_hits, e1_cases = _score(e1_payload["predictions"], labels, questions)
    new_hits, regressions = e1_hits - baseline_hits, baseline_hits - e1_hits
    slices: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for case_id, question in questions.items():
        if case_id not in baseline_cases:
            continue
        dimensions = {"answer_type": str(question.get("answer_type")), "requires_calculation": str(bool(question.get("requires_calculation"))).lower(), "requires_multiple_sources": str(bool(question.get("requires_multiple_sources"))).lower()}
        for dimension, value in dimensions.items():
            for variant, records in (("baseline", baseline_cases), ("e1", e1_cases)):
                slices[dimension][value][f"{variant}_cases"] += 1
                slices[dimension][value][f"{variant}_final_hits"] += int(records[case_id]["final_case_hit"])
                slices[dimension][value][f"{variant}_all_gold"] += int(records[case_id]["final_all_gold"])
    slice_payload = {dimension: {value: dict(counts) for value, counts in values.items()} for dimension, values in slices.items()}
    _write(args.out_dir / "stage-funnel-comparison.json", {"baseline": baseline, "e1": e1, "deltas": {key: e1[key] - baseline[key] for key in ("bm25_source_recall_at_200", "dense_source_recall_at_200", "union_source_recall_at_200", "rrf_source_recall_at_40", "reranker_source_recall_at_20", "strict_final_source_recall_at_5", "final_case_hit_at_5", "final_all_gold_coverage", "multi_evidence_complete_coverage", "mrr_at_5")}})
    _write(args.out_dir / "strict-source-change-report.json", {"baseline_strict_hit_count": len(baseline_hits), "e1_strict_hit_count": len(e1_hits), "new_gold_source_count": len(new_hits), "regressed_gold_source_count": len(regressions), "new_gold_sources": [{"case_id": case_id, "candidate_key": key} for case_id, key in sorted(new_hits)], "regressed_gold_sources": [{"case_id": case_id, "candidate_key": key} for case_id, key in sorted(regressions)], "original_13_hit_retained_count": len(baseline_hits & e1_hits), "original_13_hit_regressed_count": len(regressions)})
    _write(args.out_dir / "slice-metrics.json", slice_payload)
    _write(args.out_dir / "no-answer-status.json", {"no_answer_evaluation": "not_run", "no_answer_case_count": 8, "reason": "terminal transfer pre-registration excludes no-answer scoring"})
    protocol = json.loads((args.out_dir / "terminal-transfer-protocol.json").read_text(encoding="utf-8"))
    identity = json.loads((args.out_dir / "candidate-identity-integrity.json").read_text(encoding="utf-8"))
    thresholds = {"strict_final_source_recall_at_5": e1["strict_final_source_recall_at_5"] >= 20 / 80, "original_hit_regressions": len(regressions) <= 1, "rrf_source_recall_at_40": e1["rrf_source_recall_at_40"] >= 28 / 80, "final_all_gold": e1["hit_counts"]["all_gold"] >= 15, "multi_evidence_not_decreased": e1["multi_evidence_complete_coverage"] >= baseline["multi_evidence_complete_coverage"], "candidate_identity_integrity": identity["identity_loss_count"] == identity["identity_conflict_count"] == identity["duplicate_view_count"] == 0}
    passed = all(thresholds.values()) and protocol["pre_protocol_question_content_preview_count"] == 0
    decision = "pdf_sr_v2_terminal_transfer_gate_passed" if passed else "pdf_sr_v2_terminal_transfer_failed"
    acceptance = {"schema": "pdf-sr-v2/terminal-transfer/acceptance/v1", "evaluation_type": "one_shot_terminal_diagnostic_transfer", "seal_verified_before_gold_load": True, "protocol_precondition_clean": protocol["pre_protocol_question_content_preview_count"] == 0, "thresholds": thresholds, "gate_passed": passed, "production_index_writes": 0, "production_config_modified": False, "production_behavior_modified": False, "answer_generation_calls": 0, "calculator_calls": 0, "binder_calls": 0, "model_training_calls": 0, "parameter_scan": False, "per_query_oracle_selection": False, "post_score_tuning_allowed": False, "production_switch_allowed": False, "decision": decision, "next_gate": "stop_pdf_sr_v2_terminal_transfer"}
    _write(args.out_dir / "terminal-transfer-acceptance.json", acceptance)
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": "stop_pdf_sr_v2_terminal_transfer", "production_switch_allowed": False, "post_score_tuning_allowed": False})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
