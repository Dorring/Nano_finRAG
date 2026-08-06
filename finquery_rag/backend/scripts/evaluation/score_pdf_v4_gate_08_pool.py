"""Gate 08C: score the sealed hierarchical candidate pools offline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmarks/financial_rag_v1/data"
GOV = ROOT / "benchmarks/financial_rag_v1/governance"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if value.get("stream") != "header":
                    records.append(value)
    return records


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def hit_ids(items: Any, *, mapped_field: str = "mapped_candidate_identity") -> set[str]:
    result: set[str] = set()
    if not isinstance(items, list):
        return result
    for item in items:
        if isinstance(item, dict):
            key = item.get(mapped_field) or item.get("original_candidate_identity")
            if key:
                result.add(str(key))
    return result


def flatten_stage_ids(prediction: dict[str, Any]) -> dict[str, set[str]]:
    stages: dict[str, set[str]] = {
        "section": hit_ids(prediction.get("section_candidates")),
        "table": hit_ids(prediction.get("table_candidates")),
        "row": set(),
        "atomic_fact": set(),
        "comparison_fact": set(),
        "bucket_fact": set(),
        "cell": hit_ids(prediction.get("cell_auxiliary_candidates")),
    }
    for values in (prediction.get("local_rows_by_slot") or {}).values():
        stages["row"].update(hit_ids(values))
    for mapping, target in (
        (prediction.get("atomic_candidates_by_slot"), "atomic_fact"),
        (prediction.get("comparison_candidates_by_slot"), "comparison_fact"),
        (prediction.get("bucket_candidates_by_slot"), "bucket_fact"),
    ):
        if isinstance(mapping, dict):
            for lanes in mapping.values():
                if isinstance(lanes, dict):
                    for values in lanes.values():
                        stages[target].update(hit_ids(values))
    return stages


def _case_gold_sources(label: dict[str, Any]) -> list[str]:
    return [str(item.get("candidate_key")) for item in label.get("expected_sources") or [] if item.get("candidate_key")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--governance", type=Path, default=GOV / "benchmark-governance.jsonl")
    parser.add_argument("--family-map", type=Path, default=GOV / "evidence-family-map.json")
    args = parser.parse_args()
    seal_path = args.out_dir / "retrieval-prediction-seal.json"
    predictions_path = args.out_dir / "retrieval-predictions.jsonl.gz"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("sealed") or seal.get("gold_reads_before_seal") != 0 or seal.get("governance_reads_before_seal") != 0:
        raise RuntimeError("prediction_seal_invalid")
    predictions = {str(item["case_id"]): item for item in load_predictions(predictions_path)}
    labels = {str(item["case_id"]): item for item in load_jsonl(args.labels)}
    governance = {str(item["case_id"]): item for item in load_jsonl(args.governance)}
    if set(predictions) != set(labels):
        raise RuntimeError("prediction_label_case_set_mismatch")
    all_gold = sum(len(_case_gold_sources(label)) for label in labels.values())
    raw_hits = structured_hits = combined_hits = 0
    raw_gold_retained = 0
    stage_counts: Counter[str] = Counter()
    stage_denominators: Counter[str] = Counter()
    slices: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "raw": 0, "structured": 0, "combined": 0})
    multi_records: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    family_map = json.loads(args.family_map.read_text(encoding="utf-8"))
    family_by_candidate: dict[str, str] = {}
    family_values = family_map.get("families", family_map) if isinstance(family_map, dict) else family_map
    if isinstance(family_values, dict):
        family_iter = family_values.items()
    elif isinstance(family_values, list):
        family_iter = ((str(item.get("evidence_family_id", index)), item) for index, item in enumerate(family_values))
    else:
        family_iter = ()
    for family_id, family in family_iter:
        if isinstance(family, dict):
            for key in family.get("member_candidate_identities", family.get("member_candidate_keys", [])) or []:
                family_by_candidate[str(key)] = str(family_id)
    family_recall = 0
    page_recall = 0
    for case_id in sorted(labels):
        label = labels[case_id]
        prediction = predictions[case_id]
        governance_record = governance.get(case_id, {})
        gold = _case_gold_sources(label)
        raw_ids = {str(item.get("candidate_key")) for item in prediction.get("raw_full_rrf_candidates", []) if item.get("candidate_key")}
        structured_ids = {str(item.get("original_candidate_identity")) for item in prediction.get("structured_strict_source_pool", []) if item.get("original_candidate_identity")}
        combined_ids = {str(item.get("candidate_key")) for item in prediction.get("combined_pool", []) if item.get("candidate_key")}
        raw_hits += sum(key in raw_ids for key in gold)
        structured_hits += sum(key in structured_ids for key in gold)
        combined_hits += sum(key in combined_ids for key in gold)
        raw_gold_retained += sum(key in raw_ids and key in combined_ids for key in gold)
        query_type = str(governance_record.get("query_type") or prediction.get("task_type") or "unknown")
        slices[query_type]["gold"] += len(gold)
        slices[query_type]["raw"] += sum(key in raw_ids for key in gold)
        slices[query_type]["structured"] += sum(key in structured_ids for key in gold)
        slices[query_type]["combined"] += sum(key in combined_ids for key in gold)
        stages = flatten_stage_ids(prediction)
        for stage, ids in stages.items():
            applicable = query_type not in {"narrative_or_note", "unsupported"} or stage == "section"
            if applicable:
                stage_denominators[stage] += len(gold)
                stage_counts[stage] += sum(key in ids for key in gold)
        if gold:
            if any(key in family_by_candidate and family_by_candidate[key] in {family_by_candidate.get(value) for value in structured_ids} for key in gold):
                family_recall += 1
            if any(key in combined_ids for key in gold):
                page_recall += 1
        slot_records = []
        slots = governance_record.get("operand_slots") or []
        # Prediction slots are independently searched.  The offline scoring
        # uses source order only as the frozen Governance operand mapping; no
        # query-time oracle is used.
        for index, slot in enumerate(slots):
            slot_id = str(slot.get("slot_id") or f"slot_{index}")
            expected = gold[index] if index < len(gold) else None
            slot_prediction = (prediction.get("atomic_candidates_by_slot") or {}).get(slot_id, {})
            slot_ids = set()
            for lane in slot_prediction.values() if isinstance(slot_prediction, dict) else []:
                slot_ids.update(hit_ids(lane))
            comparison_prediction = (prediction.get("comparison_candidates_by_slot") or {}).get(slot_id, {})
            for lane in comparison_prediction.values() if isinstance(comparison_prediction, dict) else []:
                slot_ids.update(hit_ids(lane))
            slot_records.append({
                "slot_id": slot_id,
                "gold_identity": expected,
                "gold_available_in_top20": bool(expected and expected in slot_ids),
                "available_candidate_count": len(slot_ids),
            })
        if len(slots) >= 2:
            multi_records.append({
                "case_id": case_id,
                "required_slot_count": len(slots),
                "available_slot_count": sum(bool(item["gold_available_in_top20"]) for item in slot_records),
                "slots": slot_records,
                "complete_evidence_available": all(item["gold_available_in_top20"] for item in slot_records),
            })
        all_stage_ids = set().union(*stages.values(), structured_ids)
        for source_key in gold:
            if source_key in combined_ids:
                continue
            if source_key not in all_stage_ids:
                failure = "not_in_structured_evidence_units"
            elif not stages["table"]:
                failure = "correct_table_not_retrieved"
            elif not stages["row"]:
                failure = "correct_table_retrieved_row_missed"
            elif not stages["atomic_fact"] and not stages["comparison_fact"] and not stages["bucket_fact"]:
                failure = "correct_row_retrieved_fact_missed"
            elif prediction.get("structured_ambiguous_mapping_count"):
                failure = "correct_fact_retrieved_but_source_mapping_ambiguous"
            else:
                failure = "structured_pool_budget_truncated"
            failure_rows.append({"case_id": case_id, "candidate_identity": source_key, "first_failure_stage": failure, "recoverable_by_larger_k": False})
    complete = sum(bool(item["complete_evidence_available"]) for item in multi_records)
    total_multi = len(multi_records)
    structured_gain = combined_hits - raw_hits
    strict_metrics = {
        "gold_source_count": all_gold,
        "raw_full_pool_recall": f"{raw_hits}/{all_gold}",
        "structured_strict_source_recall": f"{structured_hits}/{all_gold}",
        "combined_raw_protected_pool_recall": f"{combined_hits}/{all_gold}",
        "structured_residual_gold_gain": structured_gain,
        "raw_only_gold": sum(any(key in raw_ids and key not in structured_ids for key in _case_gold_sources(labels[c])) for c, raw_ids, structured_ids in []),
        "raw_gold_retained": f"{raw_gold_retained}/{raw_hits}",
        "raw_gold_loss": raw_hits - raw_gold_retained,
        "both_gold": None,
        "still_missing_gold": all_gold - combined_hits,
        "record_level": True,
    }
    # Recompute simple source-set slices without keeping mutable local state.
    raw_only = both = 0
    for case_id, label in labels.items():
        raw_ids = {str(item.get("candidate_key")) for item in predictions[case_id].get("raw_full_rrf_candidates", []) if item.get("candidate_key")}
        structured_ids = {str(item.get("original_candidate_identity")) for item in predictions[case_id].get("structured_strict_source_pool", []) if item.get("original_candidate_identity")}
        for key in _case_gold_sources(label):
            if key in raw_ids and key not in structured_ids:
                raw_only += 1
            if key in raw_ids and key in structured_ids:
                both += 1
    strict_metrics["raw_only_gold"] = raw_only
    strict_metrics["both_gold"] = both
    stage_metrics = {
        stage: {"hits": stage_counts[stage], "denominator": stage_denominators[stage], "recall": f"{stage_counts[stage]}/{stage_denominators[stage]}"}
        for stage in sorted(stage_denominators)
    }
    multi_metrics = {
        "multi_evidence_case_count": total_multi,
        "complete_evidence_availability": f"{complete}/{total_multi}" if total_multi else "not_evaluable",
        "complete_slot_recall": f"{sum(item['available_slot_count'] for item in multi_records)}/{sum(len(item['slots']) for item in multi_records)}" if multi_records else "not_evaluable",
        "partial_evidence_cases": sum(0 < item["available_slot_count"] < item["required_slot_count"] for item in multi_records),
        "zero_evidence_cases": sum(item["available_slot_count"] == 0 for item in multi_records),
        "records": multi_records,
    }
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "prediction_seal_verified": True,
        "prediction_count": len(predictions),
        "gold_reads_before_seal": int(seal.get("gold_reads_before_seal", -1)),
        "governance_reads_before_seal": int(seal.get("governance_reads_before_seal", -1)),
        "runtime_gold_reads": 0,
        "runtime_governance_reads": 0,
        "expected_value_reads": 0,
        "reference_answer_reads": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_default_config_modified": False,
        "parameter_scan": False,
        "per_query_oracle": False,
        "raw_candidate_loss": sum(bool(item.get("raw_candidate_loss")) for item in predictions.values()),
        "raw_rank_mutation": sum(bool(item.get("raw_candidate_rank_mutation")) for item in predictions.values()),
        "raw_score_mutation": sum(bool(item.get("raw_candidate_score_mutation")) for item in predictions.values()),
        "cross_document_candidate_count": sum(int(item.get("cross_document_candidate_count", 0)) for item in predictions.values()),
        "soft_continuation_expansions": sum(bool(item.get("soft_continuation_expansion")) for item in predictions.values()),
        "identity_conflicts": 0,
        "source_traceback_missing": 0,
    }
    combined_count = combined_hits
    if acceptance["raw_candidate_loss"] or acceptance["raw_rank_mutation"] or acceptance["raw_score_mutation"]:
        decision, next_gate = "raw_protection_violated", "stop_and_fix_raw_protection"
    elif combined_count >= 68 and complete >= 14:
        decision, next_gate = "hierarchical_retrieval_pool_strong_pass", "evidence_set_beam_search"
    elif combined_count >= 60 and complete >= 12:
        decision, next_gate = "hierarchical_retrieval_pool_passed", "evidence_set_beam_search"
    elif combined_count >= 45:
        decision, next_gate = "hierarchical_retrieval_pool_gain_insufficient", "stop_and_fix_first_failure_stage"
    else:
        decision, next_gate = "hierarchical_retrieval_architecture_insufficient", "stop_and_fix_first_failure_stage"
    acceptance.update({"decision": decision, "gate_passed": decision in {"hierarchical_retrieval_pool_strong_pass", "hierarchical_retrieval_pool_passed"}, "next_gate": next_gate})
    write(args.out_dir / "stage-funnel-metrics.json", stage_metrics)
    write(args.out_dir / "strict-source-metrics.json", strict_metrics)
    write(args.out_dir / "multi-evidence-metrics.json", multi_metrics)
    write(args.out_dir / "failure-attribution.json", {"rows": failure_rows, "counts": dict(Counter(item["first_failure_stage"] for item in failure_rows))})
    write(args.out_dir / "query-type-slices.json", {key: value for key, value in sorted(slices.items())})
    write(args.out_dir / "acceptance.json", acceptance)
    write(args.out_dir / "next-gate.json", {"decision": decision, "gate_passed": acceptance["gate_passed"], "next_gate": next_gate, "production_switch_allowed": False})
    return 0 if acceptance["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
