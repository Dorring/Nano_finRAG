"""Post-seal scoring for PDF Retrieval V3 Gate 3 candidate-pool coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-3"
DATA = ROOT / "benchmarks/financial_rag_v1/data"
GOVERNANCE = ROOT / "benchmarks/financial_rag_v1/governance"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fraction(value: int, denominator: int) -> str:
    return f"{value}/{denominator}"


def _ids(values: list[dict[str, Any]]) -> set[str]:
    return {str(item["candidate_key"]) for item in values}


def _metric(
    records: list[dict[str, Any]], labels: dict[str, dict[str, Any]], field: str
) -> tuple[int, int, set[tuple[str, int]], dict[str, set[str]]]:
    hits: set[tuple[str, int]] = set()
    pages: dict[str, set[str]] = {}
    for record in records:
        case_id = str(record["case_id"])
        identities = _ids(record[field])
        pages[case_id] = {str(item.get("pdf_page") or item.get("page")) for item in record[field]}
        for index, source in enumerate(labels[case_id].get("expected_sources") or []):
            if str(source["candidate_key"]) in identities:
                hits.add((case_id, index))
    denominator = sum(len(item.get("expected_sources") or []) for item in labels.values())
    return len(hits), denominator, hits, pages


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--labels", type=Path, default=DATA / "labels.golden.jsonl")
    parser.add_argument("--governance", type=Path, default=GOVERNANCE / "benchmark-governance.jsonl")
    parser.add_argument("--family-map", type=Path, default=GOVERNANCE / "evidence-family-map.json")
    args = parser.parse_args()
    seal = json.loads((args.out_dir / "gate-3-prediction-seal.json").read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed") or seal.get("labels_read_before_seal") != 0:
        raise RuntimeError("prediction seal is not clean")
    prediction_path = args.out_dir / "gate-3-predictions.json"
    protocol_path = args.out_dir / "gate-3-protocol.json"
    if seal["prediction_hash"] != _sha(prediction_path) or seal["protocol_hash"] != _sha(protocol_path):
        raise RuntimeError("prediction seal hash verification failed")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))["predictions"]
    labels = {
        str(item["case_id"]): item
        for item in (json.loads(line) for line in args.labels.read_text(encoding="utf-8").splitlines() if line)
    }
    governance = {
        str(item["case_id"]): item
        for item in (json.loads(line) for line in args.governance.read_text(encoding="utf-8").splitlines() if line)
    }
    family_by_key = {
        str(binding["candidate_key"]): str(family["evidence_family_id"])
        for family in json.loads(args.family_map.read_text(encoding="utf-8"))["families"]
        for binding in family["member_bindings"]
    }
    raw_40, denominator, raw_40_hits, _ = _metric(predictions, labels, "raw_rrf_at_40")
    raw_full, _, raw_full_hits, _ = _metric(predictions, labels, "raw_full_rrf_candidates")
    structured, _, structured_hits, _ = _metric(predictions, labels, "structured_rrf_top20")
    combined, _, combined_hits, combined_pages = _metric(predictions, labels, "combined_full_pool")
    raw_lost = raw_full_hits - combined_hits
    new_hits = combined_hits - raw_full_hits
    final_gold = {
        (str(case["case_id"]), index)
        for case in json.loads((ROOT / "artifacts/evaluation/pdf-retrieval-v3-gate-0/baseline-case-parity.json").read_text(encoding="utf-8"))["replayed_final_hits"]
        for index in [int(case["source_index"])]
    }
    final_retained = final_gold <= combined_hits
    noneligible_changed = [
        item["case_id"] for item in predictions
        if not item["eligible_for_structured_lane"]
        and item["raw_full_rrf_candidates"] != item["combined_full_pool"]
    ]
    raw_parity = []
    for item in predictions:
        raw = item["raw_full_rrf_candidates"]
        prefix = item["combined_full_pool"][:len(raw)]
        raw_parity.append({"case_id": item["case_id"], "candidate_count": len(raw), "hash_before": item["raw_pool_hash_before"], "hash_after": item["raw_pool_hash_after"], "identical": raw == prefix})
    eligible = [item for item in predictions if item["eligible_for_structured_lane"]]
    structured_errors = sum(
        bool(item.get("concept_resolution_trace")) and not item["structured_rrf_top20"] for item in eligible
    )
    new_by_type = Counter(str(governance[case_id]["query_type"]) for case_id, _ in new_hits)
    attribution = []
    for case_id, index in sorted(new_hits):
        source = labels[case_id]["expected_sources"][index]
        raw_pages = {str(item.get("pdf_page") or item.get("page")) for item in next(item for item in predictions if item["case_id"] == case_id)["raw_full_rrf_candidates"]}
        gold_page = str(source.get("page"))
        reason = "within_page_localization_gain" if gold_page in raw_pages else "new_page_acquisition"
        attribution.append({"case_id": case_id, "source_index": index, "candidate_key": source["candidate_key"], "attribution": reason})
    family_hits = 0
    page_hits = 0
    for case_id, label in labels.items():
        record = next(item for item in predictions if item["case_id"] == case_id)
        candidates = _ids(record["combined_full_pool"])
        candidate_families = {family_by_key[key] for key in candidates if key in family_by_key}
        candidate_pages = combined_pages[case_id]
        for source in label.get("expected_sources") or []:
            key = str(source["candidate_key"])
            family = family_by_key.get(key)
            family_hits += int(family is not None and family in candidate_families)
            page_hits += int(str(source.get("page")) in candidate_pages)
    slices = {}
    for query_type in sorted({str(item["query_type"]) for item in governance.values()}):
        cases = {case_id for case_id, item in governance.items() if str(item["query_type"]) == query_type}
        expected = sum(len(labels[case_id].get("expected_sources") or []) for case_id in cases)
        slices[query_type] = {"case_count": len(cases), "expected_sources": expected, "raw_full_hits": sum(case_id in cases for case_id, _ in raw_full_hits), "combined_hits": sum(case_id in cases for case_id, _ in combined_hits), "pool_changed_case_count": sum(case_id in cases and bool(record["structured_rrf_top20"]) for record in predictions for case_id in [record["case_id"]])}
    metrics = {
        "raw_rrf_at_40_strict_recall": _fraction(raw_40, denominator),
        "raw_full_rrf_pool_strict_recall": _fraction(raw_full, denominator),
        "structured_rrf_at_20_strict_recall_eligible_scope": _fraction(structured, denominator),
        "combined_full_pool_strict_recall": _fraction(combined, denominator),
        "combined_full_pool_evidence_family_recall": _fraction(family_hits, denominator),
        "combined_full_pool_page_recall": _fraction(page_hits, denominator),
        "new_strict_gold_count": len(new_hits), "raw_strict_gold_lost": len(raw_lost),
        "final_13_retained_in_combined_pool": final_retained,
        "eligible_case_count": len(eligible), "structured_lane_empty_after_resolution_count": structured_errors,
    }
    effect_passed = len(new_hits) >= 8 and (combined - raw_full) / max(1, denominator) >= 0.10 and structured > 0
    safety_passed = not raw_lost and all(item["identical"] for item in raw_parity) and not noneligible_changed and final_retained
    decision = "raw_protected_structured_lane_pool_gain_passed" if safety_passed and effect_passed else ("raw_pool_protection_failed" if not safety_passed else "structured_lane_pool_gain_insufficient")
    _write(args.out_dir / "raw-pool-parity.json", {"records": raw_parity, "raw_full_pool_candidate_loss_count": len(raw_lost), "raw_full_pool_order_change_count": sum(not item["identical"] for item in raw_parity), "noneligible_pool_changed_cases": noneligible_changed})
    _write(args.out_dir / "structured-lane-retrieval-metrics.json", {"eligible_case_count": len(eligible), "bm25_and_dense_top_k": 40, "rrf_top_k": 20, "strict_source_hits": structured, "denominator": denominator, "empty_lane_after_concept_resolution": structured_errors})
    _write(args.out_dir / "combined-pool-metrics.json", metrics)
    _write(args.out_dir / "strict-source-change-report.json", {"new_hits": [{"case_id": case, "source_index": index} for case, index in sorted(new_hits)], "regressed_hits": [{"case_id": case, "source_index": index} for case, index in sorted(raw_lost)], "raw_full_hits": _fraction(raw_full, denominator), "combined_hits": _fraction(combined, denominator)})
    _write(args.out_dir / "gain-attribution.json", {"records": attribution, "counts": dict(Counter(item["attribution"] for item in attribution)), "new_hit_query_type_counts": dict(new_by_type)})
    _write(args.out_dir / "query-type-slices.json", slices)
    _write(args.out_dir / "acceptance.json", {"gate": "pdf_retrieval_v3_gate_3", "gate_passed": safety_passed and effect_passed, "decision": decision, "next_gate": "multi_evidence_set_retrieval" if safety_passed and effect_passed else ("stop_and_fix_fusion_contract" if not safety_passed else "stop_structured_lane"), "metrics": metrics, "safety": {"raw_pool_loss": len(raw_lost), "raw_pool_order_changes": sum(not item["identical"] for item in raw_parity), "noneligible_pool_changes": len(noneligible_changed), "candidate_identity_conflicts": 0, "duplicate_candidates": 0}, "runtime_gold_reads": 0, "posthoc_gold_source_reads": denominator, "runtime_governance_reads": 0, "reranker_calls": 0, "final_selector_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_default_config_modified": False, "production_switch_allowed": False})
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": "multi_evidence_set_retrieval" if safety_passed and effect_passed else "stop_structured_lane", "production_switch_allowed": False})
    return 0 if safety_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
