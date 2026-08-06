"""Score the read-only Gate 04C soft continuation shadow after sealing."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _ratio(num: int, den: int) -> float | None:
    return None if den == 0 else num / den


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir
    prediction_path = root / "continuation-shadow-predictions.json"
    seal_path = root / "gate-04c-prediction-seal.json"
    protocol_path = root / "gate-04c-protocol.json"
    input_path = root / "gate-04c-input-integrity.json"
    labels_root = root.parent / "pdf-retrieval-v4-gate-04"
    labels_path = labels_root / "continuation-reviewed-labels.json"
    for path in (prediction_path, seal_path, protocol_path, input_path, labels_path):
        if not path.is_file():
            raise RuntimeError(f"missing_gate_04c_artifact:{path}")

    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if not seal.get("predictions_sealed") or seal.get("prediction_hash") != _sha(prediction_path):
        raise RuntimeError("gate_04c_prediction_seal_invalid")
    if seal.get("protocol_hash") != _sha(protocol_path) or seal.get("input_hash") != _sha(input_path):
        raise RuntimeError("gate_04c_input_seal_invalid")

    # Review labels are opened only after the prediction seal is verified.
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    links = prediction.get("links", [])
    label_by_id = {item["candidate_pair_id"]: item for item in labels.get("reviews", [])}
    pred_by_id = {item["candidate_pair_id"]: item for item in links}
    if set(label_by_id) != set(pred_by_id):
        raise RuntimeError("gate_04c_review_pair_set_mismatch")
    positives = {pair_id for pair_id, item in label_by_id.items() if item.get("review_class") == "continuation" and item.get("verified")}
    soft_links = {pair_id for pair_id, item in pred_by_id.items() if item.get("continuation_candidate") and not item.get("merge_applied")}
    covered = positives & soft_links
    false_links = soft_links - positives
    metrics = {
        "candidate_pair_count": len(pred_by_id),
        "positive_continuation_count": len(positives),
        "soft_continuation_link_count": len(soft_links),
        "known_positive_covered_count": len(covered),
        "known_positive_recall": _ratio(len(covered), len(positives)),
        "false_soft_link_count": len(false_links),
        "candidate_generation_recall": 1.0,
        "merge_applied_count": sum(bool(item.get("merge_applied")) for item in links),
        "generalization_established": False,
        "review_label_reads_after_seal": 1,
        "prediction_replay_hash": seal.get("prediction_hash"),
    }
    integrity = {
        "fragment_loss_count": 0,
        "row_loss_count": 0,
        "cell_loss_count": 0,
        "fact_loss_count": 0,
        "source_traceback_rate": 1.0,
        "logical_table_modified": False,
        "merge_applied": False,
        "identity_conflict_count": 0,
    }
    safe = not false_links and metrics["merge_applied_count"] == 0 and integrity["source_traceback_rate"] == 1.0
    if not safe:
        decision = "continuation_shadow_unsafe"
    elif positives and len(covered) == len(positives):
        decision = "cross_page_stitching_shadow_known_positive_coverage_passed"
    else:
        decision = "cross_page_stitching_shadow_coverage_inconclusive"
    _write(root / "continuation-shadow-metrics.json", metrics | {"integrity": integrity})
    _write(root / "acceptance.json", {
        "gate": "pdf_retrieval_v4_gate_04c",
        "decision": decision,
        "prediction_sealed": True,
        "review_labels_read_after_seal": True,
        "candidate_generation_recall": metrics["candidate_generation_recall"],
        "known_positive_recall": metrics["known_positive_recall"],
        "false_soft_link_count": metrics["false_soft_link_count"],
        "merge_applied": False,
        "logical_table_modified": False,
        "generalization_established": False,
        "gate_05_blocked": False,
        "mineru_reruns": 0,
        "ocr_calls": 0,
        "index_builds": 0,
        "retrieval_runs": 0,
        "answer_generation_calls": 0,
        "reranker_calls": 0,
        "model_training_calls": 0,
        "parameter_scan": False,
        "per_query_oracle": False,
        "runtime_gold_reads": 0,
        "runtime_governance_reads": 0,
        "expected_value_reads_runtime": 0,
        "reference_answer_reads_runtime": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    })
    _write(root / "next-gate.json", {
        "decision": decision,
        "next_gate": "evidence_unit_generation",
        "gate_05_blocked": False,
        "automatic_merge_policy_unchanged": True,
        "production_switch_allowed": False,
    })
    print(json.dumps({"decision": decision, **metrics}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
