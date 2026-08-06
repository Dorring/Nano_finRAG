"""Build a soft continuation-link shadow without changing Gate 04 Logical Tables."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04"
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-04c"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _soft_link(item: dict[str, Any]) -> tuple[bool, float, list[str]]:
    features = item.get("features", {})
    if item.get("hard_blockers"):
        return False, 0.0, [f"blocked:{value}" for value in item["hard_blockers"]]
    # Generic shape for a continuation fragment that has no repeated header:
    # a large bottom fragment followed by a small top fragment with the same
    # columns and local accounting scope. This is a shadow link, not a merge.
    signals = {
        "same_section": bool(features.get("same_section")),
        "same_statement": bool(features.get("same_statement")),
        "column_count_compatible": bool(features.get("column_count_compatible")),
        "column_geometry": float(features.get("column_band_similarity", 0.0)) >= 0.90,
        "scale_compatible": bool(features.get("scale_compatible")),
        "currency_compatible": bool(features.get("currency_compatible")),
        "period_compatible": bool(features.get("period_set_compatible")),
        "left_bottom": bool(features.get("left_near_page_bottom")),
        "right_top": bool(features.get("right_near_page_top")),
        "row_style": bool(features.get("row_label_style_compatible")),
        "unrepeated_header": not bool(features.get("header_fingerprint_equal")),
    }
    # The row-count asymmetry is supplied by the candidate generator in the
    # shadow feature contract when available; absent values fail closed.
    left_rows, right_rows = features.get("left_row_count"), features.get("right_row_count")
    signals["fragment_size_asymmetry"] = bool(left_rows and right_rows and left_rows >= 15 and right_rows <= 10)
    right_label = str(features.get("right_first_row_label") or "").strip().lower()
    signals["right_starts_with_data_row"] = features.get("right_first_row_role") not in {None, "header", "separator"}
    signals["right_not_unit_only_row"] = not bool(re.search(r"\b(?:in|except)\s+(?:usd\s+)?(?:millions?|thousands?)\b", right_label))
    active = [name for name, value in signals.items() if value]
    allowed = all(signals[name] for name in ("same_section", "same_statement", "column_count_compatible", "column_geometry", "scale_compatible", "currency_compatible", "period_compatible", "left_bottom", "right_top", "row_style", "unrepeated_header", "fragment_size_asymmetry", "right_starts_with_data_row", "right_not_unit_only_row"))
    return allowed, round(len(active) / len(signals), 6), active


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--code-commit", default="working-tree")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    candidate_path = args.input / "continuation-candidates.json"
    prediction_seal_path = args.input / "gate-04-prediction-seal.json"
    for path in (candidate_path, prediction_seal_path, args.input / "logical-table-integrity.json"):
        if not path.is_file():
            raise RuntimeError(f"missing_gate_04c_input:{path.name}")
    seal = json.loads(prediction_seal_path.read_text(encoding="utf-8"))
    prediction_path = args.input / "gate-04-predictions.json"
    if not seal.get("predictions_sealed") or seal.get("prediction_hash") != _sha(prediction_path):
        raise RuntimeError("gate_04_prediction_seal_invalid")
    candidates = json.loads(candidate_path.read_text(encoding="utf-8")).get("candidates", [])
    links = []
    for item in candidates:
        accepted, confidence, signals = _soft_link(item)
        links.append({"candidate_pair_id": item["candidate_pair_id"], "document_id": item["document_id"], "left_fragment_id": item["left_fragment_id"], "right_fragment_id": item["right_fragment_id"], "continuation_candidate": accepted, "continuation_confidence": confidence, "continuation_signals": signals, "continuation_group_id": "continuation-group:" + _hash([item["document_id"], item["left_fragment_id"], item["right_fragment_id"]]) if accepted else None, "merge_applied": False})
    protocol = {"gate": "pdf_retrieval_v4_gate_04c", "evaluation_type": "shadow_only_post_benchmark_iterative_evaluation", "code_commit": args.code_commit, "input_gate": "pdf_retrieval_v4_gate_04", "gate_04_prediction_hash": _sha(prediction_path), "candidate_hash": _sha(candidate_path), "merge_applied": False, "automatic_merge_policy_changed": False, "oracle_blind": True, "mineru_reruns": 0, "ocr_calls": 0, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0, "production_switch_allowed": False}
    input_integrity = {"gate_04_prediction_sha256": _sha(prediction_path), "gate_04_seal_sha256": _sha(prediction_seal_path), "candidate_sha256": _sha(candidate_path), "logical_table_integrity_sha256": _sha(args.input / "logical-table-integrity.json")}
    _write(args.out / "gate-04c-protocol.json", protocol)
    _write(args.out / "gate-04c-input-integrity.json", input_integrity)
    _write(args.out / "continuation-shadow-predictions.json", {"candidate_count": len(links), "links": links, "merge_applied": False})
    prediction_path_out = args.out / "continuation-shadow-predictions.json"
    _write(args.out / "gate-04c-prediction-seal.json", {"prediction_count": len(links), "oracle_reads_before_seal": 0, "review_label_reads_before_seal": 0, "input_hash": _sha(args.out / "gate-04c-input-integrity.json"), "protocol_hash": _sha(args.out / "gate-04c-protocol.json"), "prediction_hash": _sha(prediction_path_out), "predictions_sealed": True, "merge_applied": False})
    _write(args.out / "acceptance.json", {"gate": "pdf_retrieval_v4_gate_04c", "prediction_sealed": True, "decision": "pending_posthoc_scoring", "next_gate": "score_continuation_shadow", "merge_applied": False, "gate_05_blocked": False, "production_switch_allowed": False, "mineru_reruns": 0, "ocr_calls": 0, "index_builds": 0, "retrieval_runs": 0, "production_index_writes": 0})
    _write(args.out / "next-gate.json", {"decision": "pending_posthoc_scoring", "next_gate": "score_continuation_shadow", "gate_05_blocked": False})
    print(json.dumps({"candidate_count": len(links), "soft_link_count": sum(item["continuation_candidate"] for item in links), "merge_applied": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
