#!/usr/bin/env python3
"""Audit whether sealed Gate 08 R3 predictions can support Gate 08 R4.

This is a read-only preflight. It never opens indexes, runs retrieval, reads gold,
or produces fusion predictions. R4 F2 requires per-slot Raw and Structured family
rankings; the audit fails closed when those frozen inputs are absent.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
R3_RS_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs"
R4_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r4"

RRF_K = 60
FINAL_POOL_K = 40
SLOT_TOP_K = 20
STRUCTURED_PROTECTED_K = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def audit_r3_inputs(
    predictions: list[dict[str, Any]],
    *,
    prediction_path: Path,
    seal: dict[str, Any],
) -> dict[str, Any]:
    """Return the deterministic R4 frozen-input contract audit."""
    multi_cases = [record for record in predictions if record.get("is_multi_slot")]
    missing_slot_definitions: list[str] = []
    missing_slot_family_rankings: list[str] = []
    family_rank_fields_missing: list[str] = []

    for record in predictions:
        case_id = str(record.get("case_id") or "")
        for family in ("candidate_raw", "structured_expanded"):
            fused = (record.get(family) or {}).get("fused")
            if not isinstance(fused, list):
                family_rank_fields_missing.append(f"{case_id}:{family}")
        if not record.get("is_multi_slot"):
            continue
        if not (record.get("slot_definitions") or record.get("slot_queries")):
            missing_slot_definitions.append(case_id)
        if not record.get("slot_family_rankings"):
            missing_slot_family_rankings.append(case_id)

    file_hash = _sha256(prediction_path)
    declared_hash = str(seal.get("prediction_hash") or "")
    seal_verified = (
        seal.get("sealed") is True
        and seal.get("prediction_count") == len(predictions)
        and declared_hash == file_hash
    )
    f1_replayable = not family_rank_fields_missing
    f2_replayable = (
        f1_replayable
        and not missing_slot_definitions
        and not missing_slot_family_rankings
    )

    return {
        "schema": "pdf-retrieval-v4/gate-08-r4/input-integrity/v1",
        "r3_prediction_path": str(prediction_path),
        "r3_prediction_sha256": file_hash,
        "r3_seal_declared_prediction_hash": declared_hash,
        "r3_seal_verified": seal_verified,
        "prediction_count": len(predictions),
        "single_slot_case_count": len(predictions) - len(multi_cases),
        "multi_slot_case_count": len(multi_cases),
        "family_rank_fields_missing_count": len(family_rank_fields_missing),
        "family_rank_fields_missing": family_rank_fields_missing,
        "multi_slot_missing_slot_definitions_count": len(missing_slot_definitions),
        "multi_slot_missing_slot_definitions": missing_slot_definitions,
        "multi_slot_missing_slot_family_rankings_count": len(
            missing_slot_family_rankings
        ),
        "multi_slot_missing_slot_family_rankings": missing_slot_family_rankings,
        "combined_pool_preserves_slot_trace": False,
        "combined_pool_serialization_contract": (
            "candidate_key/source/rank only; slot_id, slot_rank, supporting_slots "
            "are discarded by R3 _build_combined_pool"
        ),
        "f1_single_slot_replayable": f1_replayable,
        "f2_full_lane_preserving_replayable": f2_replayable,
        "formal_prediction_seal_allowed": f2_replayable,
        "blocker": None
        if f2_replayable
        else "sealed_r3_missing_slot_local_family_rankings",
    }


def validate_composite_input(
    composite_path: Path,
    *,
    original_prediction_path: Path,
    original_seal_path: Path,
) -> dict[str, Any]:
    """Validate the sealed R3 + R3-RS composite without reading indexes."""
    if not composite_path.is_file():
        return {
            "present": False,
            "sealed": False,
            "hashes_verified": False,
            "coverage_verified": False,
            "parity_verified": False,
            "sidecar_records_verified": False,
        }

    composite = _load_json(composite_path)
    sidecar_path = composite_path.parent / "slot-local-rankings.jsonl.gz"
    sidecar_seal_path = composite_path.parent / "prediction-seal.json"
    if not sidecar_path.is_file() or not sidecar_seal_path.is_file():
        return {
            "present": True,
            "sealed": False,
            "hashes_verified": False,
            "coverage_verified": False,
            "parity_verified": False,
            "sidecar_records_verified": False,
        }
    sidecar_seal = _load_json(sidecar_seal_path)
    original = composite.get("original_r3") or {}
    sidecar = composite.get("slot_local_sidecar") or {}
    coverage = composite.get("coverage") or {}
    parity = composite.get("slot_replay_parity") or {}
    hashes_verified = (
        original.get("prediction_sha256") == _sha256(original_prediction_path)
        and original.get("seal_sha256") == _sha256(original_seal_path)
        and sidecar.get("prediction_sha256") == _sha256(sidecar_path)
        and sidecar.get("seal_sha256") == _sha256(sidecar_seal_path)
        and sidecar_seal.get("slot_sidecar_sha256") == _sha256(sidecar_path)
    )
    coverage_verified = coverage == {
        "single_slot_cases": 54,
        "multi_slot_cases": 18,
        "total_cases": 72,
    }
    parity_verified = parity == {
        "e1": "18/18",
        "e2_expanded": "18/18",
        "e3_expanded": "18/18",
    }
    records_verified = sidecar_seal.get("slot_sidecar_records") == 18
    sealed = (
        composite.get("sealed") is True
        and sidecar_seal.get("sealed") is True
        and hashes_verified
        and coverage_verified
        and parity_verified
        and records_verified
    )
    return {
        "present": True,
        "sealed": sealed,
        "hashes_verified": hashes_verified,
        "coverage_verified": coverage_verified,
        "parity_verified": parity_verified,
        "sidecar_records_verified": records_verified,
        "composite_manifest_sha256": _sha256(composite_path),
        "sidecar_prediction_sha256": _sha256(sidecar_path),
        "sidecar_seal_sha256": _sha256(sidecar_seal_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=R3_DIR / "predictions.jsonl.gz",
    )
    parser.add_argument(
        "--seal",
        type=Path,
        default=R3_DIR / "prediction-seal.json",
    )
    parser.add_argument(
        "--composite",
        type=Path,
        default=R3_RS_DIR / "r4-composite-input-manifest.json",
    )
    parser.add_argument("--out-dir", type=Path, default=R4_DIR)
    args = parser.parse_args()

    predictions = _load_predictions(args.predictions)
    seal = _load_json(args.seal)
    integrity = audit_r3_inputs(
        predictions,
        prediction_path=args.predictions,
        seal=seal,
    )
    composite_integrity = validate_composite_input(
        args.composite,
        original_prediction_path=args.predictions,
        original_seal_path=args.seal,
    )
    integrity["composite_input"] = composite_integrity
    if composite_integrity["sealed"]:
        integrity["f2_full_lane_preserving_replayable"] = True
        integrity["formal_prediction_seal_allowed"] = True
        integrity["blocker"] = None

    protocol = {
        "schema": "pdf-retrieval-v4/gate-08-r4/preflight/v1",
        "gate": "pdf_retrieval_v4_gate_08_r4",
        "phase": "lane_preserving_fusion_input_preflight",
        "evaluation_type": "post_benchmark_iterative_evaluation",
        "frozen_inputs": ["gate_08_r3_r4_composite_seal"],
        "required_r3_fields": [
            "candidate_raw.fused",
            "structured_expanded.fused",
            "is_multi_slot",
            "slot_definitions",
            "slot_family_rankings.raw",
            "slot_family_rankings.structured",
        ],
        "rrf_k": RRF_K,
        "final_pool_k": FINAL_POOL_K,
        "slot_top_k": SLOT_TOP_K,
        "structured_protected_k": STRUCTURED_PROTECTED_K,
        "parameter_scan": False,
        "quota_scan": False,
        "bm25_searches": 0,
        "dense_searches": 0,
        "embedding_calls": 0,
        "index_builds": 0,
        "index_reads": 0,
        "gold_reads": 0,
        "governance_reads": 0,
        "reranker_calls": 0,
        "calculator_calls": 0,
        "generator_calls": 0,
        "production_index_writes": 0,
        "production_switch_allowed": False,
    }

    blocked = not integrity["f2_full_lane_preserving_replayable"]
    decision = (
        "lane_preserving_fusion_input_contract_blocked"
        if blocked
        else "lane_preserving_fusion_input_contract_passed"
    )
    next_gate = (
        "stop_and_reseal_slot_local_family_rankings"
        if blocked
        else "run_lane_preserving_fusion_prediction"
    )
    acceptance = {
        "gate": "pdf_retrieval_v4_gate_08_r4",
        "decision": decision,
        "next_gate": next_gate,
        "gate_passed": not blocked,
        "prediction_generated": False,
        "prediction_sealed": False,
        "frozen_metrics": {
            "e0": "42/80",
            "e1_candidate_raw": "47/80",
            "e2_expanded": "57/80",
            "e3_early_fusion": "52/80",
            "family_union": "58/80",
            "gross_fusion_loss": 8,
            "net_union_gap": 6,
            "fusion_synergy_gain": 2,
        },
        "input_integrity": {
            "r3_seal_verified": integrity["r3_seal_verified"],
            "single_slot_f1_replayable": integrity["f1_single_slot_replayable"],
            "full_f2_replayable": integrity[
                "f2_full_lane_preserving_replayable"
            ],
            "composite_seal_verified": composite_integrity["sealed"],
            "multi_slot_case_count": integrity["multi_slot_case_count"],
            "missing_slot_family_rankings": integrity[
                "multi_slot_missing_slot_family_rankings_count"
            ],
        },
        "safety": {
            key: protocol[key]
            for key in (
                "bm25_searches",
                "dense_searches",
                "embedding_calls",
                "index_builds",
                "index_reads",
                "gold_reads",
                "governance_reads",
                "reranker_calls",
                "calculator_calls",
                "generator_calls",
                "production_index_writes",
                "production_switch_allowed",
                "parameter_scan",
                "quota_scan",
            )
        },
        "blocking_rationale": None
        if not blocked
        else (
            "F2 requires slot-local Raw/Structured family rankings before the "
            "unchanged build_slot_pool round-robin. R3 sealed predictions contain "
            "neither slot definitions nor slot-local family rankings for all "
            f"{integrity['multi_slot_case_count']} multi-slot cases. Reusing global "
            "family ranks or combined pools would violate the frozen R4 contract."
        ),
    }
    next_gate_payload = {
        "current_gate": "pdf_retrieval_v4_gate_08_r4",
        "decision": decision,
        "next_gate": next_gate,
        "required_remediation": [
            "Persist slot definitions in the prediction-stage output",
            "Persist per-slot candidate_raw fused rankings",
            "Persist per-slot structured_expanded fused rankings",
            "Preserve slot_id, slot_rank, and supporting_slots before combined-pool serialization",
            "Create a new sealed prediction under an explicitly amended protocol",
        ],
        "forbidden_remediation": [
            "infer slots from gold or governance",
            "substitute global family ranks for slot-local ranks",
            "claim F2 from the serialized combined pool",
            "run unsealed retrieval inside R4",
        ],
        "production_switch_allowed": False,
    }

    _write_json(args.out_dir / "input-protocol.json", protocol)
    _write_json(args.out_dir / "input-integrity.json", integrity)
    _write_json(args.out_dir / "input-acceptance.json", acceptance)
    _write_json(args.out_dir / "input-next-gate.json", next_gate_payload)

    print(f"decision={decision}")
    print(f"multi_slot_cases={integrity['multi_slot_case_count']}")
    print(
        "missing_slot_family_rankings="
        f"{integrity['multi_slot_missing_slot_family_rankings_count']}"
    )
    print(f"next_gate={next_gate}")
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
