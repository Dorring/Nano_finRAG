"""Tests for the Gate 08 R4 frozen-input preflight."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.evaluation.audit_pdf_v4_gate_08_r4_input_contract import (
    audit_r3_inputs,
)

ROOT = Path(__file__).resolve().parents[2]
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
R4_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r4"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_predictions() -> list[dict]:
    with gzip.open(R3_DIR / "predictions.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_r3_seal_is_verified() -> None:
    integrity = _load_json(R4_DIR / "input-integrity.json")

    assert integrity["r3_seal_verified"] is True
    assert integrity["prediction_count"] == 72


def test_single_slot_family_rankings_are_replayable() -> None:
    integrity = _load_json(R4_DIR / "input-integrity.json")

    assert integrity["family_rank_fields_missing_count"] == 0
    assert integrity["f1_single_slot_replayable"] is True


def test_multi_slot_rankings_are_missing_from_frozen_input() -> None:
    integrity = _load_json(R4_DIR / "input-integrity.json")

    assert integrity["multi_slot_case_count"] == 18
    assert integrity["multi_slot_missing_slot_definitions_count"] == 18
    assert integrity["multi_slot_missing_slot_family_rankings_count"] == 18
    assert integrity["composite_input"]["sealed"] is True
    assert integrity["f2_full_lane_preserving_replayable"] is True
    assert integrity["formal_prediction_seal_allowed"] is True


def test_combined_pool_cannot_recover_slot_trace() -> None:
    integrity = _load_json(R4_DIR / "input-integrity.json")

    assert integrity["combined_pool_preserves_slot_trace"] is False
    assert "slot_id" in integrity["combined_pool_serialization_contract"]


def test_audit_is_deterministic() -> None:
    predictions = _load_predictions()
    seal = _load_json(R3_DIR / "prediction-seal.json")
    first = audit_r3_inputs(
        predictions,
        prediction_path=R3_DIR / "predictions.jsonl.gz",
        seal=seal,
    )
    second = audit_r3_inputs(
        predictions,
        prediction_path=R3_DIR / "predictions.jsonl.gz",
        seal=seal,
    )

    assert first == second


def test_gate_accepts_only_verified_composite_input() -> None:
    acceptance = _load_json(R4_DIR / "input-acceptance.json")
    next_gate = _load_json(R4_DIR / "input-next-gate.json")

    assert acceptance["decision"] == "lane_preserving_fusion_input_contract_passed"
    assert acceptance["gate_passed"] is True
    assert acceptance["prediction_generated"] is False
    assert acceptance["prediction_sealed"] is False
    assert next_gate["next_gate"] == "run_lane_preserving_fusion_prediction"


def test_composite_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    composite_path = tmp_path / "r4-composite-input-manifest.json"
    composite_path.write_text(
        (ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs/r4-composite-input-manifest.json").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
    )
    from scripts.evaluation.audit_pdf_v4_gate_08_r4_input_contract import (
        validate_composite_input,
    )

    result = validate_composite_input(
        composite_path,
        original_prediction_path=R3_DIR / "predictions.jsonl.gz",
        original_seal_path=R3_DIR / "prediction-seal.json",
    )
    assert result["sealed"] is False


def test_no_retrieval_or_tuning_was_run() -> None:
    acceptance = _load_json(R4_DIR / "input-acceptance.json")
    safety = acceptance["safety"]

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
    ):
        assert safety[key] == 0
    assert safety["parameter_scan"] is False
    assert safety["quota_scan"] is False
    assert safety["production_switch_allowed"] is False


def test_frozen_budgets_are_unchanged() -> None:
    protocol = _load_json(R4_DIR / "input-protocol.json")

    assert protocol["rrf_k"] == 60
    assert protocol["final_pool_k"] == 40
    assert protocol["slot_top_k"] == 20
    assert protocol["structured_protected_k"] == 20
