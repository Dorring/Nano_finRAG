"""Contract tests for Gate 08 R3-RS slot-local ranking reseal."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
RS_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3-rs"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _records() -> list[dict]:
    with gzip.open(RS_DIR / "slot-local-rankings.jsonl.gz", "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_original_r3_prediction_not_mutated() -> None:
    seal = _json(R3_DIR / "prediction-seal.json")
    rs_seal = _json(RS_DIR / "prediction-seal.json")

    assert _sha256(R3_DIR / "predictions.jsonl.gz") == seal["prediction_hash"]
    assert rs_seal["original_r3_prediction_sha256"] == seal["prediction_hash"]
    assert rs_seal["original_r3_prediction_immutable"] is True


def test_replay_scope_exactly_18_multislot_cases() -> None:
    records = _records()

    assert len(records) == 18
    assert len({record["case_id"] for record in records}) == 18
    assert all(record["is_multi_slot"] for record in records)


def test_slot_definitions_and_query_hashes_persisted() -> None:
    for record in _records():
        assert len(record["slot_definitions"]) >= 2
        for order, slot in enumerate(record["slot_definitions"]):
            assert slot["slot_order"] == order
            assert slot["slot_id"]
            assert slot["query_plan_id"] == record["query_plan_id"]
            assert slot["query_sha256"] == hashlib.sha256(
                slot["query_text"].encode("utf-8")
            ).hexdigest()


def test_all_family_rankings_persisted() -> None:
    for record in _records():
        definitions = {slot["slot_id"] for slot in record["slot_definitions"]}
        assert set(record["slot_family_rankings"]) == definitions
        for ranking in record["slot_family_rankings"].values():
            for family in ("raw", "structured"):
                assert len(ranking[family]["bm25"]) <= 50
                assert len(ranking[family]["dense"]) <= 50
                assert len(ranking[family]["fused"]) >= 50
            assert len(ranking["early_cross_family_fused"]) >= 50


def test_reconstructed_pool_preserves_slot_trace() -> None:
    for record in _records():
        for family in ("e1", "e2_expanded", "e3_expanded"):
            pool = record["reconstructed_candidate_pools"][family]
            assert len(pool) <= 40
            assert [item["pool_rank"] for item in pool] == list(
                range(1, len(pool) + 1)
            )
            assert all(item["slot_id"] for item in pool)
            assert all(item["slot_rank"] >= 1 for item in pool)
            assert all(item["supporting_slots"] for item in pool)


def test_e1_e2_e3_multislot_exact_parity() -> None:
    parity = _json(RS_DIR / "slot-replay-parity.json")

    assert parity["case_count"] == 18
    assert parity["all_exact"] is True
    assert parity["parity_counts"] == {
        "e1": 18,
        "e2_expanded": 18,
        "e3_expanded": 18,
    }
    assert all(all(value for key, value in case.items() if key != "case_id") for case in parity["cases"])


def test_query_contract_has_no_drift() -> None:
    parity = _json(RS_DIR / "query-contract-parity.json")

    assert parity["semantic_diff_empty"] is True
    assert parity["all_source_hashes_exact"] is True


def test_seal_and_composite_manifest() -> None:
    seal = _json(RS_DIR / "prediction-seal.json")
    composite = _json(RS_DIR / "r4-composite-input-manifest.json")

    assert seal["sealed"] is True
    assert seal["slot_sidecar_records"] == 18
    assert seal["slot_sidecar_sha256"] == _sha256(
        RS_DIR / "slot-local-rankings.jsonl.gz"
    )
    assert composite["sealed"] is True
    assert composite["coverage"] == {
        "single_slot_cases": 54,
        "multi_slot_cases": 18,
        "total_cases": 72,
    }
    assert composite["slot_replay_parity"] == {
        "e1": "18/18",
        "e2_expanded": "18/18",
        "e3_expanded": "18/18",
    }


def test_search_accounting_is_truthful() -> None:
    seal = _json(RS_DIR / "prediction-seal.json")
    accounting = seal["search_accounting"]

    assert accounting["total_slots"] > 18
    assert accounting["raw_bm25_searches"] == accounting["total_slots"]
    assert accounting["raw_dense_searches"] == accounting["total_slots"]
    assert accounting["structured_bm25_searches"] == accounting["total_slots"]
    assert accounting["structured_dense_searches"] == accounting["total_slots"]
    assert accounting["total_lane_searches"] == 4 * accounting["total_slots"]
    assert accounting["index_reads"] > 0
    assert accounting["index_builds"] == 0


def test_no_forbidden_reads_or_mutations() -> None:
    seal = _json(RS_DIR / "prediction-seal.json")
    protocol = _json(RS_DIR / "protocol.json")

    assert seal["gold_reads_before_seal"] == 0
    assert seal["governance_reads_before_seal"] == 0
    assert seal["reference_answer_reads_before_seal"] == 0
    assert seal["expected_value_reads_before_seal"] == 0
    assert seal["parameter_scan"] is False
    assert seal["quota_scan"] is False
    assert seal["index_builds"] == 0
    assert seal["production_index_writes"] == 0
    assert protocol["index_mutations"] == 0
    assert protocol["bridge_mutations"] == 0


def test_acceptance_closes_r4_input_contract() -> None:
    acceptance = _json(RS_DIR / "acceptance.json")

    assert acceptance["decision"] == "slot_local_family_rankings_resealed"
    assert acceptance["next_gate"] == "lane_preserving_fusion"
    assert acceptance["gate_passed"] is True
