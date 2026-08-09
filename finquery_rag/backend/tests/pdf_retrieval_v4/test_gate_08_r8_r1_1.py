from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R1 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r1"
R11 = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r1-1"


def load(name: str) -> dict:
    return json.loads((R11 / name).read_text())


def test_original_r8_r1_prediction_hash_immutable() -> None:
    prediction = R1 / "candidate-top50-predictions.jsonl.gz"
    expected = json.loads((R1 / "prediction-seal.json").read_text())[
        "prediction_sha256"
    ]
    assert hashlib.sha256(prediction.read_bytes()).hexdigest() == expected


def test_identity_catalogs_not_mixed_for_ten_loss_claim() -> None:
    audit = load("strict-identity-contract-audit.json")
    assert audit["identity_contract_consistent"] is False
    assert audit["labels_golden"]["identity_count"] == 80
    assert audit["gate1_governance"]["identity_count"] == 73
    assert audit["identity_disagreement_count"] == 13


def test_net_gap_not_mislabeled_as_gross_loss() -> None:
    audit = load("strict-identity-contract-audit.json")["labels_golden"]
    assert audit["net_gap"] == 5
    assert audit["gross_loss"] == 6
    assert audit["selector_synergy"] == 1
    assert audit["net_gap"] == audit["gross_loss"] - audit["selector_synergy"]


def test_every_auditable_loss_has_one_first_failure() -> None:
    audit = load("boundary-loss-audit.json")
    records = audit["records"]
    assert len(records) == audit["auditable_same_identity_loss_count"] == 6
    assert all(record["first_failure_stage"] for record in records)


def test_raw_regressions_have_causal_trace() -> None:
    records = {
        (record["case_id"], record["source_index"]): record
        for record in load("boundary-loss-audit.json")["records"]
    }
    assert records[("pfe_fy2024_005", 0)]["production_raw_rank"] == 30
    tesla = records[("tsla_fy2025_007", 1)]
    assert tesla["production_raw_rank"] == 10
    assert tesla["top_level_candidate_rank"] == 28
    assert tesla["first_failure_stage"] == "main_query_residual_displacement"


def test_r1_2_is_blocked_until_identity_contract_closes() -> None:
    acceptance = load("acceptance.json")
    assert acceptance["decision"] == "boundary_loss_identity_contract_blocked"
    assert acceptance["requested_ten_loss_audit_completed"] is False
    assert acceptance["next_gate"] == "stop_and_fix_strict_gold_identity_contract"


def test_post_seal_audit_did_not_rerun_prediction_or_fusion() -> None:
    acceptance = load("acceptance.json")
    assert acceptance["prediction_reruns"] == 0
    assert acceptance["fusion_reruns"] == 0
    assert acceptance["retrieval_runs"] == 0
