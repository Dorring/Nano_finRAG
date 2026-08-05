import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/pdf-sr-v2-terminal-transfer"


def _sha(name: str) -> str:
    return hashlib.sha256((OUT / name).read_bytes()).hexdigest()


def test_prediction_seal_verifies_without_gold_reads() -> None:
    seal = json.loads((OUT / "prediction-seal.json").read_text())
    assert seal["predictions_sealed"] is True
    assert seal["prediction_count"] == 72
    assert seal["labels_read_before_seal"] == 0
    assert seal["protocol_hash"] == _sha("terminal-transfer-protocol.json")
    assert seal["baseline_prediction_hash"] == _sha("baseline-predictions.json")
    assert seal["e1_prediction_hash"] == _sha("e1-predictions.json")


def test_shadow_candidate_identity_is_one_to_one() -> None:
    identity = json.loads((OUT / "candidate-identity-integrity.json").read_text())
    assert identity["original_identity_count"] == identity["shadow_view_count"]
    assert identity["identity_loss_count"] == 0
    assert identity["identity_conflict_count"] == 0
    assert identity["duplicate_view_count"] == 0


def test_terminal_gate_fails_closed_and_forbids_tuning() -> None:
    acceptance = json.loads((OUT / "terminal-transfer-acceptance.json").read_text())
    assert acceptance["seal_verified_before_gold_load"] is True
    assert acceptance["thresholds"]["baseline_replay_parity_13_of_80"] is False
    assert acceptance["gate_passed"] is False
    assert acceptance["production_switch_allowed"] is False
    assert acceptance["post_score_tuning_allowed"] is False
    assert acceptance["next_gate"] == "stop_pdf_sr_v2_terminal_transfer"


def test_no_answer_is_explicitly_not_run() -> None:
    status = json.loads((OUT / "no-answer-status.json").read_text())
    assert status["no_answer_evaluation"] == "not_run"
