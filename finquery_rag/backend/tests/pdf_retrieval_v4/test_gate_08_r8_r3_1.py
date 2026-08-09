from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.structure_aware_rerank_view import (
    build_rerank_query_view,
    build_slot_rerank_query_view,
)

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "artifacts").exists():
    ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/evaluation"
R3 = BASE / "pdf-retrieval-v4-gate-08-r8-r3"
R31A = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a"
R31 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_historical_r3_predictions_immutable() -> None:
    assert sha(R3 / "rerank-predictions.jsonl.gz") == "d6409ee87e6e3c2c6f0c2f1dfbe4bcca7b54dfc089e60c8dfa9b1e7040da88f6"


def test_context_registry_has_exact_top100_identity() -> None:
    seal = json.loads((R31A / "prediction-seal.json").read_text())
    assert seal["sealed"] is True
    assert seal["cases"] == 72 and seal["candidate_occurrences"] == 7200
    assert seal["candidate_added"] == seal["candidate_removed"] == seal["candidate_mutation"] == 0
    assert seal["grade_a_with_authoritative_context"] == seal["grade_a_occurrences"] == 5251
    assert seal["gold_reads_before_seal"] == 0


def test_context_status_is_fail_closed() -> None:
    records = load(R31A / "top100-authoritative-context-v2.jsonl.gz")
    statuses = {candidate["context_status"] for record in records for candidate in record["candidates"]}
    assert statuses == {"authoritative_structured", "ambiguous_not_attached", "unmapped"}
    for record in records:
        for candidate in record["candidates"]:
            if candidate["context_status"] != "authoritative_structured":
                assert candidate["authoritative_evidence"] == []


def test_query_views_are_exact_r3_replay() -> None:
    old = {item["case_id"]: item for item in load(BASE / "pdf-retrieval-v4-gate-08-r8-r3-p0/rerank-input-views.jsonl.gz")}
    new = {item["case_id"]: item for item in load(R31A / "rerank-input-views-v2.jsonl.gz")}
    for case_id in old:
        assert old[case_id]["query_view_sha256"] == new[case_id]["query_view_sha256"]
        assert [item["candidate_key"] for item in old[case_id]["candidates"]] == [item["candidate_key"] for item in new[case_id]["candidates"]]


def test_prediction_reuses_exact_model_and_candidates() -> None:
    seal = json.loads((R31 / "prediction-seal.json").read_text())
    assert seal["model_revision"] == "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    assert seal["top100_sha256"] == "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"
    assert seal["max_length"] == 8192 and seal["dtype"] == "bfloat16"
    assert seal["prediction_sha256"] == sha(R31 / "rerank-predictions.jsonl.gz")
    assert seal["candidate_added"] == seal["candidate_removed"] == 0


def test_no_retrieval_gold_or_scan_before_replay_seal() -> None:
    protocol = json.loads((R31 / "protocol.json").read_text())
    for field in ("retrieval_runs", "index_reads", "embedding_calls", "gold_reads_before_seal", "governance_reads_before_seal", "reference_answer_reads", "expected_value_reads", "generation_calls"):
        assert protocol[field] == 0
    for field in ("model_scan", "instruction_scan", "weight_scan"):
        assert protocol[field] is False
    assert protocol["only_changed_variable"] == "document_context_source_v1_to_v2"


def test_formal_r3_1_decision() -> None:
    acceptance = json.loads((R31 / "acceptance.json").read_text())
    assert acceptance["metrics"]["strict_source_binding_recall_at_100"] == "68/80"
    assert acceptance["metrics"]["strict_source_binding_recall_at_5"] == "40/80"
    assert acceptance["candidate_mutation"] == 0
    assert acceptance["grade_a_context_coverage_complete"] is True
    assert acceptance["decision"] == "full_context_cross_encoder_insufficient"
    assert acceptance["next_gate"] == "qwen3_reranker_4b_capacity_escalation"


def test_slot_query_builder_is_implemented_but_not_scored() -> None:
    plan = {"raw_question": "growth?", "task_type": "calculation", "operation": "growth_rate", "operand_slots": []}
    slot = {"raw_metric_phrase": "revenue", "period": "FY2025", "required_evidence_shape": "atomic_fact"}
    view = build_slot_rerank_query_view(plan, slot)
    assert "revenue" in view and "FY2025" in view
    assert view != build_rerank_query_view({**plan, "operand_slots": [slot]})
    assert "[FOCUS OPERAND]" in view
    assert "Evidence supporting only another operand" in view
    assert not (R31 / "slot-aware-score.json").exists()
