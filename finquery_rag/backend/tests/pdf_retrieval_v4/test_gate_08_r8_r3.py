from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.structure_aware_rerank_view import (
    RERANK_INSTRUCTION,
    build_rerank_document_view,
    build_rerank_query_view,
)

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "artifacts").exists():
    ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/evaluation"
P0 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-p0"
R3 = BASE / "pdf-retrieval-v4-gate-08-r8-r3"
RERANKER_SOURCE = ROOT / "src/pdf_retrieval_v4/qwen3_reranker.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_gzip(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_query_view_uses_plan_without_identity_or_gold() -> None:
    view = build_rerank_query_view({
        "raw_question": "What was revenue?", "task_type": "calculation",
        "operation": "growth_rate", "operand_slots": [{"raw_metric_phrase": "revenue", "period": "FY2025", "required_evidence_shape": "atomic_fact"}],
    })
    assert "What was revenue?" in view and "growth_rate" in view and "FY2025" in view
    assert "case_id" not in view and "source_index" not in view and "candidate_key" not in view


def test_document_view_omits_missing_fields_and_supports_raw_only() -> None:
    view = build_rerank_document_view({"document_id": "doc", "raw_text": "raw evidence", "metadata": {}}, [])
    assert "Document: doc" in view and "raw evidence" in view
    assert "None" not in view and "null" not in view


def test_numeric_value_originates_from_candidate_evidence() -> None:
    attachment = [{"evidence_type": "atomic_fact", "context": {}, "semantic_payload": {"value_normalized": 245, "leaf_metric": "Revenue"}}]
    view = build_rerank_document_view({"document_id": "doc", "raw_text": "text", "metadata": {}}, attachment)
    assert "Value: 245" in view and "Metric: Revenue" in view


def test_p0_contract_exact_and_sealed() -> None:
    seal = json.loads((P0 / "input-seal.json").read_text())
    assert seal["sealed"] is True
    assert seal["cases"] == 72 and seal["candidates"] == 7200
    assert seal["candidate_added"] == seal["candidate_removed"] == seal["candidate_mutation"] == 0
    assert seal["gold_reads"] == seal["reference_answer_reads"] == 0
    assert seal["rerank_input_views_sha256"] == sha(P0 / "rerank-input-views.jsonl.gz")


def test_model_and_prediction_seal_are_exact() -> None:
    seal = json.loads((R3 / "prediction-seal.json").read_text())
    assert seal["sealed"] is True
    assert seal["model_id"] == "Qwen/Qwen3-Reranker-0.6B"
    assert seal["model_revision"] == "e61197ed45024b0ed8a2d74b80b4d909f1255473"
    assert seal["max_length"] == 8192 and seal["dtype"] == "bfloat16"
    assert seal["prediction_sha256"] == sha(R3 / "rerank-predictions.jsonl.gz")
    assert seal["p0_input_views_sha256"] == sha(P0 / "rerank-input-views.jsonl.gz")


def test_prediction_is_exact_top100_permutation() -> None:
    predictions = {item["case_id"]: item for item in load_gzip(R3 / "rerank-predictions.jsonl.gz")}
    views = {item["case_id"]: item for item in load_gzip(P0 / "rerank-input-views.jsonl.gz")}
    assert len(predictions) == 72
    for case_id, record in predictions.items():
        ranked = record["ranked_candidates"]
        assert len(ranked) == 100
        assert {item["candidate_key"] for item in ranked} == {item["candidate_key"] for item in views[case_id]["candidates"]}
        assert [item["post_rerank_rank"] for item in ranked] == list(range(1, 101))
        assert ranked == sorted(ranked, key=lambda item: (-item["reranker_score"], item["pre_rerank_rank"], item["candidate_key"]))


def test_no_generation_search_scan_or_gold_before_seal() -> None:
    protocol = json.loads((R3 / "protocol.json").read_text())
    for field in ("retrieval_runs", "index_reads", "embedding_calls", "gold_reads_before_seal", "governance_reads_before_seal", "generation_calls", "calculator_calls"):
        assert protocol[field] == 0
    for field in ("model_scan", "instruction_scan", "max_length_scan", "weight_scan", "query_rewrite"):
        assert protocol[field] is False


def test_formal_score_uses_80_bindings_and_blocks_escalation() -> None:
    acceptance = json.loads((R3 / "acceptance.json").read_text())
    assert acceptance["metrics"]["strict_source_binding_recall_at_100"] == "68/80"
    assert acceptance["metrics"]["strict_source_binding_recall_at_5"] == "42/80"
    assert acceptance["candidate_mutation"] == 0
    assert acceptance["decision"] == "structure_aware_cross_encoder_insufficient"
    assert acceptance["next_gate"] == "rerank_failure_audit"
    assert acceptance["final_retrieval_target_reached"] is False


def test_fixed_instruction_and_max_length_contract() -> None:
    assert "financial-report question" in RERANK_INSTRUCTION
    source = RERANKER_SOURCE.read_text()
    assert "max_length: int = 8192" in source
    assert "generation: bool = False" in source
