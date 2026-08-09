from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.slot_aware_neural_composition import compose_slot_aware_top5

ROOT = Path(__file__).resolve().parents[2]
if not (ROOT / "artifacts").exists():
    ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/evaluation"
R31A = BASE / "pdf-retrieval-v4-gate-08-r8-r3-1a"
R32 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_input_is_exact_r3_1_v2() -> None:
    seal = json.loads((R32 / "prediction-seal.json").read_text())
    assert seal["r3_1a_views_sha256"] == sha(R31A / "rerank-input-views-v2.jsonl.gz")
    assert seal["top100_sha256"] == "ced014c357a3c8862a5ae3723a4c618556156542fd63aa2a8fd4bf766b60f01a"
    integrity = json.loads((R32 / "input-integrity.json").read_text())
    assert integrity["candidate_identity_exact"] == "7200/7200"
    assert integrity["query_view_mutation"] == integrity["document_view_mutation"] == 0


def test_model_revision_and_files_are_frozen() -> None:
    manifest = json.loads((R32 / "model-manifest.json").read_text())
    assert manifest["model_id"] == "Qwen/Qwen3-Reranker-4B"
    assert manifest["model_revision"] == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert manifest["max_length"] == 8192 and manifest["dtype"] == "bfloat16"
    assert manifest["generation"] is False
    assert set(manifest["file_sha256"]) >= {"config.json", "tokenizer.json", "tokenizer_config.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"}


def test_batch_invariance_smoke_and_formal_batch() -> None:
    runtime = json.loads((R32 / "runtime-manifest.json").read_text())
    assert runtime["batch_invariance_smoke_pairs"] == 16
    assert runtime["batch_rank_order_parity"] is True
    assert runtime["batch_size"] == 1
    assert runtime["max_score_delta"] == 0.0


def test_predictions_are_complete_top100_permutations() -> None:
    predictions = {item["case_id"]: item for item in load(R32 / "rerank-predictions.jsonl.gz")}
    views = {item["case_id"]: item for item in load(R31A / "rerank-input-views-v2.jsonl.gz")}
    assert len(predictions) == 72
    for case_id, record in predictions.items():
        ranked = record["ranked_candidates"]
        assert len(ranked) == 100
        assert {item["candidate_key"] for item in ranked} == {item["candidate_key"] for item in views[case_id]["candidates"]}
        assert ranked == sorted(ranked, key=lambda item: (-item["reranker_score"], item["pre_rerank_rank"], item["candidate_key"]))


def test_prediction_protocol_has_no_forbidden_mutation() -> None:
    protocol = json.loads((R32 / "protocol.json").read_text())
    for field in ("candidate_added", "candidate_removed", "candidate_mutation", "query_view_mutation", "document_view_mutation", "retrieval_runs", "index_reads", "embedding_calls", "bridge_runs", "semantic_graph_runs", "gold_reads_before_seal", "governance_reads_before_seal", "slot_aware_scoring", "calculator_calls", "generator_calls"):
        assert protocol[field] == 0 or protocol[field] is False
    for field in ("instruction_scan", "model_scan", "max_length_scan"):
        assert protocol[field] is False
    assert protocol["only_changed_variable"] == "reranker_model_capacity_0_6b_to_4b"


def test_formal_score_and_capacity_delta() -> None:
    acceptance = json.loads((R32 / "acceptance.json").read_text())
    metrics = acceptance["metrics"]
    assert metrics["strict_source_binding_recall_at_5"] == "43/80"
    assert metrics["strict_source_binding_recall_at_100"] == "68/80"
    assert metrics["net_recall_at_5_gain_vs_r3_1"] == 3
    assert metrics["4b_promoted_gold"] == 13 and metrics["4b_demoted_gold"] == 10
    assert acceptance["decision"] == "qwen3_reranker_4b_insufficient"
    assert acceptance["next_gate"] == "4b_failure_attribution"


def test_failure_attribution_is_complete_for_top100_misses() -> None:
    audit = json.loads((R32 / "failure-attribution.json").read_text())
    assert audit["record_count"] == 25
    assert sum(audit["counts"].values()) == 25
    assert audit["prediction_mutation"] == 0


def test_slot_aware_fixed_top1_dedup_main_fill() -> None:
    main = [{"candidate_key": key} for key in ("a", "b", "c", "d", "e", "f")]
    slots = {"left": [{"candidate_key": "x"}], "right": [{"candidate_key": "x"}]}
    selected = compose_slot_aware_top5(main, slots)
    assert [item["candidate_key"] for item in selected] == ["x", "a", "b", "c", "d"]
    assert [item["final_rank"] for item in selected] == [1, 2, 3, 4, 5]
