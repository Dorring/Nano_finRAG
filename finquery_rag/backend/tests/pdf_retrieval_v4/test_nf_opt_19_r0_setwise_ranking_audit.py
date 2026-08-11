"""Focused contract tests for NF-OPT-19 R0 sealed diagnostics."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


BACKEND = Path(__file__).resolve().parents[2]
ARTIFACT = BACKEND / "artifacts" / "evaluation" / "nf-opt-19-r0-setwise-ranking-audit"
QWEN = BACKEND / "artifacts" / "evaluation" / "pdf-retrieval-v4-gate-08-r8-r3-3" / "main_rerank_predictions.jsonl.gz"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _gzip_rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_r0_is_post_seal_and_identity_safe():
    decision = _json(ARTIFACT / "decision.json")
    assert decision["model_execution"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["ranking_mutation"] is False
    assert decision["production_switch_allowed"] is False
    assert decision["c0"] + decision["c1"] + decision["c2"] == 80
    assert (decision["c0"], decision["c1"], decision["c2"]) == (43, 25, 12)
    assert decision["candidate_identity_mismatch"] == 0
    assert decision["qwen_rank_identity_mismatch"] == 0
    assert decision["top100_sha_match"] is True


def test_semantic_collapse_preserves_qwen_order_and_never_adds_candidates():
    qwen_rows = {row["case_id"]: row["ranked_candidates"] for row in _gzip_rows(QWEN)}
    collapsed_rows = {row["case_id"]: row["ranked_candidates"] for row in _gzip_rows(ARTIFACT / "semantic-collapse-predictions.jsonl.gz")}
    assert set(collapsed_rows) == set(qwen_rows)
    for case_id, rows in collapsed_rows.items():
        source_ids = {row["candidate_key"] for row in qwen_rows[case_id]}
        assert {row["candidate_key"] for row in rows} <= source_ids
        original_ranks = [row["original_qwen_rank"] for row in rows]
        assert original_ranks == sorted(original_ranks)
        assert [row["collapsed_rank"] for row in rows] == list(range(1, len(rows) + 1))


def test_collapse_seal_precedes_gold_and_semantic_identity_is_separate():
    seal = _json(ARTIFACT / "semantic-collapse-seal.json")
    metrics = _json(ARTIFACT / "semantic-collapse-metrics.json")
    assert seal["gold_reads_before_seal"] == 0
    assert seal["candidate_added"] == 0
    assert seal["ranking_mutation"] is False
    assert metrics["strict_physical_recall_at_5"]["before"]["hits"] == 43
    assert metrics["semantic_recall_at_5"]["before"]["hits"] == 49
    assert metrics["strict_physical_recall_at_5"]["before"]["total"] == 80
    assert metrics["semantic_recall_at_5"]["before"]["total"] == 80


def test_collapse_seal_hash_is_reproducible_for_current_artifact():
    seal = _json(ARTIFACT / "semantic-collapse-seal.json")
    digest = hashlib.sha256((ARTIFACT / "semantic-collapse-predictions.jsonl.gz").read_bytes()).hexdigest()
    assert digest == seal["prediction_sha256"]
