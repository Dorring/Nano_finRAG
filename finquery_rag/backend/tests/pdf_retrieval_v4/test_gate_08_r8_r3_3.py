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
P0 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3-p0"
R32 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-2"
R33 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_statement_hint_is_emitted_only_when_present() -> None:
    base = {"raw_question": "revenue?", "operand_slots": []}
    assert "Statement Hint:" not in build_rerank_query_view(base)
    assert "Statement Hint: income statement" in build_rerank_query_view({**base, "statement_hint": "income statement"})


def test_slot_query_is_explicitly_operand_focused() -> None:
    plan = {"raw_question": "growth from 2023 to 2024?", "operation": "growth_rate", "operand_slots": []}
    slot = {"slot_id": "base", "role": "denominator", "raw_metric_phrase": "revenue", "concept_candidates": ["net sales"], "period": "FY2023", "temporal_kind": "duration", "required_evidence_shape": "atomic_fact"}
    view = build_slot_rerank_query_view(plan, slot)
    assert "[FOCUS OPERAND]" in view
    assert "Role: denominator" in view and "Metric: revenue" in view and "Period: FY2023" in view
    assert "Evidence supporting only another operand" in view


def test_p0_reuses_all_main_scores_and_builds_exact_slots() -> None:
    seal = json.loads((P0 / "input-seal.json").read_text())
    assert seal["cases"] == 72 and seal["candidate_occurrences"] == 7200
    assert seal["multi_slot_cases"] == 18 and seal["slot_count"] == 36 and seal["slot_pair_count"] == 3600
    assert seal["statement_hint_nonempty_cases"] == 0
    assert seal["main_query_changed_cases"] == 0 and seal["main_query_reused_cases"] == 72
    assert seal["gold_reads_before_seal"] == 0


def test_main_scores_are_exact_r3_2_reuse() -> None:
    assert sha(R32 / "rerank-predictions.jsonl.gz") == "7e058da966f554c8e898cecfd5401ce735e995daec1852f7750fa8d8e9e88da6"
    source = {item["case_id"]: item for item in load(R32 / "rerank-predictions.jsonl.gz")}
    reused = {item["case_id"]: item for item in load(R33 / "main_rerank_predictions.jsonl.gz")}
    for case_id in source:
        assert source[case_id]["ranked_candidates"] == reused[case_id]["ranked_candidates"]
        assert reused[case_id]["score_reused"] is True


def test_slot_predictions_are_36_by_100_and_candidate_bounded() -> None:
    slots = load(R33 / "slot_rerank_predictions.jsonl.gz")
    p0 = {item["case_id"]: item for item in load(P0 / "queryplan-rerank-input-views.jsonl.gz")}
    assert len(slots) == 36
    for record in slots:
        ranked = record["ranked_candidates"]
        assert len(ranked) == 100
        assert {item["candidate_key"] for item in ranked} == {item["candidate_key"] for item in p0[record["case_id"]]["candidates"]}
        assert [item["slot_rank"] for item in ranked] == list(range(1, 101))


def test_final_top5_fixed_composition_and_no_single_regression() -> None:
    records = load(R33 / "slot_aware_top5_predictions.jsonl.gz")
    assert len(records) == 72
    for record in records:
        assert len(record["candidates"]) == 5
        assert len({item["candidate_key"] for item in record["candidates"]}) == 5
        if not record["is_multi_slot"]:
            assert {item["selection_source"] for item in record["candidates"]} == {"main"}
    acceptance = json.loads((R33 / "acceptance.json").read_text())
    assert acceptance["metrics"]["single_source_question_hit_at_5"] == "30/56"
    assert acceptance["metrics"]["single_source_regression_vs_r3_2"] == 0


def test_prediction_protocol_has_no_forbidden_operations() -> None:
    protocol = json.loads((R33 / "protocol.json").read_text())
    for field in ("gold_reads_before_seal", "governance_reads_before_seal", "reference_answer_reads", "expected_value_reads", "retrieval_runs", "embedding_calls", "index_reads", "bridge_runs", "candidate_added", "candidate_removed", "production_writes"):
        assert protocol[field] == 0
    for field in ("model_scan", "prompt_scan", "slot_quota_scan", "model_8b"):
        assert protocol[field] is False
    assert protocol["slot_top_n"] == 1 and protocol["final_top_k"] == 5


def test_formal_result_stops_physical_source_optimization() -> None:
    acceptance = json.loads((R33 / "acceptance.json").read_text())
    assert acceptance["metrics"]["strict_source_binding_recall_at_5"] == "43/80"
    assert acceptance["metrics"]["multi_evidence_complete_at_5"] == "4/16"
    assert acceptance["metrics"]["calculation_complete_at_5"] == "4/11"
    assert acceptance["decision"] == "queryplan_guided_reranking_below_physical_source_target"
    assert acceptance["next_gate"] == "semantic_evidence_recall_contract"
    assert acceptance["final_retrieval_target_reached"] is False
