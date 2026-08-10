from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.semantic_evidence_set import (
    build_access_universe,
    build_semantic_classes,
    match_slots,
    minimum_candidate_cover,
    operand_projection,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
OUT = BASE / "pdf-retrieval-v4-gate-09-r5"
R33 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3"


def load(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fact(fact_id: str, period: str, value: str, candidate_key: str, currency: str = "USD") -> tuple[dict, dict]:
    semantic_fact = {
        "semantic_fact_id": fact_id,
        "document_id": "issuer_fy2025",
        "normalized_metric": "revenue",
        "normalized_period": period,
        "normalized_segment": "",
        "normalized_bucket": "",
        "normalized_base_value": value,
        "normalized_scale": "1000000",
        "normalized_currency": currency,
        "physical_provenance": [{"authoritative_evidence_id": f"atomic:{fact_id}"}],
    }
    registry = {
        "candidate_key": candidate_key,
        "semantic_fact_ids": [fact_id],
        "semantic_facts": [semantic_fact],
    }
    return semantic_fact, registry


def plan(periods: list[str]) -> dict:
    return {
        "plan_id": "plan",
        "document_scope": ["issuer_fy2025"],
        "task_type": "calculation_multi_operand" if len(periods) > 1 else "table_single_fact",
        "operation": "growth_rate" if len(periods) > 1 else None,
        "operand_slots": [
            {
                "slot_id": f"slot_{index}",
                "role": "operand",
                "raw_metric_phrase": "revenue",
                "concept_candidates": [],
                "period": period,
                "segment_label": None,
                "bucket_label": None,
                "required_evidence_shape": "atomic_fact",
            }
            for index, period in enumerate(periods)
        ],
    }


def test_same_fact_different_candidates_collapses_to_one_class() -> None:
    semantic_fact, left = fact("fact-a", "fy2025", "100", "candidate-a")
    right = {**left, "candidate_key": "candidate-b", "semantic_facts": [{**semantic_fact}]}
    access = [
        {"candidate_key": "candidate-a", "main_rank": 1, "slot_ranks": {}, "access_sources": ["main_top10"]},
        {"candidate_key": "candidate-b", "main_rank": 2, "slot_ranks": {}, "access_sources": ["main_top10"]},
    ]
    classes = build_semantic_classes(access, {"candidate-a": left, "candidate-b": right})
    assert len(classes) == 1
    assert classes[0]["supporting_candidate_keys"] == ["candidate-a", "candidate-b"]


def test_different_values_or_periods_never_collapse() -> None:
    _, left = fact("fact-a", "fy2025", "100", "candidate-a")
    _, right = fact("fact-b", "fy2025", "101", "candidate-b")
    _, prior = fact("fact-c", "fy2024", "100", "candidate-c")
    access = [
        {"candidate_key": key, "main_rank": rank, "slot_ranks": {}, "access_sources": ["main_top10"]}
        for rank, key in enumerate(("candidate-a", "candidate-b", "candidate-c"), 1)
    ]
    classes = build_semantic_classes(access, {"candidate-a": left, "candidate-b": right, "candidate-c": prior})
    assert len(classes) == 3


def test_conflicting_semantic_classes_fail_closed_as_ambiguous() -> None:
    _, left = fact("fact-a", "fy2025", "100", "candidate-a")
    _, right = fact("fact-b", "fy2025", "101", "candidate-b")
    access = [
        {"candidate_key": "candidate-a", "main_rank": 1, "slot_ranks": {}, "access_sources": ["main_top10"]},
        {"candidate_key": "candidate-b", "main_rank": 2, "slot_ranks": {}, "access_sources": ["main_top10"]},
    ]
    classes = build_semantic_classes(access, {"candidate-a": left, "candidate-b": right})
    matches = match_slots(plan(["fy2025"]), classes)
    assert matches[0]["slot_status"] == "runtime_operand_ambiguity"


def test_one_candidate_can_cover_two_slots() -> None:
    current, registry = fact("fact-current", "fy2025", "100", "candidate-a")
    previous, _ = fact("fact-previous", "fy2024", "90", "candidate-a")
    registry["semantic_fact_ids"].append("fact-previous")
    registry["semantic_facts"].append(previous)
    access = [{"candidate_key": "candidate-a", "main_rank": 1, "slot_ranks": {}, "access_sources": ["main_top10"]}]
    classes = build_semantic_classes(access, {"candidate-a": registry})
    query_plan = plan(["fy2024", "fy2025"])
    matches = match_slots(query_plan, classes)
    cover = minimum_candidate_cover(matches, classes, access)
    assert cover["complete"] is True
    assert cover["selected_candidate_keys"] == ["candidate-a"]
    assert cover["evidence_item_count"] == 1
    assert current["semantic_fact_id"] in cover["covered_semantic_fact_ids"]


def test_calculation_projection_requires_value_scale_and_currency() -> None:
    _, complete = fact("fact-a", "fy2025", "100", "candidate-a")
    _, missing_currency = fact("fact-b", "fy2024", "90", "candidate-b", currency="")
    access = [
        {"candidate_key": "candidate-a", "main_rank": 1, "slot_ranks": {}, "access_sources": ["main_top10"]},
        {"candidate_key": "candidate-b", "main_rank": 2, "slot_ranks": {}, "access_sources": ["main_top10"]},
    ]
    classes = build_semantic_classes(access, {"candidate-a": complete, "candidate-b": missing_currency})
    query_plan = plan(["fy2024", "fy2025"])
    matches = match_slots(query_plan, classes)
    projection = operand_projection(query_plan, matches, classes)
    assert projection["calculation_runtime_ready"] is False
    assert projection["operands"]["slot_0"]["currency"] is None


def test_single_and_multi_access_routes_are_fixed() -> None:
    main = [{"candidate_key": f"main-{index}"} for index in range(1, 101)]
    slot = {"slot_0": [{"candidate_key": f"slot-{index}"} for index in range(1, 101)]}
    single = build_access_universe(plan(["fy2025"]), main, slot)
    multi_plan = plan(["fy2024", "fy2025"])
    multi_slot = {
        "slot_0": [{"candidate_key": f"slot0-{index}"} for index in range(1, 101)],
        "slot_1": [{"candidate_key": f"slot1-{index}"} for index in range(1, 101)],
    }
    multi = build_access_universe(multi_plan, main, multi_slot)
    assert {item["candidate_key"] for item in single} == {f"main-{index}" for index in range(1, 11)}
    assert len(multi) == 30


def test_prediction_seal_is_zero_gold_and_inputs_are_exact() -> None:
    seal = json.loads((OUT / "prediction-seal.json").read_text(encoding="utf-8"))
    assert seal["sealed"] is True
    assert seal["case_count"] == 72
    assert seal["gold_reads_before_seal"] == seal["strict_binding_reads_before_seal"] == 0
    assert seal["retrieval_runs"] == seal["reranker_calls"] == seal["embedding_calls"] == 0
    assert seal["candidate_mutation"] == seal["semantic_registry_mutation"] == 0
    assert seal["r3_3_main_prediction_sha256"] == sha256(R33 / "main_rerank_predictions.jsonl.gz")
    assert seal["r3_3_slot_prediction_sha256"] == sha256(R33 / "slot_rerank_predictions.jsonl.gz")


def test_every_evidence_set_is_within_frozen_access_and_budget() -> None:
    access = {row["case_id"]: row for row in load(OUT / "evidence-access-universe.jsonl.gz")}
    sets = load(OUT / "evidence-set-predictions.jsonl.gz")
    for record in sets:
        allowed = {item["candidate_key"] for item in access[record["case_id"]]["candidates"]}
        selected = set(record["selected_candidate_keys"])
        assert selected <= allowed
        assert record["evidence_item_count"] == len(selected)
        assert record["evidence_item_count"] <= 5


def test_formal_gate09_r5_result_is_fail_closed() -> None:
    acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
    false_binding = json.loads((OUT / "false-binding-audit.json").read_text(encoding="utf-8"))
    assert acceptance["main_only_semantic_access"] == "61/80"
    assert acceptance["slot_augmented_semantic_access"] == "62/80"
    assert acceptance["semantic_evidence_set_recall"] == "11/80"
    assert acceptance["calculation_runtime_ready"] == "0/11"
    assert acceptance["decision"] == "top10_semantic_evidence_set_insufficient"
    assert acceptance["next_gate"] == "deterministic_operand_binding_contract_repair"
    assert false_binding["false_slot_binding"] == 1
