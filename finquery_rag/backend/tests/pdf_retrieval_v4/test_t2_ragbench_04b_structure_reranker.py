from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_04b_structure_reranker.py"
ARTIFACT = ROOT / "artifacts/evaluation/t2-ragbench-04b-structure-aware-reranker"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_04b", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def candidate_row(rank: int, *, metric: int = 0, period: float = 1.0, table: int = 0) -> dict:
    return {
        "query_id": "q",
        "candidate_context_id": f"c{rank}",
        "bm25_rank": rank,
        "features": {
            "metric_exact_match": metric,
            "metric_normalized_match": metric,
            "period_any_match": int(period > 0),
            "required_period_coverage": period,
            "metric_in_row_label": table,
            "period_in_table_header": float(table),
            "row_header_coherence": table,
            "multi_period_coverage": period,
            "operation_evidence_compatibility": table,
        },
    }


def test_candidate_universe_and_r50_are_unchanged() -> None:
    comparison = read_json("method-comparison.json")
    baseline = comparison["metrics"]["bm25"]["hits"]["50"]
    assert baseline == 1857
    for metrics in comparison["metrics"].values():
        assert metrics["hits"]["50"] == baseline


def test_test_and_transfer_gold_are_locked() -> None:
    decision = read_json("decision.json")
    manifest = read_json("b2-training-manifest.json")
    assert decision["primary_test_gold_reads"] == 0
    assert decision["convfinqa_gold_reads"] == 0
    assert manifest["test_gold_used"] is False
    assert manifest["convfinqa_gold_used"] is False


def test_guarded_promotion_uses_one_rank5_boundary_slot() -> None:
    rows = [candidate_row(rank) for rank in range(1, 51)]
    rows[5] = candidate_row(6, metric=1, table=1)
    query = {
        "period_requirement_present": True,
        "normalized_metric_terms": ["revenue"],
        "operation_intent": "direct_fact",
    }
    order, diagnostics = module.guarded_order(rows, {"q": query}, 10)
    assert order["q"][:6] == ["c1", "c2", "c3", "c4", "c6", "c5"]
    assert diagnostics["max_promotions_per_query"] == 1
    assert diagnostics["promotion_position"] == "rank5_boundary"


def test_noneligible_period_conflict_cannot_promote() -> None:
    rows = [candidate_row(rank) for rank in range(1, 51)]
    rows[5] = candidate_row(6, metric=1, period=0.0, table=1)
    query = {
        "period_requirement_present": True,
        "normalized_metric_terms": ["revenue"],
        "operation_intent": "direct_fact",
    }
    order, _ = module.guarded_order(rows, {"q": query}, 10)
    assert order["q"][:6] == ["c1", "c2", "c3", "c4", "c5", "c6"]


def test_noneligible_metric_mismatch_cannot_promote() -> None:
    rows = [candidate_row(rank) for rank in range(1, 51)]
    rows[5] = candidate_row(6, metric=0, table=1)
    query = {
        "period_requirement_present": True,
        "normalized_metric_terms": ["revenue"],
        "operation_intent": "direct_fact",
    }
    order, _ = module.guarded_order(rows, {"q": query}, 10)
    assert order["q"][:6] == ["c1", "c2", "c3", "c4", "c5", "c6"]


def test_b2_training_contract_is_train_only_and_fixed() -> None:
    manifest = read_json("b2-training-manifest.json")
    model = read_json("b2-model.json")
    assert manifest["training_split"] == "Primary Train"
    assert manifest["training_queries"] == 15314
    assert manifest["dev_gold_used_during_fit"] is False
    assert model["C"] == 1.0
    assert model["random_seed"] == 20250810
    assert model["max_iter"] == 200
    assert model["features"] == list(module.B2_FEATURES)
    assert model["negative_ranks"]["fixed_remaining"] == [11, 21, 31, 41, 50]


def test_b2_tie_break_is_deterministic() -> None:
    class EqualModel:
        @staticmethod
        def decision_function(_rows):
            return [0.0]

    rows = [candidate_row(2), candidate_row(1)]
    rows[0]["candidate_context_id"] = "b"
    rows[1]["candidate_context_id"] = "a"
    order = module.b2_order(rows, EqualModel())
    assert order["q"] == ["a", "b"]


def test_selection_and_lock_state_are_explicit() -> None:
    decision = read_json("decision.json")
    assert decision["structure_reranker_selected"] is False
    assert decision["selected_method"] is None
    assert decision["next_gate"] == "t2_04_method_reconsideration"
    assert decision["empty_questions_retained"] is True
    assert decision["retrieval_rerun"] is False
    assert decision["llm_execution"] is False


def test_selected_manifest_hash_is_reproducible() -> None:
    manifest = ARTIFACT / "selected-method-manifest.json"
    expected = (ARTIFACT / "selected-method-sha256.txt").read_text().strip()
    assert module.sha256_file(manifest) == expected


def test_b1_contract_has_only_preregistered_scopes() -> None:
    contract = read_json("b1-guarded-contract.json")
    assert contract["promotion_scopes"] == {
        "b1_a": "ranks 6-10",
        "b1_b": "ranks 6-20",
    }
    assert contract["max_promotions_per_query"] == 1
    assert contract["promotion_position"] == "rank5_boundary"
