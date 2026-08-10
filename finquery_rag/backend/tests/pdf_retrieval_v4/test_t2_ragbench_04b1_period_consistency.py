from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_04b1_period_consistency.py"
ARTIFACT = ROOT / "artifacts/evaluation/t2-ragbench-04b1-period-consistency-freeze"

sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_04b1_period_consistency", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def candidate_row(rank: int, coverage: float) -> dict:
    return {
        "query_id": "q",
        "candidate_context_id": f"c{rank}",
        "bm25_rank": rank,
        "features": {"required_period_coverage": coverage},
    }


def test_pcr_v1_contract_uses_sealed_feature_and_bm25_top50() -> None:
    contract = read_json("pcr-v1-contract.json")
    assert contract["method"] == "Financial Period-Consistency Reranker V1"
    assert contract["short_name"] == "PCR-V1"
    assert contract["candidate_source"] == "frozen_bm25_top50"
    assert contract["candidate_depth"] == 50
    assert contract["feature"] == "required_period_coverage"
    assert contract["feature_seal"] == module.SEAL_SHA
    assert contract["new_feature_extraction"] is False
    assert contract["feature_weight_search"] is False
    assert contract["scope_search"] is False


def test_pcr_bucket_order_and_bm25_tie_break() -> None:
    rows = [
        candidate_row(1, 0.0),
        candidate_row(2, 0.5),
        candidate_row(3, 1.0),
        candidate_row(4, 0.5),
    ]
    order = module.pcr_order(rows)["q"]
    assert order == ["c3", "c2", "c4", "c1"]


def test_no_period_neutral_bucket_preserves_exact_bm25_order() -> None:
    rows = [candidate_row(2, 0.0), candidate_row(1, 0.0), candidate_row(3, 0.0)]
    assert module.pcr_order(rows) == module.bm25_order(rows)
    contract = read_json("pcr-v1-contract.json")
    assert contract["no_period_order_invariant"] is True
    assert contract["no_period_query_count"] == 65


def test_dev_reproduction_matches_sealed_reference() -> None:
    result = read_json("dev-reproduction.json")
    assert result["reproduction_passed"] is True
    assert result["candidate_set_invariant"] is True
    assert result["r_at_50_invariant"] is True
    assert result["bm25"]["hits"]["5"] == 1447
    assert result["pcr_v1"]["hits"]["5"] == 1478
    assert result["pcr_v1"]["recall_pct"]["@5"] == 72.987654
    assert result["pcr_v1"]["hits"]["50"] == 1857
    assert result["movement"] == {
        "rescued_at_5": 64,
        "damaged_at_5": 33,
        "net_top5_gain": 31,
        "unchanged_at_5": 1928,
    }


def test_subset_results_are_recorded_without_method_changes() -> None:
    subset = read_json("subset-analysis.json")
    assert subset["FinQA"]["query_count"] == 883
    assert subset["TAT-DQA"]["query_count"] == 1142
    assert subset["FinQA"]["pcr_v1"]["recall_pct"]["@5"] == 89.127973
    assert subset["TAT-DQA"]["pcr_v1"]["recall_pct"]["@5"] == 60.507881
    assert subset["FinQA"]["pcr_v1"]["hits"]["50"] == subset["FinQA"]["bm25"]["hits"]["50"]
    assert subset["TAT-DQA"]["pcr_v1"]["hits"]["50"] == subset["TAT-DQA"]["bm25"]["hits"]["50"]


def test_selected_method_and_locked_gold_contract() -> None:
    selected = read_json("selected-method.json")
    manifest = ARTIFACT / "selected-method-manifest.json"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    decision = read_json("decision.json")
    assert selected["selected"] is True
    assert selected["feature"] == "required_period_coverage"
    assert selected["dev_recall_at_5"] == 72.987654
    assert selected["test_gold_used"] is False
    assert selected["convfinqa_gold_used"] is False
    assert manifest_data["primary_test_gold_reads"] == 0
    assert manifest_data["convfinqa_gold_reads"] == 0
    assert decision["primary_test_gold_reads"] == 0
    assert decision["convfinqa_gold_reads"] == 0
    assert decision["pcr_v1_selected"] is True
    assert decision["structure_reranker_selected"] is True
    assert decision["next_gate"] == "t2_04c_frozen_test_evaluation"
    expected_hash = (ARTIFACT / "selected-method-sha256.txt").read_text().strip()
    assert module.sha256_obj(manifest_data) == expected_hash


def test_historical_guarded_reranker_result_remains_distinct() -> None:
    decision = read_json("decision.json")
    assert decision["historical_t2_04b_structure_reranker_selected"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["new_feature_extraction"] is False
