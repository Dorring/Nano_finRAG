from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_04a2_structure_signal_audit.py"
ARTIFACT = ROOT / "artifacts/evaluation/t2-ragbench-04a2-structure-signal-audit"
sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location(
    "t2_04a2_structure_signal_audit", MODULE_PATH
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_primary_contract_and_candidate_depth_are_frozen() -> None:
    decision = read_json("decision.json")
    seal = read_json("feature-seal.json")
    assert decision["primary_train_queries"] == 15_314
    assert decision["primary_dev_queries"] == 2_025
    assert decision["primary_test_queries"] == 2_291
    assert decision["candidate_depth"] == 50
    assert seal["candidate_feature_count"] == (15_314 + 2_025) * 50
    assert seal["query_count"] == 15_314 + 2_025


def test_gold_access_contract_is_sealed_and_test_tracks_locked() -> None:
    seal = read_json("feature-seal.json")
    decision = read_json("decision.json")
    assert seal["sealed"] is True
    assert seal["gold_reads_before_feature_seal"] == 0
    assert seal["primary_test_gold_reads"] == 0
    assert seal["convfinqa_gold_reads"] == 0
    assert decision["gold_reads_before_feature_seal"] == 0
    assert decision["primary_test_gold_reads"] == 0
    assert decision["convfinqa_gold_reads"] == 0
    assert decision["test_structure_scoring"] is False
    assert decision["convfinqa_structure_scoring"] is False


def test_feature_contract_is_gold_independent_and_fixed() -> None:
    contract = read_json("feature-contract.json")
    assert contract["gold_independent"] is True
    assert contract["candidate_source"] == "T2-01 frozen BM25 Top50"
    assert contract["structure_match_count_components"] == [
        "entity_normalized_match",
        "metric_normalized_match",
        "required_period_coverage_full",
        "metric_in_row_label",
        "period_in_table_header",
        "row_header_coherence",
        "operation_evidence_compatibility",
    ]
    assert contract["tie_break"] == [
        "feature_value_desc",
        "bm25_rank_asc",
        "context_id_asc",
    ]


def test_diagnostic_rerank_preserves_candidate_recall_at_50() -> None:
    ranking = read_json("single-feature-dev-ranking.json")
    baseline = ranking["bm25"]["hits"]["50"]
    assert baseline == 1_857
    for payload in ranking["features"].values():
        assert payload["r_at_50_invariant"] is True
        assert payload["metrics"]["hits"]["50"] == baseline


def test_subset_analysis_is_scoped_and_reconciles() -> None:
    subsets = read_json("subset-analysis.json")
    assert subsets["FinQA"]["query_count"] == 883
    assert subsets["TAT-DQA"]["query_count"] == 1_142
    assert sum(item["query_count"] for item in subsets.values()) == 2_025
    assert subsets["FinQA"]["bm25"]["recall_pct"]["@5"] == 84.824462
    assert subsets["TAT-DQA"]["bm25"]["recall_pct"]["@5"] == 61.120841
    for result in subsets.values():
        for feature in result["features"].values():
            assert feature["r_at_50_invariant"] is True


def test_empty_queries_are_retained_in_feature_coverage() -> None:
    coverage = read_json("extraction-coverage.json")
    assert coverage["query"]["TAT-DQA"]["empty_question_count"] == 4
    assert coverage["query"]["FinQA"]["empty_question_count"] == 0


def test_decision_requires_non_entity_signal_and_is_explicit() -> None:
    decision = read_json("decision.json")
    assert decision["structure_signal_supported"] is True
    assert decision["next_gate"] == "t2_04b_structure_aware_reranker"
    assert decision["dev_bm25_recall_at_5"] == 71.45679
    assert decision["dev_structure_count_recall_at_5"] == 57.62963
    assert decision["dev_structure_count_gain_pp"] == -13.82716


def test_deterministic_normalization_contract() -> None:
    assert module.extract_periods("FY 2023 and Q4 2024")[0] == [
        "2023",
        "2024",
        "q4",
    ]
    assert module.normalize_tokens("Services net sales") == (
        "net",
        "sale",
        "service",
    )
    assert module.normalize_tokens("gross margin") != module.normalize_tokens(
        "gross profit"
    )


def test_structure_features_do_not_add_candidates() -> None:
    ranking = read_json("single-feature-dev-ranking.json")
    for payload in ranking["features"].values():
        assert payload["metrics"]["count"] == ranking["bm25"]["count"]
        assert payload["metrics"]["hits"]["50"] == ranking["bm25"]["hits"]["50"]


def test_feature_files_are_sealed() -> None:
    seal = read_json("feature-seal.json")
    assert set(seal["feature_files"]) == {
        "feature-contract.json",
        "query-structure.jsonl.gz",
        "context-structure.jsonl.gz",
        "train-candidate-features.jsonl.gz",
        "dev-candidate-features.jsonl.gz",
    }
    assert seal["candidate_mutation"] == 0
    assert seal["retrieval_rerun"] is False
    assert seal["model_execution"] is False
