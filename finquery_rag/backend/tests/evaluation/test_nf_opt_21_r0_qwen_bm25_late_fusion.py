"""Focused contract tests for the sealed NF-OPT-21 R0 shadow artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "nf-opt-21-r0-qwen-bm25-late-fusion"


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def load_rows(name: str):
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_decision_is_shadow_and_no_execution():
    decision = load_json("decision.json")
    assert decision["evaluation_role"] == "development_shadow_late_fusion"
    assert decision["fresh_blind_evaluation"] is False
    assert decision["model_execution"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["production_switch_allowed"] is False


def test_frozen_input_integrity():
    contract = load_json("frozen-input-contract.json")
    assert contract["candidate_identity_mismatch"] == 0
    assert contract["bm25_rank_mismatch"] == 0
    assert contract["qwen_rank_unchanged"] is True
    assert contract["qwen_score_unchanged"] is True
    assert contract["bm25_rank_source"] == "qwen.pre_rerank_rank equals bounded.rank"
    assert contract["fusion_candidate_depth"] == 10


def test_prediction_rows_and_top10_identity():
    qwen_rows = load_rows("lrrf-predictions.jsonl.gz")
    assert len(qwen_rows) == 72
    for name in ("lrrf-predictions.jsonl.gz", "plrf-predictions.jsonl.gz"):
        rows = load_rows(name)
        assert len(rows) == 72
        for row in rows:
            candidates = sorted(row["ranked_candidates"], key=lambda item: item["fusion_rank"])
            assert len(candidates) == 100
            assert [item["fusion_rank"] for item in candidates] == list(range(1, 101))
            assert row["fusion_candidate_depth"] == 10
            assert {item["original_qwen_rank"] for item in candidates[:10]} == set(range(1, 11))
            assert {item["original_qwen_rank"] for item in candidates} == set(range(1, 101))


def test_plrf_protects_qwen_ranks_one_to_four():
    qwen = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: item["original_qwen_rank"]) for row in load_rows("lrrf-predictions.jsonl.gz")}
    plrf = {row["case_id"]: sorted(row["ranked_candidates"], key=lambda item: item["fusion_rank"]) for row in load_rows("plrf-predictions.jsonl.gz")}
    for case_id in qwen:
        assert [item["candidate_key"] for item in qwen[case_id][:4]] == [item["candidate_key"] for item in plrf[case_id][:4]]


def test_variant_seals_are_gold_blind():
    for name in ("lrrf-prediction-seal.json", "plrf-prediction-seal.json"):
        seal = load_json(name)
        assert seal["gold_reads_during_prediction"] == 0
        assert seal["sealed"] is True
        assert seal["rows"] == 72
        assert seal["candidate_count_per_query"] == 100


def test_candidate_set_invariants_and_c2_not_rescued():
    decision = load_json("decision.json")
    assert decision["top10_candidate_set_invariant"] is True
    assert decision["top20_candidate_set_invariant"] is True
    assert decision["top50_candidate_set_invariant"] is True
    assert decision["top100_candidate_set_invariant"] is True
    cohort = load_json("cohort-analysis.json")
    for variant in ("lrrf_v1", "plrf_v1"):
        assert cohort[variant]["strict_cohorts"]["C2"]["rescued"] == 0


def test_frozen_rrf_and_selection_contract():
    assert load_json("lrrf-v1-contract.json")["rrf_k"] == 60
    assert load_json("plrf-v1-contract.json")["rrf_k"] == 60
    decision = load_json("decision.json")
    assert decision["selected_variant"] == "lrrf_v1"
    assert decision["late_fusion_materially_effective"] == "marginal"


def test_denominators_and_baseline_metrics():
    strict = load_json("strict-metrics.json")
    assert strict["strict_sources"] == 80
    assert strict["qwen"]["@5"]["hits"] == 43
    assert strict["qwen"]["@10"]["hits"] == 60
    assert strict["qwen"]["@100"]["hits"] == 68
    multi = load_json("multi-evidence-analysis.json")
    assert multi["denominator"] == 16
    assert multi["variants"]["qwen"]["@5"]["all"] == 4
    assert multi["variants"]["qwen"]["@10"]["all"] == 9
    calc = load_json("calculation-slot-analysis.json")
    assert calc["denominator"] == 11
    assert calc["variants"]["qwen"]["@5"]["all_slots"] == 5
    assert calc["variants"]["qwen"]["@10"]["all_slots"] == 7


def test_no_gold_cohort_identifiers_in_runtime_source():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "run_nf_opt_21_r0_qwen_bm25_late_fusion.py"
    source = script.resolve().read_text(encoding="utf-8").split("# Post-seal only", 1)[0]
    assert "P1" not in source
    assert "C1" not in source
    assert "gold_candidate_id" not in source
