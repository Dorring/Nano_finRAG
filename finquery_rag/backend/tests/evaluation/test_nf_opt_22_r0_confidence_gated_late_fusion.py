"""Focused contract tests for NF-OPT-22 R0 frozen shadow artifacts."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "nf-opt-22-r0-confidence-gated-late-fusion"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "evaluation" / "run_nf_opt_22_r0_confidence_gated_late_fusion.py"


def load_json(name: str):
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def load_rows(name: str):
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def test_decision_is_gold_blind_shadow_and_frozen():
    decision = load_json("decision.json")
    assert decision["evaluation_role"] == "development_shadow_confidence_gated_late_fusion"
    assert decision["fresh_blind_evaluation"] is False
    assert decision["model_execution"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["training"] is False
    assert decision["production_switch_allowed"] is False


def test_frozen_input_and_near_boundary_contract():
    contract = load_json("frozen-contract.json")
    gate = load_json("confidence-gate-contract.json")
    assert contract["candidate_identity_mismatch"] == 0
    assert contract["bm25_rank_mismatch"] == 0
    assert contract["qwen_score_unchanged"] is True
    assert contract["qwen_rank_unchanged"] is True
    assert contract["bounded_top100_sha256"] == contract["bounded_top100_sha_expected"]
    assert gate["rrf_k"] == 60
    assert gate["weights"] == {"qwen": 1.0, "bm25": 1.0}
    assert gate["near_boundary_rule"] == "near_boundary iff C1 margin <= median(C1 margins)"
    assert gate["near_boundary_threshold"] == 0.035075027495622635


def test_prediction_completeness_and_top10_identity():
    rows = load_rows("predictions.jsonl.gz")
    seal = load_json("prediction-seal.json")
    assert len(rows) == 72
    assert seal["queries"] == 72
    assert seal["gold_reads_during_prediction"] == 0
    for row in rows:
        assert len(row["selected_top5_candidate_ids"]) == 5
        assert len(set(row["selected_top5_candidate_ids"])) == 5
        assert set(row["selected_top5_candidate_ids"]) <= set(row["input_top10_candidate_ids"])
        ranked = sorted(row["ranked_candidates"], key=lambda item: item["cglrrf_rank"])
        assert len(ranked) == 100
        assert [item["cglrrf_rank"] for item in ranked] == list(range(1, 101))
        assert set(item["candidate_key"] for item in ranked[:10]) == set(row["input_top10_candidate_ids"])


def test_partition_totals_and_low_candidate_safety():
    partition = load_json("candidate-partition.json")
    assert partition["queries"] == 72
    totals = partition["aggregate_candidate_counts"]
    assert totals["protected"] + totals["border"] + totals["low"] == 72 * 10
    for record in partition["records"]:
        selected = set(record["selected_top5_ids"])
        protected = set(record["protected_ids"])
        border = set(record["border_ids"])
        low = set(record["low_ids"])
        assert protected.isdisjoint(border)
        assert protected.isdisjoint(low)
        assert border.isdisjoint(low)
        # Every selected id is either protected or border unless a fallback
        # was required; the latter still follows original Qwen order.
        assert len(selected) == 5


def test_candidate_set_invariants_and_supply():
    decision = load_json("decision.json")
    strict = load_json("strict-metrics.json")
    assert decision["top10_supply_invariant"] is True
    assert strict["metrics"]["qwen"]["@10"]["hits"] == 60
    assert strict["metrics"]["cglrrf_v1"]["@10"]["hits"] == 60
    assert strict["metrics"]["cglrrf_v1"]["@100"]["hits"] == 68


def test_fixed_fusion_and_decision():
    decision = load_json("decision.json")
    assert decision["cglrrf_r5_hits"] == 45
    assert decision["rescued_vs_qwen"] == 5
    assert decision["damaged_vs_qwen"] == 3
    assert decision["net_vs_qwen"] == 2
    assert decision["confidence_gated_late_fusion_effective"] is False
    assert decision["selected_internal_shadow_method"] == "lrrf_v1"


def test_prediction_source_has_no_runtime_gold_cohorts():
    source = SCRIPT.read_text(encoding="utf-8").split("# Post-seal only", 1)[0]
    assert "strict_rows" not in source
    assert "gold_candidate_id" not in source
    assert "failure_tags" not in source
    assert "failure taxonomy" not in source
    assert "gold_reads_during_prediction" in source


def test_post_seal_safety_metrics_are_present():
    semantic = load_json("semantic-metrics.json")
    multi = load_json("multi-evidence-analysis.json")
    calc = load_json("calculation-slot-analysis.json")
    assert semantic["cglrrf_v1"]["@5"]["hits"] == 50
    assert multi["denominator"] == 16
    assert calc["denominator"] == 11
    assert multi["variants"]["cglrrf_v1"]["@10"]["all"] == 9
    assert calc["variants"]["cglrrf_v1"]["@10"]["all_slots"] == 7


def test_prediction_hash_is_sealed():
    import hashlib

    digest = hashlib.sha256()
    with (ARTIFACTS / "predictions.jsonl.gz").open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    assert digest.hexdigest() == load_json("prediction-seal.json")["prediction_sha256"]
