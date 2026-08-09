"""Tests for Gate 08 R3.1 Fusion Attribution Closure."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R3_DIR = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r3"
SCORING_DIR = R3_DIR / "scoring"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rank_regression_metric_and_categories() -> None:
    data = _load_json(SCORING_DIR / "structured-expansion-rank-regression.json")

    assert data["metric_name"] == "structured_expansion_rank_regression"
    assert "not raw_lane_dilution" in data.get("note", "").lower()
    for category in ("improved", "unchanged", "worsened", "new_entry", "dropped_out"):
        assert category in data


def test_cross_family_fusion_stage_is_renamed_and_verified() -> None:
    data = _load_json(R3_DIR / "first-failure-attribution-corrected.json")
    stages = data["failure_stage_counts"]

    assert "structured_fused_top40_but_post_filter_excluded" not in stages
    assert "structured_top40_lost_after_cross_family_fusion" in stages
    for detail in data["failure_details"]:
        if detail["first_failure_stage"] != "structured_top40_lost_after_cross_family_fusion":
            continue
        assert detail["structured_expanded_rank"] <= 40
        cross_family_rank = detail.get("cross_family_rrf_rank")
        assert cross_family_rank is None or cross_family_rank > 40


def test_failure_attribution_is_complete() -> None:
    data = _load_json(R3_DIR / "first-failure-attribution-corrected.json")

    assert data["total_gold_sources"] == 80
    assert data["in_pool"] + data["missed"] == 80
    assert sum(data["failure_stage_counts"].values()) == data["missed"]


def test_raw_parity_uses_authoritative_metrics() -> None:
    data = _load_json(SCORING_DIR / "raw-parity-corrected.json")

    assert data["bm25_source_recall_200"] == "37/80"
    assert data["dense_source_recall_200"] == "14/80"
    assert data["rrf_recall_40"] == "20/80"
    assert data["raw_full_pool"] == "31/80"
    assert data["bm25_recomputed"] is False
    assert data["bm25_authoritative_baseline"] == 37


def test_gold_fusion_loss_matrix_contract() -> None:
    data = _load_json(SCORING_DIR / "gold-fusion-loss-matrix.json")

    assert data["total_gold_sources"] == 80
    assert len(data["matrix"]) == 80
    for entry in data["matrix"]:
        assert "structured_expanded_rank" in entry
        assert "raw_candidate_rank" in entry
        assert "cross_family_rrf_rank" in entry
        expected_structured_loss = entry["in_e2_expanded"] and not entry["in_e3_expanded"]
        expected_raw_loss = entry["in_e1"] and not entry["in_e3_expanded"]
        assert entry["structured_recovered_but_e3_lost"] == expected_structured_loss
        assert entry["raw_recovered_but_e3_lost"] == expected_raw_loss


def test_family_union_ceiling_and_budget_loss() -> None:
    data = _load_json(SCORING_DIR / "family-union-ceiling.json")

    assert data["e1_only"] + data["e2_expanded_only"] + data["both"] + data["neither"] == 80
    assert data["family_union_gold"] == data["e1_only"] + data["e2_expanded_only"] + data["both"]
    assert data["fusion_budget_loss"] == len(data["loss_details"])
    assert all(
        (detail["in_e1"] or detail["in_e2_expanded"]) and not detail["in_e3_expanded"]
        for detail in data["loss_details"]
    )
    assert data["family_union_gold"] == 58
    assert data["e3_expanded_gold"] == 52
    assert data["fusion_budget_loss"] == 8


def test_fusion_loss_categories_are_mutually_exclusive() -> None:
    data = _load_json(SCORING_DIR / "fusion-loss-classification.json")
    expected_categories = {
        "cross_family_rrf_displacement",
        "structured_rank_preserved_but_raw_competition",
        "raw_rank_preserved_but_structured_competition",
        "multi_slot_family_budget_loss",
        "duplicate_candidate_budget_waste",
        "candidate_family_score_tie",
    }

    assert data["mutually_exclusive"] is True
    assert set(data["categories"]) == expected_categories
    assert sum(data["classification_counts"].values()) == data["total_fusion_losses"] == 8
    classified_ids = [
        (detail["case_id"], detail["source_index"], detail["candidate_key"])
        for detail in data["classification_details"]
    ]
    assert len(classified_ids) == len(set(classified_ids)) == 8


def test_acceptance_is_diagnostic_only_and_records_fixes() -> None:
    data = _load_json(R3_DIR / "fusion-attribution-acceptance.json")

    assert data["diagnostic_only"] is True
    assert all(data["fixes_applied"].values())
    assert data["corrected_raw_parity"] == {
        "bm25_source_recall_200": "37/80",
        "dense_source_recall_200": "14/80",
        "raw_full_pool": "31/80",
        "rrf_recall_40": "20/80",
    }


def test_core_judgment_recommends_r4() -> None:
    next_gate = _load_json(R3_DIR / "fusion-attribution-next-gate.json")
    closure = _load_json(R3_DIR / "fusion-attribution-closure.json")

    assert next_gate["fusion_budget_loss"] >= 3
    assert next_gate["decision"] == "fusion_attribution_closure_recommends_r4"
    assert next_gate["next_gate"] == "candidate_family_fusion_closure"
    assert closure["core_judgment"]["decision"] == next_gate["decision"]
    assert closure["core_judgment"]["next_gate"] == next_gate["next_gate"]


def test_closure_performed_no_search_or_index_work() -> None:
    data = _load_json(R3_DIR / "fusion-attribution-closure.json")

    assert data["diagnostic_only"] is True
    assert data["new_index_builds"] == 0
    assert data["bm25_searches"] == 0
    assert data["dense_searches"] == 0
    assert data["embedding_calls"] == 0
