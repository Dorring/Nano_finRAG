"""Focused contract tests for the sealed NF-OPT-20 R0 audit."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    BACKEND_ROOT
    / "artifacts"
    / "evaluation"
    / "nf-opt-20-r0-pointwise-discrimination-audit"
)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def test_nf_opt_20_decision_contract_is_frozen() -> None:
    decision = read_json("decision.json")
    assert decision["gate"] == "NF-OPT-20-R0"
    assert decision["model_execution"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["ranking_mutation"] is False
    assert decision["production_switch_allowed"] is False
    assert decision["c1_cases"] == 25
    assert (decision["c0"], decision["c1"], decision["c2"]) == (43, 25, 12)
    assert decision["near_boundary"] == 13
    assert decision["clear_loss"] == 12
    assert decision["near_boundary"] + decision["clear_loss"] == decision["c1_cases"]
    assert decision["candidate_identity_mismatch"] == 0
    assert decision["qwen_rank_mismatch"] == 0
    assert decision["qwen_scores_unchanged"] is True
    assert decision["qwen_ranks_unchanged"] is True


def test_nf_opt_20_c1_and_recoverability_partitions() -> None:
    with gzip.open(ARTIFACT_ROOT / "c1-pairs.jsonl.gz", "rt", encoding="utf-8") as stream:
        pairs = [json.loads(line) for line in stream if line.strip()]
    assert len(pairs) == 25
    assert {row["cohort"] for row in pairs} == {"near_boundary", "clear_loss"}
    assert sum(row["cohort"] == "near_boundary" for row in pairs) == 13
    assert sum(row["cohort"] == "clear_loss" for row in pairs) == 12
    assert all(row["competitors"] for row in pairs)

    counts = read_json("recoverability-classes.json")["counts"]
    assert set(counts) == {"P1", "P2", "P3", "P4"}
    assert sum(counts.values()) == 25

    movement = read_json("bm25-vs-qwen-movement.json")
    assert movement["denominator"] == 25
    assert sum(movement["overall"].values()) == 25


def test_nf_opt_20_review_package_is_diagnostic_only() -> None:
    package = read_json("human-review-package.json")
    assert package["review_status"] == "diagnostic_only"
    assert package["c1_total"] == 25
    assert all(record["review_status"] == "diagnostic_only" for record in package["records"])
    assert all("physical_binding" in record["gold"] for record in package["records"])


def test_nf_opt_20_t2_contrast_is_aggregate_only() -> None:
    contrast = read_json("t2-contrast-analysis.json")
    assert contrast["source"] == "sealed_t2_05_aggregate_artifacts"
    assert set(contrast["subsets"]) == {"FinQA", "TAT-DQA"}
    assert set(contrast["query_types"]) == {"difference", "percentage", "percentage_change", "ratio"}
    assert "internal_case_id" not in json.dumps(contrast)
