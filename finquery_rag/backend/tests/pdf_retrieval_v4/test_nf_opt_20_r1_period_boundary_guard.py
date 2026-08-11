"""Focused contract tests for NF-OPT-20 R1 BPG-V1."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.evaluation.run_nf_opt_20_r1_period_boundary_guard import period_status


BACKEND_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = (
    BACKEND_ROOT
    / "artifacts"
    / "evaluation"
    / "nf-opt-20-r1-period-boundary-guard"
)
QWEN_PATH = (
    BACKEND_ROOT
    / "artifacts"
    / "evaluation"
    / "pdf-retrieval-v4-gate-08-r8-r3-3"
    / "main_rerank_predictions.jsonl.gz"
)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT_ROOT / name).read_text(encoding="utf-8"))


def read_gzip(name: str) -> list[dict]:
    with gzip.open(ARTIFACT_ROOT / name, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_qwen() -> dict[str, list[str]]:
    with gzip.open(QWEN_PATH, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return {
        row["case_id"]: [item["candidate_key"] for item in sorted(row["ranked_candidates"], key=lambda x: x["post_rerank_rank"])]
        for row in rows
    }


def test_bpg_decision_and_prediction_seal_contract() -> None:
    decision = read_json("decision.json")
    manifest = read_json("prediction-manifest.json")
    seal = read_json("prediction-seal.json")
    assert decision["gate"] == "NF-OPT-20-R1"
    assert decision["evaluation_role"] == "development_shadow_calibration"
    assert decision["fresh_blind_evaluation"] is False
    assert decision["model_execution"] is False
    assert decision["retrieval_rerun"] is False
    assert decision["production_switch_allowed"] is False
    assert decision["strict_sources"] == 80
    assert decision["queries_mutated"] <= 72
    assert manifest["rows"] == 72
    assert manifest["gold_reads_during_prediction"] == 0
    assert seal["sealed"] is True


def test_bpg_preserves_top100_and_qwen_order_when_no_eligibility() -> None:
    predictions = read_gzip("predictions.jsonl.gz")
    qwen = read_qwen()
    assert len(predictions) == 72
    for row in predictions:
        case_id = row["case_id"]
        candidates = sorted(row["ranked_candidates"], key=lambda item: item["bpg_rank"])
        keys = [item["candidate_key"] for item in candidates]
        assert len(keys) == 100
        assert set(keys) == set(qwen[case_id])
        assert keys[:4] == qwen[case_id][:4]
        assert sum(item["bpg_role"] == "promoted_challenger" for item in candidates) <= 1
        assert sum(item["bpg_role"] == "demoted_incumbent" for item in candidates) <= 1


def test_bpg_period_unknown_is_fail_closed_and_safety_metrics_hold() -> None:
    audit = read_json("period-resolution-audit.json")
    eligibility = read_json("eligibility-audit.json")
    decision = read_json("decision.json")
    assert audit["gold_reads_before_seal"] == 0
    assert all(record["query_period"]["status"] in {"explicit_single_period", "explicit_multi_period", "no_explicit_period", "unresolved"} for record in audit["records"])
    for record in eligibility["records"]:
        if record["query_period_status"] in {"no_explicit_period", "unresolved"}:
            assert record["mutated"] is False
    assert decision["hard_safety"]["top10_candidate_set_invariant"] is True
    assert decision["hard_safety"]["top100_candidate_set_invariant"] is True
    assert decision["hard_safety"]["rank1_4_unchanged"] is True


def test_bpg_metrics_have_no_regression_and_decision_is_shadow_only() -> None:
    decision = read_json("decision.json")
    strict = read_json("strict-metrics.json")
    semantic = read_json("semantic-metrics.json")
    assert strict["before_qwen"]["@5"]["hits"] == 43
    assert strict["after_bpg"]["@5"]["hits"] == 43
    assert strict["before_qwen"]["@100"]["hits"] == strict["after_bpg"]["@100"]["hits"] == 68
    assert semantic["before_qwen"]["@5"]["hits"] == semantic["after_bpg"]["@5"]["hits"] == 49
    assert semantic["before_qwen"]["@10"]["hits"] == semantic["after_bpg"]["@10"]["hits"] == 61
    assert decision["development_shadow_result"] is True
    assert decision["production_switch_allowed"] is False


def test_period_contract_is_fail_closed_for_unknown_and_unresolved() -> None:
    no_period = {"status": "no_explicit_period", "required_periods": []}
    unresolved = {"status": "unresolved", "required_periods": []}
    explicit = {"status": "explicit_single_period", "required_periods": ["fy2025"]}
    assert period_status(no_period, {"fy2025"}, False) == "NEUTRAL"
    assert period_status(unresolved, {"fy2025"}, False) == "NEUTRAL"
    assert period_status(explicit, set(), False) == "NEUTRAL"
    assert period_status(explicit, {"fy2025"}, False) == "FULL_MATCH"
    assert period_status(explicit, {"fy2024"}, False) == "EXPLICIT_CONFLICT"
