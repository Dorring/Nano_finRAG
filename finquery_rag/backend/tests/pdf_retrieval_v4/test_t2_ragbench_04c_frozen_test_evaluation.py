from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = ROOT / "scripts" / "evaluation"
MODULE_PATH = SCRIPT_DIR / "run_t2_ragbench_04c_frozen_test_evaluation.py"
ARTIFACT = ROOT / "artifacts/evaluation/t2-ragbench-04c-frozen-test-evaluation"

sys.path.insert(0, str(SCRIPT_DIR))
spec = importlib.util.spec_from_file_location("t2_04c", MODULE_PATH)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def read_json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="utf-8"))


def test_frozen_method_and_feature_seal_are_verified() -> None:
    summary = read_json("external-track-final-summary.json")
    assert summary["method_hash"] == module.METHOD_HASH
    assert summary["feature_seal"] == module.FEATURE_SEAL_SHA
    assert summary["external_structure_method_validated"] is True


def test_prediction_counts_and_gold_preseal_contract() -> None:
    primary = read_json("primary-test-prediction-seal.json")
    conv = read_json("convfinqa-transfer-prediction-seal.json")
    assert primary["prediction_count"] == module.PRIMARY_TEST_COUNT
    assert conv["prediction_count"] == module.CONVFINQA_COUNT
    assert primary["gold_reads_before_seal"] == 0
    assert conv["gold_reads_before_seal"] == 0
    assert primary["method_hash"] == conv["method_hash"] == module.METHOD_HASH
    assert primary["feature_seal"] == conv["feature_seal"] == module.FEATURE_SEAL_SHA


def test_primary_metrics_and_r50_invariant() -> None:
    metrics = read_json("primary-test-metrics.json")
    assert metrics["bm25"]["count"] == module.PRIMARY_TEST_COUNT
    assert metrics["pcr_v1"]["count"] == module.PRIMARY_TEST_COUNT
    assert metrics["bm25"]["recall_pct"]["@5"] == 73.723265
    assert metrics["pcr_v1"]["recall_pct"]["@5"] == 74.552597
    assert metrics["bm25"]["hits"]["50"] == metrics["pcr_v1"]["hits"]["50"] == 2113
    movement = read_json("primary-test-rank-movement.json")
    assert movement["rescued_at_5"] == 59
    assert movement["damaged_at_5"] == 40
    assert movement["net_top5_gain"] == 19
    assert movement["rescue_precision"] == 0.59596


def test_no_period_cohort_is_exactly_bm25() -> None:
    primary = read_json("primary-test-period-cohort.json")
    conv = read_json("convfinqa-transfer-period-cohort.json")
    for cohort in (primary["no_period_requirement"], conv["no_period_requirement"]):
        assert cohort["ranking_identical"] is True
        assert cohort["bm25"] == cohort["pcr_v1"]


def test_candidate_set_is_unchanged_for_every_prediction() -> None:
    retrieval = (
        ROOT
        / "artifacts/evaluation/t2-ragbench-01-standard-retrieval/bm25-predictions.jsonl.gz"
    )
    bm25 = {}
    for line in gzip.open(retrieval, "rt", encoding="utf-8"):
        row = json.loads(line)
        if row["query_id"].startswith(("finqa_test_", "tatqa_test_", "convfinqa_")):
            bm25[row["query_id"]] = [item["context_id"] for item in row["ranked_contexts"][:50]]
    for filename in ("primary-test-predictions.jsonl.gz", "convfinqa-transfer-predictions.jsonl.gz"):
        path = ARTIFACT / filename
        rows = list(gzip.open(path, "rt", encoding="utf-8"))
        assert rows
        for line in rows:
            row = json.loads(line)
            predicted = [item["context_id"] for item in row["ranked_contexts"]]
            assert len(predicted) == 50
            assert set(predicted) == set(bm25[row["query_id"]])


def test_subset_and_transfer_metrics_are_recorded() -> None:
    subset = read_json("primary-test-subset-analysis.json")
    assert subset["FinQA"]["pcr_v1"]["recall_pct"]["@5"] == 89.537925
    assert subset["TAT-DQA"]["pcr_v1"]["recall_pct"]["@5"] == 59.527972
    transfer = read_json("convfinqa-transfer-metrics.json")
    assert transfer["bm25"]["recall_pct"]["@5"] == 83.718913
    assert transfer["pcr_v1"]["recall_pct"]["@5"] == 88.114517
    movement = read_json("convfinqa-transfer-rank-movement.json")
    assert movement["rescued_at_5"] == 197
    assert movement["damaged_at_5"] == 45
    assert movement["net_top5_gain"] == 152


def test_final_summary_is_deterministic_and_transfer_supported() -> None:
    summary = read_json("external-track-final-summary.json")
    assert summary["primary_test_queries"] == 2291
    assert summary["convfinqa_queries"] == 3458
    assert summary["pcr_v1_recall_at_5"] == 0.7455259700000001
    assert summary["convfinqa_pcr_recall_at_5"] == 0.88114517
    assert summary["transfer_supported"] is True
    decision = read_json("decision.json")
    assert decision["candidate_universe_unchanged"] is True
    assert decision["primary_r50_invariant"] is True
    assert decision["convfinqa_r50_invariant"] is True
    assert decision["next_gate"] == "external_track_complete"
