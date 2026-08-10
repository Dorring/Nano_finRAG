from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "artifacts/evaluation"
CLOSURE = EVAL / "t2-ragbench-00-r1"
BASELINE = EVAL / "t2-ragbench-01-standard-retrieval"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_published_raw_closure_keeps_all_rows_and_known_anomaly() -> None:
    acceptance = read_json(CLOSURE / "acceptance.json")
    anomaly = read_json(CLOSURE / "empty-question-audit.json")
    contract = read_json(CLOSURE / "published-anomaly-contract.json")
    assert acceptance["dataset_materialization_accepted"] is True
    assert acceptance["headline_denominator"] == 23088
    assert acceptance["empty_question_rows"] == 11
    assert acceptance["silent_exclusion_allowed"] is False
    assert acceptance["question_repair_allowed"] is False
    assert anomaly["count"] == 11
    assert all(row["question_raw_value"] == "" for row in anomaly["rows"])
    assert all(row["question_python_type"] == "str" for row in anomaly["rows"])
    assert contract["published_raw_track_ready"] is True


def test_official_query_contract_preserves_empty_question_fstring() -> None:
    contract = read_json(CLOSURE / "official-query-contract.json")
    assert contract["query_template"] == "f'{company_name} : {question}'"
    assert contract["empty_question_behavior"] == "Company Name : "
    assert contract["query_count"] == 23088


def test_t2_01_prediction_seal_is_complete_and_frozen() -> None:
    seal = read_json(BASELINE / "prediction-seal.json")
    manifest = read_json(BASELINE / "prediction-manifest.json")
    assert seal["sealed"] is True
    assert seal["prediction_count"] == 23088
    assert seal["candidate_budget"] == 100
    assert seal["gold_scoring_reads_before_seal"] == 0
    assert seal["pdf_parsing"] == 0
    assert seal["chunking"] == 0
    assert seal["cross_encoder"] == 0
    assert seal["llm"] == 0
    assert seal["parameter_scan"] is False
    assert manifest["query_count"] == 23088
    assert manifest["corpus_count"] == 7318
    for baseline in ("bm25", "dense", "hybrid"):
        path = BASELINE / f"{baseline}-predictions.jsonl.gz"
        assert digest(path) == seal["output_sha256"][baseline]
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            assert sum(1 for _ in handle) == 23088


def test_t2_01_context_id_scoring_has_no_identity_errors() -> None:
    acceptance = read_json(BASELINE / "acceptance.json")
    audit = read_json(BASELINE / "identity-score-audit.json")
    assert acceptance["decision"] == "first_valid_public_benchmark_measurement"
    assert acceptance["published_rows"] == 23088
    assert acceptance["gold_identity"] == "context_id"
    assert acceptance["gold_identity_errors"] == 0
    assert acceptance["baselines_completed"] == ["bm25", "dense", "hybrid"]
    assert all(
        all(value == 0 for value in values.values())
        for values in audit["by_baseline"].values()
    )


def test_weighted_metrics_keep_published_denominator() -> None:
    weighted = read_json(BASELINE / "weighted-metrics.json")
    empty = read_json(BASELINE / "empty-question-diagnostic.json")
    assert weighted["denominator"] == 23088
    assert set(weighted["baselines"]) == {"bm25", "dense", "hybrid"}
    assert empty["published_denominator"] == 23088
    assert empty["nonempty_diagnostic_denominator"] == 23077
    assert empty["diagnostic_only"] is True

