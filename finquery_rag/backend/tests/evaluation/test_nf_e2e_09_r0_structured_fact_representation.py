"""Focused NF-E2E-09 R0 contract tests."""
from __future__ import annotations

import gzip
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts/evaluation/nf-e2e-09-r0-structured-financial-fact-representation"


def read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def read_jsonl_gz(name: str) -> list[dict]:
    with gzip.open(ARTIFACTS / name, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_runner():
    path = ROOT / "scripts/evaluation/run_nf_e2e_09_r0_structured_fact_representation.py"
    spec = importlib.util.spec_from_file_location("nf09_runner", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_frozen_input_and_execution_guards() -> None:
    contract = read_json("frozen-input-contract.json")
    assert contract["evaluation_role"] == "development_shadow_structured_fact_representation_review"
    assert contract["fresh_blind_evaluation"] is False
    assert contract["sada_top100"] == {"hits": 78, "total": 80, "recall": 97.5}
    assert contract["top5"]["candidates"] == 46 * 5
    assert contract["top5"]["token_budget"] == 1100
    assert contract["top5"]["order_unchanged"] is True
    assert contract["nf_opt_26_manifest_sha256"] == "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
    assert contract["model_calls"] == contract["retrieval_calls"] == contract["reranker_calls"] == 0
    assert contract["pdf_reparse"] is False
    assert contract["production_switch_allowed"] is False


def test_financial_fact_schema_and_contract_are_sealed() -> None:
    schema = read_json("financial-fact-v1.schema.json")
    contract = read_json("financial-fact-v1-contract.json")
    assert schema["title"] == "FinancialFactV1"
    assert "question_id" not in json.dumps(schema)
    assert contract["fact_id"]["question_independent"] is True
    assert contract["provenance_complete_requires"]
    assert sha256(ARTIFACTS / "financial-fact-v1-contract.json") == (ARTIFACTS / "financial-fact-v1-contract.sha256").read_text(encoding="utf-8").strip()


def test_materialization_is_query_independent_and_fail_closed() -> None:
    facts = read_jsonl_gz("financial-facts-v1.jsonl.gz")
    seal = read_json("financial-facts-v1-seal.json")
    assert seal["complete"] is True
    assert seal["question_reads_during_materialization"] == 0
    assert seal["gold_reads_during_materialization"] == 0
    assert seal["model_calls"] == seal["retrieval_calls"] == seal["reranker_calls"] == 0
    assert seal["pdf_reparse"] is False
    assert sha256(ARTIFACTS / "financial-facts-v1.jsonl.gz") == seal["financial_facts_sha256"]
    assert facts
    assert all(item["provenance_complete"] is True for item in facts)
    assert all("question_id" not in item and "case_id" not in item for item in facts)
    assert all(item["fact_id"] for item in facts)
    assert len({item["fact_id"] for item in facts}) == len(facts)
    assert all(item["candidate_id"] in item.get("candidate_ids", []) for item in facts)

    runner = load_runner()
    signature = inspect.signature(runner.materialize_candidate)
    assert "question" not in signature.parameters
    assert "query" not in signature.parameters
    assert "gold" not in signature.parameters


def test_coverage_and_relation_integrity() -> None:
    decision = read_json("decision.json")
    query = read_json("query-level-coverage.json")
    candidate = read_json("candidate-level-coverage.json")
    relation = read_json("relation-integrity.json")
    assert query["denominator"] == 46
    assert query["counts"]["financial_fact_available"] == 39
    assert query["counts"]["typed_metric"] == 39
    assert query["counts"]["typed_period"] == 39
    assert query["counts"]["typed_numeric"] == 39
    assert query["counts"]["full_provenance"] == 39
    assert candidate["unique_top5_candidates"] == 175
    assert candidate["facts_materialized_deduplicated"] == 169
    assert candidate["provenance_complete_facts"] == 169
    assert candidate["numeric_parse_success"] == 169
    assert relation["fail"] == 0
    assert relation["percent"] == 100.0
    assert relation["fabricated_cross_candidate_facts"] == 0
    assert decision["structured_fact_representation_effective"] is True
    assert decision["financial_fact_v1_frozen"] is True
    assert decision["next_gate"] == "dfs_v1_retry_on_frozen_financial_fact"


def test_representation_audit_preserves_prior_failure_and_no_downstream_replay() -> None:
    lineage = read_json("representation-field-lineage.json")
    failures = read_json("structured-fact-failure-taxonomy.json")
    no_structured = read_json("no-structured-fact-root-cause.json")
    materialization = read_json("materialization-failure-taxonomy.json")
    baseline = read_json("baseline-vs-financial-fact-v1.json")
    assert lineage["query_independent"] is True
    assert lineage["question_reads_during_materialization"] == 0
    assert failures["denominator"] == 43
    assert failures["counts"]["SF0_parsed_numeric_value_missing"] == 43
    assert no_structured["denominator"] == 3
    assert materialization["fabricated_facts"] == 0
    assert materialization["failures"]["MF0_success"] == 90
    assert baseline["downstream_evaluation"] is False
