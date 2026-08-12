"""Focused NF-V2-02 shadow materialization contract tests."""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
from pathlib import Path

from scripts.evaluation import run_nf_e2e_09_r0_structured_fact_representation as nf09


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/nf-v2-02-top20-financial-fact-expansion"
OLD = ROOT / "artifacts/evaluation/nf-e2e-09-r0-structured-financial-fact-representation"


def load_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_top20_input_is_frozen_and_top5_is_subset() -> None:
    seal = json.loads((OUT / "top20-candidate-seal.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "frozen-input-contract.json").read_text(encoding="utf-8"))
    assert seal["candidate_occurrences"] == 72 * 20
    assert seal["retrieval_recomputed"] is False
    assert seal["reranker_recomputed"] is False
    assert contract["top5_is_subset_of_top20"] is True
    assert contract["candidate_order_unchanged"] is True


def test_financial_fact_contract_and_sffm_are_unchanged() -> None:
    reference = json.loads((OUT / "financial-fact-contract-reference.json").read_text(encoding="utf-8"))
    assert reference["schema"] == "FinancialFactV1"
    assert reference["contract_sha256"] == "7a253b443962c5f372dd897c49c057a19b553e92314faadc31eefc82b27b54eb"
    assert reference["financial_fact_v1_modified"] is False
    assert reference["sffm_v1_modified"] is False
    assert "question" not in reference["query_independent_api"]
    assert "gold" not in reference["query_independent_api"].casefold()


def test_materializer_does_not_accept_question_or_gold() -> None:
    parameters = inspect.signature(nf09.materialize_candidate).parameters
    assert list(parameters) == ["candidate", "atomic_index"]
    assert "question" not in parameters
    assert "gold" not in parameters


def test_same_candidate_facts_and_fact_ids_are_preserved() -> None:
    old = load_gz(OLD / "financial-facts-v1.jsonl.gz")
    new = load_gz(OUT / "top20-materialized-facts.jsonl.gz")
    old_by_id = {row["fact_id"]: row for row in old}
    new_by_id = {row["fact_id"]: row for row in new}
    assert set(old_by_id) <= set(new_by_id)
    for fact_id, old_row in old_by_id.items():
        new_row = new_by_id[fact_id]
        ignored = {"candidate_id", "candidate_ids"}
        assert {key: value for key, value in old_row.items() if key not in ignored} == {
            key: value for key, value in new_row.items() if key not in ignored
        }


def test_relation_integrity_and_fabrication_guards() -> None:
    relation = json.loads((OUT / "relation-integrity.json").read_text(encoding="utf-8"))
    fabrication = json.loads((OUT / "fabrication-safety.json").read_text(encoding="utf-8"))
    assert relation["relation_integrity_fail"] == 0
    assert fabrication["fabricated_cross_candidate_facts"] == 0
    assert fabrication["cross_candidate_composition"] is False


def test_top20_materialization_seal_has_no_query_or_gold_reads() -> None:
    seal = json.loads((OUT / "top20-materialization-seal.json").read_text(encoding="utf-8"))
    assert seal["question_reads_during_materialization"] == 0
    assert seal["gold_reads_during_materialization"] == 0
    assert seal["model_calls"] == 0
    assert seal["retrieval_calls"] == 0
    assert seal["reranker_calls"] == 0
    digest = hashlib.sha256((OUT / "top20-materialized-facts.jsonl.gz").read_bytes()).hexdigest()
    assert digest == seal["financial_facts_sha256"]


def test_no_downstream_component_was_called() -> None:
    decision = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    projection = json.loads((OUT / "binder-packet-projection.json").read_text(encoding="utf-8"))
    assert decision["model_calls"] == 0
    assert decision["retrieval_execution"] is False
    assert decision["reranker_execution"] is False
    assert projection["binder_called"] is False
    assert decision["production_switch_allowed"] is False
