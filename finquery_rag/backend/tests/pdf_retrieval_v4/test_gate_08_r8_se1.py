from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from src.pdf_retrieval_v4.runtime_semantic_fact_identity import (
    build_semantic_fact,
    expand_authoritative_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/evaluation"
P0 = BASE / "pdf-retrieval-v4-gate-08-r8-se1-p0"
SE1 = BASE / "pdf-retrieval-v4-gate-08-r8-se1"
R33 = BASE / "pdf-retrieval-v4-gate-08-r8-r3-3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_semantic_identity_excludes_physical_provenance() -> None:
    fields = {
        "document_id": "issuer_fy2025",
        "normalized_metric": "Revenue",
        "normalized_period": "FY2025",
        "normalized_base_value": "100",
        "normalized_scale": "1000000",
        "normalized_currency": "USD",
    }
    left = build_semantic_fact(fields, {"row_id": "row:a", "pdf_page": 1})
    right = build_semantic_fact(fields, {"row_id": "row:b", "pdf_page": 9})
    assert left is not None and right is not None
    assert left["semantic_fact_id"] == right["semantic_fact_id"]


def test_row_matrix_expands_to_distinct_cell_facts() -> None:
    matrix = {
        "document_id": "issuer_fy2025",
        "metric_path": "Revenue",
        "scale": 1000000,
        "currency_code": "USD",
        "row_id": "row:a",
        "dimensions": [
            {"normalized_period": "FY2025", "value_normalized": "100", "cell_id": "cell:a"},
            {"normalized_period": "FY2024", "value_normalized": "90", "cell_id": "cell:b"},
        ],
    }
    facts, status = expand_authoritative_evidence("matrix:test", matrix)
    assert status == "expanded"
    assert len(facts) == 2
    assert len({fact["semantic_fact_id"] for fact in facts}) == 2
    assert {fact["normalized_period"] for fact in facts} == {"fy2024", "fy2025"}


def test_incomplete_comparison_fails_closed() -> None:
    comparison = {
        "document_id": "issuer_fy2025",
        "metric_path": "Revenue",
        "base_period": "FY2024",
        "base_value": "90",
        "compared_period": "FY2025",
        "compared_value": None,
    }
    facts, status = expand_authoritative_evidence("comparison:test", comparison)
    assert facts == []
    assert status == "insufficient_for_expansion"


def test_p0_registry_is_zero_gold_and_candidate_exact() -> None:
    seal = json.loads((P0 / "prediction-seal.json").read_text(encoding="utf-8"))
    registry = P0 / "candidate-semantic-fact-registry.jsonl.gz"
    assert seal["sealed"] is True
    assert seal["case_count"] == 72
    assert seal["candidate_occurrence_count"] == 7200
    assert seal["candidate_mutation"] == 0
    assert seal["gold_reads_before_seal"] == 0
    assert seal["strict_source_binding_reads_before_seal"] == 0
    assert seal["semantic_graph_runs"] == seal["bridge_runs"] == 0
    assert seal["retrieval_runs"] == seal["embedding_calls"] == seal["reranker_calls"] == 0
    assert seal["registry_sha256"] == sha256(registry)
    with gzip.open(registry, "rt", encoding="utf-8") as handle:
        assert sum(1 for _ in handle) == seal["unique_candidate_count"]


def test_r3_3_predictions_remain_immutable() -> None:
    p0 = json.loads((P0 / "prediction-seal.json").read_text(encoding="utf-8"))
    assert p0["r3_3_commit"] == "c8a4ef33103e40625ed7e0853d4c93cc7f6b18cd"
    assert p0["r3_3_main_prediction_sha256"] == sha256(R33 / "main_rerank_predictions.jsonl.gz")
    assert p0["r3_3_prediction_seal_sha256"] == sha256(R33 / "prediction-seal.json")


def test_three_metric_granularities_are_not_conflated() -> None:
    strict = json.loads((SE1 / "strict-physical-recall.json").read_text(encoding="utf-8"))
    family = json.loads((SE1 / "evidence-family-recall.json").read_text(encoding="utf-8"))
    semantic = json.loads((SE1 / "semantic-fact-recall.json").read_text(encoding="utf-8"))
    assert strict["recall_at_5"] == "43/80"
    assert family["recall_at_5"] == "43/80"
    assert semantic["recall_at_5"] == "49/80"
    assert semantic["recall_at_10"] == "61/80"


def test_semantic_rescues_are_pending_and_false_equivalence_is_zero() -> None:
    review = json.loads((SE1 / "semantic-rescue-review-package.json").read_text(encoding="utf-8"))
    audit = json.loads((SE1 / "false-equivalence-audit.json").read_text(encoding="utf-8"))
    assert review["rescue_count"] == 6
    assert review["review_status"] == "pending"
    assert review["verified"] is False
    assert all(record["review_status"] == "pending" and record["verified"] is False for record in review["records"])
    assert audit["automatic_false_equivalence"] == 0


def test_semantic_acceptance_uses_conservative_80_binding_denominator() -> None:
    acceptance = json.loads((SE1 / "acceptance.json").read_text(encoding="utf-8"))
    coverage = json.loads((SE1 / "semantic-target-coverage.json").read_text(encoding="utf-8"))
    assert acceptance["historical_strict_physical_source_recall_at_5"] == "43/80"
    assert acceptance["semantic_target_coverage"] == "48/80"
    assert acceptance["semantic_fact_recall_at_5"] == "49/80"
    assert acceptance["semantic_fact_recall_at_10"] == "61/80"
    assert acceptance["next_gate"] == "top10_evidence_set_contract"
    assert acceptance["retrieval_optimization_stopped"] is True
    assert coverage["counts"]["resolved"] == 48
    assert sum(coverage["counts"].values()) == 80
