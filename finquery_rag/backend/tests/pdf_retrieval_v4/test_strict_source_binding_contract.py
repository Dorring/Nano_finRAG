from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "artifacts/evaluation/pdf-retrieval-v4-strict-source-contract"
RESCORE = ROOT / "artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r1-1b"


def load(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text())


def test_source_binding_contract_is_exact_80() -> None:
    audit = load(CONTRACT, "binding-parity.json")
    assert audit["source_binding_count"] == "80/80"
    assert audit["source_index_parity"] == "80/80"
    assert audit["candidate_key_parity"] == "80/80"
    assert audit["document_id_parity"] == "80/80"
    assert audit["page_parity"] == "80/80"
    assert audit["evidence_id_parity"] == "80/80"


def test_source_binding_contract_has_no_mutation() -> None:
    audit = load(CONTRACT, "binding-parity.json")
    assert audit["missing_bindings"] == 0
    assert audit["extra_bindings"] == 0
    assert audit["reordered_bindings"] == 0
    assert audit["mismatches"] == []


def test_binding_ids_are_unique_and_sidecar_hash_is_sealed() -> None:
    path = CONTRACT / "strict-gold-source-bindings.jsonl"
    bindings = [json.loads(line) for line in path.open()]
    assert len(bindings) == len({item["binding_id"] for item in bindings}) == 80
    assert hashlib.sha256(path.read_bytes()).hexdigest() == load(
        CONTRACT, "binding-parity.json"
    )["sidecar_sha256"]


def test_unique_candidate_set_is_diagnostic_only() -> None:
    audit = load(CONTRACT, "binding-parity.json")
    assert (
        audit["strict_gold_identities_status"]
        == "strict_gold_unique_candidate_set_diagnostic_only"
    )
    assert audit["retrieval_recall_scoring_unit"] == "case_id_source_index_candidate_key"


def test_r8_r1_unified_binding_rescore() -> None:
    metrics = load(RESCORE, "unified-binding-metrics.json")
    assert metrics["r8_r1"]["strict_source_binding_recall_at_50"] == "55/80"
    assert metrics["unbounded_presence"] == "60/80"
    assert metrics["union_to_top50_conversion"] == "55/60"
    assert (metrics["gross_loss"], metrics["selector_synergy"], metrics["net_gap"]) == (
        6,
        1,
        5,
    )


def test_r8_r1_raw_retention_uses_binding_denominator() -> None:
    metrics = load(RESCORE, "unified-binding-metrics.json")
    assert metrics["production_raw_own_recall_at_50"] == "24/80"
    assert metrics["raw_retained"] == "22/24"
    assert metrics["raw_regression"] == 2


def test_rescore_does_not_rerun_predictions_or_retrieval() -> None:
    acceptance = load(RESCORE, "acceptance.json")
    assert acceptance["prediction_reruns"] == 0
    assert acceptance["fusion_reruns"] == 0
    assert acceptance["retrieval_runs"] == 0
    assert acceptance["index_reads"] == 0
    assert acceptance["embedding_calls"] == 0


def test_r1_2_is_next_but_r1_raw_gate_remains_blocked() -> None:
    acceptance = load(RESCORE, "acceptance.json")
    assert acceptance["decision"] == "bounded_candidate_raw_regression_blocked"
    assert acceptance["next_gate"] == "support_count_invariant_candidate_fusion"
