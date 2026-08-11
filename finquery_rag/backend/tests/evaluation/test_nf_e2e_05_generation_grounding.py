"""Focused NF-E2E-05 contract and seal checks.

The replay artifacts are intentionally read-only here.  These tests verify
that the one-shot shadow run preserved the frozen retrieval/calculation
contracts, that GGIA was identity-preserving, and that scoring happened only
after a complete output seal.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "evaluation" / (
    "nf-e2e-05-r0-generation-grounding-recovery"
)


def _read_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_stage_a_exposes_contract_defects_without_gold_access() -> None:
    decision = _read_json("generation-grounding-decision.json")
    visibility = _read_json("generator-evidence-visibility.json")

    assert decision["generation_input_contract_defect_supported"] is True
    assert decision["generation_grounding_objective_mismatch_supported"] is True
    assert decision["gold_reads_during_runtime"] == 0
    assert visibility["generator_receives_stable_evidence_ids"] is False
    assert visibility["generator_receives_citation_labels"] is False
    assert visibility["generator_receives_abstention_contract"] is False
    assert visibility["citation_namespace_compatible"] is False


def test_grounded_instruction_is_sealed_and_exact() -> None:
    expected = (
        "Answer the financial question using only the provided evidence.\n\n"
        "Every factual claim in the answer must be directly supported by the "
        "provided evidence. Cite the supporting evidence using the citation "
        "identifiers exactly as provided in the context.\n\n"
        "Do not cite evidence that does not directly support the claim.\n\n"
        "If the provided evidence is insufficient to answer the question "
        "reliably, return the system's safe-response form instead of "
        "inferring or guessing.\n\n"
        "When an authoritative deterministic calculation result is provided, "
        "preserve its numeric value, period, unit, and supporting evidence "
        "exactly."
    )
    instruction = OUT / "grounded-generation-instruction-v1.txt"
    assert instruction.read_text(encoding="utf-8") == expected
    assert _sha256(instruction) == (OUT / "grounded-generation-instruction-v1.sha256").read_text(
        encoding="utf-8"
    ).strip()


def test_ggia_mapping_is_deterministic_and_identity_preserving() -> None:
    contract = _read_json("ggia-v1-contract.json")
    manifest = _read_json("ggia-v1-mapping-manifest.json")
    seal = _read_json("output-seal.json")

    assert contract["assigns_deterministic_local_numeric_labels"] is True
    assert contract["preserves_evidence_order"] is True
    assert contract["preserves_evidence_text"] is True
    assert contract["preserves_physical_source_identity"] is True
    assert contract["adds_evidence"] is False
    assert contract["drops_evidence"] is False
    assert contract["reorders_evidence"] is False
    assert contract["gold_access"] is False
    assert len(manifest["rows"]) == 72
    assert manifest["gold_reads_during_mapping"] == 0
    for row in manifest["rows"]:
        assert row["added_evidence"] == 0
        assert row["dropped_evidence"] == 0
        assert row["reordered"] is False
        assert row["evidence_text_preserved"] is True
        labels = row["labels"]
        assert [item["label"] for item in labels] == [
            f"[{index}]" for index in range(1, len(labels) + 1)
        ]
        assert all(item["candidate_id"] for item in labels)
        assert all(item["physical_source_id"] for item in labels)
    assert seal["ggia_mapping_sha256"] == manifest["mapping_sha256"]


def test_complete_output_is_sealed_before_gold_scoring() -> None:
    seal = _read_json("output-seal.json")
    runtime = seal["runtime"]
    input_sha = (OUT / "shadow-input-manifest.sha256").read_text(encoding="utf-8").strip()

    assert seal["complete"] is True
    assert seal["case_count"] == 72
    assert seal["gold_reads_during_execution"] == 0
    assert seal["input_manifest_sha256"] == input_sha
    assert seal["instruction_sha256"] == (
        OUT / "grounded-generation-instruction-v1.sha256"
    ).read_text(encoding="utf-8").strip()
    assert runtime["case_count"] == 72
    assert runtime["model_calls"] == 0
    assert runtime["model_chat_completion_requests"] == 0
    assert runtime["gold_reads_during_execution"] == 0


def test_safety_and_frozen_calculation_contract_are_preserved() -> None:
    decision = _read_json("decision.json")
    calc = _read_json("calculation-preservation.json")
    baseline = _read_json("baseline-vs-ggc.json")

    assert decision["ggia_v1_executed"] is True
    assert decision["ggc_v1_executed"] is True
    assert decision["production_switch_allowed"] is False
    assert decision["generation_grounding_recovery_effective"] is False
    assert decision["dominant_bottleneck_after_recovery"] == "citation_binding"
    assert calc["calculator_response_byte_equivalent"] == 5
    assert calc["calculator_strict_correct"] == 5
    assert calc["final_numeric_correct"] == 5
    assert calc["final_period_correct"] == 5
    assert calc["final_unit_correct"] == 5
    assert calc["false_binding"] == 0
    assert calc["false_execution"] == 0
    assert calc["executed_incorrect"] == 0
    assert baseline["baseline_nf_e2e_04"] == baseline["ggc_v1"]
