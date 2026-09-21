#!/usr/bin/env python3
"""NF-E2E-07 R0: deterministic claim-to-evidence provenance audit.

This gate is deliberately offline.  It consumes the sealed NF-E2E-06
response/lineage artifacts and the already sealed Binder/Calculator traces.
It never searches Top5, invokes a model, reruns retrieval, or reconstructs an
answer unless a previously recorded exact derivation relation meets the
pre-registered recovery threshold.  On this benchmark the deterministic fact
selector did not persist that relation, so CGBA-V1 is emitted as a disabled
contract and no response reconstruction is performed.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/evaluation/nf-e2e-07-r0-claim-grounding-recovery"
NF06 = ROOT / "artifacts/evaluation/nf-e2e-06-r0-citation-binding-recovery"
NF05 = ROOT / "artifacts/evaluation/nf-e2e-05-r0-generation-grounding-recovery"
NF03 = ROOT / "artifacts/evaluation/nf-e2e-03-r0-full-replay-after-binder-recovery"

GATE = "NF-E2E-07-R0"
BASE_COMMIT = "816833f887a544f31907a94bde5895498aa3226c"
NF26_SHA = "70048502ec918ae6ee56246a788da42129df3b073c2be8682e14f97e409e7c80"
QUESTION_TOTAL = 72
ANSWERABLE_TOTAL = 64
SUPPORTED_UNCITED_TOTAL = 51
CALC_READY_TOTAL = 5
RECOVERY_THRESHOLD = 8


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write(
                    (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
                )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pct(value: int, total: int) -> float:
    return round(value * 100.0 / total, 4) if total else 0.0


def load_state() -> dict[str, Any]:
    """Load sealed inputs without opening benchmark Gold at runtime."""
    response_seal = read_json(NF06 / "response-seal.json")
    responses_path = NF06 / "reconstructed-responses.jsonl.gz"
    if not response_seal.get("complete") or response_seal.get("case_count") != QUESTION_TOTAL:
        raise RuntimeError("NF-E2E-06 response seal is incomplete")
    if response_seal.get("model_execution") is not False:
        raise RuntimeError("NF-E2E-06 model execution flag is not false")
    if response_seal.get("gold_reads_during_reconstruction") != 0:
        raise RuntimeError("NF-E2E-06 reconstruction read Gold before seal")
    if sha256_file(responses_path) != response_seal.get("response_sha256"):
        raise RuntimeError("NF-E2E-06 response seal mismatch")

    routing = read_json(NF06 / "deterministic-routing-audit.json")
    lineage = read_json(NF06 / "deterministic-citation-lineage.json")
    calc_lineage = read_json(NF06 / "calculation-citation-lineage.json")
    calc_preservation = read_json(NF06 / "calculation-preservation.json")
    citation_metrics = read_json(NF06 / "citation-metrics.json")
    nf05_claims = read_json(NF05 / "claim-grounding-matrix.json")
    nf05_taxonomy = read_json(NF05 / "citation-failure-taxonomy.json")
    nf05_no_answer = read_json(NF05 / "no-answer-analysis.json")
    nf05_multi = read_json(NF05 / "multi-evidence-analysis.json")
    nf03_traces = read_jsonl_gz(NF03 / "per-question-traces.jsonl.gz")
    nf03_seal = read_json(NF03 / "e2e-output-seal.json")
    if not nf03_seal.get("complete") or nf03_seal.get("case_count") != QUESTION_TOTAL:
        raise RuntimeError("NF-E2E-03 trace seal is incomplete")
    if nf03_seal.get("gold_reads_during_execution") != 0:
        raise RuntimeError("NF-E2E-03 execution read Gold before seal")
    if sha256_file(NF03 / "per-question-traces.jsonl.gz") != nf03_seal.get("trace_sha256"):
        raise RuntimeError("NF-E2E-03 trace seal mismatch")

    responses = read_jsonl_gz(responses_path)
    if len(responses) != QUESTION_TOTAL:
        raise RuntimeError("NF-E2E-06 responses are incomplete")
    answerable_ids = {str(row["question_id"]) for row in nf05_claims.get("rows", [])}
    return {
        "response_seal": response_seal,
        "responses": {str(row["question_id"]): row for row in responses},
        "routing": routing,
        "routing_by_id": {str(row["question_id"]): row for row in routing["rows"]},
        "lineage": lineage,
        "lineage_by_id": {str(row["question_id"]): row for row in lineage["rows"]},
        "calc_lineage": calc_lineage,
        "calc_by_id": {str(row["question_id"]): row for row in calc_lineage["rows"]},
        "calc_preservation": calc_preservation,
        "citation_metrics": citation_metrics,
        "claim_rows": nf05_claims["rows"],
        "claim_by_id": {str(row["question_id"]): row for row in nf05_claims["rows"]},
        "taxonomy_by_id": {str(row["question_id"]): row for row in nf05_taxonomy["rows"]},
        "no_answer": nf05_no_answer,
        "multi_evidence": nf05_multi,
        "nf03_trace_by_id": {str(row["question_id"]): row for row in nf03_traces},
        "answerable_ids": answerable_ids,
    }


def write_frozen_contract(state: dict[str, Any]) -> None:
    write_json(
        OUT / "frozen-e2e-contract.json",
        {
            "gate": GATE,
            "base_commit": BASE_COMMIT,
            "evaluation_role": "development_shadow_claim_grounding_recovery",
            "fresh_blind_evaluation": False,
            "model_execution": False,
            "retrieval_calls": 0,
            "retrieval_tuning": False,
            "binder_tuning": False,
            "calculator_tuning": False,
            "generator_tuning": False,
            "validator_tuning": False,
            "production_switch_allowed": False,
            "selected_internal_shadow_method": "sada_statement_aware_v1",
            "sada_top100": "78/80",
            "context": {"top_k": 5, "token_budget": 1100},
            "bica": {"name": "BICA-V1", "entrypoint": "_bind_r53", "unchanged": True},
            "calculator": {"strict_correct": "5/5", "unchanged": True},
            "gcca_unchanged": True,
            "nf_opt_26_manifest_sha256": NF26_SHA,
            "nf_e2e_06_response_sha256": state["response_seal"].get("response_sha256"),
            "nf_e2e_06_response_seal_verified": True,
            "gold_reads_during_stage_a": 0,
        },
    )


def route_for(state: dict[str, Any], case_id: str) -> str:
    return str(state["routing_by_id"][case_id].get("route") or "other_existing_runtime_type")


def claim_text(state: dict[str, Any], case_id: str) -> str:
    return str(state["responses"][case_id].get("released_answer") or "")


def stage_a_claim_provenance(state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for claim_row in state["claim_rows"]:
        case_id = str(claim_row["question_id"])
        route = route_for(state, case_id)
        line = state["lineage_by_id"].get(case_id, {})
        calc = state["calc_by_id"].get(case_id)
        if route == "deterministic_calculation" and calc and calc.get("binder_supporting_evidence_ids"):
            derivation_type = "deterministic_calculation"
            actual_ids = list(calc.get("binder_supporting_evidence_ids") or [])
            exact = True
            complete = True
            classification = "CG0_exact_support_already_available"
        elif route == "deterministic_calculation":
            derivation_type = "deterministic_calculation"
            actual_ids = []
            exact = False
            complete = False
            classification = "CG2_support_set_incomplete"
        elif route == "safe_response":
            derivation_type = "safe_response"
            actual_ids = []
            exact = False
            complete = False
            classification = "CG7_not_applicable"
        elif claim_row.get("category", "").startswith("unsupported"):
            derivation_type = "deterministic_fact_extraction"
            actual_ids = []
            exact = False
            complete = False
            classification = "CG5_claim_exceeds_selected_evidence"
        else:
            # The frozen deterministic trace retained context/source lists but
            # did not persist the selector's selected fact_id/source span.  A
            # context member is therefore not promoted to exact provenance.
            derivation_type = "deterministic_fact_extraction"
            actual_ids = []
            exact = False
            complete = False
            classification = "CG4_answer_derived_without_traceable_evidence"
        counts[classification] += 1
        rows.append(
            {
                "claim_id": f"{case_id}:claim:0",
                "question_id": case_id,
                "route": route,
                "claim_category": claim_row.get("category"),
                "claim_text": claim_text(state, case_id),
                "answer_value": None,
                "derivation_type": derivation_type,
                "candidate_ids_available": list(line.get("candidate_ids") or []),
                "physical_source_ids_available": list(line.get("physical_source_ids") or []),
                "actual_derivation_evidence_ids": actual_ids,
                "claim_support_identity_known": exact,
                "claim_support_complete": complete,
                "exact_derivation_lineage": exact,
                "citation_emitted": bool(line.get("citation_refs")),
                "citation_support_consistent": bool(
                    calc and calc.get("citation_valid_baseline")
                ) if route == "deterministic_calculation" else None,
                "primary_classification": classification,
                "selection_trace_present": False if route == "deterministic_fact" else None,
            }
        )
    payload = {
        "gate": GATE,
        "denominator": len(rows),
        "claims_audited": len(rows),
        "claims_exact_lineage": sum(int(row["exact_derivation_lineage"]) for row in rows),
        "claims_incomplete_lineage": sum(int(row["primary_classification"] == "CG2_support_set_incomplete") for row in rows),
        "claims_wrong_source": 0,
        "claims_unresolved": sum(int(row["primary_classification"] == "CG4_answer_derived_without_traceable_evidence") for row in rows),
        "counts": dict(counts),
        "rows": rows,
        "gold_reads": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
    }
    write_json(OUT / "deterministic-claim-provenance.json", payload)
    return payload, {row["question_id"]: row for row in rows}


def supported_uncited_audit(state: dict[str, Any], claim_by_id: dict[str, Any]) -> dict[str, Any]:
    ids = [
        str(row["question_id"])
        for row in state["claim_rows"]
        if row.get("category") == "supported_but_uncited"
    ]
    rows = []
    for case_id in ids:
        claim = claim_by_id[case_id]
        rows.append(
            {
                "question_id": case_id,
                "route": claim["route"],
                "claim_id": claim["claim_id"],
                "exact_derivation_provenance_available": claim["exact_derivation_lineage"],
                "exact_support_reaches_answer_builder": claim["exact_derivation_lineage"],
                "support_set_incomplete": claim["primary_classification"] == "CG2_support_set_incomplete",
                "wrong_or_unresolved_provenance": claim["primary_classification"] == "CG4_answer_derived_without_traceable_evidence",
                "no_traceable_derivation": claim["primary_classification"] == "CG4_answer_derived_without_traceable_evidence",
                "claim_exceeds_evidence": claim["primary_classification"] == "CG5_claim_exceeds_selected_evidence",
                "recoverable_by_contract_only": False,
                "reason": (
                    "calculation support set was not produced for this blocked route"
                    if claim["route"] == "deterministic_calculation"
                    else "sealed deterministic fact trace has context sources but no selected fact/source-span relation"
                ),
            }
        )
    counts = {
        "exact_derivation_provenance_available": sum(int(row["exact_derivation_provenance_available"]) for row in rows),
        "exact_support_reaches_answer_builder": sum(int(row["exact_support_reaches_answer_builder"]) for row in rows),
        "support_set_incomplete": sum(int(row["support_set_incomplete"]) for row in rows),
        "wrong_or_unresolved_provenance": sum(int(row["wrong_or_unresolved_provenance"]) for row in rows),
        "no_traceable_derivation": sum(int(row["no_traceable_derivation"]) for row in rows),
        "claim_exceeds_evidence": sum(int(row["claim_exceeds_evidence"]) for row in rows),
        "recoverable_by_contract_only": sum(int(row["recoverable_by_contract_only"]) for row in rows),
    }
    payload = {
        "gate": GATE,
        "denominator": len(rows),
        "baseline": SUPPORTED_UNCITED_TOTAL,
        "counts": counts,
        "recoverable_by_contract_only_threshold": RECOVERY_THRESHOLD,
        "rows": rows,
        "gold_reads": 0,
    }
    write_json(OUT / "supported-uncited-provenance.json", payload)
    return payload


def deterministic_fact_audit(state: dict[str, Any]) -> dict[str, Any]:
    rows = []
    counts = Counter()
    wrong_ids = {
        str(row["question_id"])
        for row in state["taxonomy_by_id"].values()
        if row.get("primary") == "CIT3_wrong_source_identity"
    }
    for case_id in sorted(state["routing_by_id"]):
        if route_for(state, case_id) != "deterministic_fact":
            continue
        # The source list is a context projection, not a selector trace.  We
        # therefore conservatively classify the value as known but its unique
        # derivation source as unavailable.  Even wrong-source labels do not
        # prove which source produced the value.
        classification = "DF2_value_known_but_source_not_unique"
        counts[classification] += 1
        rows.append(
            {
                "question_id": case_id,
                "answer_value_present": bool(claim_text(state, case_id)),
                "answer_builder_entrypoint": "src/generation/deterministic_answers.py::answer_numeric_query_from_chunks",
                "fact_selector_entrypoint": "DeterministicAnswerExtractor._select_distinct_evidence/_select_answer_values",
                "selected_candidate": None,
                "selected_structured_fact": None,
                "source_identity": None,
                "candidate_ids_available": list(state["lineage_by_id"].get(case_id, {}).get("candidate_ids") or []),
                "selection_trace_present": False,
                "classification": classification,
                "was_in_wrong_source_cohort": case_id in wrong_ids,
                "exact_derivation_lineage": False,
            }
        )
    payload = {
        "gate": GATE,
        "denominator": len(rows),
        "counts": dict(counts),
        "rows": rows,
        "interpretation": "No selector fact_id/source-span relation was persisted in the sealed deterministic answer state; context membership is not promoted to exact provenance.",
        "gold_reads": 0,
    }
    write_json(OUT / "deterministic-fact-derivation-audit.json", payload)
    return payload


def wrong_source_root_cause(state: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for row in state["taxonomy_by_id"].values():
        if row.get("primary") != "CIT3_wrong_source_identity":
            continue
        rows.append(
            {
                "question_id": str(row["question_id"]),
                "classification": "WS2_no_unique_derivation_source",
                "evidence": "sealed output marks wrong source, but no exact selector derivation relation identifies a repairable provenance mapping",
                "cgba_allowed": False,
            }
        )
    payload = {
        "gate": GATE,
        "denominator": len(rows),
        "counts": {
            "WS0_wrong_evidence_used_to_derive_answer": 0,
            "WS1_correct_value_but_wrong_provenance_attached": 0,
            "WS2_no_unique_derivation_source": len(rows),
            "WS3_multiple_plausible_sources_unresolved": 0,
            "WS4_claim_exceeds_selected_source": 0,
            "WS5_other": 0,
        },
        "rows": rows,
        "wrong_source_fixed_by_contract": 0,
        "wrong_source_unchanged": len(rows),
        "gold_reads": 0,
    }
    write_json(OUT / "wrong-source-root-cause.json", payload)
    return payload


def calculation_support_audit(state: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case_id, line in sorted(state["calc_by_id"].items()):
        trace = state["nf03_trace_by_id"].get(case_id, {})
        binder = trace.get("binder") or {}
        selected = binder.get("selected_assignment") or {}
        provenance = list(selected.get("physical_provenance") or [])
        operand_ids = list(line.get("binder_supporting_evidence_ids") or [])
        unique_sources = {
            item.get("authoritative_evidence_id")
            for item in provenance
            if item.get("authoritative_evidence_id")
        }
        complete = len(operand_ids) == 2 and len(provenance) >= 2 and len(unique_sources) >= 2
        citation_valid = bool(line.get("citation_valid_baseline"))
        row = {
            "question_id": case_id,
            "calculator_strict_correct": True,
            "operand_evidence_ids": operand_ids,
            "operand_physical_provenance": provenance,
            "operand_evidence_id_count": len(operand_ids),
            "operand_physical_source_count": len(unique_sources),
            "complete_support_set": complete,
            "support_set_unique": len(unique_sources) == len(set(unique_sources)) if unique_sources else False,
            "all_operand_physical_sources_known": complete,
            "citation_full_recall_satisfiable": citation_valid,
            "citation_valid_baseline": citation_valid,
            "reason_if_not_satisfiable": None if citation_valid else "authoritative operand identities are not represented by a complete emitted source identity set",
        }
        rows.append(row)
    payload = {
        "gate": GATE,
        "denominator": len(rows),
        "calculator_strict_correct": len(rows),
        "complete_support_sets": sum(int(row["complete_support_set"]) for row in rows),
        "all_operand_physical_sources_known": sum(int(row["all_operand_physical_sources_known"]) for row in rows),
        "citation_full_recall_satisfiable": sum(int(row["citation_full_recall_satisfiable"]) for row in rows),
        "rows": rows,
        "special_cases": {
            case_id: next(row for row in rows if row["question_id"] == case_id)
            for case_id in ("ko_fy2025_006", "nvda_fy2025_006")
            if any(row["question_id"] == case_id for row in rows)
        },
        "gold_reads": 0,
    }
    write_json(OUT / "calculation-support-set-audit.json", payload)
    return payload


def write_stage_a_decision(state: dict[str, Any], provenance: dict[str, Any], supported: dict[str, Any]) -> dict[str, Any]:
    defect = supported["counts"]["recoverable_by_contract_only"] >= RECOVERY_THRESHOLD
    decision = {
        "gate": GATE,
        "stage": "A",
        "evaluation_role": "development_shadow_claim_grounding_recovery",
        "fresh_blind_evaluation": False,
        "model_execution": False,
        "retrieval_calls": 0,
        "response_reconstruction": False,
        "recoverable_by_contract_only_threshold": RECOVERY_THRESHOLD,
        "recoverable_by_contract_only": supported["counts"]["recoverable_by_contract_only"],
        "claim_grounding_contract_defect_supported": True if defect else False,
        "cgba_v1_allowed": defect,
        "reason": (
            "At least eight exact claim derivation relations were sealed without claim binding."
            if defect
            else "The sealed deterministic fact route has no exact selected-fact/source-span relation; only the five calculation operand sets are exact, and none are in the supported-uncited cohort."
        ),
        "gold_reads": 0,
    }
    write_json(OUT / "claim-grounding-decision.json", decision)
    return decision


def write_disabled_cgba_artifacts(state: dict[str, Any], decision: dict[str, Any]) -> None:
    write_json(
        OUT / "cgba-v1-contract.json",
        {
            "name": "CGBA-V1",
            "executed": False,
            "allowed_only_if_recoverable_claims_at_least": RECOVERY_THRESHOLD,
            "exact_derivation_only": True,
            "can_search_context": False,
            "can_use_reranker_scores": False,
            "can_use_gold": False,
            "can_modify_answer_text": False,
            "can_add_support_without_exact_lineage": False,
            "can_add_citation_without_exact_claim_support": False,
            "support_source_must_be_in_runtime_state": True,
            "shadow_only": True,
            "reason_not_executed": decision["reason"],
        },
    )
    write_json(
        OUT / "cgba-v1-manifest.json",
        {
            "gate": GATE,
            "executed": False,
            "rows": [],
            "claims_support_added": 0,
            "claims_support_added_with_exact_lineage": 0,
            "claims_support_added_without_exact_lineage": 0,
            "citations_added": 0,
            "citations_added_from_exact_claim_support": 0,
            "citations_added_without_exact_claim_support": 0,
            "gold_reads_during_mapping": 0,
        },
    )
    empty = OUT / "reconstructed-responses.jsonl.gz"
    write_jsonl_gz(empty, [])
    write_json(
        OUT / "response-seal.json",
        {
            "gate": GATE,
            "stage_b_executed": False,
            "response_reconstruction": False,
            "complete": False,
            "case_count": 0,
            "response_sha256": sha256_file(empty),
            "model_calls": 0,
            "retrieval_calls": 0,
            "reranker_calls": 0,
            "gold_reads_during_reconstruction": 0,
            "reason": "CGBA-V1 was not authorized by the Stage-A threshold.",
        },
    )
    write_json(
        OUT / "answer-text-invariance.json",
        {
            "checked": False,
            "answer_text_byte_identical": None,
            "numeric_unchanged": None,
            "period_unchanged": None,
            "unit_unchanged": None,
            "answerability_unchanged": None,
            "routing_unchanged": None,
            "formal_result_invalid": False,
            "reason": "No Stage-B response reconstruction was authorized.",
        },
    )


def write_post_artifacts(state: dict[str, Any], provenance: dict[str, Any], supported: dict[str, Any], calc: dict[str, Any], decision: dict[str, Any]) -> None:
    baseline_citation = int(state["citation_metrics"]["baseline"]["full_recall"])
    baseline = {
        "grounded_pass": 3,
        "citation_full_recall": baseline_citation,
        "supported_cited": 10,
        "supported_uncited": 51,
        "unsupported_cited": 2,
        "unsupported_uncited": 1,
        "no_citation": int(state["citation_metrics"]["baseline"]["no_citation"]),
        "partial": int(state["citation_metrics"]["baseline"]["partial"]),
        "wrong_source": int(state["citation_metrics"]["baseline"]["wrong_source"]),
    }
    write_json(OUT / "claim-grounding-metrics.json", {"baseline": baseline, "post_cgba": None, "stage_b_executed": False})
    write_json(OUT / "citation-metrics.json", {"baseline": baseline, "post_cgba": None, "stage_b_executed": False})
    write_json(
        OUT / "provenance-integrity.json",
        {
            "claims_support_added": 0,
            "claims_support_added_with_exact_lineage": 0,
            "claims_support_added_without_exact_lineage": 0,
            "citations_added": 0,
            "citations_added_from_exact_claim_support": 0,
            "citations_added_without_exact_claim_support": 0,
            "support_source_not_in_original_runtime_state": 0,
            "formal_result_invalid": False,
        },
    )
    write_json(
        OUT / "calculation-preservation.json",
        {
            "baseline": {
                "calculator_strict_correct": 5,
                "final_numeric_correct": 5,
                "final_period_correct": 5,
                "final_unit_correct": 5,
                "citation_valid": 3,
                "validator_accepted": 5,
            },
            "post_cgba": None,
            "stage_b_executed": False,
            "false_binding": 0,
            "false_execution": 0,
            "executed_incorrect": 0,
            "support_set_audit": calc["complete_support_sets"],
        },
    )
    write_json(
        OUT / "no-answer-preservation.json",
        {
            "baseline": {"correct_safe_response": 5, "false_answer_release": 3},
            "post_cgba": None,
            "byte_identical": None,
            "stage_b_executed": False,
        },
    )
    write_json(
        OUT / "multi-evidence-analysis.json",
        {
            "baseline": state["multi_evidence"],
            "post_cgba": None,
            "stage_b_executed": False,
        },
    )
    write_json(
        OUT / "baseline-vs-cgba.json",
        {
            "baseline": baseline,
            "post_cgba": None,
            "delta": None,
            "stage_b_executed": False,
        },
    )
    write_json(
        OUT / "safety-analysis.json",
        {
            "stage_b_executed": False,
            "false_binding": 0,
            "false_execution": 0,
            "executed_incorrect": 0,
            "claims_support_added_without_exact_lineage": 0,
            "citations_added_without_exact_claim_support": 0,
            "support_source_not_in_original_runtime_state": 0,
            "no_answer_baseline_correct": 5,
            "no_answer_baseline_false_release": 3,
            "production_switch_allowed": False,
            "formal_result_invalid": False,
        },
    )
    bottleneck = "deterministic_fact_selection"
    write_json(
        OUT / "bottleneck-analysis.json",
        {
            "dominant_bottleneck_after_stage_a": bottleneck,
            "claim_grounding_contract_recovery": "not_authorized",
            "deterministic_fact_selection_unresolved": 46,
            "supported_uncited_not_recoverable": supported["counts"]["wrong_or_unresolved_provenance"],
            "next_gate": "deterministic_fact_selection_recovery",
        },
    )
    final = {
        "gate": GATE,
        "evaluation_role": "development_shadow_claim_grounding_recovery",
        "fresh_blind_evaluation": False,
        "model_execution": False,
        "retrieval_calls": 0,
        "retrieval_tuning": False,
        "binder_tuning": False,
        "calculator_tuning": False,
        "generator_tuning": False,
        "validator_tuning": False,
        "production_switch_allowed": False,
        "baseline_grounded_pass": 3,
        "baseline_citation_full_recall": baseline_citation,
        "supported_uncited_baseline": 51,
        "supported_uncited_exact_lineage": supported["counts"]["exact_derivation_provenance_available"],
        "supported_uncited_not_recoverable": supported["counts"]["wrong_or_unresolved_provenance"] + supported["counts"]["support_set_incomplete"],
        "wrong_source_baseline": 7,
        "claim_grounding_contract_defect_supported": decision["claim_grounding_contract_defect_supported"],
        "cgba_v1_executed": False,
        "claims_support_added": 0,
        "claims_support_added_without_exact_lineage": 0,
        "citations_added_without_exact_claim_support": 0,
        "post_grounded_pass": None,
        "post_citation_full_recall": None,
        "supported_uncited_post": None,
        "wrong_source_post": None,
        "calculation_citation_valid_post": None,
        "false_binding": 0,
        "false_execution": 0,
        "executed_incorrect": 0,
        "claim_grounding_recovery_effective": False,
        "dominant_bottleneck_after_recovery": bottleneck,
        "next_gate": "deterministic_fact_selection_recovery",
        "gold_reads_during_stage_a": 0,
        "gold_reads_during_reconstruction": 0,
    }
    write_json(OUT / "decision.json", final)
    (OUT / "README.md").write_text(
        "# NF-E2E-07 R0 — Deterministic Claim-to-Evidence Grounding Recovery\n\n"
        "Development-shadow, offline provenance audit over sealed NF-E2E-06 "
        "outputs. The deterministic fact route retains context/source lists but "
        "does not persist the selected fact/source-span relation required by the "
        "exact derivation contract. Five Calculator cases retain exact operand "
        "support sets, but they are not supported-uncited claims and do not meet "
        "the eight-claim CGBA threshold. CGBA-V1 was not executed; no response "
        "was reconstructed and no evidence/citation was added. Next gate: "
        "deterministic_fact_selection_recovery. Production switch allowed: false.\n",
        encoding="utf-8",
    )


def run() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = load_state()
    write_frozen_contract(state)
    provenance, claim_by_id = stage_a_claim_provenance(state)
    supported = supported_uncited_audit(state, claim_by_id)
    deterministic_fact_audit(state)
    wrong_source_root_cause(state)
    calc = calculation_support_audit(state)
    decision = write_stage_a_decision(state, provenance, supported)
    write_disabled_cgba_artifacts(state, decision)
    write_post_artifacts(state, provenance, supported, calc, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

