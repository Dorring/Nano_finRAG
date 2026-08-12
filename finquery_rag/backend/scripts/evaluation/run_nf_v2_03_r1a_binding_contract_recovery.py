#!/usr/bin/env python3
"""NF-V2-03 R1A: offline Binding Validator and bindability contract audit."""

from __future__ import annotations

import gzip
import hashlib
import inspect
import json
import re
import sys
import ast
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Intent  # noqa: E402
from rag_v2.evidence.binder_provider import _binding_from_payload  # noqa: E402
from rag_v2.evidence.binding_validator import validate_binding  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


BASE_COMMIT = "ead160d"
MODEL = "qwen3.7-plus"
FORMAL = ROOT / "artifacts/evaluation/nf-v2-03-formal-attempt-3"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1a-binding-contract-recovery"
QUESTION_TOTAL = 72
DIRECT_TOTAL = 56
HISTORICAL_TOTAL = 46

BV_NAMES = {
    "BV0_PASS",
    "BV1_unknown_slot_id",
    "BV2_unknown_fact_id",
    "BV3_fact_outside_query_packet",
    "BV4_fact_not_provenance_complete",
    "BV5_cross_query_fact",
    "BV6_duplicate_invalid_binding",
    "BV7_invalid_status_binding_consistency",
    "BV8_missing_slot_contract_violation",
    "BV9_ambiguous_slot_contract_violation",
    "BV10_cardinality_violation",
    "BV11_invalid_candidate_source_relation",
    "BV12_semantic_metric_check_in_validator",
    "BV13_semantic_period_check_in_validator",
    "BV14_semantic_role_check_in_validator",
    "BV15_other",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm_text(value: Any) -> str:
    text = str(value or "").casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def norm_period(value: Any) -> str:
    return norm_text(value).replace("fy ", "fy")


def load_inputs() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seal = read_json(FORMAL / "binder-prediction-seal.json")
    prediction_path = FORMAL / "binder-predictions.jsonl.gz"
    if not seal.get("sealed") or seal.get("gold_reads_before_prediction_seal") != 0:
        raise RuntimeError("Attempt-3 prediction seal is not valid")
    if sha256_file(prediction_path) != seal.get("prediction_sha256"):
        raise RuntimeError("Attempt-3 prediction SHA mismatch")
    frozen = legacy.load_frozen_inputs()
    predictions = read_jsonl_gz(prediction_path)
    if len(predictions) != QUESTION_TOTAL:
        raise RuntimeError("Attempt-3 prediction count mismatch")
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in legacy.LABELS.read_text(encoding="utf-8").splitlines()) if row}
    return frozen, predictions, labels


def classify_reason(reason: str) -> str:
    if reason.startswith(("unknown_slot:", "unknown_missing_slot", "unknown_ambiguous_slot")):
        return "BV1_unknown_slot_id"
    if reason.startswith("unknown_fact:"):
        return "BV2_unknown_fact_id"
    if reason.startswith("incomplete_provenance:"):
        return "BV4_fact_not_provenance_complete"
    if reason in {"duplicate_fact_across_slots"}:
        return "BV6_duplicate_invalid_binding"
    if reason.startswith("empty_fact_binding:") or reason in {"bound_has_error_fields", "invalid_without_reasons"}:
        return "BV7_invalid_status_binding_consistency"
    if reason == "missing_without_slots":
        return "BV8_missing_slot_contract_violation"
    if reason == "ambiguous_without_slots":
        return "BV9_ambiguous_slot_contract_violation"
    if reason in {"bound_slot_cardinality_mismatch", "bound_fact_cardinality_mismatch"}:
        return "BV10_cardinality_violation"
    if "candidate" in reason and "source" in reason:
        return "BV11_invalid_candidate_source_relation"
    return "BV15_other"


def raw_slot_status(binding: Mapping[str, Any], allowed_slots: set[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for slot_id in binding.get("slot_bindings", {}):
        if slot_id not in allowed_slots:
            status = "INVALID"
        elif slot_id in set(binding.get("ambiguous_slots", [])):
            status = "AMBIGUOUS"
        elif slot_id in set(binding.get("missing_slots", [])):
            status = "MISSING"
        else:
            status = "BOUND"
        counts[status] += 1
    for slot_id in binding.get("missing_slots", []):
        counts["MISSING"] += 1
    for slot_id in binding.get("ambiguous_slots", []):
        counts["AMBIGUOUS"] += 1
    return counts


def validator_audit(frozen: dict[str, Any], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    query_status: Counter[str] = Counter()
    query_final_status: Counter[str] = Counter()
    slot_status: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    query_reason_counts: Counter[str] = Counter()
    status_matrix: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for row in predictions:
        question_id = str(row["question_id"])
        binding_payload = row["binding"]
        binding = _binding_from_payload(binding_payload)
        request = frozen["requests"][question_id]
        validation = validate_binding(binding, request.plan, request.facts)
        raw_status = str(binding.status)
        query_status[raw_status] += 1
        query_final_status[validation.final_status] += 1
        allowed_slots = {slot.slot_id for slot in request.plan.required_slots}
        slot_status.update(raw_slot_status(binding_payload, allowed_slots))
        mapped = [classify_reason(reason) for reason in validation.reasons]
        if validation.passed:
            query_reason_counts["BV0_PASS"] += 1
        else:
            for category in mapped:
                reason_counts[category] += 1
            query_reason_counts[mapped[0] if mapped else "BV15_other"] += 1
        status_matrix.setdefault(raw_status, {"schema_valid": 0, "validator_pass": 0, "validator_fail": 0})
        status_matrix[raw_status]["schema_valid"] += int(bool(row.get("binding_schema_valid")))
        status_matrix[raw_status]["validator_pass"] += int(validation.passed)
        status_matrix[raw_status]["validator_fail"] += int(not validation.passed)
        rows.append({
            "question_id": question_id,
            "raw_status": raw_status,
            "final_status": validation.final_status,
            "validator_pass": validation.passed,
            "reasons": list(validation.reasons),
            "categories": mapped,
        })
    for name in BV_NAMES:
        reason_counts.setdefault(name, 0)
        query_reason_counts.setdefault(name, 0)
    source_text = inspect.getsource(validate_binding)
    parsed = ast.parse(source_text)
    function = parsed.body[0]
    if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) and function.body and isinstance(function.body[0], ast.Expr) and isinstance(getattr(function.body[0], "value", None), ast.Constant) and isinstance(function.body[0].value.value, str):
        function.body = function.body[1:]
    executable_source = ast.unparse(function)
    semantic_tokens = ("normalized_metric", "metric equality", "period equality", "expected source", "role equality", "Gold")
    semantic_leakage = any(token.casefold() in executable_source.casefold() for token in semantic_tokens)
    return {
        "query_status": dict(sorted(query_status.items())),
        "query_final_status": dict(sorted(query_final_status.items())),
        "slot_status": dict(sorted(slot_status.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "query_reason_counts": dict(sorted(query_reason_counts.items())),
        "status_matrix": status_matrix,
        "rows": rows,
        "validator_semantic_leakage": semantic_leakage,
        "validator_source_tokens_found": [token for token in semantic_tokens if token.casefold() in executable_source.casefold()],
        "before_validator_pass": read_json(FORMAL / "binding-validator-results.json").get("passed"),
        "after_validator_pass": sum(int(row["validator_pass"]) for row in rows),
    }


def expected_sources(slot: Any, label: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    sources = list(label.get("expected_sources") or [])
    matching = [source for source in sources if norm_period(source.get("period")) == norm_period(slot.period) and norm_text(source.get("row_label")) == norm_text(slot.metric)]
    return matching or [source for source in sources if norm_period(source.get("period")) == norm_period(slot.period)]


def fact_candidate_ids(fact: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}


def funnel_row(question_id: str, request: Any, label: Mapping[str, Any], top20: list[str]) -> dict[str, Any]:
    facts = list(request.facts)
    slots = list(request.plan.required_slots)
    sources = [source for slot in slots for source in expected_sources(slot, label)]
    source_keys = {str(source.get("candidate_key")) for source in sources if source.get("candidate_key")}
    d1 = bool(facts)
    d2 = bool(source_keys & {str(item) for item in top20})
    source_facts = [fact for fact in facts if fact_candidate_ids(fact) & source_keys]
    d3 = bool(source_facts)
    metric_facts = [fact for fact in source_facts if any(legacy.metric_matches(fact, slot, source) for slot in slots for source in sources)]
    d4 = bool(metric_facts)
    period_facts = [fact for fact in metric_facts if any(legacy.period_matches(fact, slot, source) for slot in slots for source in sources)]
    d5 = bool(period_facts)
    metric_rows = read_json(legacy.METRIC_REVIEW).get("metric_matches", [])
    metric_contract = {(str(row["question_id"]), int(row["slot_index"])): bool(row.get("match", {}).get("matched")) for row in metric_rows}
    d6 = all(any(legacy.strict_fact_for_slot(fact, slot, label, metric_contract_ok=metric_contract.get((question_id, index), False)) for fact in facts) for index, slot in enumerate(slots)) if slots else False
    stage = "D6_strict_bindable" if d6 else next((name for name, value in (("D1_provenance_supply", d1), ("D2_gold_source_admitted", d2), ("D3_gold_source_fact", d3), ("D4_metric_compatible", d4), ("D5_period_compatible", d5)) if not value), "D5_period_compatible")
    return {
        "question_id": question_id,
        "fact_count": len(facts),
        "gold_source_keys": sorted(source_keys),
        "d1_provenance_supply": d1,
        "d2_gold_source_admitted": d2,
        "d3_gold_source_fact": d3,
        "d4_metric_compatible": d4,
        "d5_period_compatible": d5,
        "d6_strict_bindable": d6,
        "first_loss_stage": None if d6 else stage,
        "source_fact_count": len(source_facts),
        "metric_fact_count": len(metric_facts),
        "period_fact_count": len(period_facts),
    }


def funnel_report(frozen: dict[str, Any], labels: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    rows = [funnel_row(question_id, frozen["requests"][question_id], labels[question_id], frozen["top20_order"].get(question_id, [])) for question_id in ids]
    return {
        "denominator": len(rows),
        "D0_total": len(rows),
        "D1_provenance_supply": sum(int(row["d1_provenance_supply"]) for row in rows),
        "D2_gold_source_admitted": sum(int(row["d2_gold_source_admitted"]) for row in rows),
        "D3_gold_source_fact": sum(int(row["d3_gold_source_fact"]) for row in rows),
        "D4_metric_compatible": sum(int(row["d4_metric_compatible"]) for row in rows),
        "D5_period_compatible": sum(int(row["d5_period_compatible"]) for row in rows),
        "D6_strict_bindable": sum(int(row["d6_strict_bindable"]) for row in rows),
        "first_loss_counts": dict(sorted(Counter(row["first_loss_stage"] for row in rows if row["first_loss_stage"]).items())),
        "rows": rows,
    }


def source_identity_audit(frozen: dict[str, Any], labels: dict[str, Any]) -> dict[str, Any]:
    compared = 0
    candidate_matches = 0
    physical_source_matches = 0
    legacy_evidence_ids = 0
    examples: list[dict[str, Any]] = []
    for question_id, request in frozen["requests"].items():
        label = labels[question_id]
        for source in label.get("expected_sources") or []:
            key = str(source.get("candidate_key") or "")
            if not key:
                continue
            compared += 1
            fact_match = [fact for fact in request.facts if key in fact_candidate_ids(fact)]
            candidate_matches += int(bool(fact_match))
            physical_source_matches += sum(int(str(fact.get("physical_source_id")) == key) for fact in fact_match)
            legacy_evidence_ids += int(bool(source.get("evidence_id")))
            if len(examples) < 10:
                examples.append({"question_id": question_id, "gold_candidate_key": key, "fact_candidate_ids": sorted({candidate for fact in fact_match for candidate in fact_candidate_ids(fact)}), "fact_physical_source_ids": sorted({str(fact.get("physical_source_id")) for fact in fact_match})})
    source_code = inspect.getsource(legacy.source_matches)
    # The frozen source contract is candidate_key -> candidate_id/candidate_ids.
    # physical_source_id is a separate lineage identifier and must not be used
    # as the Gold candidate namespace.
    mismatch = "physical_source_id" in source_code or "evidence_id" in source_code
    return {
        "compared_expected_sources": compared,
        "candidate_key_to_fact_candidate_id_matches": candidate_matches,
        "candidate_key_to_physical_source_id_matches": physical_source_matches,
        "legacy_evidence_id_present": legacy_evidence_ids,
        "source_identity_namespace_mismatch": mismatch,
        "canonical_contract": "Gold candidate_key -> FinancialFact candidate_id/candidate_ids; physical_source_id remains separate lineage identity",
        "evaluator_source_match_code": source_code,
        "examples": examples,
    }


def main() -> int:
    frozen, predictions, labels = load_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    audit = validator_audit(frozen, predictions)
    direct_ids = sorted(question_id for question_id, item in frozen["plans"].items() if item["plan"].intent is Intent.DIRECT_FACT)
    historical_ids = [row["question_id"] for row in sorted(read_json(legacy.NF09 / "query-level-coverage.json").get("rows", []), key=lambda row: row["question_id"]) if row["question_id"] in frozen["requests"]][:HISTORICAL_TOTAL]
    direct = funnel_report(frozen, labels, direct_ids)
    historical = funnel_report(frozen, labels, historical_ids)
    source_audit = source_identity_audit(frozen, labels)
    write_json(OUT / "validator-breakdown.json", audit)
    write_json(OUT / "validator-responsibility-audit.json", {"validator_semantic_leakage": audit["validator_semantic_leakage"], "semantic_checks_removed": [], "structural_checks_preserved": ["slot", "fact", "packet", "provenance", "status", "cardinality"], "validator_source": inspect.getsource(validate_binding)})
    write_json(OUT / "binding-status-audit.json", {"query_level_raw": audit["query_status"], "query_level_final": audit["query_final_status"], "slot_level_raw": audit["slot_status"], "schema_valid_plus_validator": audit["status_matrix"]})
    write_json(OUT / "direct-bindability-funnel.json", direct)
    write_json(OUT / "historical-46-funnel.json", historical)
    write_json(OUT / "source-identity-audit.json", source_audit)
    write_json(OUT / "offline-attempt-3-replay.json", {"prediction_sha256_verified": True, "structural_validator_before": 10, "structural_validator_after": audit["after_validator_pass"], "direct_strict_bindable_before": 4, "direct_strict_bindable_after": direct["D6_strict_bindable"], "historical_strict_bindable_after": historical["D6_strict_bindable"], "semantic_scoring_reinterpreted": False, "model_calls": 0})
    recovered = not audit["validator_semantic_leakage"] and not source_audit["source_identity_namespace_mismatch"] and not audit["query_reason_counts"].get("BV15_other")
    decision = {"gate": "NF-V2-03-R1A", "base_commit": BASE_COMMIT, "binder_model": MODEL, "model_calls": 0, "prediction_sha256_verified": True, "validator_semantic_leakage": audit["validator_semantic_leakage"], "source_identity_namespace_mismatch": source_audit["source_identity_namespace_mismatch"], "binding_contract_recovered": recovered, "formal_attempt_4": "eligible_not_executed", "next_gate": "nf_v2_03_formal_attempt_4" if recovered else "nf_v2_03_binding_contract_failure_review", "production_default": "V1", "production_switch_allowed": False}
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("# NF-V2-03 R1A\n\nOffline audit only; no Qwen call was made. Attempt-3 predictions were SHA-verified before post-seal label attribution. Structural validator responsibility and the source-identity namespace were audited separately from strict semantic bindability.\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
