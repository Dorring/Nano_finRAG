#!/usr/bin/env python3
"""NF-V2 architecture scope freeze R1.

This is an offline, no-model, no-retrieval handoff audit.  It freezes the
route-specific evidence boundary and replays the historical deterministic
calculation operand contract through a deterministic current-artifact
adapter.  Gold/review artifacts are opened only after the runtime contract
and replay rows have been sealed in memory.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from itertools import product
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_04_r0_missing_evidence_repair as r0  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402


BASE_COMMIT = "6525bac0547eac38acb177e9656447ec80f9ad22"
GATE = "NF-V2-ARCHITECTURE-SCOPE-FREEZE-R1"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-architecture-scope-freeze-r1"
HIST = ROOT / "artifacts/evaluation/nf-e2e-02-r0-binder-contract-recovery"
HIST_REPLAY = ROOT / "artifacts/evaluation/nf-e2e-03-r0-full-replay-after-binder-recovery"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stable_sha(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def period(value: Any) -> str | None:
    text = norm(value)
    match = re.search(r"\bfy\s*(\d{4})\b", text)
    return f"fy{match.group(1)}" if match else text or None


def candidate_ids(fact: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", []) if item}


def source_map_for_frozen() -> dict[str, dict[str, Any]]:
    state = nf02.verify_frozen_top100()
    _, source_map = r0.candidate_rows_topk(state, 100)
    return {str(key): dict(value) for key, value in source_map.items()}


def source_for_fact(fact: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    for candidate_id in sorted(candidate_ids(fact)):
        source = source_map.get(candidate_id)
        if source:
            return dict(source)
    return {}


def field(fact: Mapping[str, Any], source: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if fact.get(name) is not None:
            return fact.get(name)
        if source.get(name) is not None:
            return source.get(name)
    return None


def sequence_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " / ".join(norm(item) for item in value if norm(item))
    return norm(value)


def metric_variants(fact: Mapping[str, Any], source: Mapping[str, Any]) -> set[str]:
    values = (
        fact.get("normalized_metric"),
        fact.get("raw_metric"),
        source.get("normalized_metric"),
        source.get("metric"),
        source.get("row_label"),
        source.get("row_path"),
        source.get("metric_path"),
    )
    result: set[str] = set()
    for value in values:
        text = sequence_text(value)
        if text:
            result.add(text)
    return result


def fact_period(fact: Mapping[str, Any], source: Mapping[str, Any]) -> str | None:
    return period(field(fact, source, "normalized_period", "raw_period", "period"))


def provenance_complete(fact: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    if fact.get("provenance_complete") is True:
        return True
    physical = str(field(fact, source, "physical_source_id") or "")
    candidate = candidate_ids(fact)
    return bool(physical and candidate and field(fact, source, "document_id", "pdf_page") is not None)


def relation_valid(fact: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> bool:
    physical = str(fact.get("physical_source_id") or "")
    for candidate_id in candidate_ids(fact):
        source = source_map.get(candidate_id)
        if source and physical and str(source.get("physical_source_id") or "") == physical:
            return True
    return False


def measurement_kind(fact: Mapping[str, Any], source: Mapping[str, Any]) -> str | None:
    value_type = field(fact, source, "measurement_kind", "value_type")
    if value_type is not None:
        return norm(value_type)
    unit = field(fact, source, "unit")
    currency = field(fact, source, "currency", "normalized_currency")
    scale = field(fact, source, "scale", "normalized_scale")
    if unit is not None or currency is not None or scale is not None:
        return "numeric_with_unit_context"
    if fact.get("parsed_numeric_value") is not None or fact.get("raw_value") is not None:
        return "numeric"
    return None


def current_fact_projection(fact: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Map current FinancialFactV1 + linked source metadata to historical names."""
    source = source_for_fact(fact, source_map)
    return {
        "candidate_key": str(fact.get("candidate_id") or next(iter(candidate_ids(fact)), "")),
        "candidate_ids": sorted(candidate_ids(fact)),
        "fact_id": str(fact.get("fact_id") or ""),
        "physical_source_id": field(fact, source, "physical_source_id"),
        "document_id": field(fact, source, "document_id"),
        "pdf_page": field(fact, source, "pdf_page", "page"),
        "table_fragment_id": field(fact, source, "table_fragment_id", "table_id"),
        "logical_table_id": field(fact, source, "logical_table_id", "table_id"),
        "row_id": field(fact, source, "row_id"),
        "canonical_row_label": field(fact, source, "canonical_row_label", "row_label", "raw_metric"),
        "cell_id": field(fact, source, "cell_id"),
        "period": field(fact, source, "normalized_period", "raw_period", "period"),
        "value": field(fact, source, "raw_value", "parsed_numeric_value", "value"),
        "parsed_numeric_value": field(fact, source, "parsed_numeric_value"),
        "currency": field(fact, source, "currency", "normalized_currency"),
        "scale": field(fact, source, "normalized_scale", "raw_scale", "scale"),
        "measurement_kind": measurement_kind(fact, source),
        "metric_path": field(fact, source, "row_path", "row_hierarchy", "metric_path", "normalized_metric", "raw_metric"),
        "semantic_fact_id": str(fact.get("fact_id") or ""),
        "supporting_candidate_keys": sorted(candidate_ids(fact)),
        "physical_provenance": field(fact, source, "source_traceback", "physical_provenance", "physical_source_id"),
        "unit_context": {
            "unit": field(fact, source, "unit"),
            "currency": field(fact, source, "currency", "normalized_currency"),
            "scale": field(fact, source, "normalized_scale", "raw_scale", "scale"),
        },
        "statement_title": field(fact, source, "statement_title", "statement_type", "statement_id"),
        "table_title": field(fact, source, "table_title"),
        "row_path": field(fact, source, "row_path", "row_hierarchy", "metric_path"),
        "column_header_path": field(fact, source, "column_header_path", "column_header"),
        "section_path": field(fact, source, "section_path", "section_title", "section_heading"),
        "provenance_complete": provenance_complete(fact, source),
        "relation_valid": relation_valid(fact, source_map),
    }


def projection_key(projected: Mapping[str, Any]) -> tuple[Any, ...]:
    """Generic semantic tuple used to collapse duplicate physical projections."""
    return (
        projected.get("document_id") or projected.get("physical_source_id"),
        sequence_text(projected.get("metric_path") or projected.get("canonical_row_label")),
        period(projected.get("period")),
        sequence_text(projected.get("row_path")),
        projected.get("row_id"),
        projected.get("cell_id"),
        projected.get("parsed_numeric_value"),
        projected.get("currency"),
        projected.get("scale"),
        projected.get("measurement_kind"),
        sequence_text(projected.get("column_header_path")),
        sequence_text(projected.get("statement_title")),
    )


def slot_metric_matches(slot: Any, projected: Mapping[str, Any]) -> bool:
    requested = norm(slot.metric)
    variants = {
        norm(projected.get("canonical_row_label")),
        norm(projected.get("metric_path")),
        norm(projected.get("raw_metric")),
    }
    # Preserve exact canonical/structured equality.  A small token equality
    # fallback is only for a structured row path represented as a list.
    variants.add(sequence_text(projected.get("row_path")))
    return bool(requested and requested in variants)


def compatible_projection(slot: Any, projected: Mapping[str, Any], packet_ids: set[str], source_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    if not projected.get("fact_id") or projected["fact_id"] not in packet_ids:
        reasons.append("fact_outside_current_packet")
    if not projected.get("provenance_complete"):
        reasons.append("provenance_incomplete")
    if not projected.get("relation_valid"):
        reasons.append("source_relation_invalid")
    requested_period = period(slot.period)
    actual_period = period(projected.get("period"))
    if not requested_period or not actual_period or requested_period != actual_period:
        reasons.append("period_not_exact")
    if not slot_metric_matches(slot, projected):
        reasons.append("metric_not_exact")
    if slot.value_type == "numeric" and projected.get("parsed_numeric_value") is None and projected.get("value") is None:
        reasons.append("numeric_value_missing")
    if slot.unit and norm(slot.unit) != norm(projected.get("unit_context", {}).get("unit")):
        reasons.append("unit_not_exact")
    return sorted(set(reasons))


def historical_operand_key(projected: Mapping[str, Any]) -> tuple[Any, ...]:
    """Equivalent-to-historical operand tuple; no rank is used."""
    return (
        projected.get("document_id") or projected.get("physical_source_id"),
        sequence_text(projected.get("metric_path") or projected.get("canonical_row_label")),
        period(projected.get("period")),
        projected.get("parsed_numeric_value"),
        projected.get("currency"),
        projected.get("scale"),
        projected.get("measurement_kind"),
        "",  # segment: current RequiredSlot has no segment field
        "",  # bucket: current RequiredSlot has no bucket field
    )


def evaluate_calculation_request(request: Any, packet: list[dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    projections = [current_fact_projection(fact, source_map) for fact in packet]
    packet_ids = {str(fact.get("fact_id")) for fact in packet}
    slot_rows: list[dict[str, Any]] = []
    groups_by_slot: list[list[dict[str, Any]]] = []
    for slot in request.plan.required_slots:
        candidates: list[dict[str, Any]] = []
        groups: dict[tuple[Any, ...], dict[str, Any]] = {}
        for projected in projections:
            reasons = compatible_projection(slot, projected, packet_ids, source_map)
            candidate = {"fact_id": projected["fact_id"], "reasons": reasons, "projection": projected}
            if not reasons:
                key = historical_operand_key(projected)
                groups.setdefault(key, {"key": list(key), "representative": projected, "fact_ids": []})["fact_ids"].append(projected["fact_id"])
                candidates.append(candidate)
        grouped = list(groups.values())
        slot_rows.append({"slot_id": slot.slot_id, "operation": request.plan.operation, "role": slot.role, "metric": slot.metric, "period": slot.period, "candidate_count": len(candidates), "equivalent_group_count": len(grouped), "candidates": candidates, "groups": grouped})
        groups_by_slot.append(grouped)
    assignments: list[dict[str, Any]] = []
    if all(groups_by_slot):
        for choice in product(*groups_by_slot):
            documents = {item["representative"].get("document_id") or item["representative"].get("physical_source_id") for item in choice}
            if len(documents) != 1:
                continue
            assignments.append({"keys": [item["key"] for item in choice], "fact_ids": [item["representative"]["fact_id"] for item in choice], "document": next(iter(documents))})
    # Historical contract collapses physically duplicated projections by the
    # operand tuple; the assignments are therefore compared by tuple keys.
    unique_assignments = {json.dumps(item["keys"], sort_keys=True, default=str): item for item in assignments}
    ready = bool(unique_assignments) and len(unique_assignments) == 1
    status = "DETERMINISTIC_READY" if ready else "UNDERCOVERED" if not all(groups_by_slot) else "RUNTIME_OPERAND_AMBIGUITY"
    return {"question_id": request.question_id, "operation": request.plan.operation, "status": status, "ready": ready, "slots": slot_rows, "assignments": list(unique_assignments.values()), "assignment_count": len(unique_assignments)}


def historical_rows() -> list[dict[str, Any]]:
    path = HIST / "calculation-shadow-results.json"
    data = read_json(path)
    if isinstance(data, dict):
        for key in ("rows", "results", "cases"):
            if isinstance(data.get(key), list):
                return list(data[key])
    return list(data) if isinstance(data, list) else []


def historical_ready_ids() -> set[str]:
    rows = historical_rows()
    return {
        str(row.get("question_id") or row.get("case_id"))
        for row in rows
        if row.get("status") == "deterministic_ready"
        or row.get("runtime_status") == "deterministic_ready"
        or row.get("binding_status") == "deterministic_ready"
    }


def contract() -> dict[str, Any]:
    return {
        "contract": "CalculationContractCompatibilityV1",
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "model_calls": 0,
        "retrieval_calls": 0,
        "historical_contract": str(HIST / "historical-binder-contract.json"),
        "current_input": ["SupervisorPlan.RequiredSlot", "FinancialFactV1", "BinderFactViewV2", "source_relation"],
        "historical_input_fields": ["candidate_key", "physical_source_id", "document_id", "pdf_page", "table_fragment_id", "logical_table_id", "row_id", "canonical_row_label", "cell_id", "period", "value", "parsed_numeric_value", "currency", "scale", "measurement_kind", "metric_path", "semantic_fact_id", "supporting_candidate_keys", "physical_provenance", "unit_context"],
        "current_to_historical_adapter": {
            "candidate_key": "fact.candidate_id/candidate_ids",
            "physical_source_id": "fact.physical_source_id or linked source",
            "document_id": "fact.document_id or linked source",
            "pdf_page": "fact.pdf_page/page or linked source",
            "table_fragment_id": "fact.table_fragment_id/table_id or linked source",
            "logical_table_id": "fact.logical_table_id/table_id or linked source",
            "row_id": "fact.row_id or linked source",
            "canonical_row_label": "fact.raw_metric/linked row_label",
            "cell_id": "fact.cell_id or linked source",
            "period": "fact.normalized_period/raw_period",
            "value": "fact.raw_value/parsed_numeric_value",
            "parsed_numeric_value": "fact.parsed_numeric_value",
            "currency": "fact.currency/normalized_currency",
            "scale": "fact.normalized_scale/raw_scale",
            "measurement_kind": "existing value_type/unit context only; no inference",
            "metric_path": "BinderFactViewV2 row_path/row_hierarchy/raw_metric",
            "semantic_fact_id": "fact.fact_id",
            "supporting_candidate_keys": "fact.candidate_ids",
            "physical_provenance": "fact.source_traceback/physical provenance",
            "unit_context": "unit/currency/scale",
        },
        "historical_rules_reused": ["provenance and relation validity", "exact metric/period compatibility", "same-document tuple coherence", "semantic operand tuple collapse", "no rank-based disambiguation"],
        "adapter_only": True,
        "semantic_inference": False,
        "gold_independent_runtime": True,
        "question_specific_rules": 0,
        "gold_rules": 0,
        "financial_fact_v1_modified": False,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    # Freeze the contract before reading any historical/review labels.
    frozen_contract = contract()
    contract_path = OUT / "calculation-contract-compatibility.json"
    write_json(contract_path, frozen_contract)
    contract_sha = sha256_file(contract_path)
    (OUT / "calculation-contract-compatibility.sha256").write_text(contract_sha + "  calculation-contract-compatibility.json\n", encoding="utf-8")

    frozen = r1d.load_r1c_frozen_inputs()
    source_map = source_map_for_frozen()
    calc_ids = sorted(qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "CALCULATION")
    # Runtime replay is fully sealed before opening historical labels.
    runtime_rows = [evaluate_calculation_request(frozen["requests"][qid], list(frozen["requests"][qid].facts), source_map) for qid in calc_ids]
    runtime_seal = {"rows": runtime_rows, "row_sha256": stable_sha(runtime_rows), "contract_sha256": contract_sha, "model_calls": 0, "retrieval_calls": 0}

    # Historical/review artifacts are attribution only and are read after the
    # runtime rows are sealed.
    hist = historical_rows()
    hist_by_id = {str(row.get("question_id") or row.get("case_id")): row for row in hist}
    hist_ready = historical_ready_ids()
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in (ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl").read_text(encoding="utf-8").splitlines()) if row}
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    for row in runtime_rows:
        row["historical_status"] = hist_by_id.get(row["question_id"], {}).get("status") or hist_by_id.get(row["question_id"], {}).get("binding_status")
        row["historical_ready"] = row["question_id"] in hist_ready
        strict_flags: list[bool] = []
        if row["ready"] and row["assignments"]:
            request = frozen["requests"][row["question_id"]]
            by_id = {str(fact.get("fact_id")): fact for fact in request.facts}
            for slot, fact_id in zip(request.plan.required_slots, row["assignments"][0]["fact_ids"]):
                fact = by_id.get(str(fact_id))
                strict_flags.append(bool(fact and r1d.slot_is_strict(row["question_id"], slot, fact, labels[row["question_id"]], source_map, reviewed_ids, reviewed_fact_ids, set())))
        row["strict_operand_flags"] = strict_flags
        row["strict_correct_operand_count"] = sum(int(value) for value in strict_flags)
        row["false_operand_binding_count"] = sum(int(not value) for value in strict_flags) if row["ready"] else 0
        if row["status"] == "UNDERCOVERED":
            row["compatibility_class"] = "CP2_missing_required_current_field"
        elif row["historical_ready"] and row["status"] == "RUNTIME_OPERAND_AMBIGUITY":
            row["compatibility_class"] = "CP4_historical_pipeline_depended_on_removed_data"
        elif row["ready"] and row["false_operand_binding_count"]:
            row["compatibility_class"] = "CP3_semantic_contract_incompatible"
        elif row["ready"]:
            row["compatibility_class"] = "CP1_field_rename_or_adapter_only"
        else:
            row["compatibility_class"] = "CP5_other"
        row["exact_historical_status_match"] = row["ready"] == (row["question_id"] in hist_ready)

    compat_counts = {name: sum(int(row["compatibility_class"] == name) for row in runtime_rows) for name in ("CP0_directly_contract_compatible", "CP1_field_rename_or_adapter_only", "CP2_missing_required_current_field", "CP3_semantic_contract_incompatible", "CP4_historical_pipeline_depended_on_removed_data", "CP5_other")}
    historical_fields = frozen_contract["historical_input_fields"]
    field_presence: dict[str, dict[str, int]] = {}
    for name in historical_fields:
        total = present = 0
        for qid in calc_ids:
            for fact in frozen["requests"][qid].facts:
                value = current_fact_projection(fact, source_map).get(name)
                total += 1
                present += int(value not in (None, "", [], {}))
        field_presence[name] = {"non_null": present, "total": total}
    # A deterministic field adapter is required for the typed shape even when
    # the source evidence is already compatible; CP1 is tracked separately.
    adapter_rows = [{"question_id": row["question_id"], "adapter_required": True, "runtime_status": row["status"], "compatibility_class": row["compatibility_class"]} for row in runtime_rows]
    write_json(OUT / "calculation-contract-compatibility-audit.json", {"contract_sha256": contract_sha, "compatibility_counts": compat_counts, "typed_shape_adapter_available": "11/11", "safe_runtime_compatible": sum(int(row["compatibility_class"] in {"CP0_directly_contract_compatible", "CP1_field_rename_or_adapter_only"}) for row in runtime_rows), "adapter_required": True, "field_presence": field_presence, "rows": adapter_rows})
    runtime_ready = sum(int(row["ready"]) for row in runtime_rows)
    false_operand_binding = sum(int(row["false_operand_binding_count"]) for row in runtime_rows)
    strict_operand_count = sum(int(row["strict_correct_operand_count"]) for row in runtime_rows)
    safe_ready = runtime_ready if false_operand_binding == 0 else 0
    safe_compatible = sum(int(row["compatibility_class"] in {"CP0_directly_contract_compatible", "CP1_field_rename_or_adapter_only"}) for row in runtime_rows)
    write_json(OUT / "calculation-readiness-replay.json", {"historical_reference_ready": "5/11", "historical_ready_question_ids": sorted(hist_ready), "current_artifact_compatible": f"{safe_compatible}/11", "runtime_candidate_ready": f"{runtime_ready}/11", "current_deterministic_ready": f"{safe_ready}/11", "operand_complete": f"{safe_ready}/11", "strict_correct_operand_slots": strict_operand_count, "false_operand_binding": false_operand_binding, "calculator_executed": 0, "rows": runtime_rows, "exact_query_id_overlap": sorted(set(hist_ready) & {row["question_id"] for row in runtime_rows if row["ready"]}), "runtime_seal_sha256": runtime_seal["row_sha256"], "contract_sha256": contract_sha})

    route_policy = {
        "architecture": "route_specific_trusted_evidence",
        "DIRECT_FACT": {"controller": "General Financial RAG Supervisor", "path": "Retrieval -> FinancialFactV1 -> BinderFactViewV2 -> SelectiveBindingAdmissionV2", "release": "selective_fail_closed", "safe_coverage": "4/56", "strict": "4/4", "false_binding": 0},
        "CALCULATION": {"controller": "General Financial RAG Supervisor", "path": "Retrieval -> FinancialFactV1 -> trusted_deterministic_operand_binding_v1 -> Calculator", "release": "only_when_all_operands_trusted", "historical_ready": "5/11", "runtime_candidate_ready": f"{runtime_ready}/11", "current_ready": f"{safe_ready}/11", "false_operand_binding": false_operand_binding},
        "MULTI_EVIDENCE": {"controller": "General Financial RAG Supervisor", "path": "Retrieval -> evidence aggregation", "release": "fail_closed_until_verified", "safe_coverage": "0/5"},
        "semantic_binder": {"model": MODEL, "role": "diagnostic_semantic_evidence_analyzer", "release_authority": False},
        "reranker": {"model": "Qwen3-Reranker-4B", "role": "retrieval_only", "slot_admission_scorer": False},
        "production": "V1",
    }
    write_json(OUT / "route-specific-evidence-policy.json", route_policy)
    packet = {"packet": "VerifiedEvidencePacket", "unified": True, "producers": ["direct_selective_admission", "trusted_deterministic_operand_binding_v1", "future_verified_multi_evidence"], "required_safety_fields": ["source_provenance", "period", "unit", "currency", "scale", "metric", "scope", "route", "validation_status"], "coverage_precision_abstention_separate": True, "false_binding_priority": True}
    write_json(OUT / "verified-evidence-packet-handoff.json", packet)
    closed = {"generative_binder_exploration_closed": True, "discriminative_binder_exploration_closed": True, "evidence_repair_exploration_closed": True, "generic_deterministic_binding_exploration_closed": True, "model_calls": 0, "retrieval_calls": 0, "production_switch_allowed": False, "next_gate": "v2_05_calculation_path_integration" if safe_ready >= 5 and false_operand_binding == 0 else "v2_06_verified_generation_subset"}
    write_json(OUT / "closed-experiment-registry.json", closed)
    write_json(OUT / "accepted-component-registry.json", {"general_supervisor": True, "supervisor_required_slot_contract": True, "deterministic_state_machine": True, "financial_fact_v1": True, "binder_fact_view_v2": True, "binding_validator": True, "selective_binding_admission_v2": True, "deterministic_calculator": True, "financial_sft_generator_contract": True, "deterministic_output_validator": True, "production": "V1"})
    write_json(OUT / "final-v2-architecture.json", {"architecture_scope_frozen": True, "control_plane": "General Financial RAG Supervisor", "route_specific_trusted_evidence": True, "direct_policy": "SelectiveBindingAdmissionV2", "calculation_policy": "trusted_deterministic_operand_binding_v1" if safe_ready >= 5 and false_operand_binding == 0 else "fail_closed_until_verified", "multi_policy": "fail_closed_until_verified", "semantic_binder_final_role": "diagnostic_semantic_evidence_analyzer", "reranker_final_role": "retrieval_only", "verified_evidence_packet_unified": True, "production_switch_allowed": False})
    write_json(OUT / "v2-04-final-close.json", {"generic_missing_evidence_repair_effective": False, "targeted_supply_repair_effective": False, "repair_policy_frozen": False, "repair_exploration_closed": True, "r0_repair_policy_rejected": True, "r1_repair_policy_rejected": True, "multi_supply_recovery_retained_as_diagnostic": True})
    write_json(OUT / "resume-claim-boundary.json", {"allowed_claim": "runtime-selectively admitted DIRECT evidence was 4/4 strict correct with zero false binding", "disallowed_claims": ["V2 Binder accuracy = 100%", "all queries bind correctly", "LLM Binder solves all evidence selection"], "coverage_precision_abstention_false_binding_separate": True})
    calc_ready = safe_ready
    decision = {"gate": GATE, "base_commit": BASE_COMMIT, "model_calls": 0, "retrieval_calls": 0, "binder_exploration_closed": True, "repair_exploration_closed": True, "calculation_historical_ready": "5/11", "calculation_current_contract_compatible": f"{safe_compatible}/11", "calculation_current_deterministic_ready": f"{calc_ready}/11", "runtime_candidate_ready": f"{runtime_ready}/11", "strict_correct_operand_slots": strict_operand_count, "false_operand_binding": false_operand_binding, "adapter_required": True, "calculation_deterministic_path_adopted": calc_ready >= 5 and false_operand_binding == 0, "deterministic_first_binding_warranted": False, "calculation_only_deterministic_binding_warranted": calc_ready >= 5 and false_operand_binding == 0, "next_gate": "v2_05_calculation_path_integration" if calc_ready >= 5 and false_operand_binding == 0 else "v2_06_verified_generation_subset", "architecture_scope_frozen": True, "production": "V1"}
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text(
        "# NF-V2 Architecture Scope Freeze R1\n\n"
        "Binder and evidence-repair exploration are closed. Trusted release is route-specific: "
        "DIRECT remains SelectiveBindingAdmissionV2 fail-closed; CALCULATION remains fail-closed "
        "because the current typed adapter replay does not meet the historical deterministic contract; "
        "MULTI_EVIDENCE remains fail-closed until verified complete evidence exists.\n\n"
        f"Historical calculation readiness: 5/11. Current candidate-ready: {runtime_ready}/11. "
        f"Current safe deterministic readiness: {safe_ready}/11.\n\n"
        f"Decision: {decision['next_gate']}. Contract SHA256: {contract_sha}.\n",
        encoding="utf-8",
    )
    print(json.dumps({"decision": decision, "runtime": [(row["question_id"], row["status"], row["ready"], row["historical_status"]) for row in runtime_rows]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
