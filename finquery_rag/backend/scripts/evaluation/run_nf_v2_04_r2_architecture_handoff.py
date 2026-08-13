#!/usr/bin/env python3
"""NF-V2-04 R2 deterministic-first evidence binding audit.

The contract is serialized and hashed before Gold/review labels are opened.
The evaluator then runs entirely offline over the frozen FinancialFactV1 and
BinderFactViewV2 source relation.  It never calls a provider or retriever.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding  # noqa: E402
from scripts.evaluation import run_nf_e2e_09_r0_structured_fact_representation as nf09  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_04_r0_missing_evidence_repair as r0  # noqa: E402


BASE_COMMIT = "888d3c071a23de4950a5e8330f9f32224ee6bfed"
GATE = "NF-V2-04-R2"
OUT = ROOT / "artifacts/evaluation/nf-v2-04-r2-architecture-handoff"
R1_OUT = ROOT / "artifacts/evaluation/nf-v2-04-r1-targeted-supply-repair"
V203_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r7-2-admission-contract-fix"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"
MODEL = "qwen3.7-plus"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stable_sha(value: Any) -> str:
    return hashlib.sha256((json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


def period(value: Any) -> str | None:
    value = norm(value)
    match = re.search(r"\bfy\s*(\d{4})\b", value)
    return f"fy{match.group(1)}" if match else value or None


def candidate_ids(fact: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", []) if item}


def fact_ids(facts: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(item.get("fact_id")) for item in facts if item.get("fact_id")}


def binding_from_row(row: Mapping[str, Any]) -> EvidenceBinding:
    binding = row.get("v2_binding") or row.get("binding") or {}
    return EvidenceBinding(
        status=str(binding.get("status")),
        slot_bindings={key: tuple(value) for key, value in (binding.get("slot_bindings") or {}).items()},
        missing_slots=tuple(binding.get("missing_slots") or ()),
        ambiguous_slots=tuple(binding.get("ambiguous_slots") or ()),
        invalid_reasons=tuple(binding.get("invalid_reasons") or ()),
    )


def metric_variants(fact: Mapping[str, Any], source: Mapping[str, Any] | None) -> set[str]:
    source = source or {}
    values: list[Any] = [fact.get("normalized_metric"), fact.get("raw_metric"), source.get("metric"), source.get("normalized_metric"), source.get("row_label"), source.get("row_path")]
    variants: set[str] = set()
    for value in values:
        if isinstance(value, (list, tuple)):
            variants.update(norm(item) for item in value if norm(item))
        elif norm(value):
            variants.add(norm(value))
    return variants


def exact_statement(slot: Any) -> str | None:
    for name in ("statement", "statement_id", "statement_type"):
        value = getattr(slot, name, None)
        if value:
            return norm(value)
    return None


def exact_scope(slot: Any) -> str | None:
    return norm(getattr(slot, "scope", None)) or None


def deterministic_contract() -> dict[str, Any]:
    return {
        "contract": "DeterministicBindingAuditV1",
        "version": 1,
        "model_calls": 0,
        "retrieval_calls": 0,
        "candidate_status": {"zero_eligible": "MISSING", "one_eligible": "DETERMINISTIC_BOUND", "multiple_eligible": "AMBIGUOUS"},
        "checks": {
            "DTC1": "provenance_complete must be true",
            "DTC2": "fact candidate identity must belong to the current query packet and source map",
            "DTC3": "requested period and fact period must both be known and exactly equal",
            "DTC4": "explicit scope/segment conflict rejects; unknown scope is not a match",
            "DTC5": "explicit statement conflict rejects when slot statement identity exists",
            "DTC6": "explicit unit/currency conflict rejects; constrained unknown unit/currency is not a match",
            "DTC7": "metric identity requires exact normalized canonical/structured equality; no fuzzy overlap",
            "DTC8": "candidate/source physical relation must be internally valid",
        },
        "required_dimensions": ["metric", "period"],
        "optional_dimensions": ["scope", "statement", "unit", "currency"],
        "unknown_policy": "unknown material evidence is not treated as a match",
        "role_policy": "operand role guides slot interpretation but is never evidence by itself",
        "gold_independent": True,
        "question_specific_rules": 0,
        "benchmark_aliases": 0,
        "financial_fact_v1_modified": False,
    }


def relation_reason(fact: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], fact_ids_in_packet: set[str]) -> list[str]:
    reasons: list[str] = []
    fact_id = str(fact.get("fact_id") or "")
    if not fact_id or fact_id not in fact_ids_in_packet:
        reasons.append("DTC2_fact_outside_query_packet")
    if fact.get("provenance_complete") is not True:
        reasons.append("DTC1_provenance_incomplete")
    candidates = candidate_ids(fact)
    linked = [(cid, source_map.get(cid)) for cid in candidates if source_map.get(cid) is not None]
    if not linked:
        reasons.append("DTC2_candidate_not_in_source_map")
        reasons.append("DTC8_source_relation_missing")
    else:
        relation_ok = False
        for _, source in linked:
            source_physical = str(source.get("physical_source_id") or "")
            fact_physical = str(fact.get("physical_source_id") or "")
            if source_physical and fact_physical and source_physical == fact_physical:
                relation_ok = True
        if not relation_ok:
            reasons.append("DTC8_physical_source_relation_mismatch")
    return reasons


def candidate_reasons(slot: Any, fact: Mapping[str, Any], packet: list[dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reasons = relation_reason(fact, source_map, fact_ids(packet))
    source = next((source_map[cid] for cid in candidate_ids(fact) if cid in source_map), None)
    requested_metric = norm(getattr(slot, "metric", None))
    variants = metric_variants(fact, source)
    if not requested_metric or requested_metric not in variants:
        reasons.append("DTC7_metric_identity_not_exact")
    requested_period = period(getattr(slot, "period", None))
    fact_period = period(fact.get("normalized_period") or fact.get("raw_period"))
    if not requested_period or not fact_period:
        reasons.append("DTC3_period_not_explicit")
    elif requested_period != fact_period:
        reasons.append("DTC3_explicit_period_conflict")
    if getattr(slot, "value_type", None) == "numeric" and not (fact.get("parsed_numeric_value") is not None or fact.get("raw_value")):
        reasons.append("DTC7_numeric_value_missing")
    requested_unit = norm(getattr(slot, "unit", None))
    fact_unit = norm(fact.get("unit"))
    if requested_unit and not fact_unit:
        reasons.append("DTC6_constrained_unit_unknown")
    elif requested_unit and fact_unit != requested_unit:
        reasons.append("DTC6_explicit_unit_conflict")
    requested_scope = exact_scope(slot)
    if requested_scope:
        explicit_scope = norm(fact.get("scope") or fact.get("segment") or (source or {}).get("scope") or (source or {}).get("row_path"))
        if not explicit_scope:
            reasons.append("DTC4_constrained_scope_unknown")
        elif explicit_scope != requested_scope:
            reasons.append("DTC4_explicit_scope_conflict")
    requested_statement = exact_statement(slot)
    if requested_statement:
        explicit_statement = norm(fact.get("statement_id") or (source or {}).get("statement_id"))
        if not explicit_statement:
            reasons.append("DTC5_constrained_statement_unknown")
        elif explicit_statement != requested_statement:
            reasons.append("DTC5_explicit_statement_conflict")
    return sorted(set(reasons))


def evaluate_slot(slot: Any, packet: list[dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    eligible: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    packet_ids = fact_ids(packet)
    for fact in packet:
        reasons = candidate_reasons(slot, fact, packet, source_map)
        row = {"fact_id": str(fact.get("fact_id")), "candidate_id": str(fact.get("candidate_id")), "reasons": reasons}
        if reasons:
            rejected.append(row)
        else:
            eligible.append(row)
    if len(eligible) == 0:
        status = BindingStatus.MISSING.value
    elif len(eligible) == 1:
        status = BindingStatus.BOUND.value
    else:
        status = BindingStatus.AMBIGUOUS.value
    return {"slot_id": slot.slot_id, "status": status, "eligible": eligible, "rejected": rejected, "packet_fact_count": len(packet_ids)}


def deterministic_binding(request: Any, packet: list[dict[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    slots = [evaluate_slot(slot, packet, source_map) for slot in request.plan.required_slots]
    if all(row["status"] == BindingStatus.BOUND.value for row in slots):
        status = BindingStatus.BOUND.value
        slot_bindings = {row["slot_id"]: [row["eligible"][0]["fact_id"]] for row in slots}
        missing, ambiguous = [], []
    elif any(row["status"] == BindingStatus.AMBIGUOUS.value for row in slots):
        status = BindingStatus.AMBIGUOUS.value
        slot_bindings = {}
        missing = [row["slot_id"] for row in slots if row["status"] == BindingStatus.MISSING.value]
        ambiguous = [row["slot_id"] for row in slots if row["status"] == BindingStatus.AMBIGUOUS.value]
    else:
        status = BindingStatus.MISSING.value
        slot_bindings = {}
        missing = [row["slot_id"] for row in slots]
        ambiguous = []
    return {"question_id": request.question_id, "status": status, "released": status == BindingStatus.BOUND.value, "slot_bindings": slot_bindings, "missing_slots": missing, "ambiguous_slots": ambiguous, "slots": slots}


def reconstruct_r1_packets(frozen: Mapping[str, Any], state: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Recreate only already-sealed R1 Stage-A materialization, no retrieval."""
    packet_rows = {row["question_id"]: row for row in read_jsonl_gz(R1_OUT / "stage-a-repair-packets.jsonl.gz")}
    _, atomic_index = nf09.load_atomic_facts()
    extra_by_qid: dict[str, list[dict[str, Any]]] = {}
    rows_by_case, _ = r0.candidate_rows_topk(state, 100)
    for qid, row in packet_rows.items():
        selected = set(str(item) for item in row.get("selected_candidate_ids", []))
        candidates = [candidate for candidate in rows_by_case.get(qid, []) if str(candidate.get("candidate_id")) in selected and candidate.get("physical_source_identity_complete")]
        facts: list[dict[str, Any]] = []
        for candidate in candidates:
            produced, _ = nf09.materialize_candidate(candidate, atomic_index)
            facts.extend(produced)
        extra_by_qid[qid], _ = nf09.dedup_facts(facts)
    packets: dict[str, list[dict[str, Any]]] = {}
    for qid, request in frozen["requests"].items():
        extras = extra_by_qid.get(qid, [])
        packets[qid], _ = nf09.dedup_facts(list(request.facts) + extras)
    return packets


def strict_correct(row: Mapping[str, Any], request: Any, labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], reviewed_ids: set[str], reviewed_fact_ids: Mapping[str, set[str]]) -> bool:
    binding = row["slot_bindings"]
    facts = {str(fact["fact_id"]): fact for fact in request.facts}
    for slot in request.plan.required_slots:
        selected = binding.get(slot.slot_id, [])
        if len(selected) != 1:
            return False
        fact = facts.get(str(selected[0]))
        if fact is None or not r1d.slot_is_strict(request.question_id, slot, fact, labels[request.question_id], source_map, reviewed_ids, reviewed_fact_ids, set()):
            return False
    return True


def strict_operand(slot: Any, qid: str, fact_id: str | None, packet: list[dict[str, Any]], labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]], reviewed_ids: set[str], reviewed_fact_ids: Mapping[str, set[str]]) -> bool:
    if not fact_id:
        return False
    fact = next((item for item in packet if str(item.get("fact_id")) == str(fact_id)), None)
    return bool(fact and r1d.slot_is_strict(qid, slot, fact, labels[qid], source_map, reviewed_ids, reviewed_fact_ids, set()))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = r1d.load_r1c_frozen_inputs()
    state = nf02.verify_frozen_top100()
    _, source_map = r0.candidate_rows_topk(state, 100)
    source_map = {str(key): dict(value) for key, value in source_map.items()}
    packets = reconstruct_r1_packets(frozen, state, source_map)
    contract = deterministic_contract()
    contract_path = OUT / "deterministic-binding-contract.json"
    write_json(contract_path, contract)
    contract_sha = sha256_file(contract_path)
    (OUT / "deterministic-binding-contract.sha256").write_text(contract_sha + "  deterministic-binding-contract.json\n", encoding="utf-8")
    if sha256_file(contract_path) != contract_sha:
        raise RuntimeError("deterministic contract SHA verification failed")

    # Runtime-only deterministic audit; no Gold/review labels are opened above.
    audits: dict[str, dict[str, Any]] = {}
    for qid, request in sorted(frozen["requests"].items()):
        binding = deterministic_binding(request, packets[qid], source_map)
        audits[qid] = {**binding, "intent": request.plan.intent.value, "fact_count": len(packets[qid]), "fact_packet_sha256": stable_sha(packets[qid])}

    # Gold/review labels are loaded only after the deterministic contract and
    # all pre-Gold runtime audit rows are sealed in memory.
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    direct_ids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "DIRECT_FACT"]
    calc_ids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "CALCULATION"]
    multi_ids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "MULTI_EVIDENCE"]

    direct_rows: list[dict[str, Any]] = []
    for qid in sorted(direct_ids):
        row = audits[qid]
        correct = bool(row["released"] and strict_correct(row, type("Req", (), {"question_id": qid, "plan": frozen["requests"][qid].plan, "facts": packets[qid]})(), labels, source_map, reviewed_ids, reviewed_fact_ids))
        direct_rows.append({"question_id": qid, "status": row["status"], "released": row["released"], "strict_correct": correct, "false_binding": bool(row["released"] and not correct), "eligible_fact_count": sum(len(slot["eligible"]) for slot in row["slots"]), "slots": row["slots"]})
    direct_bound = [row for row in direct_rows if row["released"]]
    direct_correct = sum(int(row["strict_correct"]) for row in direct_bound)

    calc_slot_rows: list[dict[str, Any]] = []
    calc_question_rows: list[dict[str, Any]] = []
    for qid in sorted(calc_ids):
        request = frozen["requests"][qid]
        row = audits[qid]
        operand_rows: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            slot_row = next(item for item in row["slots"] if item["slot_id"] == slot.slot_id)
            selected = slot_row["eligible"][0]["fact_id"] if slot_row["status"] == BindingStatus.BOUND.value else None
            correct = strict_operand(slot, qid, selected, packets[qid], labels, source_map, reviewed_ids, reviewed_fact_ids)
            operand_rows.append({"slot_id": slot.slot_id, "operation": request.plan.operation, "role": slot.role, "status": slot_row["status"], "selected_fact_id": selected, "strict_correct": correct, "eligible": slot_row["eligible"], "rejected": slot_row["rejected"]})
            calc_slot_rows.append({"question_id": qid, **operand_rows[-1]})
        calc_question_rows.append({"question_id": qid, "status": row["status"], "all_operands_deterministically_bound": all(item["status"] == BindingStatus.BOUND.value for item in operand_rows), "all_operands_strict_correct": all(item["strict_correct"] for item in operand_rows), "operands": operand_rows})

    multi_rows: list[dict[str, Any]] = []
    for qid in sorted(multi_ids):
        row = audits[qid]
        correct = bool(row["released"] and strict_correct(row, type("Req", (), {"question_id": qid, "plan": frozen["requests"][qid].plan, "facts": packets[qid]})(), labels, source_map, reviewed_ids, reviewed_fact_ids))
        multi_rows.append({"question_id": qid, "status": row["status"], "deterministic_complete": row["released"], "strict_correct": correct, "false_binding": bool(row["released"] and not correct), "slots": row["slots"]})

    # Current V2 admission overlap, using the sealed runtime-v2 rows only.
    runtime_path = V203_OUT / "runtime-v2-predictions.jsonl.gz"
    runtime_seal = read_json(V203_OUT / "runtime-v2-prediction-seal.json")
    if sha256_file(runtime_path) != runtime_seal.get("prediction_sha256"):
        raise RuntimeError("runtime-v2 SHA mismatch")
    current = {str(row["question_id"]): row for row in read_jsonl_gz(runtime_path)}
    both = sorted(qid for qid in direct_ids if audits[qid]["released"] and current[qid].get("released"))
    det_only = sorted(qid for qid in direct_ids if audits[qid]["released"] and not current[qid].get("released"))
    admission_only = sorted(qid for qid in direct_ids if current[qid].get("released") and not audits[qid]["released"])
    neither = sorted(qid for qid in direct_ids if not audits[qid]["released"] and not current[qid].get("released"))
    write_json(OUT / "direct-deterministic-binding.json", {"total": 56, "deterministic_bound": len(direct_bound), "missing": sum(int(row["status"] == BindingStatus.MISSING.value) for row in direct_rows), "ambiguous": sum(int(row["status"] == BindingStatus.AMBIGUOUS.value) for row in direct_rows), "strict_correct": direct_correct, "false_binding": sum(int(row["false_binding"]) for row in direct_rows), "precision": direct_correct / len(direct_bound) if direct_bound else None, "rows": direct_rows})
    write_json(OUT / "calculation-deterministic-binding.json", {"questions": 11, "bindable_operand_slots": 12, "deterministically_bound_operand_slots": sum(int(row["status"] == BindingStatus.BOUND.value) for row in calc_slot_rows), "strict_correct_operand_slots": sum(int(row["strict_correct"]) for row in calc_slot_rows), "false_operand_binding": sum(int(row["status"] == BindingStatus.BOUND.value and not row["strict_correct"]) for row in calc_slot_rows), "all_operands_deterministically_bound": sum(int(row["all_operands_deterministically_bound"]) for row in calc_question_rows), "all_operands_strict_correct": sum(int(row["all_operands_strict_correct"]) for row in calc_question_rows), "absolute_calculation_ready": sum(int(row["all_operands_deterministically_bound"]) for row in calc_question_rows), "rows": calc_question_rows, "operand_rows": calc_slot_rows})
    write_json(OUT / "multi-deterministic-binding.json", {"total": 5, "complete_supply_reference": "5/5", "deterministic_complete": sum(int(row["deterministic_complete"]) for row in multi_rows), "partial": sum(int(not row["deterministic_complete"] and row["status"] == BindingStatus.AMBIGUOUS.value) for row in multi_rows), "none": sum(int(not row["deterministic_complete"] and row["status"] == BindingStatus.MISSING.value) for row in multi_rows), "false_binding": sum(int(row["false_binding"]) for row in multi_rows), "rows": multi_rows})
    write_json(OUT / "admission-v2-vs-deterministic.json", {"admission_v2_bound": 4, "admission_v2_strict_correct": 4, "already_bound_by_admission_v2": len(both), "new_deterministic_only_bound": len(det_only), "both": both, "deterministic_only": det_only, "admission_v2_only": admission_only, "neither": neither})

    direct_det = len(direct_bound)
    calc_ready = sum(int(row["all_operands_deterministically_bound"]) for row in calc_question_rows)
    multi_det = sum(int(row["deterministic_complete"]) for row in multi_rows)
    query_calls_avoided = direct_det + calc_ready + multi_det
    avg_input_tokens = 1574861 / 62
    avg_output_tokens = 2410 / 62
    avg_latency_ms = 2066.99125
    write_json(OUT / "hybrid-binder-projection.json", {"architecture": "HybridEvidenceBinderV1", "deterministic_priority": True, "residual_model": MODEL, "direct_deterministic_calls_avoided": direct_det, "calculation_deterministic_calls_avoided": calc_ready, "multi_deterministic_calls_avoided": multi_det, "total_query_calls_avoided": query_calls_avoided, "residual_calls": 72 - query_calls_avoided, "fallback_admission": "SelectiveBindingAdmissionV2"})
    write_json(OUT / "cost-avoidance-projection.json", {"observed_reference": {"binder_calls": 62, "input_tokens": 1574861, "output_tokens": 2410, "average_latency_ms": avg_latency_ms}, "query_calls_avoided": query_calls_avoided, "estimated_input_tokens_avoided": round(query_calls_avoided * avg_input_tokens, 3), "estimated_output_tokens_avoided": round(query_calls_avoided * avg_output_tokens, 3), "estimated_latency_ms_avoided": round(query_calls_avoided * avg_latency_ms, 3), "assumption": "observed R0 Binder averages only; no external price assumptions"})

    direct_overlap = {"admission_v2_bound": "4/56", "admission_v2_correct": "4/4", "deterministic_bound": f"{direct_det}/56", "new_safe_coverage_over_admission_v2": len(det_only), "strict_correct": f"{direct_correct}/{direct_det}", "precision": direct_correct / direct_det if direct_det else None, "false_binding": sum(int(row["false_binding"]) for row in direct_rows)}
    calc_summary = {"bindable_operand_slots": 12, "deterministically_bound": sum(int(row["status"] == BindingStatus.BOUND.value) for row in calc_slot_rows), "strict_correct": sum(int(row["strict_correct"]) for row in calc_slot_rows), "false_operand_binding": sum(int(row["status"] == BindingStatus.BOUND.value and not row["strict_correct"]) for row in calc_slot_rows), "all_operands_ready": calc_ready, "absolute_ready": calc_ready}
    multi_summary = {"complete_supply": "5/5", "deterministic_complete": multi_det, "false_binding": sum(int(row["false_binding"]) for row in multi_rows)}
    write_json(OUT / "v2-04-final-summary.json", {"model_calls": 0, "retrieval_calls": 0, "r0_repair_policy_rejected": True, "r1_repair_policy_rejected": True, "generic_missing_evidence_repair_effective": False, "evidence_repair_router_frozen": False, "repair_exploration_closed": True, "direct": direct_overlap, "calculation": calc_summary, "multi": multi_summary, "contract_sha256": contract_sha})

    false_binding = sum(int(row["false_binding"]) for row in direct_rows) + sum(int(row["status"] == BindingStatus.BOUND.value and not row["strict_correct"]) for row in calc_slot_rows) + sum(int(row["false_binding"]) for row in multi_rows)
    deterministic_first = bool(direct_det >= 8 or (sum(int(row["status"] == BindingStatus.BOUND.value) for row in calc_slot_rows) >= 8 and calc_ready >= 3)) and false_binding == 0
    calc_only = bool(not deterministic_first and sum(int(row["status"] == BindingStatus.BOUND.value) for row in calc_slot_rows) >= 8 and false_binding == 0)
    decision = {"gate": GATE, "base_commit": BASE_COMMIT, "model_calls": 0, "retrieval_calls": 0, "generic_missing_evidence_repair_effective": False, "r0_repair_policy_rejected": True, "r1_repair_policy_rejected": True, "evidence_repair_router_frozen": False, "repair_exploration_closed": True, "deterministic_first_binding_warranted": deterministic_first, "calculation_only_deterministic_binding_warranted": calc_only, "deterministic_binding_insufficient": not deterministic_first and not calc_only, "next_gate": "v2_04_r2_1_hybrid_binding_implementation" if deterministic_first else "v2_05_calculation_operand_binding" if calc_only else "v2_architecture_handoff_review", "production_default": "V1", "production_switch_allowed": False, "contract_sha256": contract_sha}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"summary": "A Gold-independent deterministic binding contract was frozen before scoring. R0/R1 repair exploration is closed; deterministic binding is audited as a possible first stage and qwen3.7-plus remains residual fallback only if warranted.", "decision": decision, "contract_sha256": contract_sha})
    print(json.dumps({"decision": decision, "direct": direct_overlap, "calculation": calc_summary, "multi": multi_summary}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
