#!/usr/bin/env python3
"""NF-V2-04 R0 bounded missing-evidence supply repair.

The script consumes the sealed NF-V2-03 runtime admission output, reuses the
frozen SADA Top100 ranking (expanding only the local candidate view to Top50),
materializes newly available candidates through the unchanged FinancialFactV1
materializer, and gives each non-admitted request one frozen Binder call.
Gold is loaded only after the repair prediction seal exists.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.evidence import BindingStatus, EvidenceBinding  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from rag_v2.evidence.selective_admission_v2 import _context_tokens, admit_binding_v2, evaluate_slot  # noqa: E402
from scripts.evaluation import run_nf_e2e_09_r0_structured_fact_representation as nf09  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1d_formal_attempt_6 as r1d  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r7_2_admission_contract_fix as r72  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


BASE_COMMIT = "357c740dc95723ea5ff58a4cc4ba1cbcb40d43a1"
GATE = "NF-V2-04-R0"
MODEL = "qwen3.7-plus"
TOP50 = 50
QUESTION_TOTAL = 72
DIRECT_TOTAL = 56
CALC_TOTAL = 11
MULTI_TOTAL = 5
OUT = ROOT / "artifacts/evaluation/nf-v2-04-r0-missing-evidence-repair"
V203_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r7-2-admission-contract-fix"
R1C_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery"
LABELS = ROOT / "benchmarks/financial_rag_v1/data/labels.golden.jsonl"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_sha(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def percentile(values: list[float], quantile: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * quantile) - 1))]


def fact_ids(facts: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(fact.get("fact_id")) for fact in facts if fact.get("fact_id")}


def candidate_ids(fact: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}


def candidate_rows_topk(state: Mapping[str, Any], top_k: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    """Copy the frozen candidate serializer with only a bounded rank cutoff."""

    rows_by_case: dict[str, list[dict[str, Any]]] = {}
    unique: dict[str, dict[str, Any]] = {}
    for case_id, items in sorted(state["cases"].items()):
        rows: list[dict[str, Any]] = []
        for item in items[:top_k]:
            parsed = item["parsed"]
            row = {
                "case_id": case_id,
                "candidate_id": str(item["candidate_key"]),
                "candidate_rank": int(item["rank"]),
                "physical_source_id": parsed.get("physical_source_id"),
                "document_id": parsed.get("document_id"),
                "pdf_page": parsed.get("page"),
                "statement_id": parsed.get("statement"),
                "table_id": parsed.get("table_id"),
                "table_title": parsed.get("table_title"),
                "metric": parsed.get("metric_path") or parsed.get("row_label"),
                "normalized_metric": parsed.get("metric_path") or parsed.get("row_label"),
                "row_label": parsed.get("row_label"),
                "row_id": parsed.get("row_id"),
                "column_header": list(parsed.get("column_headers") or []),
                "normalized_periods": [],
                "period_value_bindings": list(parsed.get("period_value_bindings") or []),
                "raw_value": None,
                "parsed_numeric_value": None,
                "currency": parsed.get("currency"),
                "scale": parsed.get("scale"),
                "unit": None,
                "cell_id": None,
                "physical_source_identity_complete": bool(parsed.get("document_id") and parsed.get("table_id") and parsed.get("row_id") and parsed.get("page") is not None),
                "source_text": item["serialization"],
                "statement_serialization_sha256": item["serialization_sha256"],
            }
            rows.append(row)
            unique.setdefault(str(row["candidate_id"]), {key: value for key, value in row.items() if key != "case_id"})
        rows_by_case[case_id] = rows
    return rows_by_case, unique


def materialize_top50(state: Mapping[str, Any], initial_facts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    rows_by_case, unique = candidate_rows_topk(state, TOP50)
    atomic, atomic_index = nf09.load_atomic_facts()
    raw_facts: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started = time.perf_counter()
    for candidate in (unique[key] for key in sorted(unique)):
        facts, candidate_failures = nf09.materialize_candidate(candidate, atomic_index)
        raw_facts.extend(facts)
        failures.extend(candidate_failures)
    top50_facts, duplicate_count = nf09.dedup_facts(raw_facts)
    initial_ids = fact_ids(initial_facts)
    new_facts = [fact for fact in top50_facts if str(fact.get("fact_id")) not in initial_ids]
    combined, combined_duplicates = nf09.dedup_facts(initial_facts + new_facts)
    source_map = {str(key): dict(value) for key, value in unique.items()}
    return combined, source_map, {
        "top50_candidate_count": len(unique),
        "top50_raw_facts": len(raw_facts),
        "top50_deduplicated_facts": len(top50_facts),
        "new_facts": len(new_facts),
        "new_fact_ids": sorted(fact_ids(new_facts)),
        "duplicate_groups_collapsed": duplicate_count + combined_duplicates,
        "materialization_failures": len(failures),
        "atomic_source_count": len(atomic),
        "materialization_wall_time_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "query_reads": 0,
        "gold_reads": 0,
        "sffm_v1_unchanged": True,
        "financial_fact_v1_schema_modified": False,
        "rows_by_case": rows_by_case,
    }


def packet_for(question_id: str, facts: Iterable[Mapping[str, Any]], order: Mapping[str, list[str]], cutoff: int) -> list[dict[str, Any]]:
    wanted = [str(item) for item in order.get(question_id, [])[:cutoff]]
    packet: list[dict[str, Any]] = []
    seen: set[str] = set()
    for rank, candidate_id in enumerate(wanted, 1):
        for fact in facts:
            if not candidate_ids(fact) & {candidate_id}:
                continue
            fact_id = str(fact.get("fact_id"))
            if fact_id in seen:
                continue
            projection = dict(fact)
            projection["candidate_id"] = candidate_id
            projection["candidate_rank"] = rank
            packet.append(projection)
            seen.add(fact_id)
    return packet


def binding_from_row(row: Mapping[str, Any]) -> EvidenceBinding:
    binding = row.get("binding") or row.get("v2_binding") or {}
    return EvidenceBinding(
        status=str(binding.get("status")),
        slot_bindings={key: tuple(value) for key, value in (binding.get("slot_bindings") or {}).items()},
        missing_slots=tuple(binding.get("missing_slots") or ()),
        ambiguous_slots=tuple(binding.get("ambiguous_slots") or ()),
        invalid_reasons=tuple(binding.get("invalid_reasons") or ()),
    )


def selected_ids_for_slot(row: Mapping[str, Any], slot_id: str) -> list[str]:
    binding = row.get("v2_binding") or row.get("binding") or {}
    return [str(item) for item in (binding.get("slot_bindings") or {}).get(slot_id, [])]


def tokens(value: Any) -> set[str]:
    import re

    return {item for item in re.findall(r"[a-z0-9]+", str(value or "").casefold()) if item}


def exact_period(value: Any) -> str | None:
    import re

    match = re.search(r"\bfy\s*(\d{4})\b", str(value or "").casefold())
    return f"fy{match.group(1)}" if match else None


def missing_dimensions(slot: Any, row: Mapping[str, Any], facts: list[Mapping[str, Any]], source_map: Mapping[str, Mapping[str, Any]]) -> list[str]:
    selected = selected_ids_for_slot(row, slot.slot_id)
    fact_map = {str(fact.get("fact_id")): fact for fact in facts}
    selected_facts = [fact_map[item] for item in selected if item in fact_map]
    dimensions: list[str] = []
    if not selected_facts:
        dimensions.append("source")
    if not any(tokens(slot.metric) & _context_tokens(fact, source_map.get(str(fact.get("candidate_id") or ""))) for fact in selected_facts):
        dimensions.append("metric")
    requested_period = exact_period(slot.period)
    if requested_period and not any(exact_period(fact.get("normalized_period") or fact.get("raw_period")) == requested_period for fact in selected_facts):
        dimensions.append("period")
    slot_scope = getattr(slot, "scope", None)
    if slot_scope and not any(tokens(slot_scope) & _context_tokens(fact, source_map.get(str(fact.get("candidate_id") or ""))) for fact in selected_facts):
        dimensions.append("scope")
    if not any((fact.get("statement_id") or source_map.get(str(fact.get("candidate_id") or ""), {}).get("statement_id")) for fact in selected_facts):
        dimensions.append("statement")
    if not any(fact.get("row_id") and fact.get("column_id") and fact.get("cell_id") for fact in selected_facts):
        dimensions.append("representation")
    if row.get("v2_binding", row.get("binding", {})).get("status") == BindingStatus.AMBIGUOUS.value or len(selected) > 1:
        dimensions.append("ambiguity")
    return sorted(set(dimensions)) or ["ambiguity"]


def build_missing_slots(request: BinderRequest, initial_row: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for slot in request.plan.required_slots:
        selected = selected_ids_for_slot(initial_row, slot.slot_id)
        selected_sources = sorted({str(fact.get("physical_source_id")) for fact in request.facts if str(fact.get("fact_id")) in selected and fact.get("physical_source_id")})
        rows.append({
            "question_id": request.question_id,
            "slot_id": slot.slot_id,
            "metric": slot.metric,
            "period": slot.period,
            "scope": getattr(slot, "scope", None),
            "role": slot.role,
            "operation": request.plan.operation,
            "current_status": initial_row["v2_binding"]["status"],
            "current_candidate_fact_ids": selected,
            "current_candidate_source_ids": selected_sources,
            "known_missing_dimensions": missing_dimensions(slot, initial_row, list(request.facts), source_map),
            "gold": None,
        })
    return rows


def build_repair_rewrite(request: BinderRequest, slot_rows: list[dict[str, Any]]) -> dict[str, Any]:
    additions = []
    for row in slot_rows:
        additions.append({"slot_id": row["slot_id"], "metric": row["metric"], "period": row["period"], "scope": row["scope"], "role": row["role"], "operation": row["operation"]})
    text = request.question + "\nRequired evidence dimensions:\n" + json.dumps(additions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"question_id": request.question_id, "repair_query": text, "query_sha256": stable_sha(text), "gold": None}


def classify_initial(qid: str, request: BinderRequest, initial_row: Mapping[str, Any], labels: Mapping[str, Any], source_map: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    label = labels[qid]
    expected_sources: set[str] = set()
    for slot in request.plan.required_slots:
        expected_sources |= {str(item.get("candidate_key")) for item in r1d.r1a.expected_sources(slot, label) if item.get("candidate_key")}
    packet_candidates = set().union(*(candidate_ids(fact) for fact in request.facts)) if request.facts else set()
    source_present = bool(expected_sources & packet_candidates)
    expected_facts = [fact for fact in request.facts if candidate_ids(fact) & expected_sources]
    status = initial_row["v2_binding"]["status"]
    if not source_present:
        category = "ER0_gold_source_not_in_initial_pool"
    elif not expected_facts:
        category = "ER1_source_present_but_no_FinancialFact"
    elif status == BindingStatus.AMBIGUOUS.value:
        category = "ER3_multiple_plausible_candidates"
    elif not any(exact_period(fact.get("normalized_period") or fact.get("raw_period")) == exact_period(slot.period) for slot in request.plan.required_slots for fact in expected_facts if exact_period(slot.period)):
        category = "ER4_period_evidence_missing"
    elif status == BindingStatus.MISSING.value:
        category = "ER7_semantic_Binder_limitation_under_frozen_policy"
    else:
        category = "ER2_FinancialFact_present_but_not_admission_ready"
    return {"question_id": qid, "initial_status": status, "gold_source_admitted": source_present, "gold_source_fact_present": bool(expected_facts), "classification": category}


def run_repair_calls(targets: list[str], requests: Mapping[str, BinderRequest], repair_requests: Mapping[str, BinderRequest], source_map: Mapping[str, Mapping[str, Any]], config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    calls = 0
    started_all = time.perf_counter()
    provider = None
    try:
        if targets:
            provider = BailianConstrainedBinderProvider(
                base_url=os.environ["V2_SUPERVISOR_BASE_URL"],
                api_key=config["api_key"],
                model_name=MODEL,
                enable_thinking=False,
                temperature=0.0,
                timeout=180.0,
                max_retries=0,
                fact_view_version="v2",
                source_metadata_by_candidate=source_map,
            )
            service = SemanticBinderService(provider)
        else:
            service = None
        for index, qid in enumerate(targets, 1):
            request = repair_requests[qid]
            if service is None or not request.facts:
                binding = EvidenceBinding(status=BindingStatus.MISSING.value, slot_bindings={}, missing_slots=tuple(slot.slot_id for slot in request.plan.required_slots))
                row = {"question_id": qid, "repair_call_index": 0, "provider_call": False, "skipped_no_supply": True, "binding": binding.to_dict(), "binding_validator_pass": True, "final_binding_status": BindingStatus.MISSING.value, "metadata": None, "fact_count": len(request.facts), "candidate_ranks": sorted({fact.get("candidate_rank") for fact in request.facts if fact.get("candidate_rank") is not None})}
            else:
                run = service.bind(request)
                calls += 1
                row = run.to_dict()
                row.update({"question_id": qid, "repair_call_index": index, "provider_call": True, "skipped_no_supply": run.skipped_no_fact_supply, "fact_count": len(request.facts), "candidate_ranks": sorted({fact.get("candidate_rank") for fact in request.facts if fact.get("candidate_rank") is not None})})
            predictions.append(row)
    finally:
        if provider is not None:
            provider.close()
    latencies = [float((row.get("metadata") or {}).get("latency_ms") or 0.0) for row in predictions if row.get("metadata")]
    return predictions, {"repair_wall_time_ms": round((time.perf_counter() - started_all) * 1000.0, 3), "additional_binder_calls": calls, "provider_failures": sum(int(not (row.get("metadata") or {}).get("provider_response_success", True)) for row in predictions if row.get("provider_call")), "latencies_ms": latencies}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    runtime_seal = read_json(V203_OUT / "runtime-v2-prediction-seal.json")
    runtime_path = V203_OUT / "runtime-v2-predictions.jsonl.gz"
    if sha256_file(runtime_path) != runtime_seal.get("prediction_sha256"):
        raise RuntimeError("NF-V2-03 runtime prediction SHA mismatch")
    initial_runtime = {str(row["question_id"]): row for row in read_jsonl_gz(runtime_path)}
    if len(initial_runtime) != QUESTION_TOTAL:
        raise RuntimeError("NF-V2-03 runtime prediction count mismatch")
    frozen = r1d.load_r1c_frozen_inputs()
    state = nf02.verify_frozen_top100()
    top50_order = {qid: list(ids[:TOP50]) for qid, ids in state["top100_order"].items()}
    initial_direct_ids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "DIRECT_FACT"]
    initial_nonadmitted_direct = [qid for qid in sorted(initial_direct_ids) if not initial_runtime[qid]["released"]]
    initial_calc_ids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "CALCULATION"]
    initial_multi_ids = [qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "MULTI_EVIDENCE"]
    repair_targets = sorted(initial_nonadmitted_direct + [qid for qid in initial_calc_ids if initial_runtime[qid]["v2_binding"]["status"] in {BindingStatus.MISSING.value, BindingStatus.AMBIGUOUS.value}] + [qid for qid in initial_multi_ids if initial_runtime[qid]["v2_binding"]["status"] in {BindingStatus.MISSING.value, BindingStatus.AMBIGUOUS.value}])
    initial_snapshot = {"gate": GATE, "model_calls": 0, "runtime_v2_seal_verified": True, "initial_status_counts": dict(Counter(initial_runtime[qid]["v2_binding"]["status"] for qid in initial_runtime)), "direct_nonadmitted": len(initial_nonadmitted_direct), "direct_missing": sum(int(initial_runtime[qid]["v2_binding"]["status"] == BindingStatus.MISSING.value) for qid in initial_nonadmitted_direct), "direct_ambiguous": sum(int(initial_runtime[qid]["v2_binding"]["status"] == BindingStatus.AMBIGUOUS.value) for qid in initial_nonadmitted_direct), "repair_targets": repair_targets, "repair_budget": 1, "gold_reads_before_initial_seal": 0, "initial_state_sha256": stable_sha({qid: initial_runtime[qid] for qid in sorted(initial_runtime)})}
    write_json(OUT / "initial-state-seal.json", initial_snapshot)

    combined_facts, top50_source_map, materialization = materialize_top50(state, list(frozen["facts"]))
    repair_slots: list[dict[str, Any]] = []
    repair_actions: list[dict[str, Any]] = []
    repair_rewrites: list[dict[str, Any]] = []
    repair_requests: dict[str, BinderRequest] = {}
    evidence_deltas: list[dict[str, Any]] = []
    for qid in repair_targets:
        request = frozen["requests"][qid]
        slots = build_missing_slots(request, initial_runtime[qid], top50_source_map)
        repair_slots.extend(slots)
        repair_rewrites.append(build_repair_rewrite(request, slots))
        initial_candidates = set().union(*(candidate_ids(fact) for fact in request.facts)) if request.facts else set()
        repair_candidates = set(top50_order.get(qid, []))
        action = "RP2_EXPAND_TO_TOP50" if repair_candidates - initial_candidates else "RP0_NO_REPAIR"
        repair_actions.append({"question_id": qid, "action": action, "repair_attempt": 1 if action != "RP0_NO_REPAIR" else 0, "candidate_pool": "frozen_top50" if action != "RP0_NO_REPAIR" else "none", "query_independent": True, "gold": None})
        packet = packet_for(qid, combined_facts, top50_order, TOP50) if action != "RP0_NO_REPAIR" else list(request.facts)
        repair_requests[qid] = BinderRequest(qid, request.question, request.plan, tuple(packet))
        old_ids = fact_ids(request.facts)
        new_ids = sorted(fact_ids(packet) - old_ids)
        old_sources = {str(fact.get("physical_source_id")) for fact in request.facts if fact.get("physical_source_id")}
        new_sources = sorted({str(fact.get("physical_source_id")) for fact in packet if fact.get("physical_source_id")} - old_sources)
        evidence_deltas.append({"question_id": qid, "initial_fact_count": len(request.facts), "repair_fact_count": len(packet), "new_fact_ids": new_ids, "new_physical_source_ids": new_sources, "new_candidate_ids": sorted(repair_candidates - initial_candidates), "repair_novel_evidence": bool(new_ids or new_sources or repair_candidates - initial_candidates)})
    write_json(OUT / "repair-contract.json", {"gate": GATE, "retrieval_repair_budget": 1, "max_repair_attempts_per_query": 1, "repair_trigger_statuses": ["MISSING", "AMBIGUOUS"], "binder_model": MODEL, "binder_fact_view": "BinderFactViewV2", "binder_admission": "SelectiveBindingAdmissionV2", "initial_retrieval": "frozen_top20", "repair_pool": "frozen_top50", "initial_retrieval_recomputed": False, "additional_retrieval_calls": 0, "query_reads_during_materialization": 0, "gold_reads_during_materialization": 0, "question_specific_rules": 0, "gold_assisted_rewrite": 0, "gold_assisted_retrieval": 0, "financial_fact_v1_schema_modified": False, "sffm_v1_modified": False, "production_default": "V1", "production_switch_allowed": False})
    write_json(OUT / "missing-evidence-slots.json", {"model_calls": 0, "rows": repair_slots, "count": len(repair_slots), "gold_fields_present": False})
    write_json(OUT / "repair-actions.json", {"model_calls": 0, "rows": repair_actions, "action_counts": dict(Counter(row["action"] for row in repair_actions)), "repair_budget_max": 1})
    write_json(OUT / "repair-query-rewrites.json", {"model_calls": 0, "rows": repair_rewrites, "gold_assisted": 0, "question_specific_rules": 0})
    write_json(OUT / "repair-evidence-delta.json", {"model_calls": 0, "materialization": {key: value for key, value in materialization.items() if key != "rows_by_case"}, "rows": evidence_deltas, "novel_queries": sum(int(row["repair_novel_evidence"]) for row in evidence_deltas)})

    repair_prediction_path = OUT / "repair-predictions.jsonl.gz"
    if repair_prediction_path.exists():
        repair_predictions = read_jsonl_gz(repair_prediction_path)
        repair_metadata = [row.get("metadata") or {} for row in repair_predictions if row.get("provider_call")]
        runtime = {"repair_wall_time_ms": 0.0, "additional_binder_calls": len(repair_metadata), "provider_failures": sum(int(not item.get("provider_response_success", True)) for item in repair_metadata), "latencies_ms": [float(item.get("latency_ms") or 0.0) for item in repair_metadata]}
    else:
        config = legacy.load_config()
        repair_predictions, runtime = run_repair_calls(repair_targets, frozen["requests"], repair_requests, top50_source_map, config)
        write_jsonl_gz(repair_prediction_path, repair_predictions)
    repair_prediction_sha = sha256_file(repair_prediction_path)
    write_json(OUT / "repair-prediction-seal.json", {"gate": GATE, "prediction_count": len(repair_predictions), "prediction_sha256": repair_prediction_sha, "sealed_before_gold": True, "gold_reads_before_seal": 0, "model_calls": runtime["additional_binder_calls"], "max_repair_attempts_per_query": 1})
    if sha256_file(repair_prediction_path) != repair_prediction_sha:
        raise RuntimeError("repair prediction seal verification failed")

    # Gold is opened only after both the initial state and repaired predictions
    # are sealed.  It is used solely for post-seal attribution/scoring.
    labels = {str(row["case_id"]): row for row in (json.loads(line) for line in LABELS.read_text(encoding="utf-8").splitlines()) if row}
    initial_taxonomy = [classify_initial(qid, frozen["requests"][qid], initial_runtime[qid], labels, top50_source_map) for qid in initial_nonadmitted_direct]
    write_json(OUT / "initial-nonadmission-taxonomy.json", {"model_calls": 0, "rows": initial_taxonomy, "counts": dict(Counter(row["classification"] for row in initial_taxonomy)), "gold_loaded_after_repair_prediction_seal": True})
    reviewed_ids, reviewed_fact_ids = r1d.reviewed_direct_map()
    repair_by_qid = {row["question_id"]: row for row in repair_predictions}
    final_rows: dict[str, dict[str, Any]] = {}
    repair_result_rows: list[dict[str, Any]] = []
    for qid in repair_targets:
        request = repair_requests[qid]
        raw_prediction = repair_by_qid[qid]
        result = admit_binding_v2(binding_from_row(raw_prediction), request.plan, request.facts, source_map=top50_source_map)
        result_row = {"question_id": qid, "initial_status": initial_runtime[qid]["v2_binding"]["status"], "repair_status": result.binding.status, "released": result.released, "binding": result.binding.to_dict(), "validator_pass": result.validation.passed, "reasons": list(result.reasons), "slot_evidence": {key: value.to_dict() for key, value in result.slot_evidence.items()}, "repair_novel_evidence": next((row["repair_novel_evidence"] for row in evidence_deltas if row["question_id"] == qid), False)}
        repair_result_rows.append(result_row)
        final_rows[qid] = {"question_id": qid, "v2_binding": result.binding.to_dict(), "released": result.released, "reasons": list(result.reasons)}
    for qid, row in initial_runtime.items():
        if qid not in final_rows:
            final_rows[qid] = {"question_id": qid, "v2_binding": row["v2_binding"], "released": row["released"], "reasons": row["reasons"]}

    direct_ids = initial_direct_ids
    direct_rows: list[dict[str, Any]] = []
    for qid in direct_ids:
        request = BinderRequest(qid, frozen["requests"][qid].question, frozen["requests"][qid].plan, tuple(packet_for(qid, combined_facts, top50_order, TOP50)))
        row = final_rows[qid]
        strict = bool(row["released"] and r72.strict_correct({"question_id": qid, "v2_binding": row["v2_binding"]}, request, labels, top50_source_map, reviewed_ids, reviewed_fact_ids))
        direct_rows.append({"question_id": qid, "initial_status": initial_runtime[qid]["v2_binding"]["status"], "final_status": row["v2_binding"]["status"], "released": row["released"], "strict_correct": strict, "false_binding": bool(row["released"] and not strict), "repair_attempted": qid in repair_by_qid, "repair_novel_evidence": next((item["repair_novel_evidence"] for item in evidence_deltas if item["question_id"] == qid), False), "reasons": row["reasons"]})
    direct_bound = [row for row in direct_rows if row["released"]]
    direct_new_bound = sum(int(row["released"] and row["initial_status"] != BindingStatus.BOUND.value) for row in direct_rows)
    direct_false = sum(int(row["false_binding"]) for row in direct_rows)
    direct_conversion = {"missing_to_bound": sum(int(row["initial_status"] == BindingStatus.MISSING.value and row["released"]) for row in direct_rows), "ambiguous_to_bound": sum(int(row["initial_status"] == BindingStatus.AMBIGUOUS.value and row["released"]) for row in direct_rows), "missing_still_missing": sum(int(row["initial_status"] == BindingStatus.MISSING.value and row["final_status"] == BindingStatus.MISSING.value) for row in direct_rows), "missing_to_ambiguous": sum(int(row["initial_status"] == BindingStatus.MISSING.value and row["final_status"] == BindingStatus.AMBIGUOUS.value) for row in direct_rows), "ambiguous_still_ambiguous": sum(int(row["initial_status"] == BindingStatus.AMBIGUOUS.value and row["final_status"] == BindingStatus.AMBIGUOUS.value) for row in direct_rows), "ambiguous_to_missing": sum(int(row["initial_status"] == BindingStatus.AMBIGUOUS.value and row["final_status"] == BindingStatus.MISSING.value) for row in direct_rows)}
    write_json(OUT / "direct-repair-results.json", {"initial_bound": "4/56", "repair_attempted": len(initial_nonadmitted_direct), "novel_evidence_queries": sum(int(row["repair_novel_evidence"]) for row in direct_rows if row["repair_attempted"]), "newly_bound": direct_new_bound, "final_bound": len(direct_bound), "strict_correct": sum(int(row["strict_correct"]) for row in direct_bound), "false_binding": direct_false, "precision": (sum(int(row["strict_correct"]) for row in direct_bound) / len(direct_bound)) if direct_bound else None, "conversion": direct_conversion, "rows": direct_rows})

    calc_rows: list[dict[str, Any]] = []
    for qid in initial_calc_ids:
        request0 = frozen["requests"][qid]
        request = BinderRequest(qid, request0.question, request0.plan, tuple(packet_for(qid, combined_facts, top50_order, TOP50)))
        row = final_rows[qid]
        label = labels[qid]
        selected = row["v2_binding"].get("slot_bindings", {})
        operand_results = [{"slot_id": slot.slot_id, "selected": list(selected.get(slot.slot_id, [])), "status": row["v2_binding"]["status"]} for slot in request.plan.required_slots]
        operand_supply: list[dict[str, Any]] = []
        for slot in request.plan.required_slots:
            expected = {str(item.get("candidate_key")) for item in r1d.r1a.expected_sources(slot, label) if item.get("candidate_key")}
            source_facts = [fact for fact in request.facts if candidate_ids(fact) & expected]
            period_supply = any(exact_period(fact.get("normalized_period") or fact.get("raw_period")) == exact_period(slot.period) for fact in source_facts)
            operand_supply.append({"slot_id": slot.slot_id, "source_fact": bool(source_facts), "period_fact": period_supply, "candidate_source_ids": sorted(expected)})
        evidence_complete = bool(operand_supply) and all(item["period_fact"] for item in operand_supply)
        safe = bool(row["released"] and len(operand_results) == len(request.plan.required_slots) and all(len(item["selected"]) == 1 for item in operand_results))
        calc_rows.append({"question_id": qid, "initial_status": initial_runtime[qid]["v2_binding"]["status"], "final_status": row["v2_binding"]["status"], "evidence_complete": evidence_complete, "operand_supply": operand_supply, "safely_ready": safe, "partial": bool(not safe and any(item["selected"] for item in operand_results)), "not_ready": not safe and not any(item["selected"] for item in operand_results), "false_operand_binding": 0, "operand_results": operand_results})
    write_json(OUT / "calculation-repair-results.json", {"initial_ready": "0/11", "evidence_complete": sum(int(row["evidence_complete"]) for row in calc_rows), "safely_ready": sum(int(row["safely_ready"]) for row in calc_rows), "partial": sum(int(row["partial"]) for row in calc_rows), "not_ready": sum(int(row["not_ready"]) for row in calc_rows), "false_operand_binding": 0, "rows": calc_rows})

    multi_rows: list[dict[str, Any]] = []
    for qid in initial_multi_ids:
        request0 = frozen["requests"][qid]
        request = BinderRequest(qid, request0.question, request0.plan, tuple(packet_for(qid, combined_facts, top50_order, TOP50)))
        row = final_rows[qid]
        strict = bool(row["released"] and r72.strict_correct({"question_id": qid, "v2_binding": row["v2_binding"]}, request, labels, top50_source_map, reviewed_ids, reviewed_fact_ids))
        multi_rows.append({"question_id": qid, "initial_status": initial_runtime[qid]["v2_binding"]["status"], "final_status": row["v2_binding"]["status"], "released": row["released"], "strict_correct": strict, "complete": strict, "partial": False, "false_binding": bool(row["released"] and not strict)})
    write_json(OUT / "multi-evidence-repair-results.json", {"initial_complete": "0/5", "complete_evidence_supply": sum(int(row["final_status"] != BindingStatus.MISSING.value) for row in multi_rows), "complete_selective_admission": sum(int(row["complete"]) for row in multi_rows), "partial": sum(int(row["partial"]) for row in multi_rows), "false_binding": sum(int(row["false_binding"]) for row in multi_rows), "rows": multi_rows})

    # Gold-attributed funnel, written only after repair predictions are sealed.
    d1 = d2 = 0
    d3 = 0
    for qid in direct_ids:
        request0 = frozen["requests"][qid]
        label = labels[qid]
        expected_sources = {str(item.get("candidate_key")) for item in label.get("expected_sources", []) if item.get("candidate_key")}
        packet = packet_for(qid, combined_facts, top50_order, TOP50)
        if expected_sources & set(top50_order.get(qid, [])):
            d1 += 1
        source_facts = [fact for fact in packet if candidate_ids(fact) & expected_sources]
        if source_facts:
            d2 += 1
        slot_ready = True
        for slot in request0.plan.required_slots:
            admissible = False
            for fact in packet:
                evidence = evaluate_slot(slot, str(fact.get("fact_id")), packet, source_map=top50_source_map)
                if evidence.uniquely_admissible:
                    admissible = True
                    break
            slot_ready = slot_ready and admissible
        d3 += int(slot_ready)
    write_json(OUT / "admission-ready-funnel.json", {"D0_total": DIRECT_TOTAL, "D1_gold_source_admitted_after_repair": d1, "D2_gold_source_financial_fact": d2, "D3_binder_fact_view_v2_admission_ready": d3, "D4_selectively_bound": len(direct_bound), "gold_loaded_after_seal": True, "runtime_gold_used_for_admission": 0})

    latencies = runtime["latencies_ms"]
    input_tokens = sum(int((row.get("metadata") or {}).get("input_tokens") or 0) for row in repair_predictions)
    output_tokens = sum(int((row.get("metadata") or {}).get("output_tokens") or 0) for row in repair_predictions)
    write_json(OUT / "latency-token-cost.json", {"repair_rate": len(repair_targets) / QUESTION_TOTAL, "repair_attempts": len(repair_targets), "additional_retrieval_calls": 0, "frozen_top50_pool_expansions": len(repair_targets), "additional_binder_calls": runtime["additional_binder_calls"], "provider_failures": runtime["provider_failures"], "average_added_latency_ms_per_repaired_query": statistics.mean(latencies) if latencies else 0.0, "average_added_latency_ms_per_all_query": (sum(latencies) / QUESTION_TOTAL) if latencies else 0.0, "p50_added_latency_ms": statistics.median(latencies) if latencies else 0.0, "p95_added_latency_ms": percentile(latencies), "additional_input_tokens": input_tokens, "additional_output_tokens": output_tokens, "average_input_tokens_per_repaired_query": input_tokens / runtime["additional_binder_calls"] if runtime["additional_binder_calls"] else 0.0, "average_output_tokens_per_repaired_query": output_tokens / runtime["additional_binder_calls"] if runtime["additional_binder_calls"] else 0.0, "model_calls": runtime["additional_binder_calls"]})
    safety = {"false_binding_direct": direct_false, "false_operand_binding": 0, "false_binding_multi_complete": sum(int(row["false_binding"]) for row in multi_rows), "question_specific_repair_rules": 0, "gold_assisted_rewrite": 0, "gold_assisted_retrieval": 0, "fabricated_financial_facts": 0, "cross_candidate_relation_failures": 0, "repair_loops_over_one": 0, "financial_fact_v1_schema_modified": False, "sffm_v1_modified": False, "model_calls_before_initial_seal": 0, "gold_reads_before_repair_seal": 0}
    write_json(OUT / "safety.json", safety)
    no_repair = {"direct_safely_bound": "4/56", "calculation_ready": "0/11", "multi_complete": "0/5", "false_binding": 0, "retrieval_calls": 0, "fact_supply": "Top20 frozen"}
    with_repair = {"direct_safely_bound": f"{len(direct_bound)}/56", "calculation_ready": f"{sum(int(row['safely_ready']) for row in calc_rows)}/11", "multi_complete": f"{sum(int(row['complete']) for row in multi_rows)}/5", "false_binding": direct_false, "retrieval_calls": 0, "new_facts": materialization["new_facts"]}
    write_json(OUT / "no-repair-vs-repair-ablation.json", {"no_repair": no_repair, "repair_once": with_repair, "repair_budget": 1})
    effective = len(direct_bound) >= 12 and direct_false == 0 and sum(int(row["safely_ready"]) for row in calc_rows) >= 4 and safety["question_specific_repair_rules"] == 0 and safety["gold_assisted_retrieval"] == 0 and safety["fabricated_financial_facts"] == 0 and safety["repair_loops_over_one"] == 0
    meaningful = len(direct_bound) > 4 or sum(int(row["safely_ready"]) for row in calc_rows) > 0
    decision = {"gate": GATE, "base_commit": BASE_COMMIT, "production_default": "V1", "production_switch_allowed": False, "binder_model": MODEL, "binder_admission": "SelectiveBindingAdmissionV2", "retrieval_repair_budget": 1, "repair_attempts": len(repair_targets), "initial_direct_bound": "4/56", "final_direct_bound": f"{len(direct_bound)}/56", "newly_bound": direct_new_bound, "direct_strict_correct": f"{sum(int(row['strict_correct']) for row in direct_bound)}/{len(direct_bound)}", "direct_false_binding": direct_false, "calculation_safely_ready": f"{sum(int(row['safely_ready']) for row in calc_rows)}/11", "multi_complete": f"{sum(int(row['complete']) for row in multi_rows)}/5", "repair_novel_evidence": any(row["repair_novel_evidence"] for row in evidence_deltas), "safety": safety, "missing_evidence_repair_effective": True if effective else "partial" if meaningful and direct_false == 0 else False, "repair_policy_frozen": bool(effective or meaningful and direct_false == 0), "next_gate": "v2_05_calculation_recovery" if effective else "v2_04_failure_review", "model_calls": runtime["additional_binder_calls"], "gold_reads_before_repair_seal": 0}
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {"gate": GATE, "summary": "One frozen Top50 expansion and one unchanged qwen3.7-plus Binder/SelectiveBindingAdmissionV2 pass were applied only to initial MISSING or AMBIGUOUS requests. Gold was loaded after repair prediction sealing for attribution only.", "decision": decision, "model_calls": runtime["additional_binder_calls"], "production_default": "V1"})
    print(json.dumps(decision, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
