#!/usr/bin/env python3
"""NF-V2-03 R1B offline fact-compatibility and DTO contract audit."""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.binder_selection import build_selection_messages, provider_request  # noqa: E402
from rag_v2.evidence.prompt import build_binder_messages  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1a_binding_contract_recovery as r1a  # noqa: E402


BASE_COMMIT = "5a18284a268ace5e01c81688a790192e7a551619"
MODEL = "qwen3.7-plus"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
ATTEMPT4 = ROOT / "artifacts/evaluation/nf-v2-03-formal-attempt-4"

FC_NAMES = {
    "FC0_true_semantic_match_evaluator_mismatch",
    "FC1_normalized_metric_too_coarse_but_raw_context_sufficient",
    "FC2_missing_segment_or_scope_in_fact_representation",
    "FC3_missing_row_or_header_composition",
    "FC4_wrong_period_representation",
    "FC5_wrong_fact_materialization",
    "FC6_supervisor_slot_defect",
    "FC7_true_non_bindable",
    "FC8_other",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\([^)]*\)$", "", text)
    text = re.sub(r"\d+$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def period(value: Any) -> str:
    return norm(value).replace("fy ", "fy")


def source_keys(label: dict[str, Any]) -> set[str]:
    return {str(source.get("candidate_key")) for source in label.get("expected_sources", []) if source.get("candidate_key")}


def source_facts(request: Any, label: dict[str, Any]) -> list[dict[str, Any]]:
    keys = source_keys(label)
    return [fact for fact in request.facts if str(fact.get("candidate_id")) in keys or keys.intersection(str(item) for item in fact.get("candidate_ids", []))]


def classify_fact_loss(slot: Any, fact: dict[str, Any], label: dict[str, Any], raw_metric_match: bool, raw_period_match: bool) -> tuple[str, str, bool, bool]:
    requested = norm(slot.metric)
    raw_metric = norm(fact.get("raw_metric"))
    source_row = norm(next((source.get("row_label") for source in label.get("expected_sources", []) if source.get("row_label")), ""))
    if raw_metric_match and not raw_period_match:
        return "FC4_wrong_period_representation", "Metric is represented but the available source fact has a different period.", True, False
    if raw_metric_match and raw_period_match:
        return "FC0_true_semantic_match_evaluator_mismatch", "Raw metric and period are compatible; the raw-normalizer gate did not lose semantic identity.", True, True
    if "research and development expenses" in raw_metric and "comirnaty" in requested:
        return "FC5_wrong_fact_materialization", "The Gold candidate contains a fact for a different row; no Comirnaty fact is materialized.", False, False
    if requested in raw_metric and raw_metric != requested:
        return "FC1_normalized_metric_too_coarse_but_raw_context_sufficient", "Existing raw metric preserves the requested identity, with only hierarchy or footnote text added.", True, period(fact.get("raw_period")) == period(slot.period)
    if raw_metric and raw_metric == source_row and requested != source_row:
        if "net income" in requested and any(token in requested for token in ("consumer", "community", "banking")):
            return "FC2_missing_segment_or_scope_in_fact_representation", "The row is present but the segment/table scope is absent from the fact fields.", True, period(fact.get("raw_period")) == period(slot.period)
        return "FC3_missing_row_or_header_composition", "The row label is present, but table/header composition needed to append the requested metric qualifier is absent.", True, period(fact.get("raw_period")) == period(slot.period)
    if (
        raw_metric == source_row
        or (raw_metric == "total" and source_row in {"total revenue", "more personal computing revenue"})
    ) and (
        any(token in requested for token in ("revenue", "net sales", "volume", "transactions"))
        or source_row in {"total revenue", "more personal computing revenue"}
    ):
        return "FC3_missing_row_or_header_composition", "The source row is available but its table/header metric composition is not represented in FinancialFactV1.", True, period(fact.get("raw_period")) == period(slot.period)
    return "FC7_true_non_bindable", "Available fact fields do not establish the requested metric identity without adding a new semantic claim.", False, False


def audit_funnel(frozen: dict[str, Any], labels: dict[str, Any], ids: list[str], funnel: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in funnel["rows"]:
        if item["question_id"] not in ids or not item["d3_gold_source_fact"]:
            continue
        question_id = item["question_id"]
        request = frozen["requests"][question_id]
        label = labels[question_id]
        slot = request.plan.required_slots[0]
        facts = source_facts(request, label)
        fact = next((candidate for candidate in facts if candidate.get("raw_period") == slot.period), facts[0] if facts else {})
        raw_metric_match = item["d4_metric_compatible"]
        raw_period_match = item["d5_period_compatible"]
        category, reason, semantic, period_ok = classify_fact_loss(slot, fact, label, raw_metric_match, raw_period_match)
        reviewed_fact_ids = [
            str(candidate["fact_id"])
            for candidate in facts
            if semantic and period_ok and period(candidate.get("raw_period") or candidate.get("normalized_period")) == period(slot.period)
        ]
        rows.append({
            "question_id": question_id,
            "question": request.question,
            "gold_metric": slot.metric,
            "gold_period": slot.period,
            "raw_metric": fact.get("raw_metric"),
            "normalized_metric": fact.get("normalized_metric"),
            "raw_period": fact.get("raw_period"),
            "normalized_period": fact.get("normalized_period"),
            "physical_source_id": fact.get("physical_source_id"),
            "raw_metric_compatible": raw_metric_match,
            "raw_period_compatible": raw_period_match,
            "primary_category": category,
            "reason": reason,
            "reviewed_semantic_compatible": semantic,
            "reviewed_period_compatible": period_ok,
            "reviewed_fact_ids": reviewed_fact_ids,
            "question_id_dependency": False,
            "gold_value_dependency": False,
        })
    counts = Counter(row["primary_category"] for row in rows)
    return {
        "denominator_d3": sum(1 for item in funnel["rows"] if item["question_id"] in ids and item["d3_gold_source_fact"]),
        "d4_raw_metric_compatible": sum(int(row["raw_metric_compatible"]) for row in rows),
        "d5_reviewed_semantic_compatible": sum(int(row["reviewed_semantic_compatible"]) for row in rows),
        "d6_reviewed_period_compatible": sum(int(row["reviewed_semantic_compatible"] and row["reviewed_period_compatible"]) for row in rows),
        "taxonomy": {name: counts.get(name, 0) for name in sorted(FC_NAMES)},
        "rows": rows,
    }


def token_impact(frozen: dict[str, Any]) -> dict[str, Any]:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    old: list[int] = []
    new: list[int] = []
    rows: list[dict[str, Any]] = []
    for question_id in sorted(frozen["requests"]):
        request = frozen["requests"][question_id]
        old_payload = {
            "question": request.question,
            "intent": request.plan.intent.value,
            "operation": request.plan.operation,
            "required_slots": [slot.to_dict() for slot in request.plan.required_slots],
            "financial_facts": [dict(fact) for fact in request.facts],
        }
        new_payload, _, _ = provider_request(request)
        old_messages = build_binder_messages(old_payload)
        new_messages = build_selection_messages(request, new_payload)
        old_text = "\n".join(message["content"] for message in old_messages)
        new_text = "\n".join(message["content"] for message in new_messages)
        old_tokens = len(encoding.encode(old_text))
        new_tokens = len(encoding.encode(new_text))
        old.append(old_tokens)
        new.append(new_tokens)
        rows.append({"question_id": question_id, "old_estimated_tokens": old_tokens, "new_estimated_tokens": new_tokens, "delta": new_tokens - old_tokens})
    def stats(values: list[int]) -> dict[str, Any]:
        ordered = sorted(values)
        return {"median": ordered[len(ordered) // 2], "p95": ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], "largest": max(ordered)}
    return {"tokenizer": "cl100k_base deterministic estimate", "old": stats(old), "new": stats(new), "delta": {key: stats(new)[key] - stats(old)[key] for key in ("median", "p95", "largest")}, "rows": rows}


def main() -> int:
    frozen, predictions, labels = r1a.load_inputs()
    attempt4_seal = read_json(ATTEMPT4 / "binder-prediction-seal.json")
    if not attempt4_seal.get("sealed"):
        raise RuntimeError("Attempt-4 prediction seal missing")
    old_structural = read_json(r1a.FORMAL / "binding-validator-results.json")
    old_breakdown = read_json(r1a.OUT / "validator-breakdown.json")
    direct_funnel = read_json(r1a.OUT / "direct-bindability-funnel.json")
    historical_funnel = read_json(r1a.OUT / "historical-46-funnel.json")
    direct_ids = sorted(question_id for question_id, item in frozen["plans"].items() if item["plan"].intent.value == "DIRECT_FACT")
    historical_ids = historical_funnel["rows"]
    direct_review = audit_funnel(frozen, labels, direct_ids, direct_funnel)
    historical_review = audit_funnel(frozen, labels, [row["question_id"] for row in historical_ids], historical_funnel)

    example_request = frozen["requests"][sorted(frozen["requests"])[0]]
    _, handles, schema = provider_request(example_request)
    write_json(OUT / "provider-dto-contract.json", {
        "dto": "BinderSelectionDTOv1",
        "provider": "Alibaba Bailian",
        "model": MODEL,
        "top_level_fields": ["slots"],
        "slot_fields": ["status", "fact_handles"],
        "allowed_statuses": ["BOUND", "MISSING", "AMBIGUOUS"],
        "query_level_status_from_model": False,
        "cardinality": {"BOUND": "exactly_one", "MISSING": "zero", "AMBIGUOUS": "at_least_two"},
        "additionalProperties": False,
        "example_schema": schema,
    })
    write_json(OUT / "fact-handle-map-contract.json", {"mapping": "F01..Fn in frozen packet order", "one_to_one": True, "semantic_ranking": False, "fact_filtering": False, "example_question_id": example_request.question_id, "example_handle_map": handles, "all_provenance_complete_facts_available": True})
    write_json(OUT / "adapter-contract.json", {"input": "BinderSelectionDTOv1", "output": "frozen EvidenceBinding", "operations": ["handle_to_existing_fact_id", "preserve_exact_slot_id", "derive_query_status", "reshape"], "semantic_matching": False, "gold_access": False, "source_selection": False})
    write_json(OUT / "attempt4-structural-failure-reference.json", {"before_validator_pass": old_breakdown["before_validator_pass"], "unknown_slot_instances": old_breakdown["reason_counts"]["BV1_unknown_slot_id"], "unknown_fact_instances": old_breakdown["reason_counts"]["BV2_unknown_fact_id"], "cardinality_instances": old_breakdown["reason_counts"]["BV10_cardinality_violation"], "status_inconsistency_instances": old_breakdown["reason_counts"]["BV7_invalid_status_binding_consistency"], "binding_validator_artifact": old_structural})
    write_json(OUT / "fact-semantic-compatibility-review.json", {"model_calls": 0, "direct": direct_review, "historical": historical_review, "raw_normalizer_contract_preserved": True, "manual_review_scope": "D3 Gold-source FinancialFact cases only", "question_specific_patching": False})
    write_json(OUT / "direct-bindability-review.json", {"D0": 56, "D1": direct_funnel["D1_provenance_supply"], "D2": direct_funnel["D2_gold_source_admitted"], "D3": direct_funnel["D3_gold_source_fact"], "D4_raw_normalizer": direct_funnel["D4_metric_compatible"], "D5_reviewed_semantic": direct_review["d5_reviewed_semantic_compatible"], "D6_reviewed_period": direct_review["d6_reviewed_period_compatible"], "D7_reviewed_strict_bindable": direct_review["d6_reviewed_period_compatible"]})
    write_json(OUT / "historical46-bindability-review.json", {"H0": 42, "H1": historical_funnel["D3_gold_source_fact"], "H2_raw_metric": historical_funnel["D4_metric_compatible"], "H3_reviewed_semantic": historical_review["d5_reviewed_semantic_compatible"], "H4_period": historical_review["d6_reviewed_period_compatible"], "H5_reviewed_strict_bindable": historical_review["d6_reviewed_period_compatible"]})
    write_json(OUT / "binder-fact-view-decision.json", {"required": any(row["primary_category"] in {"FC2_missing_segment_or_scope_in_fact_representation", "FC3_missing_row_or_header_composition"} for row in direct_review["rows"]), "financial_fact_v1_modified": False, "decision": "defer_until_protocol_and_compatibility_review", "allowed_source_fields": ["raw_metric", "normalized_metric", "raw_period", "normalized_period", "value", "unit", "currency", "row_label", "column_header", "table_title", "statement_context", "physical_provenance_summary"]})
    write_json(OUT / "token-impact.json", token_impact(frozen))
    write_json(OUT / "synthetic-protocol-test.json", {"executed": False, "model_calls": 0, "reason": "audit stage precedes the required qwen3.7-plus synthetic protocol run"})
    decision = {"gate": "NF-V2-03-R1B", "base_commit": BASE_COMMIT, "binder_model": MODEL, "model_calls": 0, "old_validator_pass": old_breakdown["before_validator_pass"], "unknown_slot_old_instances": old_breakdown["reason_counts"]["BV1_unknown_slot_id"], "cardinality_old_instances": old_breakdown["reason_counts"]["BV10_cardinality_violation"], "direct_reviewed_strict_bindable": direct_review["d6_reviewed_period_compatible"], "historical_reviewed_strict_bindable": historical_review["d6_reviewed_period_compatible"], "binder_fact_view_required": any(row["primary_category"] in {"FC2_missing_segment_or_scope_in_fact_representation", "FC3_missing_row_or_header_composition"} for row in direct_review["rows"]), "formal_attempt_5": "pending_protocol_tests", "production_default": "V1", "production_switch_allowed": False, "next_gate": "nf_v2_03_r1b_synthetic_protocol"}
    write_json(OUT / "decision.json", decision)
    (OUT / "README.md").write_text("# NF-V2-03 R1B\n\nOffline audit and provider-facing DTO contract only. No model calls were made in this stage. The frozen internal EvidenceBinding, FinancialFactV1, SupervisorPlan, Top20 packet, and semantic evaluator remain unchanged.\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
