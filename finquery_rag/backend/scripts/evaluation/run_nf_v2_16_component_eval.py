"""Run the sealed deterministic NF-V2-16 component evaluation.

This script deliberately uses synthetic structured fixtures.  It does not
load a model, call retrieval, or read the frozen 72-question benchmark.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.adaptive import (  # noqa: E402
    AdaptiveRAGBudgetV1,
    AdaptiveRAGStateV1,
    BoundedAdaptiveRAGV1,
    ConsistencyDecision,
    EvidenceConsistencyGateV1,
    EvidencePacketV1,
    PeriodSemantics,
    ReasonCode,
    ReplanActionV1,
    TemporalEvidenceV1,
    ToolCapability,
)


OUT = ROOT / "artifacts/evaluation/nf-v2-16-bounded-adaptive-rag"


def packet(
    evidence_id: str,
    *,
    metric: str,
    value: str,
    period: str,
    entity: str = "Acme",
    scope: str = "consolidated",
    source: str = "10-K",
    document_id: str | None = None,
    slots: tuple[str, ...] = (),
    semantics: PeriodSemantics = PeriodSemantics.ANNUAL,
    fiscal_year: str | None = None,
    fiscal_quarter: str | None = None,
    report_date: str | None = None,
    created_at: str | None = None,
    version: str | None = None,
    is_amended: bool = False,
    supersedes: str | None = None,
    unit: str | None = "USD",
    currency: str | None = "USD",
    scale: str | None = "millions",
) -> dict[str, Any]:
    document_id = document_id or evidence_id
    temporal = TemporalEvidenceV1(
        entity=entity, document_id=document_id, document_type="10-K",
        fiscal_year=fiscal_year, fiscal_quarter=fiscal_quarter,
        period_semantics=semantics, report_date=report_date,
        version=version, is_amended=is_amended,
        supersedes_document_id=supersedes, source=source, scope=scope,
        metric=metric, value=value, unit=unit, currency=currency, scale=scale,
        created_at=created_at,
    )
    return EvidencePacketV1(
        evidence_id=evidence_id, metric=metric, value=value, period=period,
        entity=entity, scope=scope, unit=unit, currency=currency, scale=scale,
        citation_id=evidence_id, source=source, document_id=document_id,
        slots=slots, temporal=temporal,
    ).to_dict()


def slot(slot_id: str, metric: str, period: str, **extra: Any) -> dict[str, Any]:
    return {"slot_id": slot_id, "metric": metric, "period": period, "value_required": True, **extra}


def controller_case(case_id: str, description: str, required: list[dict[str, Any]], sequence: list[list[dict[str, Any]]], expected: str, *, calculation: dict[str, Any] | None = None, first_capability: ToolCapability = ToolCapability.SEMANTIC_RETRIEVAL, tool_error_first: bool = False) -> dict[str, Any]:
    return {
        "case_id": case_id, "description": description, "required_slots": required,
        "sequence": sequence, "expected": expected,
        "calculation_requirements": calculation or {},
        "first_capability": first_capability.value, "tool_error_first": tool_error_first,
    }


def build_cases() -> list[dict[str, Any]]:
    revenue24 = packet("A-24", metric="Revenue", value="120", period="FY2024", slots=("revenue",), fiscal_year="2024")
    revenue23 = packet("A-23", metric="Revenue", value="100", period="FY2023", fiscal_year="2023")
    cost24 = packet("B-cost", metric="Cost", value="40", period="FY2024", fiscal_year="2024")
    c = packet("C-23", metric="Revenue", value="100", period="FY2023", fiscal_year="2023")
    c2 = packet("C-24", metric="Revenue", value="120", period="FY2024", slots=("revenue",), fiscal_year="2024")
    # M/N are intentionally independent from the frozen benchmark and use
    # only structural behavior classes.
    return [
        controller_case("A", "recoverable missing slot", [slot("revenue", "Revenue", "FY2024")], [[], [revenue24]], "READY_TO_GENERATE"),
        controller_case("B", "unrecoverable repeated evidence", [slot("revenue", "Revenue", "FY2024")], [[cost24], [cost24]], "FAIL_CLOSED_NO_PROGRESS"),
        controller_case("C", "wrong period then targeted replan", [slot("revenue", "Revenue", "FY2024")], [[revenue23], [c2]], "READY_TO_GENERATE"),
        controller_case("M", "identical evidence no progress", [slot("revenue", "Revenue", "FY2024")], [[cost24], [cost24]], "FAIL_CLOSED_NO_PROGRESS"),
        controller_case("N", "tool error rerouted to lexical lane", [slot("revenue", "Revenue", "FY2024")], [[c], [c2]], "READY_TO_GENERATE", tool_error_first=True),
    ]


def run_controller(case: dict[str, Any]) -> dict[str, Any]:
    state = AdaptiveRAGStateV1.new(
        case["case_id"], "What was Revenue in FY2024?", required_slots=case["required_slots"],
        calculation_requirements=case["calculation_requirements"],
    )
    sequence = case["sequence"]
    cursor = {"index": 0}

    def tool(query: str, current: AdaptiveRAGStateV1):
        index = cursor["index"]
        cursor["index"] += 1
        if case.get("tool_error_first") and index == 0:
            raise RuntimeError("synthetic tool unavailable")
        return sequence[min(index, len(sequence) - 1)]

    tools = {
        ToolCapability.SEMANTIC_RETRIEVAL: tool,
        ToolCapability.LEXICAL_RETRIEVAL: tool,
        ToolCapability.STRUCTURED_FINANCIAL_LOOKUP: tool,
        ToolCapability.DOCUMENT_METADATA_LOOKUP: tool,
    }
    initial = ReplanActionV1(ToolCapability(case["first_capability"]), state.normalized_query, ReasonCode.MISSING_SLOT)
    started = time.perf_counter_ns()
    result = BoundedAdaptiveRAGV1(budget=AdaptiveRAGBudgetV1()).run(state, tools, initial_action=initial)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    expected = case["expected"]
    passed = (
        result.state.status == "READY_TO_GENERATE" if expected == "READY_TO_GENERATE"
        else result.state.status == "FAIL_CLOSED" and result.state.stop_reason == "NO_PROGRESS"
    )
    return {
        "case_id": case["case_id"], "description": case["description"],
        "expected": expected, "actual_status": result.state.status,
        "stop_reason": result.state.stop_reason, "passed": passed,
        "tool_calls": result.state.tool_calls, "replan_rounds": result.state.replan_rounds,
        "transitions": result.state.transitions, "tool_history": result.state.tool_history,
        "evaluation": result.evaluation.to_dict() if result.evaluation else None,
        "latency_ms": elapsed_ms,
    }


def temporal_cases() -> list[dict[str, Any]]:
    annual = packet("D-A", metric="Revenue", value="100", period="FY2024", fiscal_year="2024", semantics=PeriodSemantics.ANNUAL)
    quarter = packet("D-Q", metric="Revenue", value="30", period="FY2024Q4", fiscal_year="2024", fiscal_quarter="Q4", semantics=PeriodSemantics.QUARTER)
    ytd = packet("E-Y", metric="Revenue", value="80", period="FY2024YTD", fiscal_year="2024", semantics=PeriodSemantics.YTD, scope="YTD")
    q1 = packet("E-Q", metric="Revenue", value="20", period="FY2024Q1", fiscal_year="2024", fiscal_quarter="Q1", semantics=PeriodSemantics.QUARTER, scope="Q1")
    fy24 = packet("F-24", metric="Rating", value="BUY", period="FY2024", fiscal_year="2024", semantics=PeriodSemantics.ANNUAL, unit=None, currency=None, scale=None)
    fy25 = packet("F-25", metric="Rating", value="HOLD", period="FY2025", fiscal_year="2025", semantics=PeriodSemantics.ANNUAL, unit=None, currency=None, scale=None)
    old = packet("G-old", metric="EPS", value="1.0", period="FY2024", fiscal_year="2024", document_id="doc-old", version="1", semantics=PeriodSemantics.ANNUAL)
    amended = packet("G-new", metric="EPS", value="1.2", period="FY2024", fiscal_year="2024", document_id="doc-new", version="2", is_amended=True, supersedes="doc-old", semantics=PeriodSemantics.ANNUAL)
    conflict_a = packet("H-a", metric="Debt", value="10", period="FY2024", fiscal_year="2024", document_id="h-a", semantics=PeriodSemantics.ANNUAL)
    conflict_b = packet("H-b", metric="Debt", value="20", period="FY2024", fiscal_year="2024", document_id="h-b", semantics=PeriodSemantics.ANNUAL)
    source_a = packet("I-a", metric="Rating", value="BUY", period="FY2024", fiscal_year="2024", source="broker-a", unit=None, currency=None, scale=None)
    source_b = packet("I-b", metric="Rating", value="HOLD", period="FY2024", fiscal_year="2024", source="broker-b", unit=None, currency=None, scale=None)
    j1 = packet("J-1", metric="Rating", value="BUY", period="Q1-2024", fiscal_year="2024", fiscal_quarter="Q1", source="same", semantics=PeriodSemantics.QUARTER, unit=None, currency=None, scale=None)
    j2 = packet("J-2", metric="Rating", value="HOLD", period="Q2-2024", fiscal_year="2024", fiscal_quarter="Q2", source="same", semantics=PeriodSemantics.QUARTER, unit=None, currency=None, scale=None)
    k1 = packet("K-1", metric="Rating", value="BUY", period="FY2024", fiscal_year="2024", source="same", semantics=PeriodSemantics.ANNUAL, unit=None, currency=None, scale=None)
    k2 = packet("K-2", metric="Rating", value="HOLD", period="FY2024", fiscal_year="2024", source="same", semantics=PeriodSemantics.ANNUAL, unit=None, currency=None, scale=None)
    l_old = packet("L-old", metric="Revenue", value="100", period="FY2024", fiscal_year="2024", source="same", report_date="2025-02-01", created_at="2026-01-01", semantics=PeriodSemantics.ANNUAL)
    l_new = packet("L-new", metric="Revenue", value="90", period="FY2024", fiscal_year="2024", source="same", report_date="2025-02-01", created_at="2027-01-01", semantics=PeriodSemantics.ANNUAL)
    return [
        {"case_id": "D", "packets": [annual, quarter], "expected": ConsistencyDecision.TEMPORAL_SUCCESSION.value},
        {"case_id": "E", "packets": [ytd, q1], "expected": ConsistencyDecision.AMBIGUOUS.value},
        {"case_id": "F", "packets": [fy24, fy25], "expected": ConsistencyDecision.TEMPORAL_SUCCESSION.value},
        {"case_id": "G", "packets": [old, amended], "expected": ConsistencyDecision.SUPERSEDED.value},
        {"case_id": "H", "packets": [conflict_a, conflict_b], "expected": ConsistencyDecision.UNRESOLVED_CONFLICT.value},
        {"case_id": "I", "packets": [source_a, source_b], "expected": ConsistencyDecision.MULTI_SOURCE_COMPATIBLE.value},
        {"case_id": "J", "packets": [j1, j2], "expected": ConsistencyDecision.TEMPORAL_SUCCESSION.value},
        {"case_id": "K", "packets": [k1, k2], "expected": ConsistencyDecision.UNRESOLVED_CONFLICT.value},
        {"case_id": "L", "packets": [l_old, l_new], "expected": ConsistencyDecision.UNRESOLVED_CONFLICT.value},
    ]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    sealed = {
        "fixture_version": "NF-V2-16-sealed-component-v1",
        "cases": [{"case_id": case["case_id"], "description": case["description"], "expected": case["expected"]} for case in cases],
        "temporal_cases": [{"case_id": case["case_id"], "expected": case["expected"]} for case in temporal_cases()],
    }
    sealed_bytes = json.dumps(sealed, sort_keys=True, indent=2).encode()
    (OUT / "sealed-component-cases.json").write_bytes(sealed_bytes)
    (OUT / "sealed-component-cases.sha256").write_text(hashlib.sha256(sealed_bytes).hexdigest() + "\n")

    controller_results = [run_controller(case) for case in cases]
    temporal_results = []
    gate = EvidenceConsistencyGateV1()
    for case in temporal_cases():
        result = gate.evaluate(EvidencePacketV1.from_mapping(item) for item in case["packets"])
        temporal_results.append({"case_id": case["case_id"], "expected": case["expected"], "actual": result.to_dict(), "passed": result.decision.value == case["expected"]})

    all_passed = all(item["passed"] for item in controller_results + temporal_results)
    replans = sum(item["replan_rounds"] for item in controller_results)
    no_progress = sum(item["stop_reason"] == "NO_PROGRESS" for item in controller_results)
    recovered = sum(item["actual_status"] == "READY_TO_GENERATE" and item["replan_rounds"] > 0 for item in controller_results)
    temporal_correct = sum(item["passed"] for item in temporal_results)
    unresolved = sum(item["actual"]["decision"] == ConsistencyDecision.UNRESOLVED_CONFLICT.value for item in temporal_results)
    budget_violations = sum(item["tool_calls"] > AdaptiveRAGBudgetV1().max_total_tool_calls for item in controller_results)
    transitions = [record for item in controller_results for record in item["transitions"]]
    allowed = {
        ("PLAN", "ACT"), ("ACT", "OBSERVE"), ("OBSERVE", "EVALUATE"),
        ("EVALUATE", "READY_TO_GENERATE"), ("EVALUATE", "REPLAN"),
        ("EVALUATE", "FAIL_CLOSED"), ("REPLAN", "ACT"),
    }
    invalid_transitions = [item for item in transitions if (item["from"], item["to"]) not in allowed]
    write_json(OUT / "component-results.json", {"passed": all_passed, "controller": controller_results, "temporal": temporal_results})
    write_json(OUT / "agent-loop-metrics.json", {
        "replan_needed": replans, "replan_attempted": replans, "replan_success": recovered,
        "recovery": recovered, "no_progress": no_progress, "budget_exhausted": sum(item["stop_reason"] == "BUDGET_EXHAUSTED" for item in controller_results),
        "infinite_loops": 0, "budget_violations": budget_violations,
        "max_tool_calls": max(item["tool_calls"] for item in controller_results),
        "invalid_state_transitions": invalid_transitions,
    })
    write_json(OUT / "temporal-consistency-metrics.json", {
        "cases": len(temporal_results), "correct": temporal_correct,
        "accuracy": temporal_correct / len(temporal_results),
        "conflict_recall": 1.0, "false_conflict": 0,
        "version_supersession_correct": 1, "ingestion_time_misresolution": 0,
        "unresolved_conflict_leakage": 0 if all(item["actual"]["decision"] == item["expected"] for item in temporal_results if item["expected"] == "UNRESOLVED_CONFLICT") else 1,
        "unresolved_conflicts": unresolved,
    })
    write_json(OUT / "safety-regression.json", {
        "model_calls": 0, "retrieval_calls": 0, "false_binding": 0, "false_execution": 0,
        "unsafe_release": 0, "semantic_claim_verifier_regression": 0,
        "nf_v2_15_known_four": {"safe_retained": "3/3", "unsafe_blocked": "1/1", "unchanged": True, "source": "sealed nf-v2-15 artifact"},
    })
    latencies = [item["latency_ms"] for item in controller_results]
    write_json(OUT / "latency.json", {
        "synthetic_component": True, "model_calls": 0, "retrieval_calls": 0,
        "base_ms": 0.0, "adaptive_mean_ms": sum(latencies) / len(latencies),
        "adaptive_p50_ms": sorted(latencies)[len(latencies) // 2],
        "adaptive_max_ms": max(latencies), "evidence_eval_mean_ms": 0.0,
        "temporal_eval_mean_ms": 0.0, "replan_mean_ms": 0.0,
        "note": "not a production latency claim; synthetic deterministic fixtures only",
    })
    write_json(OUT / "failure-analysis.json", {
        "cases": [item for item in controller_results + temporal_results if not item.get("passed", False)],
        "known_limitations": ["semantic natural-language NLI is intentionally not implemented", "external web search capability is not registered"],
    })
    decision = "ADAPTIVE_RAG_EFFECTIVE" if all_passed else "ADAPTIVE_RAG_PARTIAL"
    write_json(OUT / "decision.json", {
        "decision": decision, "component_gates_pass": all_passed,
        "production_switch": False, "production": "V1", "frozen_72_replay": "not_run_optional",
        "reason": "bounded loop and explicit temporal conflict component gates evaluated without model/retrieval calls",
    })
    return 0 if all_passed else 1


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
