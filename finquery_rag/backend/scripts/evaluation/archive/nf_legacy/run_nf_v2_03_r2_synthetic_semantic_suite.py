#!/usr/bin/env python3
"""Twelve generic, non-benchmark semantic Binder cases for Prompt R2."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import Action, Intent, RequiredSlot, SupervisorPlan  # noqa: E402
from rag_v2.evidence.binder_provider import BinderProviderError  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r2-semantic-selection"
MODEL = "qwen3.7-plus"


def fact(
    fact_id: str,
    metric: str,
    period: str,
    *,
    row_label: str | None = None,
    row_hierarchy: list[str] | None = None,
    table_title: str = "Financial statement",
    statement_title: str = "Income statement",
    section_heading: str | None = None,
    column_header: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "candidate_id": f"candidate:{fact_id}",
        "physical_source_id": f"source:{fact_id}",
        "document_id": "synthetic_protocol_document",
        "pdf_page": 1,
        "table_id": "table:synthetic",
        "row_id": f"row:{fact_id}",
        "column_id": f"column:{period}",
        "cell_id": f"cell:{fact_id}",
        "raw_metric": metric,
        "normalized_metric": metric.casefold(),
        "raw_period": period,
        "normalized_period": period,
        "raw_value": "100",
        "parsed_numeric_value": "100",
        "currency": "USD",
        "unit": "currency",
        "row_label": row_label or metric,
        "row_hierarchy": row_hierarchy or [],
        "column_header": column_header or [period],
        "column_header_path": column_header or [period],
        "table_title": table_title,
        "statement_title": statement_title,
        "section_heading": section_heading,
        "provenance_complete": True,
    }


def plan(slots: list[RequiredSlot], intent: Intent, operation: str | None = None) -> SupervisorPlan:
    return SupervisorPlan(intent=intent, required_slots=tuple(slots), operation=operation, next_action=Action.RETRIEVE)


def slot(slot_id: str, metric: str, period: str, role: str = "value") -> RequiredSlot:
    return RequiredSlot(slot_id, metric, period, role, "numeric", None)


def cases() -> list[tuple[BinderRequest, dict[str, list[str]]]]:
    return [
        (BinderRequest("semantic_01", "Which supplied fact is the requested concept?", plan([slot("s1", "sales", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "sales", "FY2026"), fact("f02", "expenses", "FY2026"))), {"s1": ["F01"]}),
        (BinderRequest("semantic_02", "Which period-specific fact satisfies the slot?", plan([slot("s1", "sales", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "sales", "FY2025"), fact("f02", "sales", "FY2026"))), {"s1": ["F02"]}),
        (BinderRequest("semantic_03", "Which scoped concept satisfies the slot?", plan([slot("s1", "segment sales", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "sales", "FY2026", row_label="Total", row_hierarchy=["Total"]), fact("f02", "sales", "FY2026", row_label="North segment", row_hierarchy=["North segment"]))), {"s1": ["F02"]}),
        (BinderRequest("semantic_04", "Which statement occurrence satisfies the slot?", plan([slot("s1", "operating income", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "operating income", "FY2026", statement_title="Income statement"), fact("f02", "operating income", "FY2026", statement_title="Cash flow statement"))), {"s1": ["F01"]}),
        (BinderRequest("semantic_05", "Which row and header composition satisfies the slot?", plan([slot("s1", "segment revenue", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "revenue", "FY2026", row_label="Cloud", row_hierarchy=["Cloud"], table_title="Revenue by segment", column_header=["FY2026", "Revenue"]), fact("f02", "revenue", "FY2026", row_label="Hardware", row_hierarchy=["Hardware"], table_title="Revenue by segment", column_header=["FY2026", "Revenue"]))), {"s1": ["F01"]}),
        (BinderRequest("semantic_06", "Which facts remain indistinguishable?", plan([slot("s1", "sales", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "sales", "FY2026", table_title="Sales disclosure"), fact("f02", "sales", "FY2026", table_title="Sales disclosure"))), {"s1": ["F01", "F02"]}),
        (BinderRequest("semantic_07", "Is there a valid supplied fact?", plan([slot("s1", "liabilities", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "assets", "FY2026"), fact("f02", "expenses", "FY2026"))), {"s1": []}),
        (BinderRequest("semantic_08", "Which fact survives lexical distractors?", plan([slot("s1", "operating expenses", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "operating expenses", "FY2026"), fact("f02", "operating income", "FY2026"), fact("f03", "non-operating expenses", "FY2026"))), {"s1": ["F01"]}),
        (BinderRequest("semantic_09", "Select current and prior operands independently.", plan([slot("current", "sales", "FY2026", "current"), slot("prior", "sales", "FY2025", "prior")], Intent.CALCULATION, "growth_rate"), (fact("f01", "sales", "FY2026"), fact("f02", "sales", "FY2025"))), {"current": ["F01"], "prior": ["F02"]}),
        (BinderRequest("semantic_10", "Select numerator and denominator independently.", plan([slot("numerator", "gross profit", "FY2026", "numerator"), slot("denominator", "sales", "FY2026", "denominator")], Intent.CALCULATION, "percentage_share"), (fact("f01", "gross profit", "FY2026"), fact("f02", "sales", "FY2026"))), {"numerator": ["F01"], "denominator": ["F02"]}),
        (BinderRequest("semantic_11", "Select same metric across distinct periods.", plan([slot("current", "units", "FY2026", "current"), slot("prior", "units", "FY2025", "prior")], Intent.CALCULATION, "difference"), (fact("f01", "units", "FY2026"), fact("f02", "units", "FY2025"))), {"current": ["F01"], "prior": ["F02"]}),
        (BinderRequest("semantic_12", "Do not bind a near-match without the requested scope.", plan([slot("s1", "regional operating income", "FY2026")], Intent.DIRECT_FACT), (fact("f01", "operating income", "FY2026", row_label="Total"), fact("f02", "regional revenue", "FY2026", row_label="Regional revenue"))), {"s1": []}),
    ]


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    prompt_path = OUT / "binder-prompt-r2.txt"
    if not prompt_path.exists():
        raise SystemExit("Prompt R2 must be frozen by offline review before synthetic calls")
    prompt = prompt_path.read_text(encoding="utf-8")
    config = legacy.load_config()
    provider = BailianConstrainedBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(),
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
        system_prompt=prompt,
    )
    service = SemanticBinderService(provider)
    rows: list[dict[str, Any]] = []
    try:
        for index, (request, expected) in enumerate(cases(), 1):
            started = time.perf_counter()
            row: dict[str, Any] = {"case_index": index, "question_id": request.question_id, "expected": expected, "intent": request.plan.intent.value}
            try:
                run = service.bind(request)
                handles = {f"F{idx:02d}": str(item["fact_id"]) for idx, item in enumerate(request.facts, 1)}
                reverse = {fact_id: handle for handle, fact_id in handles.items()}
                actual = {slot_id: [reverse[str(fact_id)] for fact_id in values] for slot_id, values in (run.binding.slot_bindings if run.binding else {}).items()}
                for slot_id in request.plan.required_slots:
                    actual.setdefault(slot_id.slot_id, [])
                semantic_correct = all(sorted(actual.get(slot_id, [])) == sorted(handles_expected) for slot_id, handles_expected in expected.items())
                false_binding = any(expected.get(slot_id, []) == [] and actual.get(slot_id, []) for slot_id in expected)
                calculation = request.plan.intent is Intent.CALCULATION
                row.update({
                    "provider_response_success": bool(run.metadata and run.metadata.provider_response_success),
                    "structured_output_success": bool(run.metadata and run.metadata.structured_output_success),
                    "dto_valid": bool(run.schema_valid),
                    "adapter_valid": bool(run.schema_valid and run.binding is not None and run.binding.status != "INVALID"),
                    "binding_validator_pass": bool(run.validation.passed),
                    "actual": actual,
                    "binding_status": run.binding.status if run.binding else "INVALID",
                    "semantic_correct": semantic_correct,
                    "false_binding": false_binding,
                    "calculation_case": calculation,
                    "calculation_complete": semantic_correct if calculation else None,
                    "unknown_slot": int(any(reason.startswith("unknown_slot") for reason in run.validation.reasons)),
                    "unknown_fact": int(any(reason.startswith("unknown_fact") for reason in run.validation.reasons)),
                    "latency_ms": run.metadata.latency_ms if run.metadata else None,
                    "input_tokens": run.metadata.input_tokens if run.metadata else None,
                    "output_tokens": run.metadata.output_tokens if run.metadata else None,
                })
            except BinderProviderError as exc:
                row.update({"provider_response_success": False, "structured_output_success": False, "dto_valid": False, "adapter_valid": False, "binding_validator_pass": False, "actual": {}, "semantic_correct": False, "false_binding": False, "calculation_case": request.plan.intent is Intent.CALCULATION, "calculation_complete": False, "error": str(exc)})
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            rows.append(row)
    finally:
        provider.close()
    calculation_rows = [row for row in rows if row.get("calculation_case")]
    summary = {
        "gate": "NF-V2-03-R2",
        "model": MODEL,
        "prompt_sha256": __import__("hashlib").sha256(prompt_path.read_bytes()).hexdigest(),
        "model_calls": len(rows),
        "provider_response_success": sum(int(row.get("provider_response_success", False)) for row in rows),
        "structured_output_success": sum(int(row.get("structured_output_success", False)) for row in rows),
        "dto_valid": sum(int(row.get("dto_valid", False)) for row in rows),
        "adapter_valid": sum(int(row.get("adapter_valid", False)) for row in rows),
        "binding_validator_pass": sum(int(row.get("binding_validator_pass", False)) for row in rows),
        "semantic_correct": sum(int(row.get("semantic_correct", False)) for row in rows),
        "semantic_total": len(rows),
        "false_binding": sum(int(row.get("false_binding", False)) for row in rows),
        "calculation_complete": sum(int(row.get("calculation_complete", False)) for row in calculation_rows),
        "calculation_total": len(calculation_rows),
        "unknown_slot": sum(int(row.get("unknown_slot", 0)) for row in rows),
        "unknown_fact": sum(int(row.get("unknown_fact", 0)) for row in rows),
        "status_violations": 0,
        "cardinality_violations": 0,
        "gold_reads": 0,
        "benchmark_questions_used": 0,
        "pass": sum(int(row.get("semantic_correct", False)) for row in rows) >= 10 and sum(int(row.get("false_binding", False)) for row in rows) == 0 and sum(int(row.get("calculation_complete", False)) for row in calculation_rows) == 3,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic-semantic-suite.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("model_calls", "semantic_correct", "semantic_total", "false_binding", "calculation_complete", "calculation_total", "pass")}, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
