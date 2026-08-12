#!/usr/bin/env python3
"""Five non-benchmark qwen3.7-plus calls for R1B DTO protocol validation."""

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
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402
from rag_v2.evidence.binder_provider import BinderProviderError  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
MODEL = "qwen3.7-plus"


def fact(fact_id: str, metric: str, period: str, value: str = "100") -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "candidate_id": f"candidate:{fact_id}",
        "physical_source_id": f"source:{fact_id}",
        "document_id": "synthetic_non_benchmark",
        "pdf_page": 1,
        "statement_id": "income_statement",
        "table_id": "table:synthetic",
        "row_id": f"row:{metric}",
        "column_id": f"column:{period}",
        "cell_id": f"cell:{fact_id}",
        "raw_metric": metric,
        "normalized_metric": metric.casefold(),
        "raw_period": period,
        "normalized_period": period,
        "raw_value": value,
        "parsed_numeric_value": value,
        "raw_currency": "USD",
        "normalized_currency": "USD",
        "raw_scale": None,
        "normalized_scale": None,
        "currency": "USD",
        "unit": "currency",
        "provenance_complete": True,
    }


def plan(slots: list[RequiredSlot], intent: Intent, operation: str | None = None) -> SupervisorPlan:
    return SupervisorPlan(intent=intent, required_slots=tuple(slots), operation=operation, next_action=Action.RETRIEVE)


def cases() -> list[BinderRequest]:
    one = RequiredSlot("slot_1", "revenue", "FY2025", "value", "numeric", None)
    current = RequiredSlot("current", "revenue", "FY2025", "current", "numeric", None)
    prior = RequiredSlot("prior", "revenue", "FY2024", "prior", "numeric", None)
    first = RequiredSlot("first", "revenue", "FY2025", "value", "numeric", None)
    second = RequiredSlot("second", "operating income", "FY2025", "value", "numeric", None)
    return [
        BinderRequest("synthetic_r1b_bound", "What was ExampleCorp revenue in FY2025?", plan([one], Intent.DIRECT_FACT), (fact("fact_revenue_2025", "revenue", "FY2025"), fact("fact_revenue_2024", "revenue", "FY2024"))),
        BinderRequest("synthetic_r1b_missing", "What was ExampleCorp revenue in FY2025?", plan([one], Intent.DIRECT_FACT), (fact("fact_expenses_2025", "operating expenses", "FY2025"),)),
        BinderRequest("synthetic_r1b_ambiguous", "What was ExampleCorp revenue in FY2025?", plan([one], Intent.DIRECT_FACT), (fact("fact_revenue_a", "revenue", "FY2025"), fact("fact_revenue_b", "revenue", "FY2025"))),
        BinderRequest("synthetic_r1b_calculation", "What was revenue growth from FY2024 to FY2025?", plan([current, prior], Intent.CALCULATION, "growth_rate"), (fact("fact_revenue_2025", "revenue", "FY2025"), fact("fact_revenue_2024", "revenue", "FY2024"))),
        BinderRequest("synthetic_r1b_multi", "Compare revenue and operating income in FY2025.", plan([first, second], Intent.MULTI_EVIDENCE), (fact("fact_revenue_2025", "revenue", "FY2025"), fact("fact_operating_income_2025", "operating income", "FY2025"))),
    ]


def main() -> int:
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        raise SystemExit("V2_SUPERVISOR_MODEL must remain qwen3.7-plus")
    config = legacy.load_config()
    provider = BailianConstrainedBinderProvider(
        base_url=os.getenv("V2_SUPERVISOR_BASE_URL", "").strip(),
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=180.0,
        max_retries=0,
    )
    service = SemanticBinderService(provider)
    rows: list[dict[str, Any]] = []
    try:
        for index, request in enumerate(cases(), 1):
            started = time.perf_counter()
            row: dict[str, Any] = {"case_index": index, "question_id": request.question_id, "fact_count": len(request.facts), "expected_protocol_shape": request.question_id.replace("synthetic_r1b_", "")}
            try:
                run = service.bind(request)
                metadata = run.metadata.to_dict() if run.metadata else {}
                row.update({
                    "provider_response_success": bool(metadata.get("provider_response_success")),
                    "structured_output_success": bool(metadata.get("structured_output_success")),
                    "dto_valid": bool(run.schema_valid),
                    "adapter_valid": bool(run.binding is not None),
                    "binding_status": run.binding.status,
                    "binding_validator_pass": bool(run.validation.passed),
                    "binding_reasons": list(run.validation.reasons),
                    "unknown_slot": int(any(reason.startswith("unknown_slot") for reason in run.validation.reasons)),
                    "unknown_fact": int(any(reason.startswith("unknown_fact") for reason in run.validation.reasons)),
                    "calculation_leakage": int(any(token in (run.raw_response or "").casefold() for token in ("answer", "result", "calculation", "value"))),
                    "latency_ms": metadata.get("latency_ms"),
                    "input_tokens": metadata.get("input_tokens"),
                    "output_tokens": metadata.get("output_tokens"),
                })
            except BinderProviderError as exc:
                row.update({"provider_response_success": False, "structured_output_success": False, "dto_valid": False, "adapter_valid": False, "binding_validator_pass": False, "error": str(exc), "unknown_slot": 0, "unknown_fact": 0, "calculation_leakage": 0})
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            rows.append(row)
    finally:
        provider.close()
    summary = {
        "model": MODEL,
        "model_calls": len(rows),
        "structured_output_success": sum(int(row.get("structured_output_success", False)) for row in rows),
        "dto_valid": sum(int(row.get("dto_valid", False)) for row in rows),
        "adapter_valid": sum(int(row.get("adapter_valid", False)) for row in rows),
        "binding_validator_pass": sum(int(row.get("binding_validator_pass", False)) for row in rows),
        "unknown_slot": sum(row.get("unknown_slot", 0) for row in rows),
        "unknown_fact": sum(row.get("unknown_fact", 0) for row in rows),
        "calculation_leakage": sum(row.get("calculation_leakage", 0) for row in rows),
        "gold_reads": 0,
        "benchmark_questions_used": 0,
        "pass": all(row.get("provider_response_success") and row.get("structured_output_success") and row.get("dto_valid") and row.get("adapter_valid") and row.get("binding_validator_pass") for row in rows) and not any(row.get("unknown_slot") or row.get("unknown_fact") or row.get("calculation_leakage") for row in rows),
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic-protocol-test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
