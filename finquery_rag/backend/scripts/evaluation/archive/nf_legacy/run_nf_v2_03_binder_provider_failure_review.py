#!/usr/bin/env python3
"""NF-V2-03 provider failure review using non-benchmark Binder packets only."""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.contracts.plan import SupervisorPlan  # noqa: E402
from rag_v2.evidence.binder_provider import BailianBinderProvider  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, SemanticBinderService  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-binder-provider-failure-review"
PACKET_SIZES = (14, 28, 36)
DEFAULT_TIMEOUT = 180.0


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan() -> SupervisorPlan:
    return SupervisorPlan.from_dict({
        "intent": "DIRECT_FACT",
        "required_slots": [{
            "slot_id": "slot_1",
            "metric": "revenue",
            "period": "FY2025",
            "role": "value",
            "value_type": "numeric",
            "unit": None,
        }],
        "operation": None,
        "next_action": "RETRIEVE",
    })


def fact(fact_id: str, metric: str, period: str) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "candidate_id": f"candidate:{fact_id}",
        "candidate_rank": 1,
        "physical_source_id": f"source:{fact_id}",
        "document_id": "examplecorp_fy2025",
        "pdf_page": 1,
        "statement_id": "statement:income",
        "logical_table_id": "logical-table:income",
        "table_id": "table:income",
        "row_id": f"row:{metric}",
        "column_id": f"column:{period}",
        "cell_id": f"cell:{fact_id}",
        "raw_metric": metric,
        "normalized_metric": metric,
        "raw_period": period,
        "normalized_period": period,
        "raw_value": "100",
        "parsed_numeric_value": "100",
        "raw_currency": "USD",
        "normalized_currency": "USD",
        "raw_scale": None,
        "normalized_scale": None,
        "currency": "USD",
        "unit": "currency",
        "provenance_complete": True,
    }


def packet(size: int) -> tuple[dict[str, Any], ...]:
    facts = [fact("F000", "revenue", "FY2025")]
    for index in range(1, size):
        period = "FY2024" if index % 2 else "FY2025"
        facts.append(fact(f"F{index:03d}", f"other metric {index}", period))
    return tuple(facts)


def classify(meta: dict[str, Any] | None, *, schema_valid: bool, raw_length: int) -> str:
    if meta is None:
        return "BT9_other"
    error = str(meta.get("error") or "").casefold()
    exception_type = str(meta.get("exception_type") or "")
    cause_type = str(meta.get("exception_cause_type") or "")
    status = meta.get("http_status")
    if isinstance(status, int) and status >= 400:
        return "BT4_HTTP_non_2xx"
    timeout_types = f"{exception_type} {cause_type}".casefold()
    if "connecttimeout" in timeout_types or "connect timeout" in error:
        return "BT1_connect_timeout"
    if "writetimeout" in timeout_types or "write timeout" in error:
        return "BT3_write_timeout"
    if "readtimeout" in timeout_types or "read timeout" in error:
        return "BT2_read_timeout"
    if "timeout" in error or "timed out" in error:
        return "BT2_read_timeout"
    if not meta.get("provider_response_success"):
        return "BT9_other"
    if not meta.get("structured_output_success"):
        if raw_length == 0:
            return "BT5_HTTP_2xx_empty_content"
        if "json" in error:
            return "BT6_HTTP_2xx_invalid_json"
        return "BT7_HTTP_2xx_schema_invalid"
    if not schema_valid:
        return "BT7_HTTP_2xx_schema_invalid"
    return "BT0_success"


def run_stress(*, repetitions: int, timeout: float, label: str) -> dict[str, Any]:
    provider_name = os.environ.get("V2_SUPERVISOR_PROVIDER", "")
    model = os.environ.get("V2_SUPERVISOR_MODEL", "")
    base_url = os.environ.get("V2_SUPERVISOR_BASE_URL", "")
    api_key = os.environ.get("V2_SUPERVISOR_API_KEY", "")
    if provider_name != "bailian" or not model or not base_url or not api_key:
        raise RuntimeError("Bailian provider environment is incomplete")
    provider = BailianBinderProvider(
        base_url=base_url.strip(),
        api_key=api_key.strip(),
        model_name=model.strip(),
        enable_thinking=False,
        temperature=0.0,
        timeout=timeout,
        max_retries=0,
    )
    service = SemanticBinderService(provider)
    rows: list[dict[str, Any]] = []
    call_index = 0
    try:
        for size in PACKET_SIZES:
            for repetition in range(1, repetitions + 1):
                call_index += 1
                request = BinderRequest(
                    f"synthetic_{label}_{size}_{repetition}",
                    "What was ExampleCorp's revenue in FY2025?",
                    plan(),
                    packet(size),
                )
                started = time.perf_counter()
                result = service.bind(request)
                elapsed = (time.perf_counter() - started) * 1000.0
                metadata = result.metadata.to_dict() if result.metadata else None
                raw_length = len(result.raw_response or provider.last_raw_response or "")
                rows.append({
                    "call_index": call_index,
                    "packet_size": size,
                    "repetition": repetition,
                    "facts_in_packet": size,
                    "input_tokens": metadata.get("input_tokens") if metadata else None,
                    "output_tokens": metadata.get("output_tokens") if metadata else None,
                    "total_tokens": metadata.get("total_tokens") if metadata else None,
                    "latency_ms": round(float(metadata.get("latency_ms", elapsed)) if metadata else elapsed, 3),
                    "provider_response_success": bool(metadata and metadata.get("provider_response_success")),
                    "structured_output_success": bool(metadata and metadata.get("structured_output_success")),
                    "schema_valid": bool(result.schema_valid),
                    "exception_type": metadata.get("exception_type") if metadata else None,
                    "exception_cause_type": metadata.get("exception_cause_type") if metadata else None,
                    "exception_cause_message": metadata.get("exception_cause_message") if metadata else None,
                    "errno": metadata.get("errno") if metadata else None,
                    "http_status": metadata.get("http_status") if metadata else None,
                    "finish_reason": metadata.get("finish_reason") if metadata else None,
                    "raw_content_length": raw_length,
                    "error": metadata.get("error") if metadata else "no_metadata",
                })
                rows[-1]["failure_class"] = classify(metadata, schema_valid=result.schema_valid, raw_length=raw_length)
    finally:
        provider.close()
    by_size: dict[str, Any] = {}
    for size in PACKET_SIZES:
        subset = [row for row in rows if row["packet_size"] == size]
        latencies = [float(row["latency_ms"]) for row in subset]
        successes = sum(row["failure_class"] == "BT0_success" for row in subset)
        ordered = sorted(latencies)
        by_size[str(size)] = {
            "calls": len(subset),
            "success": successes,
            "failure_counts": dict(Counter(row["failure_class"] for row in subset)),
            "p50_latency_ms": statistics.median(ordered) if ordered else 0.0,
            "p95_latency_ms": ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1))] if ordered else 0.0,
            "rows": subset,
        }
    return {"label": label, "model": model, "timeout_seconds": timeout, "max_retries": 0, "calls": len(rows), "by_size": by_size, "rows": rows}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    result = run_stress(repetitions=5, timeout=DEFAULT_TIMEOUT, label="baseline")
    write_json(OUT / "synthetic-stress-baseline.json", result)
    write_json(OUT / "stress-summary.json", {
        "calls": result["calls"],
        "success": sum(row["failure_class"] == "BT0_success" for row in result["rows"]),
        "failure_counts": dict(Counter(row["failure_class"] for row in result["rows"])),
        "by_size": {size: {key: value for key, value in data.items() if key != "rows"} for size, data in result["by_size"].items()},
    })
    print(json.dumps({"calls": result["calls"], "success": sum(row["failure_class"] == "BT0_success" for row in result["rows"]), "failure_counts": dict(Counter(row["failure_class"] for row in result["rows"]))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
