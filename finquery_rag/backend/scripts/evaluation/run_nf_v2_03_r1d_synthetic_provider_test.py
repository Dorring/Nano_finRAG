#!/usr/bin/env python3
"""NF-V2-03 R1D provider-schema compatibility smoke test.

This is the only model evaluation performed before Attempt 6.  It reuses the
five non-benchmark protocol cases and writes a new R1D artifact; R1C artifacts
remain immutable references.
"""

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

from rag_v2.evidence.binder_provider import BinderProviderError  # noqa: E402
from rag_v2.evidence.binder_service import SemanticBinderService  # noqa: E402
from rag_v2.evidence.constrained_binder_provider import BailianConstrainedBinderProvider  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1b_synthetic_protocol as cases_module  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1d-supply-conditioned-binder"
MODEL = "qwen3.7-plus"


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
        for index, request in enumerate(cases_module.cases(), 1):
            started = time.perf_counter()
            row: dict[str, Any] = {
                "case_index": index,
                "question_id": request.question_id,
                "fact_count": len(request.facts),
                "expected_protocol_shape": request.question_id.replace("synthetic_r1b_", ""),
            }
            try:
                run = service.bind(request)
                metadata = run.metadata.to_dict() if run.metadata else {}
                error_text = str(metadata.get("error") or "")
                row.update(
                    {
                        "provider_response_success": bool(metadata.get("provider_response_success")),
                        "structured_output_success": bool(metadata.get("structured_output_success")),
                        "dto_valid": bool(run.schema_valid),
                        "adapter_valid": bool(run.schema_valid and run.binding is not None and run.binding.status != "INVALID"),
                        "binding_validator_pass": bool(run.validation.passed),
                        "binding_status": run.binding.status,
                        "binding_reasons": list(run.validation.reasons),
                        "unknown_slot": int(any(reason.startswith("unknown_slot") for reason in run.validation.reasons)),
                        "unknown_fact": int(any(reason.startswith("unknown_fact") for reason in run.validation.reasons)),
                        "duplicate_handle": int("duplicate_fact_handle" in error_text),
                        "status_violation": int("status_violation" in error_text),
                        "calculation_leakage": int(any(token in (run.raw_response or "").casefold() for token in ("answer", "result", "calculation", "value"))),
                        "http_400": int(metadata.get("http_status") == 400),
                        "latency_ms": metadata.get("latency_ms"),
                        "input_tokens": metadata.get("input_tokens"),
                        "output_tokens": metadata.get("output_tokens"),
                        "error": metadata.get("error"),
                    }
                )
            except BinderProviderError as exc:
                metadata = provider.last_call.to_dict() if provider.last_call else {}
                error_text = str(exc)
                row.update(
                    {
                        "provider_response_success": bool(metadata.get("provider_response_success")),
                        "structured_output_success": bool(metadata.get("structured_output_success")),
                        "dto_valid": bool(metadata.get("structured_output_success")),
                        "adapter_valid": False,
                        "binding_validator_pass": False,
                        "unknown_slot": 0,
                        "unknown_fact": 0,
                        "duplicate_handle": int("duplicate_fact_handle" in error_text),
                        "status_violation": int("status_violation" in error_text),
                        "calculation_leakage": 0,
                        "http_400": int(metadata.get("http_status") == 400),
                        "error": error_text,
                    }
                )
            row["elapsed_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
            rows.append(row)
    finally:
        provider.close()

    summary = {
        "gate": "NF-V2-03-R1D",
        "model": MODEL,
        "model_calls": len(rows),
        "provider_responses": sum(int(row.get("provider_response_success", False)) for row in rows),
        "structured_output": sum(int(row.get("structured_output_success", False)) for row in rows),
        "dto_valid": sum(int(row.get("dto_valid", False)) for row in rows),
        "adapter_valid": sum(int(row.get("adapter_valid", False)) for row in rows),
        "binding_validator_pass": sum(int(row.get("binding_validator_pass", False)) for row in rows),
        "unknown_slot": sum(int(row.get("unknown_slot", 0)) for row in rows),
        "unknown_fact": sum(int(row.get("unknown_fact", 0)) for row in rows),
        "duplicate_handle": sum(int(row.get("duplicate_handle", 0)) for row in rows),
        "status_violation": sum(int(row.get("status_violation", 0)) for row in rows),
        "calculation_leakage": sum(int(row.get("calculation_leakage", 0)) for row in rows),
        "http_400": sum(int(row.get("http_400", 0)) for row in rows),
        "gold_reads": 0,
        "benchmark_questions_used": 0,
        "pass": all(
            row.get("provider_response_success")
            and row.get("structured_output_success")
            and row.get("dto_valid")
            and row.get("adapter_valid")
            and row.get("binding_validator_pass")
            for row in rows
        )
        and not any(
            row.get(key)
            for row in rows
            for key in ("unknown_slot", "unknown_fact", "duplicate_handle", "status_violation", "calculation_leakage", "http_400")
        ),
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic-provider-test.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
