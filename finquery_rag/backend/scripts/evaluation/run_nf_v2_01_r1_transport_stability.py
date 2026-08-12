#!/usr/bin/env python3
"""Post-fix synthetic stability checks for the exact formal runner path."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation.run_nf_v2_01_r1_bailian_strong_general_supervisor import load_env_config, run_smoke, write_json  # noqa: E402

OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-transport-isolation"
QUESTIONS = [
    "What was ExampleCorp's revenue in FY2025?",
    "What was ExampleCorp's operating income in FY2024?",
    "What were ExampleCorp's total assets in FY2023?",
    "What was ExampleCorp's net sales in Q1 FY2025?",
    "What was ExampleCorp's gross profit in FY2025?",
]


def scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: scrub(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return value


def scrub_json(path: Path) -> None:
    if path.exists():
        path.write_text(json.dumps(scrub(json.loads(path.read_text(encoding="utf-8"))), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_many(config: dict[str, Any], count: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sequence in range(1, count + 1):
        started = time.perf_counter()
        result = run_smoke(config)
        records.append({
            "sequence": sequence,
            "success": result.get("status") == "pass",
            "provider_response_success": result.get("provider_response_success"),
            "structured_output_success": result.get("structured_output_success"),
            "schema_valid": result.get("schema_valid"),
            "plan_validator_pass": result.get("plan_validator_pass"),
            "answer_leakage": result.get("answer_leakage"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "context": result.get("runner_context"),
            "exception_chain": result.get("exception_chain", []),
            "error": result.get("error"),
        })
    return records


def run_additional(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sequence = 0
    for question in QUESTIONS:
        for repeat in range(1, 3):
            sequence += 1
            started = time.perf_counter()
            result = run_smoke(config, question=question, expected_metric=None, expected_period=None)
            records.append({
                "sequence": sequence,
                "question": question,
                "repeat": repeat,
                "success": result.get("status") == "pass",
                "provider_response_success": result.get("provider_response_success"),
                "structured_output_success": result.get("structured_output_success"),
                "schema_valid": result.get("schema_valid"),
                "plan_validator_pass": result.get("plan_validator_pass"),
                "answer_leakage": result.get("answer_leakage"),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
                "context": result.get("runner_context"),
                "exception_chain": result.get("exception_chain", []),
                "error": result.get("error"),
            })
    return records


def summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "calls": len(records),
        "success": sum(bool(record.get("success")) for record in records),
        "provider_response_success": sum(bool(record.get("provider_response_success")) for record in records),
        "structured_output_success": sum(bool(record.get("structured_output_success")) for record in records),
        "schema_valid": sum(bool(record.get("schema_valid")) for record in records),
        "plan_validator_pass": sum(bool(record.get("plan_validator_pass")) for record in records),
        "answer_leakage": sum(bool(record.get("answer_leakage")) for record in records),
        "api_connection_errors": sum(any(level.get("type") == "APIConnectionError" for level in record.get("exception_chain", [])) for record in records),
    }


def main() -> int:
    config, error = load_env_config()
    if error or config is None:
        raise SystemExit(error or "missing config")
    run20 = run_many(config, 20)
    run10 = run_additional(config)
    write_json(OUT / "stability-run-20.json", {"status": "completed", "summary": summary(run20), "records": run20})
    write_json(OUT / "stability-run-10.json", {"status": "completed", "summary": summary(run10), "records": run10})
    scrub_json(OUT / "isolation-matrix.json")
    scrub_json(OUT / "exception-chain.json")
    scrub_json(OUT / "runner-context.json")
    write_json(OUT / "client-lifecycle-audit.json", {
        "formal_runner_provider_scope": "one BailianProvider per synthetic run_smoke call; one provider for formal 72 replay",
        "openai_client_scope": "one synchronous OpenAI client per BailianProvider",
        "explicit_close": False,
        "sync_async": "synchronous",
        "threads": False,
        "async_tasks": False,
        "processes": False,
        "configured_concurrency": 1,
        "actual_concurrency": 1,
        "client_reuse_defect": False,
        "cross_event_loop_defect": False,
        "proxy_env_difference": False,
    })
    ready = summary(run20)["success"] == 20 and summary(run10)["success"] == 10 and summary(run20)["api_connection_errors"] == 0 and summary(run10)["api_connection_errors"] == 0
    write_json(OUT / "transport-fix.json", {
        "applied": "strip surrounding CRLF/whitespace from V2_SUPERVISOR_API_KEY at runner environment boundary",
        "observed_root_cause": "Windows SSH stdin injection left a trailing CR in the Authorization header",
        "semantic_behavior_changed": False,
        "sdk_max_retries": 0,
    })
    write_json(OUT / "transport-seal.json", {"provider": "bailian", "model": config["model"], "max_retries": 0, "formal_runner_transport_ready": ready, "synthetic_runner_success": f"{summary(run20)['success'] + summary(run10)['success']}/30", "api_connection_errors": summary(run20)["api_connection_errors"] + summary(run10)["api_connection_errors"], "production_switch_allowed": False})
    write_json(OUT / "decision.json", {"gate": "NF-V2-01-R1-TRANSPORT", "base_commit": "b24133aab455207091ee7827487b056dd9aa3ea2", "model_calls": 30, "benchmark_loaded": False, "formal_72_run": False, "infrastructure_ready": ready, "dominant_failure": None if ready else "runner_transport_instability", "next_gate": "resume_nf_v2_01_r1_formal_72_evaluation" if ready else "manual_transport_fix_required"})
    (OUT / "README.md").write_text("# NF-V2-01 R1 Transport Isolation\n\nThe first matrix isolated a CRLF-tainted Authorization header. The runner now strips surrounding environment transport whitespace; post-fix checks are synthetic-only and sequential.\n", encoding="utf-8")
    print(json.dumps({"run20": summary(run20), "run10": summary(run10), "infrastructure_ready": ready}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
