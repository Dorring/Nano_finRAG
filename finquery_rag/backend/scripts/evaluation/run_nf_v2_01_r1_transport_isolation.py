#!/usr/bin/env python3
"""NF-V2-01 R1 synthetic transport isolation; never loads benchmark data."""
from __future__ import annotations

import json
import os
import platform
import statistics
import sys
import threading
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI

ROOT = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/pdf-sr-v2/finquery_rag/backend")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.supervisor.bailian_provider import BAILIAN_SUPERVISOR_RESPONSE_FORMAT, BailianProvider  # noqa: E402
from rag_v2.supervisor.prompt import build_messages  # noqa: E402
from rag_v2.supervisor.service import SupervisorService  # noqa: E402
from scripts.evaluation.run_nf_v2_01_r1_bailian_strong_general_supervisor import load_env_config, run_smoke, write_json  # noqa: E402

OUT = ROOT / "artifacts/evaluation/nf-v2-01-r1-transport-isolation"
QUESTION = "What was ExampleCorp's revenue in FY2025?"
N = 20
TIMEOUT = 180.0


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def safe_message(value: Any) -> str | None:
    return str(value)[:500] if value is not None else None


def chain_from_exception(exc: BaseException) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: BaseException | None = exc
    for _ in range(3):
        if current is None:
            break
        errno = getattr(current, "errno", None)
        chain.append({"type": type(current).__name__, "message": safe_message(current), "errno": errno if isinstance(errno, (int, str)) else None})
        current = current.__cause__ or current.__context__
    return chain


def context_base(sequence: int) -> dict[str, Any]:
    return {"process_id": os.getpid(), "thread_id": threading.get_ident(), "async_or_sync": "sync", "event_loop_id_if_async": None, "provider_instance_id": None, "openai_client_instance_id": None, "http_client_instance_id": None, "client_created_at": None, "client_closed_before_call": None, "client_closed_after_call": None, "call_sequence_number": sequence}


def http_context(client: Any, sequence: int) -> dict[str, Any]:
    return context_base(sequence) | {"openai_client_instance_id": id(client), "http_client_instance_id": id(client), "trust_env": getattr(client, "_trust_env", None), "custom_http_client": True, "client_closed_before_call": bool(getattr(client, "is_closed", False))}


def sdk_context(client: Any, sequence: int) -> dict[str, Any]:
    http_client = getattr(client, "_client", None)
    return context_base(sequence) | {"openai_client_instance_id": id(client), "http_client_instance_id": id(http_client) if http_client is not None else None, "trust_env": getattr(http_client, "_trust_env", None), "custom_http_client": http_client is not None, "client_closed_before_call": bool(getattr(http_client, "is_closed", False)) if http_client is not None else None}


def transport_environment() -> dict[str, Any]:
    return {"python_version": platform.python_version(), "openai_version": package_version("openai"), "httpx_version": package_version("httpx"), "httpcore_version": package_version("httpcore"), "HTTP_PROXY_present": bool(os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")), "HTTPS_PROXY_present": bool(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")), "ALL_PROXY_present": bool(os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")), "NO_PROXY_present": bool(os.environ.get("NO_PROXY") or os.environ.get("no_proxy")), "raw_httpx_trust_env": True, "openai_default_trust_env": True, "custom_http_client": False, "sdk_max_retries": 0}


def request_payload(config: dict[str, Any], *, structured: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": config["model"], "messages": build_messages(QUESTION), "temperature": 0.0}
    if structured:
        payload["response_format"] = BAILIAN_SUPERVISOR_RESPONSE_FORMAT
    return payload


def run_raw_httpx(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    headers = {"Authorization": f"Bearer {config['api_key']}"}
    with httpx.Client(base_url=config["base_url"], timeout=TIMEOUT, trust_env=True) as client:
        for sequence in range(1, N + 1):
            started = time.perf_counter()
            try:
                response = client.post("/chat/completions", headers=headers, json=request_payload(config, structured=False))
                records.append({"sequence": sequence, "success": 200 <= response.status_code < 300, "status_code": response.status_code, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": http_context(client, sequence), "exception_chain": []})
            except Exception as exc:
                records.append({"sequence": sequence, "success": False, "status_code": None, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": http_context(client, sequence), "exception_chain": chain_from_exception(exc)})
    return records


def run_openai_basic(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    client = OpenAI(base_url=config["base_url"], api_key=config["api_key"], timeout=TIMEOUT, max_retries=0)
    for sequence in range(1, N + 1):
        started = time.perf_counter()
        context = sdk_context(client, sequence)
        try:
            response = client.chat.completions.create(**request_payload(config, structured=False))
            records.append({"sequence": sequence, "success": bool(response.choices), "status_code": 200, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": context, "exception_chain": []})
        except Exception as exc:
            records.append({"sequence": sequence, "success": False, "status_code": getattr(exc, "status_code", None), "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": context, "exception_chain": chain_from_exception(exc)})
    client.close()
    return records


def run_provider(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    provider = BailianProvider(base_url=config["base_url"], api_key=config["api_key"], model_name=config["model"], enable_thinking=False, temperature=0.0, max_retries=0)
    service = SupervisorService(provider)
    for sequence in range(1, N + 1):
        started = time.perf_counter()
        context = provider.transport_context(call_sequence_number=sequence)
        try:
            run = service.plan(QUESTION)
            records.append({"sequence": sequence, "success": bool(run.plan_valid and run.metadata and run.metadata.structured_output_success), "status_code": 200 if run.metadata and run.metadata.provider_response_success else None, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": context | {"client_closed_after_call": provider.transport_context(call_sequence_number=sequence).get("client_closed_before_call")}, "exception_chain": provider.last_exception_chain, "plan_validator_pass": run.plan_valid})
        except Exception as exc:
            records.append({"sequence": sequence, "success": False, "status_code": getattr(exc, "status_code", None), "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": context, "exception_chain": chain_from_exception(exc), "plan_validator_pass": False})
    provider.close()
    return records


def run_formal_runner(config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sequence in range(1, N + 1):
        started = time.perf_counter()
        result = run_smoke(config)
        records.append({"sequence": sequence, "success": result.get("status") == "pass", "status_code": 200 if result.get("provider_response_success") else None, "latency_ms": round((time.perf_counter() - started) * 1000, 3), "context": result.get("runner_context") or context_base(sequence), "exception_chain": result.get("exception_chain", []), "plan_validator_pass": result.get("plan_validator_pass", False), "error": result.get("error")})
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    failures = [record for record in records if not record["success"]]
    return {"success": sum(record["success"] for record in records), "failure": len(failures), "calls": len(records), "latency_ms": {"average": statistics.mean(record["latency_ms"] for record in records) if records else None, "p50": statistics.median(record["latency_ms"] for record in records) if records else None, "max": max((record["latency_ms"] for record in records), default=None)}, "failure_types": sorted({level["type"] for record in failures for level in record.get("exception_chain", [])})}


def main() -> int:
    config, error = load_env_config()
    if error or config is None:
        raise SystemExit(error or "missing config")
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "transport-environment.json", transport_environment())
    results = {"A_raw_httpx": run_raw_httpx(config), "B_openai_sdk_basic": run_openai_basic(config), "C_bailian_provider": run_provider(config), "D_formal_runner": run_formal_runner(config)}
    write_json(OUT / "isolation-matrix.json", {name: {"summary": summarize(records), "records": records} for name, records in results.items()})
    write_json(OUT / "exception-chain.json", {name: [record for record in records if record.get("exception_chain")] for name, records in results.items()})
    write_json(OUT / "runner-context.json", {name: [record["context"] for record in records] for name, records in results.items()})
    write_json(OUT / "client-lifecycle-audit.json", {"formal_runner_provider_scope": "one BailianProvider per synthetic run_smoke call; one provider for formal 72 replay", "openai_client_scope": "one synchronous OpenAI client per BailianProvider", "explicit_close_before_diagnostic_fix": False, "sync_async": "synchronous", "threads": False, "async_tasks": False, "processes": False, "configured_concurrency": 1, "actual_concurrency": 1})
    write_json(OUT / "transport-fix.json", {"applied": "explicit SDK max_retries=0 and sanitized diagnostics", "semantic_behavior_changed": False})
    write_json(OUT / "stability-run-20.json", {"status": "pending_until_concrete_fix", "formal_runner": summarize(results["D_formal_runner"])})
    write_json(OUT / "stability-run-10.json", {"status": "not_run", "reason": "20-call matrix must identify a concrete fix first"})
    write_json(OUT / "transport-seal.json", {"provider": "bailian", "model": config["model"], "max_retries": 0, "formal_runner_transport_ready": False, "synthetic_runner_success": f"{sum(record['success'] for record in results['D_formal_runner'])}/20", "api_connection_errors": sum(any(level["type"] == "APIConnectionError" for level in record.get("exception_chain", [])) for record in results["D_formal_runner"]), "production_switch_allowed": False})
    write_json(OUT / "decision.json", {"gate": "NF-V2-01-R1-TRANSPORT", "base_commit": "b24133aab455207091ee7827487b056dd9aa3ea2", "model_calls": sum(len(records) for records in results.values()), "benchmark_loaded": False, "formal_72_run": False, "infrastructure_ready": False, "dominant_failure": "pending_matrix_interpretation", "next_gate": "manual_transport_fix_required"})
    (OUT / "README.md").write_text("# NF-V2-01 R1 Transport Isolation\n\nSynthetic-only, sequential transport matrix. No benchmark questions or labels were loaded.\n", encoding="utf-8")
    print(json.dumps({name: summarize(records) for name, records in results.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
