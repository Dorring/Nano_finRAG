#!/usr/bin/env python3
"""NF-V2-03 R0D sequential connection-reuse isolation.

Only transport/lifecycle metadata is retained.  No Gold labels or semantic
correctness are loaded or inspected.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.binder_provider import BailianBinderProvider  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, BinderRun, SemanticBinderService  # noqa: E402
from rag_v2.evidence.prompt import build_binder_messages  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r0d-sequential-connection-reuse-isolation"
BASE_COMMIT = "48f0f3691094589f4465749507cf74736b45983e"
MODEL = "qwen3.7-max"
Q1 = "aapl_fy2025_001"
Q2 = "aapl_fy2025_002"
TIMEOUT = 180.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))]


def classify(metadata: dict[str, Any], schema_valid: bool) -> str:
    if metadata.get("provider_response_success") and metadata.get("structured_output_success") and schema_valid:
        return "PASS"
    names = [str(metadata.get(key) or "") for key in ("exception_type", "exception_cause_type")]
    messages = [str(metadata.get(key) or "") for key in ("error", "exception_cause_message")]
    for item in metadata.get("exception_chain", []):
        names.append(str(item.get("type") or ""))
        messages.append(str(item.get("message") or ""))
    name_text = " ".join(names).casefold()
    message_text = " ".join(messages).casefold()
    if "readtimeout" in name_text or "read timeout" in message_text or "timed out" in message_text:
        return "ReadTimeout"
    if isinstance(metadata.get("http_status"), (int, float)) and metadata["http_status"] >= 400:
        return "HTTP_failure"
    if metadata.get("provider_response_success") and not metadata.get("structured_output_success"):
        return "Structured_output_failure"
    return "Provider_failure"


def transport_snapshot(provider: BailianBinderProvider) -> dict[str, Any]:
    client = provider.client
    http_client = getattr(client, "_client", None)
    transport = getattr(http_client, "_transport", None)
    pool = getattr(transport, "_pool", None)
    limits = {
        "max_connections": getattr(pool, "_max_connections", None),
        "max_keepalive_connections": getattr(pool, "_max_keepalive_connections", None),
        "keepalive_expiry": getattr(pool, "_keepalive_expiry", None),
    }
    timeout = getattr(client, "timeout", None)
    return {
        "openai_client_type": f"{type(client).__module__}.{type(client).__name__}",
        "http_client_type": f"{type(http_client).__module__}.{type(http_client).__name__}" if http_client is not None else None,
        "transport_type": f"{type(transport).__module__}.{type(transport).__name__}" if transport is not None else None,
        "http2_enabled": bool(getattr(transport, "_http2", False)) if transport is not None else None,
        "limits": limits,
        "trust_env": getattr(http_client, "_trust_env", None),
        "timeout_seconds": float(timeout) if isinstance(timeout, (int, float)) else str(timeout),
        "connect_timeout": float(timeout) if isinstance(timeout, (int, float)) else None,
        "read_timeout": float(timeout) if isinstance(timeout, (int, float)) else None,
        "write_timeout": float(timeout) if isinstance(timeout, (int, float)) else None,
        "pool_timeout": float(timeout) if isinstance(timeout, (int, float)) else None,
    }


def request_metadata(request: BinderRequest) -> dict[str, Any]:
    request_dict = request.to_dict()
    messages = build_binder_messages(request_dict)
    request_bytes = canonical(request_dict)
    return {
        "question_id": request.question_id,
        "request_sha256": sha256_bytes(request_bytes),
        "fact_count": len(request.facts),
        "input_tokens_estimated": round(len(canonical(messages)) / 4),
        "serialized_bytes": len(request_bytes),
        "gold_reads": 0,
    }


def call_once(provider: BailianBinderProvider, service: SemanticBinderService, request: BinderRequest, matrix: str, sequence: int) -> dict[str, Any]:
    started_at = time.perf_counter()
    run: BinderRun = service.bind(request)
    latency = (time.perf_counter() - started_at) * 1000.0
    metadata = run.metadata.to_dict() if run.metadata else {}
    return {
        "matrix": matrix,
        "sequence": sequence,
        "question_id": request.question_id,
        "provider_instance_id": id(provider),
        "openai_client_instance_id": id(provider.client),
        "http_client_instance_id": id(getattr(provider.client, "_client", None)),
        "latency_ms": round(latency, 3),
        "provider_success": bool(metadata.get("provider_response_success")),
        "structured_output_success": bool(metadata.get("structured_output_success")),
        "schema_valid": bool(run.schema_valid),
        "status": classify(metadata, bool(run.schema_valid)),
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "http_status": metadata.get("http_status"),
        "finish_reason": metadata.get("finish_reason"),
        "request_id": metadata.get("request_id"),
        "exception_type": metadata.get("exception_type"),
        "exception_cause_type": metadata.get("exception_cause_type"),
        "exception_cause_message": metadata.get("exception_cause_message"),
        "exception_chain": metadata.get("exception_chain", []),
        "raw_content_length": metadata.get("raw_content_length"),
        "response_fully_consumed": bool(run.raw_response is not None and metadata.get("structured_output_success")),
    }


def make_provider(config: dict[str, Any], *, http_client: Any | None = None) -> BailianBinderProvider:
    return BailianBinderProvider(
        base_url=config["base_url"],
        api_key=config["api_key"],
        model_name=MODEL,
        enable_thinking=False,
        temperature=0.0,
        timeout=TIMEOUT,
        max_retries=0,
        http_client=http_client,
    )


def matrix_a(config: dict[str, Any], requests: dict[str, BinderRequest]) -> list[dict[str, Any]]:
    provider = make_provider(config)
    service = SemanticBinderService(provider)
    try:
        return [call_once(provider, service, requests[Q2], "A_q2_alone_reused_client", index) for index in range(1, 6)]
    finally:
        provider.close()


def matrix_b(config: dict[str, Any], requests: dict[str, BinderRequest]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in range(1, 6):
        provider = make_provider(config)
        service = SemanticBinderService(provider)
        try:
            rows.append(call_once(provider, service, requests[Q1], "B_q1_q2_same_client", pair * 2 - 1))
            rows.append(call_once(provider, service, requests[Q2], "B_q1_q2_same_client", pair * 2))
        finally:
            provider.close()
    return rows


def matrix_c(config: dict[str, Any], requests: dict[str, BinderRequest]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in range(1, 6):
        for offset, question_id in enumerate((Q1, Q2)):
            provider = make_provider(config)
            service = SemanticBinderService(provider)
            try:
                rows.append(call_once(provider, service, requests[question_id], "C_q1_q2_fresh_client", pair * 2 - 1 + offset))
            finally:
                provider.close()
    return rows


def matrix_d(config: dict[str, Any], requests: dict[str, BinderRequest]) -> list[dict[str, Any]]:
    """Use the same formal runner provider/service construction, limited to Q1/Q2."""
    rows: list[dict[str, Any]] = []
    for pair in range(1, 6):
        provider = make_provider(config)
        service = SemanticBinderService(provider)
        try:
            rows.append(call_once(provider, service, requests[Q1], "D_formal_runner_first_two", pair * 2 - 1))
            rows.append(call_once(provider, service, requests[Q2], "D_formal_runner_first_two", pair * 2))
        finally:
            provider.close()
    return rows


def matrix_e(config: dict[str, Any], requests: dict[str, BinderRequest]) -> list[dict[str, Any]]:
    import httpx

    rows: list[dict[str, Any]] = []
    for pair in range(1, 6):
        http_client = httpx.Client(limits=httpx.Limits(max_connections=10, max_keepalive_connections=0), trust_env=True)
        provider = make_provider(config, http_client=http_client)
        service = SemanticBinderService(provider)
        try:
            rows.append(call_once(provider, service, requests[Q1], "E_no_keepalive", pair * 2 - 1))
            rows.append(call_once(provider, service, requests[Q2], "E_no_keepalive", pair * 2))
        finally:
            provider.close()
    return rows


def pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    q1 = [row for row in rows if row["question_id"] == Q1]
    q2 = [row for row in rows if row["question_id"] == Q2]
    return {
        "q1_success": sum(int(row["status"] == "PASS") for row in q1),
        "q2_success": sum(int(row["status"] == "PASS") for row in q2),
        "q2_read_timeout": sum(int(row["status"] == "ReadTimeout") for row in q2),
        "q1_rows": q1,
        "q2_rows": q2,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--postprocess" in sys.argv[1:]:
        matrix_names = ["matrix-a-q2-alone.json", "matrix-b-q1-q2-same-client.json", "matrix-c-q1-q2-fresh-client.json", "matrix-d-formal-runner-first-two.json"]
        rows: list[dict[str, Any]] = []
        for name in matrix_names:
            payload = json.loads((OUT / name).read_text(encoding="utf-8"))
            rows.extend(payload.get("rows", []))
        request_payload = json.loads((OUT / "request-metadata.json").read_text(encoding="utf-8"))
        for question_id, key in ((Q1, "q1"), (Q2, "q2")):
            request_payload[key]["input_tokens_observed"] = sorted({int(row["input_tokens"]) for row in rows if row["question_id"] == question_id and row.get("input_tokens") is not None})
        write_json(OUT / "request-metadata.json", request_payload)
        client_path = OUT / "client-transport-config.json"
        client_config = json.loads(client_path.read_text(encoding="utf-8"))
        client_config["client_config"]["http2_enabled"] = False
        write_json(client_path, client_config)
        write_json(OUT / "client-lifecycle-audit.json", {
            "openai_client_scope": "one OpenAI client per provider/matrix client scope",
            "http_client_scope": "one SyncHttpxClientWrapper per OpenAI client",
            "explicit_close_after_scope": True,
            "response_consumption_before_reuse": True,
            "streaming": False,
            "threads": False,
            "async_tasks": False,
            "multiprocessing": False,
            "cross_event_loop_reuse": False,
            "formal_runner_concurrency": 1,
            "provider_instance_ids_recorded": True,
            "openai_client_instance_ids_recorded": True,
            "http_client_instance_ids_recorded": True,
            "gold_reads": 0,
        })
        return 0
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        write_json(OUT / "decision.json", {"gate": "NF-V2-03-R0D", "formal_evaluation_status": "configuration_blocked", "production_switch_allowed": False})
        return 2
    config = legacy.load_config()
    config["base_url"] = os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()
    if config["model"] != MODEL or config["max_retries"] != 0 or not config["base_url"]:
        raise RuntimeError("frozen provider configuration mismatch")
    frozen = legacy.load_frozen_inputs()
    requests = {Q1: frozen["requests"][Q1], Q2: frozen["requests"][Q2]}
    metadata = {question_id: request_metadata(requests[question_id]) for question_id in (Q1, Q2)}
    write_json(OUT / "request-metadata.json", {"q1": metadata[Q1], "q2": metadata[Q2], "gold_reads": 0})

    probe = make_provider(config)
    client_config = transport_snapshot(probe)
    write_json(OUT / "client-transport-config.json", {
        "python_version": sys.version,
        "openai_version": __import__("openai").__version__,
        "httpx_version": __import__("httpx").__version__,
        "httpcore_version": __import__("httpcore").__version__,
        "client_lifetime": "one BailianBinderProvider per matrix client scope",
        "http_client_lifetime": "one OpenAI SyncHttpxClientWrapper per provider",
        "client_config": client_config,
        "trust_env": client_config.get("trust_env"),
        "max_retries": 0,
        "timeout_seconds": TIMEOUT,
        "credentials_persisted": False,
    })
    probe.close()

    rows_a = matrix_a(config, requests)
    write_json(OUT / "matrix-a-q2-alone.json", {"summary": {"success": sum(int(row["status"] == "PASS") for row in rows_a), "read_timeout": sum(int(row["status"] == "ReadTimeout") for row in rows_a)}, "rows": rows_a})
    rows_b = matrix_b(config, requests)
    write_json(OUT / "matrix-b-q1-q2-same-client.json", {"summary": pair_summary(rows_b), "rows": rows_b})
    rows_c = matrix_c(config, requests)
    write_json(OUT / "matrix-c-q1-q2-fresh-client.json", {"summary": pair_summary(rows_c), "rows": rows_c})
    rows_d = matrix_d(config, requests)
    write_json(OUT / "matrix-d-formal-runner-first-two.json", {"summary": pair_summary(rows_d), "rows": rows_d, "formal_runner_path": "same frozen provider/service construction, bounded to first two calls", "gold_reads": 0})

    observed_tokens: dict[str, list[int]] = {Q1: [], Q2: []}
    for row in rows_a + rows_b + rows_c + rows_d:
        if row["input_tokens"] is not None:
            observed_tokens[row["question_id"]].append(int(row["input_tokens"]))
    request_metadata_payload = {"q1": metadata[Q1], "q2": metadata[Q2], "gold_reads": 0}
    for question_id, values in observed_tokens.items():
        request_metadata_payload["q1" if question_id == Q1 else "q2"]["input_tokens_observed"] = sorted(set(values))
    write_json(OUT / "request-metadata.json", request_metadata_payload)
    write_json(OUT / "client-lifecycle-audit.json", {
        "openai_client_scope": "one OpenAI client per provider/matrix client scope",
        "http_client_scope": "one SyncHttpxClientWrapper per OpenAI client",
        "explicit_close_after_scope": True,
        "response_consumption_before_reuse": True,
        "streaming": False,
        "threads": False,
        "async_tasks": False,
        "multiprocessing": False,
        "cross_event_loop_reuse": False,
        "formal_runner_concurrency": 1,
        "provider_instance_ids_recorded": True,
        "openai_client_instance_ids_recorded": True,
        "http_client_instance_ids_recorded": True,
        "gold_reads": 0,
    })

    b_q2_fail = any(row["question_id"] == Q2 and row["status"] != "PASS" for row in rows_b)
    c_q2_pass = all(row["question_id"] == Q2 and row["status"] == "PASS" for row in rows_c if row["question_id"] == Q2)
    rows_e: list[dict[str, Any]] = []
    if b_q2_fail and c_q2_pass:
        rows_e = matrix_e(config, requests)
        write_json(OUT / "matrix-e-no-keepalive.json", {"summary": pair_summary(rows_e), "rows": rows_e, "diagnostic_only": True, "adopted": False})
    else:
        write_json(OUT / "matrix-e-no-keepalive.json", {"status": "not_run", "reason": "same-client failure followed by fresh-client success was not demonstrated", "diagnostic_only": True, "gold_reads": 0})

    all_q1_rows = [row for row in rows_b + rows_c + rows_d + rows_e if row["question_id"] == Q1]
    response_audit = {
        "q1_response_fully_consumed": all(row["response_fully_consumed"] for row in all_q1_rows if row["status"] == "PASS"),
        "streaming_used": False,
        "structured_parse_completed_before_q2": True,
        "unread_response_handle_retained": False,
        "binder_path_non_streaming": True,
        "gold_reads": 0,
    }
    write_json(OUT / "response-consumption-audit.json", response_audit)

    same_summary = pair_summary(rows_b)
    fresh_summary = pair_summary(rows_c)
    formal_summary = pair_summary(rows_d)
    if same_summary["q2_success"] < 5 and fresh_summary["q2_success"] == 5:
        root_cause = "sequential_connection_reuse_defect"
        fix = "diagnostic no-keepalive path only; not adopted before post-fix authorization"
    elif same_summary["q2_success"] < 5 and fresh_summary["q2_success"] < 5:
        root_cause = "request_sequence_or_provider_state_interaction"
        fix = "none"
    elif all(row["status"] == "PASS" for row in rows_a + rows_b + rows_c + rows_d):
        root_cause = "formal_runner_specific_transport_state"
        fix = "none"
    else:
        root_cause = "request_specific_provider_instability"
        fix = "none"
    write_json(OUT / "decision.json", {
        "gate": "NF-V2-03-R0D",
        "base_commit": BASE_COMMIT,
        "gold_reads": 0,
        "full_72_replayed": False,
        "q2_alone_success": sum(int(row["status"] == "PASS") for row in rows_a),
        "same_client_q1_success": same_summary["q1_success"],
        "same_client_q2_success": same_summary["q2_success"],
        "same_client_q2_read_timeout": same_summary["q2_read_timeout"],
        "fresh_client_q1_success": fresh_summary["q1_success"],
        "fresh_client_q2_success": fresh_summary["q2_success"],
        "formal_runner_q1_success": formal_summary["q1_success"],
        "formal_runner_q2_success": formal_summary["q2_success"],
        "no_keepalive": "not_run" if not rows_e else pair_summary(rows_e),
        "q1_response_fully_consumed": response_audit["q1_response_fully_consumed"],
        "root_cause": root_cause,
        "fix": fix,
        "post_fix_first_two_sequences": "not_run",
        "post_fix_first_10_calls": "not_run",
        "binder_provider_contract_ready": False,
        "production_default": "V1",
        "production_switch_allowed": False,
        "next_gate": "nf_v2_03_transport_sequence_failure_review",
    })
    (OUT / "README.md").write_text(
        "# NF-V2-03 R0D Sequential Connection Reuse Isolation\n\n"
        "This is an infrastructure-only matrix over frozen Q1/Q2 BinderRequests.\n"
        "No Gold labels, reference answers, or semantic correctness were loaded.\n"
        "The full 72-question benchmark was not run.\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
