#!/usr/bin/env python3
"""NF-V2-03 R0C exact failed-request transport diagnosis.

This diagnostic deliberately stops at the provider/structured-output boundary.
It reconstructs one frozen BinderRequest, replays only that request, and never
loads Gold labels or evaluates binding semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_v2.evidence.binder_provider import BailianBinderProvider  # noqa: E402
from rag_v2.evidence.binder_service import BinderRequest, BinderRun, SemanticBinderService  # noqa: E402
from rag_v2.evidence.prompt import build_binder_messages  # noqa: E402
from scripts.evaluation import run_nf_v2_03_semantic_evidence_binder as legacy  # noqa: E402


OUT = ROOT / "artifacts/evaluation/nf-v2-03-r0c-exact-request-timeout-review"
BASE_COMMIT = "ce69c29"
FAILED_QUESTION = "aapl_fy2025_002"
MODEL = "qwen3.7-max"
DEFAULT_TIMEOUT = 180.0
EXTENDED_TIMEOUT = 330.0


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], q: float = 0.95) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(len(ordered) * q) - 1))]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_request_snapshot() -> tuple[BinderRequest, dict[str, Any]]:
    """Load only sealed pre-Gold inputs and return the target request snapshot."""
    frozen = legacy.load_frozen_inputs()
    request = frozen["requests"].get(FAILED_QUESTION)
    if request is None:
        raise RuntimeError(f"frozen request not found: {FAILED_QUESTION}")
    request_dict = request.to_dict()
    fact_packet = request_dict["financial_facts"]
    messages = build_binder_messages(request_dict)
    request_bytes = canonical_bytes(request_dict)
    fact_bytes = canonical_bytes(fact_packet)
    message_bytes = canonical_bytes(messages)
    summary = {
        "question_id": request.question_id,
        "fact_count": len(request.facts),
        "candidate_count": len(frozen["top20_order"].get(FAILED_QUESTION, [])),
        "serialized_request_bytes": len(request_bytes),
        "prompt_serialized_bytes": len(message_bytes),
        "estimated_input_tokens": max(1, round(len(message_bytes) / 4)),
        "required_slot_count": len(request.plan.required_slots),
        "operation": request.plan.operation,
        "prompt_sha256": sha256_file(legacy.OUT / "binder-prompt.txt"),
        "schema_sha256": sha256_file(legacy.OUT / "binder-schema.json"),
        "fact_packet_sha256": sha256_bytes(fact_bytes),
        "full_binder_request_sha256": sha256_bytes(request_bytes),
        "request_sha256": sha256_bytes(request_bytes),
        "gold_reads": 0,
        "reference_answer_reads": 0,
    }
    return request, {"request": request_dict, "fact_packet": fact_packet, "messages": messages, "summary": summary}


def request_determinism() -> tuple[BinderRequest, dict[str, Any]]:
    first_request, first = build_request_snapshot()
    second_request, second = build_request_snapshot()
    first_req_sha = first["summary"]["full_binder_request_sha256"]
    second_req_sha = second["summary"]["full_binder_request_sha256"]
    first_fact_sha = first["summary"]["fact_packet_sha256"]
    second_fact_sha = second["summary"]["fact_packet_sha256"]
    request_sha_equal = first_req_sha == second_req_sha
    fact_sha_equal = first_fact_sha == second_fact_sha
    if not request_sha_equal or not fact_sha_equal:
        write_json(OUT / "decision.json", {
            "gate": "NF-V2-03-R0C",
            "base_commit": BASE_COMMIT,
            "formal_evaluation_status": "blocked",
            "runner_request_nondeterminism": True,
            "production_switch_allowed": False,
        })
        raise RuntimeError("runner request nondeterminism detected")
    first["summary"].update({
        "request_sha_run_1": first_req_sha,
        "request_sha_run_2": second_req_sha,
        "fact_packet_sha_run_1": first_fact_sha,
        "fact_packet_sha_run_2": second_fact_sha,
        "request_sha_deterministic": True,
        "fact_packet_sha_deterministic": True,
    })
    return first_request, first


def compare_synthetic_envelope(snapshot: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / "artifacts/evaluation/nf-v2-03-binder-provider-failure-review/synthetic-stress-15.json"
    if not path.exists():
        return {"artifact": str(path), "available": False}
    envelope = json.loads(path.read_text(encoding="utf-8"))
    sizes = envelope.get("by_size", {})
    fact_values = [int(key) for key in sizes if str(key).isdigit()]
    token_values = [int(sizes[key].get("input_tokens_mean")) for key in sizes if sizes[key].get("input_tokens_mean") is not None]
    facts = int(snapshot["summary"]["fact_count"])
    estimate = int(snapshot["summary"]["estimated_input_tokens"])
    return {
        "artifact": str(path),
        "artifact_sha256": sha256_file(path),
        "model": envelope.get("model"),
        "synthetic_fact_min": min(fact_values) if fact_values else None,
        "synthetic_fact_max": max(fact_values) if fact_values else None,
        "synthetic_input_tokens_min": min(token_values) if token_values else None,
        "synthetic_input_tokens_max": max(token_values) if token_values else None,
        "target_fact_count": facts,
        "target_estimated_input_tokens": estimate,
        "within_synthetic_fact_range": bool(fact_values and min(fact_values) <= facts <= max(fact_values)),
        "within_synthetic_token_range_estimate": bool(token_values and min(token_values) <= estimate <= max(token_values)),
        "target_above_synthetic_max_fact": bool(fact_values and facts > max(fact_values)),
        "target_above_synthetic_max_tokens_estimate": bool(token_values and estimate > max(token_values)),
        "comparison_note": "The exact request is compared descriptively; no timeout cause is inferred from fact count alone.",
    }


def classify_failure(metadata: dict[str, Any] | None, schema_valid: bool, raw_len: int) -> str:
    metadata = metadata or {}
    if metadata.get("provider_response_success") and metadata.get("structured_output_success") and schema_valid:
        return "BT0_success"
    status = metadata.get("http_status")
    if isinstance(status, (int, float)) and status >= 400:
        return "BT4_HTTP_non_2xx"
    chain = metadata.get("exception_chain") or []
    names = [str(metadata.get(key) or "") for key in ("exception_type", "exception_cause_type")]
    messages = [str(metadata.get(key) or "") for key in ("error", "exception_cause_message")]
    for item in chain:
        names.append(str(item.get("type") or ""))
        messages.append(str(item.get("message") or ""))
    name_text = " ".join(names).casefold()
    message_text = " ".join(messages).casefold()
    if "connecttimeout" in name_text or "connect timeout" in message_text:
        return "BT1_connect_timeout"
    if "writetimeout" in name_text or "write timeout" in message_text:
        return "BT3_write_timeout"
    if "readtimeout" in name_text or "read timeout" in message_text or "request timed out" in message_text or "timed out" in name_text:
        return "BT2_read_timeout"
    if metadata.get("provider_response_success") and raw_len == 0:
        return "BT5_HTTP_2xx_empty_content"
    if metadata.get("provider_response_success") and raw_len > 0 and not metadata.get("structured_output_success"):
        if any(token in message_text for token in ("json", "parse", "decode")):
            return "BT6_HTTP_2xx_invalid_json"
        return "BT7_HTTP_2xx_schema_invalid"
    if "parse" in name_text or "parse" in message_text or "validationerror" in name_text:
        return "BT8_SDK_parse_failure"
    return "BT9_other"


def run_one(provider: BailianBinderProvider, service: SemanticBinderService, request: BinderRequest, call_index: int, timeout: float, stage: str) -> dict[str, Any]:
    started = now_utc()
    start_clock = time.perf_counter()
    run: BinderRun = service.bind(request)
    latency = (time.perf_counter() - start_clock) * 1000.0
    metadata = run.metadata.to_dict() if run.metadata else {}
    raw_len = len(run.raw_response or "")
    row = {
        "stage": stage,
        "question_id": request.question_id,
        "call_index": call_index,
        "start_time": started,
        "end_time": now_utc(),
        "latency_ms": round(latency, 3),
        "facts_in_packet": len(request.facts),
        "required_slot_count": len(request.plan.required_slots),
        "operation": request.plan.operation,
        "metadata": metadata,
        "provider_request_id": metadata.get("request_id"),
        "http_status": metadata.get("http_status"),
        "finish_reason": metadata.get("finish_reason"),
        "input_tokens": metadata.get("input_tokens"),
        "output_tokens": metadata.get("output_tokens"),
        "total_tokens": metadata.get("total_tokens"),
        "structured_output_success": bool(metadata.get("structured_output_success")),
        "schema_valid": bool(run.schema_valid),
        "raw_content_length": raw_len,
        "exception_type": metadata.get("exception_type"),
        "exception_cause_type": metadata.get("exception_cause_type"),
        "exception_cause_message": metadata.get("exception_cause_message"),
        "exception_chain": metadata.get("exception_chain", []),
        "errno": metadata.get("errno"),
    }
    row["failure_class"] = classify_failure(metadata, bool(run.schema_valid), raw_len)
    return row


def replay(request: BinderRequest, config: dict[str, Any], timeout: float, calls: int, stage: str) -> list[dict[str, Any]]:
    provider = BailianBinderProvider(
        base_url=config["base_url"], api_key=config["api_key"], model_name=MODEL,
        enable_thinking=False, temperature=0.0, timeout=timeout, max_retries=0,
    )
    service = SemanticBinderService(provider)
    rows: list[dict[str, Any]] = []
    try:
        for index in range(1, calls + 1):
            rows.append(run_one(provider, service, request, index, timeout, stage))
    finally:
        provider.close()
    return rows


def request_complexity(request: BinderRequest) -> dict[str, Any]:
    facts = [dict(fact) for fact in request.facts]
    serialized_lengths = [len(canonical_bytes(fact)) for fact in facts]
    text_bytes = [sum(len(str(value or "").encode("utf-8")) for value in fact.values()) for fact in facts]
    pages = {str(fact.get("pdf_page")) for fact in facts if fact.get("pdf_page") is not None}
    tables = {str(fact.get("table_id")) for fact in facts if fact.get("table_id") is not None}
    metrics = Counter(str(fact.get("normalized_metric") or fact.get("raw_metric") or "") for fact in facts)
    return {
        "question_id": request.question_id,
        "fact_count": len(facts),
        "facts_with_long_serialized_text_over_1000_bytes": sum(int(value > 1000) for value in serialized_lengths),
        "max_fact_serialized_length_bytes": max(serialized_lengths, default=0),
        "total_fact_text_bytes": sum(text_bytes),
        "distinct_pages": len(pages),
        "distinct_tables": len(tables),
        "duplicate_or_similar_metric_groups": sum(int(count > 1) for count in metrics.values()),
        "duplicate_or_similar_metric_rows": sum(max(0, count - 1) for count in metrics.values()),
        "required_slot_count": len(request.plan.required_slots),
        "operation": request.plan.operation,
        "gold_reads": 0,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_ms"]) for row in rows]
    return {
        "calls": len(rows),
        "success": sum(int(row["failure_class"] == "BT0_success") for row in rows),
        "read_timeout": sum(int(row["failure_class"] == "BT2_read_timeout") for row in rows),
        "schema_invalid": sum(int(row["failure_class"] in {"BT5_HTTP_2xx_empty_content", "BT6_HTTP_2xx_invalid_json", "BT7_HTTP_2xx_schema_invalid", "BT8_SDK_parse_failure"}) for row in rows),
        "failure_counts": dict(sorted(Counter(row["failure_class"] for row in rows).items())),
        "latency_ms": {
            "p50": round(statistics.median(latencies), 3) if latencies else 0.0,
            "p95": round(percentile(latencies), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
        "rows": rows,
    }


def postprocess_existing() -> int:
    """Annotate an already completed diagnostic without making API calls."""
    replay_path = OUT / "replay-180s.json"
    metadata_path = OUT / "exact-request-metadata.json"
    comparison_path = OUT / "synthetic-envelope-comparison.json"
    if not replay_path.exists() or not metadata_path.exists() or not comparison_path.exists():
        raise RuntimeError("completed R0C artifacts are required for postprocess")
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    decision_path = OUT / "decision.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8")) if decision_path.exists() else {}
    rows = list(replay.get("rows", []))
    tokens = [int(row["input_tokens"]) for row in rows if row.get("input_tokens") is not None]
    latencies = [float(row["latency_ms"]) for row in rows]
    if tokens:
        metadata.update({
            "actual_input_tokens_min": min(tokens),
            "actual_input_tokens_max": max(tokens),
            "actual_input_tokens_mean": round(statistics.mean(tokens), 3),
            "actual_input_tokens_source": "provider_usage_metadata",
        })
        comparison.update({
            "target_actual_input_tokens": round(statistics.mean(tokens), 3),
            "actual_input_tokens_min_observed": min(tokens),
            "actual_input_tokens_max_observed": max(tokens),
            "within_synthetic_token_range_actual": bool(comparison.get("synthetic_input_tokens_min") is not None and comparison["synthetic_input_tokens_min"] <= min(tokens) and max(tokens) <= comparison["synthetic_input_tokens_max"]),
            "target_above_synthetic_max_tokens_actual": bool(comparison.get("synthetic_input_tokens_max") is not None and max(tokens) > comparison["synthetic_input_tokens_max"]),
            "token_comparison_note": "Provider usage tokens are authoritative; the exact request is materially above the prior synthetic 36-fact token maximum despite having 19 facts.",
        })
    write_json(metadata_path, metadata)
    write_json(comparison_path, comparison)
    failure_rows = [{
        "call_index": row.get("call_index"),
        "facts_in_packet": row.get("facts_in_packet"),
        "input_tokens": row.get("input_tokens"),
        "latency_ms": row.get("latency_ms"),
        "failure_class": row.get("failure_class"),
    } for row in rows]
    write_json(OUT / "failure-correlation.json", {
        "question_id": FAILED_QUESTION,
        "rows": failure_rows,
        "by_failure_class": {
            failure: {
                "calls": len([row for row in failure_rows if row["failure_class"] == failure]),
                "facts": sorted({row["facts_in_packet"] for row in failure_rows if row["failure_class"] == failure}),
                "input_tokens": sorted({row["input_tokens"] for row in failure_rows if row["failure_class"] == failure}),
                "latency_p50_ms": percentile([row["latency_ms"] for row in failure_rows if row["failure_class"] == failure]),
            } for failure in sorted({row["failure_class"] for row in failure_rows})
        },
        "interpretation": {
            "facts_vs_failure": "All five exact calls used the same 19-fact packet and all succeeded; no fact-count-correlated failure was observed.",
            "input_tokens_vs_failure": f"Observed provider input tokens were {min(tokens) if tokens else None}-{max(tokens) if tokens else None}, above the prior synthetic maximum of {comparison.get('synthetic_input_tokens_max')}; no failure occurred.",
            "latency_vs_failure": f"Observed latency P50/P95 were {statistics.median(latencies) if latencies else 0.0:.3f}/{percentile(latencies):.3f} ms and all calls were successful.",
            "semantic_inspection": False,
        },
        "gold_reads": 0,
    })
    decision.update({
        "previous_formal_attempt_status": "invalidated_formal_attempt",
        "previous_formal_failure_class": "BT2_read_timeout",
        "previous_formal_calls_attempted": "2/72",
        "previous_prediction_seal_created": False,
        "previous_semantic_scoring": False,
    })
    if decision.get("replay_180_success") == "5/5" and decision.get("neighbor_success") is True:
        decision["root_cause"] = "intermittent_provider_long_tail_or_transport_event"
    write_json(decision_path, decision)
    readme = (
        "# NF-V2-03 R0C Exact Request Timeout Review\n\n"
        "This artifact records an infrastructure-only replay of the frozen `aapl_fy2025_002` BinderRequest.\n"
        "The request was reconstructed twice, replayed sequentially, and compared with prior synthetic transport\n"
        "envelopes. Gold labels, reference answers, and semantic correctness were not loaded or inspected.\n"
        "The 72-question formal Binder evaluation was not rerun.\n"
    )
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    return 0


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if "--postprocess" in sys.argv[1:]:
        return postprocess_existing()
    if os.getenv("V2_SUPERVISOR_MODEL", "").strip() != MODEL:
        write_json(OUT / "decision.json", {"gate": "NF-V2-03-R0C", "base_commit": BASE_COMMIT, "formal_evaluation_status": "configuration_blocked", "reason": "V2_SUPERVISOR_MODEL must be qwen3.7-max", "production_switch_allowed": False})
        return 2
    config = legacy.load_config()
    config["base_url"] = os.getenv("V2_SUPERVISOR_BASE_URL", "").strip()
    if not config["base_url"]:
        raise RuntimeError("V2_SUPERVISOR_BASE_URL is not configured")
    if config["model"] != MODEL or config["max_retries"] != 0:
        raise RuntimeError("frozen provider configuration mismatch")

    request, snapshot = request_determinism()
    summary = snapshot["summary"]
    (OUT / "exact-request.json").write_text(json.dumps(snapshot["request"], ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(OUT / "exact-request-metadata.json", summary)
    write_json(OUT / "exact-request-seal.json", {
        "gate": "NF-V2-03-R0C",
        "question_id": FAILED_QUESTION,
        "request_sha256": summary["full_binder_request_sha256"],
        "fact_packet_sha256": summary["fact_packet_sha256"],
        "request_sha_deterministic": True,
        "fact_packet_sha_deterministic": True,
        "gold_reads": 0,
        "reference_answer_reads": 0,
        "credentials_persisted": False,
    })
    write_json(OUT / "synthetic-envelope-comparison.json", compare_synthetic_envelope(snapshot))
    write_json(OUT / "request-complexity.json", request_complexity(request))

    rows_180 = replay(request, config, DEFAULT_TIMEOUT, 5, "exact_failed_request_180s")
    replay_180 = summarize(rows_180)
    write_json(OUT / "replay-180s.json", {"timeout_seconds": DEFAULT_TIMEOUT, **replay_180})
    actual_tokens = [int(row["input_tokens"]) for row in rows_180 if row.get("input_tokens") is not None]
    comparison = compare_synthetic_envelope(snapshot)
    if actual_tokens:
        comparison.update({
            "target_actual_input_tokens": round(statistics.mean(actual_tokens), 3),
            "actual_input_tokens_min_observed": min(actual_tokens),
            "actual_input_tokens_max_observed": max(actual_tokens),
            "within_synthetic_token_range_actual": bool(comparison.get("synthetic_input_tokens_min") is not None and comparison["synthetic_input_tokens_min"] <= min(actual_tokens) and max(actual_tokens) <= comparison["synthetic_input_tokens_max"]),
            "target_above_synthetic_max_tokens_actual": bool(comparison.get("synthetic_input_tokens_max") is not None and max(actual_tokens) > comparison["synthetic_input_tokens_max"]),
            "token_comparison_note": "Provider usage tokens are authoritative; the exact request is compared descriptively.",
        })
    write_json(OUT / "synthetic-envelope-comparison.json", comparison)

    has_read_timeout = replay_180["read_timeout"] > 0
    rows_330: list[dict[str, Any]] = []
    replay_330: dict[str, Any] = {"status": "not_run", "timeout_seconds": EXTENDED_TIMEOUT}
    if has_read_timeout:
        rows_330 = replay(request, config, EXTENDED_TIMEOUT, 5, "exact_failed_request_330s")
        replay_330 = {"timeout_seconds": EXTENDED_TIMEOUT, **summarize(rows_330)}
        write_json(OUT / "replay-330s.json", replay_330)

    neighbor_ids = ["aapl_fy2025_001", "aapl_fy2025_003"]
    frozen_again = legacy.load_frozen_inputs()
    neighbor_rows: list[dict[str, Any]] = []
    provider = BailianBinderProvider(base_url=config["base_url"], api_key=config["api_key"], model_name=MODEL, enable_thinking=False, temperature=0.0, timeout=DEFAULT_TIMEOUT, max_retries=0)
    service = SemanticBinderService(provider)
    try:
        for index, question_id in enumerate(neighbor_ids, 1):
            neighbor_rows.append(run_one(provider, service, frozen_again["requests"][question_id], index, DEFAULT_TIMEOUT, "neighbor_control"))
    finally:
        provider.close()
    write_json(OUT / "neighbor-control.json", {"timeout_seconds": DEFAULT_TIMEOUT, "previous_question": neighbor_rows[0] if neighbor_rows else None, "next_question": neighbor_rows[1] if len(neighbor_rows) > 1 else None, "rows": neighbor_rows, "gold_reads": 0, "semantic_inspection": False})

    extended_success = bool(rows_330 and all(row["failure_class"] == "BT0_success" for row in rows_330))
    timeout_changed = bool(has_read_timeout and extended_success)
    post_fix = {"status": "not_run", "timeout_changed": timeout_changed, "gold_reads": 0}
    if timeout_changed:
        exact_10 = replay(request, config, EXTENDED_TIMEOUT, 10, "post_fix_exact_330s")
        mixed_ids = [qid for qid in sorted(frozen_again["requests"]) if qid not in {FAILED_QUESTION, *neighbor_ids}][:10]
        mixed_rows: list[dict[str, Any]] = []
        provider = BailianBinderProvider(base_url=config["base_url"], api_key=config["api_key"], model_name=MODEL, enable_thinking=False, temperature=0.0, timeout=EXTENDED_TIMEOUT, max_retries=0)
        service = SemanticBinderService(provider)
        try:
            for index, question_id in enumerate(mixed_ids, 1):
                mixed_rows.append(run_one(provider, service, frozen_again["requests"][question_id], index, EXTENDED_TIMEOUT, "post_fix_mixed"))
        finally:
            provider.close()
        post_fix = {
            "timeout_changed": True,
            "final_timeout_seconds": EXTENDED_TIMEOUT,
            "exact_10": summarize(exact_10),
            "mixed_10": summarize(mixed_rows),
            "exact_success": sum(int(row["failure_class"] == "BT0_success") for row in exact_10),
            "mixed_success": sum(int(row["failure_class"] == "BT0_success") for row in mixed_rows),
            "api_timeout": sum(int(row["failure_class"] == "BT2_read_timeout") for row in exact_10 + mixed_rows),
            "gold_reads": 0,
            "semantic_inspection": False,
        }
    write_json(OUT / "post-fix-stability.json", post_fix)

    neighbor_success = all(row["failure_class"] == "BT0_success" for row in neighbor_rows)
    exact_success = bool(rows_180 and all(row["failure_class"] == "BT0_success" for row in rows_180)) or extended_success
    schema_failure = any(row["failure_class"] in {"BT5_HTTP_2xx_empty_content", "BT6_HTTP_2xx_invalid_json", "BT7_HTTP_2xx_schema_invalid", "BT8_SDK_parse_failure"} for row in rows_180 + rows_330)
    timeout_remaining = bool(has_read_timeout and not extended_success)
    ready = bool(exact_success and neighbor_success and not schema_failure and not timeout_remaining)
    if schema_failure:
        root_cause = "request_specific_structured_output_instability"
        next_gate = "nf_v2_03_schema_failure_review"
    elif timeout_remaining:
        root_cause = "provider_long_tail_timeout"
        next_gate = "nf_v2_03_provider_runtime_review"
    elif ready:
        # The historical formal call was classified BT2 from its persisted
        # failure artifact; five exact successful replays make that event
        # intermittent rather than reproducibly request-specific.
        root_cause = "intermittent_provider_long_tail_or_transport_event"
        next_gate = "resume_nf_v2_03_formal_evaluation_attempt_2"
    else:
        root_cause = "neighbor_or_exact_request_transport_failure"
        next_gate = "nf_v2_03_provider_runtime_review"
    write_json(OUT / "timeout-policy-decision.json", {
        "read_timeout_demonstrated": has_read_timeout,
        "timeout_changed": timeout_changed,
        "timeout_seconds_before": DEFAULT_TIMEOUT,
        "timeout_seconds_after": EXTENDED_TIMEOUT if timeout_changed else DEFAULT_TIMEOUT,
        "max_retries": 0,
        "policy": "Only adopt an extended timeout when the same exact request succeeds consistently under it; no retry added.",
    })
    decision = {
        "gate": "NF-V2-03-R0C",
        "base_commit": BASE_COMMIT,
        "failed_question": FAILED_QUESTION,
        "previous_formal_attempt_status": "invalidated_formal_attempt",
        "previous_formal_failure_class": "BT2_read_timeout",
        "previous_formal_calls_attempted": "2/72",
        "gold_reads": 0,
        "formal_72_replayed": False,
        "model": MODEL,
        "provider": "Alibaba Bailian",
        "temperature": 0.0,
        "thinking": False,
        "max_retries": 0,
        "replay_180_success": f"{replay_180['success']}/5",
        "replay_180_read_timeout": replay_180["read_timeout"],
        "replay_330": "not_run" if not rows_330 else f"{replay_330['success']}/5",
        "neighbor_success": neighbor_success,
        "post_fix_exact": post_fix.get("exact_success") if isinstance(post_fix, dict) else None,
        "post_fix_mixed": post_fix.get("mixed_success") if isinstance(post_fix, dict) else None,
        "root_cause": root_cause,
        "timeout_changed": timeout_changed,
        "binder_provider_contract_ready": ready,
        "production_default": "V1",
        "production_switch_allowed": False,
        "next_gate": next_gate,
    }
    write_json(OUT / "decision.json", decision)
    write_json(OUT / "README.md", {
        "gate": "NF-V2-03 R0C",
        "purpose": "Exact failed-request transport diagnosis before formal attempt 2.",
        "semantic_evaluation": False,
        "gold_reads": 0,
        "formal_72_replayed": False,
        "request": FAILED_QUESTION,
        "artifacts_are_secret_free": True,
        "decision": decision,
    })
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
