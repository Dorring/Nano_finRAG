"""Deterministic, transport-only retry policy for Binder calls.

The policy deliberately sits outside the OpenAI SDK.  It never retries a
usable semantic response, a schema failure, or a Binding Validator result.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from .binder_provider import BinderCallMetadata
from .binder_service import BinderRequest, BinderRun, SemanticBinderService


RETRYABLE_EXCEPTION_TYPES = frozenset({
    "APITimeoutError",
    "APIConnectionError",
    "ReadTimeout",
    "ConnectTimeout",
})
RETRYABLE_HTTP_STATUSES = frozenset({429, 502, 503, 504})


def canonical_request_sha(request: BinderRequest) -> str:
    payload = (json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fact_packet_sha(request: BinderRequest) -> str:
    payload = (json.dumps([dict(fact) for fact in request.facts], ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _metadata_text(metadata: BinderCallMetadata) -> str:
    values = [metadata.exception_type, metadata.exception_cause_type, metadata.error, metadata.exception_cause_message]
    for item in metadata.exception_chain:
        values.extend((str(item.get("type") or ""), str(item.get("message") or "")))
    return " ".join(str(value or "") for value in values)


def classify_transport_failure(metadata: BinderCallMetadata | None) -> str | None:
    """Return the frozen retry class, or None for a non-retryable result."""
    if metadata is None:
        return None
    if metadata.provider_response_success and metadata.structured_output_success:
        return None
    if metadata.http_status in RETRYABLE_HTTP_STATUSES:
        return f"HTTP_{metadata.http_status}"
    names = {str(metadata.exception_type or ""), str(metadata.exception_cause_type or "")}
    names.update(str(item.get("type") or "") for item in metadata.exception_chain)
    if names & RETRYABLE_EXCEPTION_TYPES:
        for name in ("APITimeoutError", "APIConnectionError", "ReadTimeout", "ConnectTimeout"):
            if name in names:
                return name
    text = _metadata_text(metadata).casefold()
    if "read timeout" in text or "request timed out" in text:
        return "ReadTimeout"
    if "connect timeout" in text:
        return "ConnectTimeout"
    return None


def usable_semantic_response(run: BinderRun) -> bool:
    """A parsed provider response is usable even if Validator rejects semantics."""
    return bool(run.metadata and run.metadata.provider_response_success and run.metadata.structured_output_success and run.schema_valid)


@dataclass(frozen=True)
class TransportRetryPolicy:
    sdk_max_retries: int = 0
    semantic_attempt_budget: int = 1
    transport_retry_budget: int = 1
    retry_delay_seconds: float = 3.0
    http_timeout_seconds: float = 180.0

    def __post_init__(self) -> None:
        if self.sdk_max_retries != 0:
            raise ValueError("Binder SDK max_retries must remain 0")
        if self.semantic_attempt_budget != 1 or self.transport_retry_budget != 1:
            raise ValueError("NF-V2-03 R0E freezes one semantic response and one transport retry")
        if self.retry_delay_seconds < 0:
            raise ValueError("retry delay must be non-negative")

    @property
    def retryable_failures(self) -> tuple[str, ...]:
        return (
            "APITimeoutError",
            "APIConnectionError",
            "ReadTimeout",
            "ConnectTimeout",
            "HTTP_429",
            "HTTP_502",
            "HTTP_503",
            "HTTP_504",
        )


@dataclass(frozen=True)
class TransportAttempt:
    attempt_number: int
    attempted: bool
    provider_success: bool
    structured_output_success: bool
    schema_valid: bool
    failure_class: str | None
    latency_ms: float | None
    http_status: int | None
    request_id: str | None
    input_tokens: int | None
    output_tokens: int | None
    error: str | None

    @classmethod
    def from_run(cls, attempt_number: int, run: BinderRun) -> "TransportAttempt":
        metadata = run.metadata
        failure = None if usable_semantic_response(run) else classify_transport_failure(metadata)
        if failure is None and metadata is not None:
            if metadata.http_status is not None:
                failure = f"HTTP_{metadata.http_status}"
            elif metadata.provider_response_success and not metadata.structured_output_success:
                failure = "HTTP_2XX_SCHEMA_INVALID"
            elif metadata.exception_type:
                failure = metadata.exception_type
        return cls(
            attempt_number=attempt_number,
            attempted=True,
            provider_success=bool(metadata and metadata.provider_response_success),
            structured_output_success=bool(metadata and metadata.structured_output_success),
            schema_valid=bool(run.schema_valid),
            failure_class=failure,
            latency_ms=metadata.latency_ms if metadata else None,
            http_status=metadata.http_status if metadata else None,
            request_id=metadata.request_id if metadata else None,
            input_tokens=metadata.input_tokens if metadata else None,
            output_tokens=metadata.output_tokens if metadata else None,
            error=metadata.error if metadata else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_number": self.attempt_number,
            "attempted": self.attempted,
            "provider_success": self.provider_success,
            "structured_output_success": self.structured_output_success,
            "schema_valid": self.schema_valid,
            "failure_class": self.failure_class,
            "latency_ms": self.latency_ms,
            "http_status": self.http_status,
            "request_id": self.request_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
        }


@dataclass(frozen=True)
class TransportRetryResult:
    run: BinderRun
    attempt_1: TransportAttempt
    attempt_2: TransportAttempt | None
    recovered_by_transport_retry: bool
    semantic_response_count: int
    request_sha256: str
    retry_request_sha_matches_original: bool

    @property
    def final_provider_completion(self) -> bool:
        return usable_semantic_response(self.run)

    def to_dict(self) -> dict[str, Any]:
        attempt_2 = self.attempt_2.to_dict() if self.attempt_2 else {"attempted": False}
        attempt_2["retry_reason"] = self.attempt_1.failure_class
        return {
            "attempt_1": self.attempt_1.to_dict(),
            "attempt_2": attempt_2,
            "recovered_by_transport_retry": self.recovered_by_transport_retry,
            "semantic_response_count": self.semantic_response_count,
            "request_sha256": self.request_sha256,
            "retry_request_sha_matches_original": self.retry_request_sha_matches_original,
            "final_provider_completion": self.final_provider_completion,
        }


def bind_with_transport_retry(
    service: SemanticBinderService,
    request: BinderRequest,
    *,
    policy: TransportRetryPolicy | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    request_sha_fn: Callable[[BinderRequest], str] = canonical_request_sha,
) -> TransportRetryResult:
    """Bind once, then optionally retry one eligible transport failure.

    The same immutable BinderRequest object is passed to both calls.  A
    provider response with valid frozen EvidenceBinding JSON always ends the
    sequence, regardless of its status or Binding Validator outcome.
    """
    policy = policy or TransportRetryPolicy()
    request_sha = request_sha_fn(request)
    first = service.bind(request)
    first_attempt = TransportAttempt.from_run(1, first)
    if request.facts == ():
        first_attempt = TransportAttempt(
            attempt_number=1,
            attempted=False,
            provider_success=False,
            structured_output_success=False,
            schema_valid=first.schema_valid,
            failure_class=None,
            latency_ms=None,
            http_status=None,
            request_id=None,
            input_tokens=None,
            output_tokens=None,
            error=None,
        )
        return TransportRetryResult(first, first_attempt, None, False, 0, request_sha, True)
    if usable_semantic_response(first):
        return TransportRetryResult(first, first_attempt, None, False, int(usable_semantic_response(first)), request_sha, True)
    retry_reason = classify_transport_failure(first.metadata)
    if retry_reason is None or policy.transport_retry_budget == 0:
        return TransportRetryResult(first, first_attempt, None, False, 0, request_sha, True)
    if request_sha_fn(request) != request_sha:
        return TransportRetryResult(first, first_attempt, None, False, 0, request_sha, False)
    sleep_fn(policy.retry_delay_seconds)
    second = service.bind(request)
    second_attempt = TransportAttempt.from_run(2, second)
    request_match = request_sha_fn(request) == request_sha
    if not request_match:
        return TransportRetryResult(first, first_attempt, second_attempt, False, 0, request_sha, False)
    return TransportRetryResult(
        second,
        first_attempt,
        second_attempt,
        usable_semantic_response(second),
        int(usable_semantic_response(second)),
        request_sha,
        True,
    )


def retry_contract_dict(policy: TransportRetryPolicy | None = None) -> dict[str, Any]:
    policy = policy or TransportRetryPolicy()
    return {
        "sdk_max_retries": policy.sdk_max_retries,
        "semantic_attempt_budget": policy.semantic_attempt_budget,
        "transport_retry_budget": policy.transport_retry_budget,
        "retry_delay_seconds": policy.retry_delay_seconds,
        "http_timeout_seconds": policy.http_timeout_seconds,
        "retryable_failures": list(policy.retryable_failures),
        "semantic_output_retry_allowed": False,
    }
