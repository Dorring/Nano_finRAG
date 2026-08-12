from __future__ import annotations

import datetime as _datetime
import json
import re
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - provider extras are optional in unit tests
    OpenAI = None  # type: ignore[assignment,misc]

from rag_v2.contracts.evidence import EvidenceBinding

from .prompt import BINDER_RESPONSE_FORMAT, build_binder_messages


class BinderProviderError(RuntimeError):
    """Raised when the provider cannot return strict EvidenceBinding JSON."""


@dataclass(frozen=True)
class BinderCallMetadata:
    provider: str
    model: str
    provider_role: str
    model_role: str
    latency_ms: float
    provider_response_success: bool
    structured_output_success: bool
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    error: str | None = None
    exception_type: str | None = None
    exception_cause_type: str | None = None
    exception_cause_message: str | None = None
    errno: int | str | None = None
    http_status: int | None = None
    finish_reason: str | None = None
    raw_content_length: int | None = None
    request_id: str | None = None
    exception_chain: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "provider_role": self.provider_role,
            "model_role": self.model_role,
            "latency_ms": round(self.latency_ms, 3),
            "provider_response_success": self.provider_response_success,
            "structured_output_success": self.structured_output_success,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "error": self.error,
            "exception_type": self.exception_type,
            "exception_cause_type": self.exception_cause_type,
            "exception_cause_message": self.exception_cause_message,
            "errno": self.errno,
            "http_status": self.http_status,
            "finish_reason": self.finish_reason,
            "raw_content_length": self.raw_content_length,
            "request_id": self.request_id,
            "exception_chain": [dict(item) for item in self.exception_chain],
        }


@dataclass(frozen=True)
class BinderProviderResult:
    binding: EvidenceBinding | None
    metadata: BinderCallMetadata
    raw_response: str | None = None


class BinderProvider(Protocol):
    provider_name: str
    model_name: str
    last_call: BinderCallMetadata | None

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        ...


def _safe_message(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:500]


def _exception_chain(value: BaseException | None) -> tuple[dict[str, Any], ...]:
    """Return a short, secret-redacted exception chain for transport diagnostics."""
    chain: list[dict[str, Any]] = []
    seen: set[int] = set()
    current = value
    while current is not None and id(current) not in seen and len(chain) < 4:
        seen.add(id(current))
        item: dict[str, Any] = {
            "type": type(current).__name__,
            "message": _safe_message(current),
        }
        errno = getattr(current, "errno", None)
        if isinstance(errno, (int, str)):
            item["errno"] = errno
        chain.append(item)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def _usage_int(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) else None


def _binding_from_payload(payload: Any) -> EvidenceBinding:
    if not isinstance(payload, dict):
        raise BinderProviderError("EvidenceBinding response must be an object")
    expected = {"status", "slot_bindings", "missing_slots", "ambiguous_slots", "invalid_reasons"}
    if set(payload) != expected:
        raise BinderProviderError("EvidenceBinding response keys do not match frozen schema")
    if not isinstance(payload["slot_bindings"], dict):
        raise BinderProviderError("slot_bindings must be an object")
    if any(not isinstance(key, str) or not isinstance(value, list) for key, value in payload["slot_bindings"].items()):
        raise BinderProviderError("slot_bindings values must be arrays")
    for field in ("missing_slots", "ambiguous_slots", "invalid_reasons"):
        if not isinstance(payload[field], list) or any(not isinstance(item, str) for item in payload[field]):
            raise BinderProviderError(f"{field} must be an array of strings")
    try:
        return EvidenceBinding(
            status=payload["status"],
            slot_bindings={key: tuple(value) for key, value in payload["slot_bindings"].items()},
            missing_slots=tuple(payload["missing_slots"]),
            ambiguous_slots=tuple(payload["ambiguous_slots"]),
            invalid_reasons=tuple(payload["invalid_reasons"]),
        )
    except Exception as exc:
        raise BinderProviderError("EvidenceBinding response failed frozen contract validation") from exc


class BailianBinderProvider:
    """Alibaba Bailian strict JSON provider for evidence binding only."""

    provider_name = "bailian"
    provider_role = "evidence_binder"
    model_role = "strong_general_llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        enable_thinking: bool = False,
        temperature: float = 0.0,
        timeout: float = 180.0,
        max_retries: int = 0,
        http_client: Any | None = None,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("the Bailian binder requires the openai package")
        client_kwargs: dict[str, Any] = {
            "base_url": base_url,
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        self.client = OpenAI(**client_kwargs)
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_retries = max_retries
        self.client_created_at = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
        self.last_call: BinderCallMetadata | None = None
        self.last_raw_response: str | None = None

    def close(self) -> None:
        self.client.close()

    def bind(self, request: Mapping[str, Any]) -> BinderProviderResult:
        started = time.perf_counter()
        self.last_raw_response = None
        response: Any | None = None
        try:
            body: dict[str, Any] = {
                "model": self.model_name,
                "messages": build_binder_messages(request),
                "temperature": self.temperature,
                "response_format": BINDER_RESPONSE_FORMAT,
            }
            # Bailian's Qwen endpoint defaults to thinking enabled when this
            # extension is omitted.  The frozen Binder configuration is
            # thinking=false, so send the setting explicitly for both values
            # instead of relying on the endpoint default.
            body["extra_body"] = {"enable_thinking": self.enable_thinking}
            response = self.client.chat.completions.create(**body)
            message = response.choices[0].message if response.choices else None
            content = getattr(message, "content", None) if message is not None else None
            raw = content if isinstance(content, str) else None
            self.last_raw_response = raw
            if not raw or not raw.strip():
                raise BinderProviderError("Bailian returned an empty EvidenceBinding response")
            try:
                payload = json.loads(raw.strip())
            except json.JSONDecodeError as exc:
                raise BinderProviderError("Bailian response was not strict JSON") from exc
            binding = _binding_from_payload(payload)
            metadata = self._metadata(
                response,
                started,
                structured=True,
                raw_content_length=len(raw),
                request_id=getattr(response, "id", None),
            )
            self.last_call = metadata
            return BinderProviderResult(binding=binding, metadata=metadata, raw_response=raw)
        except BinderProviderError as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(
                response,
                started,
                structured=False,
                error=str(exc),
                provider_success=response is not None,
                exception_type=type(exc).__name__,
                exception_cause_type=type(cause).__name__ if cause is not None else None,
                exception_cause_message=_safe_message(cause) if cause is not None else None,
                raw_content_length=len(self.last_raw_response or ""),
                request_id=getattr(response, "id", None),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise
        except Exception as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(
                response,
                started,
                structured=False,
                error=_safe_message(exc),
                provider_success=False,
                exception_type=type(exc).__name__,
                exception_cause_type=type(cause).__name__ if cause is not None else None,
                exception_cause_message=_safe_message(cause) if cause is not None else None,
                errno=getattr(exc, "errno", None),
                raw_content_length=len(self.last_raw_response or ""),
                request_id=getattr(response, "id", None),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise BinderProviderError(f"Bailian binder API call failed: {_safe_message(exc)}") from exc

    def _metadata(
        self,
        response: Any,
        started: float,
        *,
        structured: bool,
        error: str | None = None,
        provider_success: bool = True,
        exception_type: str | None = None,
        exception_cause_type: str | None = None,
        exception_cause_message: str | None = None,
        errno: int | str | None = None,
        raw_content_length: int | None = None,
        request_id: str | None = None,
        exception_chain: tuple[dict[str, Any], ...] = (),
    ) -> BinderCallMetadata:
        usage = getattr(response, "usage", None) if response is not None else None
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
        choice = response.choices[0] if response is not None and getattr(response, "choices", None) else None
        finish_reason = getattr(choice, "finish_reason", None) if choice is not None else None
        http_status = getattr(response, "status_code", None) if response is not None else None
        return BinderCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            provider_role=self.provider_role,
            model_role=self.model_role,
            latency_ms=(time.perf_counter() - started) * 1000,
            provider_response_success=provider_success,
            structured_output_success=structured,
            input_tokens=_usage_int(usage, "prompt_tokens"),
            output_tokens=_usage_int(usage, "completion_tokens"),
            total_tokens=_usage_int(usage, "total_tokens"),
            reasoning_tokens=int(reasoning) if isinstance(reasoning, (int, float)) else None,
            error=error,
            exception_type=exception_type,
            exception_cause_type=exception_cause_type,
            exception_cause_message=exception_cause_message,
            errno=errno if isinstance(errno, (int, str)) else None,
            http_status=int(http_status) if isinstance(http_status, (int, float)) else None,
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            raw_content_length=raw_content_length,
            request_id=request_id,
            exception_chain=exception_chain,
        )
