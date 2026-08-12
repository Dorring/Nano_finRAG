from __future__ import annotations

import json
import datetime as _datetime
import os
import re
import threading
import time
from typing import Any

from rag_v2.contracts.plan import SupervisorPlan

from .api_provider import OpenAI
from .prompt import SUPERVISOR_PLAN_JSON_SCHEMA, build_messages
from .provider import SupervisorCallMetadata, SupervisorProviderError


BAILIAN_SUPERVISOR_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "SupervisorPlan",
        "strict": True,
        "schema": SUPERVISOR_PLAN_JSON_SCHEMA,
    },
}


def _safe_exception_message(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return text[:500]


def sanitized_exception_chain(exc: BaseException) -> list[dict[str, Any]]:
    chain: list[dict[str, Any]] = []
    current: BaseException | None = exc
    for _ in range(3):
        if current is None:
            break
        errno = getattr(current, "errno", None)
        chain.append({
            "type": type(current).__name__,
            "message": _safe_exception_message(current),
            "errno": errno if isinstance(errno, (int, str)) else None,
        })
        current = current.__cause__ or current.__context__
    return chain


def _strict_plan(raw: str) -> SupervisorPlan:
    try:
        payload = json.loads(raw.strip())
    except json.JSONDecodeError as exc:
        raise SupervisorProviderError("Bailian response was not strict JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"intent", "required_slots", "operation", "next_action"}:
        raise SupervisorProviderError("Bailian response has invalid SupervisorPlan keys")
    slots = payload.get("required_slots")
    if not isinstance(slots, list):
        raise SupervisorProviderError("Bailian required_slots is not an array")
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"slot_id", "metric", "period", "role", "value_type", "unit"}:
            raise SupervisorProviderError("Bailian slot does not satisfy the frozen schema")
    try:
        return SupervisorPlan.from_dict(payload)
    except Exception as exc:
        raise SupervisorProviderError("Bailian response failed SupervisorPlan schema validation") from exc


class BailianProvider:
    """Alibaba Bailian OpenAI-compatible strict structured-output provider."""

    provider_name = "bailian"
    provider_role = "supervisor"
    model_role = "strong_general_llm"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        enable_thinking: bool,
        temperature: float,
        max_tokens: int = 512,
        timeout: float = 180.0,
        max_retries: int = 0,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("the Bailian provider requires the openai package")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client_created_at = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
        self.client_instance_id = id(self.client)
        self.last_exception_chain: list[dict[str, Any]] = []
        self.last_call: SupervisorCallMetadata | None = None

    @staticmethod
    def _closed(client: Any) -> bool | None:
        value = getattr(client, "is_closed", None)
        if callable(value):
            try:
                return bool(value())
            except Exception:
                return None
        return value if isinstance(value, bool) else None

    def transport_context(self, *, call_sequence_number: int | None = None) -> dict[str, Any]:
        http_client = getattr(self.client, "_client", None)
        return {
            "process_id": os.getpid(),
            "thread_id": threading.get_ident(),
            "async_or_sync": "sync",
            "event_loop_id_if_async": None,
            "provider_instance_id": id(self),
            "openai_client_instance_id": self.client_instance_id,
            "http_client_instance_id": id(http_client) if http_client is not None else None,
            "client_created_at": self.client_created_at,
            "client_closed_before_call": self._closed(http_client),
            "client_closed_after_call": None,
            "call_sequence_number": call_sequence_number,
            "sdk_max_retries": self.max_retries,
            "trust_env": getattr(http_client, "_trust_env", None),
            "custom_http_client": http_client is not None,
        }

    def close(self) -> None:
        self.client.close()

    def plan(self, question: str) -> SupervisorPlan:
        started = time.perf_counter()
        raw: str | None = None
        self.last_exception_chain = []
        try:
            # Bailian JSON-Schema mode owns the output limit. Sending
            # ``max_tokens`` alongside a strict schema can make the endpoint
            # reject the request or truncate the structured object.
            request: dict[str, Any] = {
                "model": self.model_name,
                "messages": build_messages(question),
                "temperature": self.temperature,
                "response_format": BAILIAN_SUPERVISOR_RESPONSE_FORMAT,
            }
            # ``false`` is the service default for this model and omitting the
            # extension avoids an endpoint-specific connection failure while
            # keeping the frozen effective setting explicit in the seal.
            if self.enable_thinking:
                request["extra_body"] = {"enable_thinking": True}
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message if response.choices else None
            content = getattr(message, "content", None) if message is not None else None
            raw = content if isinstance(content, str) else None
            if not raw or not raw.strip():
                raise SupervisorProviderError("Bailian returned an empty structured response")
            try:
                plan = _strict_plan(raw)
            except SupervisorProviderError as exc:
                self._record(response, started, raw, structured=False, error=str(exc))
                raise
            self._record(response, started, raw, structured=True)
            return plan
        except SupervisorProviderError:
            raise
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            code = getattr(exc, "code", None)
            param = getattr(exc, "param", None)
            cause = exc.__cause__ or exc.__context__
            cause_type = type(cause).__name__ if cause is not None else None
            cause_text = _safe_exception_message(cause) if cause is not None else None
            safe_error = f"{type(exc).__name__}; status={status!r}; code={code!r}; param={param!r}; cause={cause_type!r}:{cause_text!r}"
            self.last_call = SupervisorCallMetadata(
                provider=self.provider_name,
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response=raw,
                provider_role=self.provider_role,
                model_role=self.model_role,
                provider_response_success=False,
                structured_output_success=False,
                error=safe_error,
            )
            self.last_exception_chain = sanitized_exception_chain(exc)
            raise SupervisorProviderError(f"Bailian API call failed ({safe_error})") from exc

    def _record(self, response: Any, started: float, raw: str, *, structured: bool, error: str | None = None) -> None:
        usage = getattr(response, "usage", None)
        details = getattr(usage, "completion_tokens_details", None) if usage is not None else None
        reasoning = getattr(details, "reasoning_tokens", None) if details is not None else None
        self.last_call = SupervisorCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw,
            provider_role=self.provider_role,
            model_role=self.model_role,
            provider_response_success=True,
            structured_output_success=structured,
            reasoning_tokens=int(reasoning) if isinstance(reasoning, (int, float)) else None,
            input_tokens=int(getattr(usage, "prompt_tokens")) if usage is not None and isinstance(getattr(usage, "prompt_tokens", None), (int, float)) else None,
            output_tokens=int(getattr(usage, "completion_tokens")) if usage is not None and isinstance(getattr(usage, "completion_tokens", None), (int, float)) else None,
            total_tokens=int(getattr(usage, "total_tokens")) if usage is not None and isinstance(getattr(usage, "total_tokens", None), (int, float)) else None,
            error=error,
        )
