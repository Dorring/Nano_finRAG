from __future__ import annotations

import json
import time
from typing import Any

try:  # Keep V2-00 contract imports usable in minimal test environments.
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised only without provider extras
    OpenAI = None  # type: ignore[assignment,misc]

from rag_v2.contracts.plan import SupervisorPlan

from .prompt import build_messages
from .provider import SupervisorCallMetadata, SupervisorProviderError


def _usage_value(usage: Any, name: str) -> int | None:
    value = getattr(usage, name, None) if usage is not None else None
    return int(value) if isinstance(value, (int, float)) else None


class APIProvider:
    """OpenAI-compatible Supervisor provider with one strict JSON attempt."""

    provider_name = "api"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model_name: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        timeout: float = 120.0,
        provider_role: str = "supervisor",
        model_role: str = "strong_general_llm",
        structured_output: bool = False,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("the API Supervisor provider requires the openai package")
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.provider_role = provider_role
        self.model_role = model_role
        self.structured_output = structured_output
        self.last_call: SupervisorCallMetadata | None = None

    def plan(self, question: str) -> SupervisorPlan:
        started = time.perf_counter()
        raw: str | None = None
        try:
            request: dict[str, Any] = {
                "model": self.model_name,
                "messages": build_messages(question),
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.structured_output:
                request["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**request)
            content = response.choices[0].message.content if response.choices else None
            raw = content if isinstance(content, str) else None
            if not raw or not raw.strip():
                raise SupervisorProviderError("empty structured response")
            try:
                payload = json.loads(raw.strip())
            except json.JSONDecodeError as exc:
                self._record(response, started, raw, parse_failure="strict_json_parse", error=str(exc))
                raise SupervisorProviderError("response was not strict JSON") from exc
            if not isinstance(payload, dict) or set(payload) != {"intent", "required_slots", "operation", "next_action"}:
                self._record(response, started, raw, parse_failure="schema_keys", error="top-level keys must match SupervisorPlan")
                raise SupervisorProviderError("response has extra or missing SupervisorPlan keys")
            if not isinstance(payload.get("required_slots"), list):
                self._record(response, started, raw, parse_failure="schema_slots", error="required_slots must be an array")
                raise SupervisorProviderError("required_slots must be an array")
            for slot in payload["required_slots"]:
                if not isinstance(slot, dict) or set(slot) != {"slot_id", "metric", "period", "role", "value_type", "unit"}:
                    self._record(response, started, raw, parse_failure="schema_slot_keys", error="slot keys must match RequiredSlot")
                    raise SupervisorProviderError("required slot has extra or missing keys")
            try:
                plan = SupervisorPlan.from_dict(payload)
            except Exception as exc:
                self._record(response, started, raw, parse_failure="plan_schema", error=str(exc))
                raise SupervisorProviderError("response did not satisfy SupervisorPlan schema") from exc
            self._record(response, started, raw)
            return plan
        except SupervisorProviderError:
            raise
        except Exception as exc:
            self.last_call = SupervisorCallMetadata(
                provider=self.provider_name,
                model=self.model_name,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response=raw,
                provider_role=self.provider_role,
                model_role=self.model_role,
                provider_response_success=False,
                structured_output_success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise SupervisorProviderError("Supervisor API call failed") from exc

    def _record(
        self,
        response: Any,
        started: float,
        raw: str,
        *,
        parse_failure: str | None = None,
        error: str | None = None,
    ) -> None:
        usage = getattr(response, "usage", None)
        self.last_call = SupervisorCallMetadata(
            provider=self.provider_name,
            model=self.model_name,
            latency_ms=(time.perf_counter() - started) * 1000,
            raw_response=raw,
            provider_role=self.provider_role,
            model_role=self.model_role,
            provider_response_success=True,
            structured_output_success=parse_failure is None,
            input_tokens=_usage_value(usage, "prompt_tokens"),
            output_tokens=_usage_value(usage, "completion_tokens"),
            total_tokens=_usage_value(usage, "total_tokens"),
            parse_failure=parse_failure,
            error=error,
        )
