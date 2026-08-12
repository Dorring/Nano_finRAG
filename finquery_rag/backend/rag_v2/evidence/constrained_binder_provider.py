"""Bailian provider using the query-local BinderSelectionDTOv1 boundary."""

from __future__ import annotations

import json
import time
from typing import Any

from rag_v2.contracts.plan import SupervisorPlan

from .binder_provider import (
    BinderProviderError,
    BinderProviderResult,
    BailianBinderProvider,
    _exception_chain,
    _exception_http_status,
    _safe_message,
)
from .binder_service import BinderRequest
from .binder_selection import (
    BinderSelectionDTOv1,
    DuplicateFactHandleError,
    build_selection_messages,
    parse_selection,
    provider_request,
    selection_to_binding,
)


class BinderAdapterError(BinderProviderError):
    """The provider response parsed, but deterministic adaptation failed."""

    schema_valid = True
    adapter_failure = True
    classification = DuplicateFactHandleError.classification


class BailianConstrainedBinderProvider(BailianBinderProvider):
    """Strict Bailian adapter; internal EvidenceBinding remains unchanged."""

    provider_role = "evidence_binder"

    def __init__(self, *args: Any, system_prompt: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.system_prompt = system_prompt

    @staticmethod
    def _coerce_request(request: Any) -> BinderRequest:
        if isinstance(request, BinderRequest):
            return request
        if not isinstance(request, dict):
            raise BinderProviderError("constrained binder request must be a BinderRequest or object")
        try:
            return BinderRequest(
                question_id=str(request["question_id"]),
                question=str(request["question"]),
                plan=SupervisorPlan.from_dict({
                    "intent": request["intent"],
                    "required_slots": request["required_slots"],
                    "operation": request["operation"],
                    "next_action": "RETRIEVE",
                }),
                facts=tuple(request.get("financial_facts") or ()),
            )
        except Exception as exc:
            raise BinderProviderError("constrained binder request failed frozen contract reconstruction") from exc

    def bind(self, request: Any) -> BinderProviderResult:
        request = self._coerce_request(request)
        started = time.perf_counter()
        self.last_raw_response = None
        response: Any | None = None
        payload, handles, schema = provider_request(request)
        try:
            body = {
                "model": self.model_name,
                "messages": build_selection_messages(request, payload, system_prompt=self.system_prompt),
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "BinderSelectionDTOv1",
                        "strict": True,
                        "schema": schema,
                    },
                },
                "extra_body": {"enable_thinking": self.enable_thinking},
            }
            response = self.client.chat.completions.create(**body)
            message = response.choices[0].message if response.choices else None
            raw = getattr(message, "content", None) if message is not None else None
            self.last_raw_response = raw if isinstance(raw, str) else None
            if not self.last_raw_response or not self.last_raw_response.strip():
                raise BinderProviderError("Bailian returned an empty BinderSelectionDTOv1 response")
            try:
                parsed = json.loads(self.last_raw_response.strip())
            except json.JSONDecodeError as exc:
                raise BinderProviderError("Bailian BinderSelectionDTOv1 response was not strict JSON") from exc
            dto: BinderSelectionDTOv1 = parse_selection(parsed, request.plan, handles)
            binding = selection_to_binding(dto, request, handles)
            metadata = self._metadata(
                response,
                started,
                structured=True,
                raw_content_length=len(self.last_raw_response),
                request_id=getattr(response, "id", None),
            )
            self.last_call = metadata
            return BinderProviderResult(binding=binding, metadata=metadata, raw_response=self.last_raw_response)
        except DuplicateFactHandleError as exc:
            cause = exc.__cause__ or exc.__context__
            metadata = self._metadata(
                response,
                started,
                structured=True,
                error=str(exc),
                provider_success=response is not None,
                exception_type=type(exc).__name__,
                exception_cause_type=type(cause).__name__ if cause is not None else None,
                exception_cause_message=_safe_message(cause) if cause is not None else None,
                raw_content_length=len(self.last_raw_response or ""),
                request_id=getattr(response, "id", None),
                http_status=_exception_http_status(exc),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise BinderAdapterError(f"{DuplicateFactHandleError.classification}:{exc}") from exc
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
                http_status=_exception_http_status(exc),
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
                http_status=_exception_http_status(exc),
                exception_chain=_exception_chain(exc),
            )
            self.last_call = metadata
            raise BinderProviderError(f"Bailian constrained binder API call failed: {_safe_message(exc)}") from exc
