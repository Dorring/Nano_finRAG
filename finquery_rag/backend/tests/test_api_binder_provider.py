"""Unit tests for APIBinderProvider (OpenAI-compatible evidence binder).

These tests use unittest.mock to avoid real network calls.  The module under
test (binder_provider) imports openai lazily so the tests are CPU-safe even
when the openai package is installed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from rag_v2.evidence.binder_provider import (
    APIBinderProvider,
    BinderProviderError,
    _binding_from_payload,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _valid_payload() -> dict[str, Any]:
    return {
        "status": "BOUND",
        "slot_bindings": {"revenue": ["evidence-1"]},
        "missing_slots": [],
        "ambiguous_slots": [],
        "invalid_reasons": [],
    }


def _make_response(content: str | None, finish_reason: str = "stop") -> Any:
    """Build a minimal mock chat completion response."""
    message = SimpleNamespace(content=content)
    # Explicitly set reasoning_content to a non-None value to verify it is ignored.
    message.reasoning_content = "private chain of thought"  # must never be read
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        completion_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage, id="resp-001")


def _make_provider(
    http_client: Any = None,
    *,
    max_tokens: int = 1024,
    enable_thinking: bool | None = None,
) -> APIBinderProvider:
    with patch("rag_v2.evidence.binder_provider.OpenAI") as mock_openai_cls:
        mock_openai_cls.return_value = MagicMock()
        provider = APIBinderProvider(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-test",
            model_name="deepseek-chat",
            max_tokens=max_tokens,
            http_client=http_client,
            enable_thinking=enable_thinking,
        )
    return provider


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


def test_bind_returns_correct_evidence_binding_on_valid_json() -> None:
    provider = _make_provider()
    payload = _valid_payload()
    response = _make_response(json.dumps(payload))
    provider.client.chat.completions.create.return_value = response

    result = provider.bind({"slots": [{"slot_id": "revenue"}], "candidates": []})

    assert result.binding is not None
    assert result.binding.status == "BOUND"
    assert result.binding.slot_bindings == {"revenue": ("evidence-1",)}
    assert result.metadata.provider == "api"
    assert result.metadata.provider_response_success is True
    assert result.metadata.structured_output_success is True
    assert result.metadata.input_tokens == 10
    assert result.metadata.output_tokens == 20
    assert result.raw_response == json.dumps(payload)


def test_api_provider_requests_json_mode_and_validates_schema_locally() -> None:
    """DeepSeek supports JSON Mode; the frozen schema remains local enforcement."""
    provider = _make_provider()
    provider.client.chat.completions.create.return_value = _make_response(json.dumps(_valid_payload()))

    provider.bind({"slots": [], "candidates": []})

    call = provider.client.chat.completions.create.call_args
    assert call is not None
    assert call.kwargs["response_format"] == {"type": "json_object"}
    assert call.kwargs["max_tokens"] == 1024


def test_api_provider_respects_configured_output_budget() -> None:
    provider = _make_provider(max_tokens=1536)
    provider.client.chat.completions.create.return_value = _make_response(
        json.dumps(_valid_payload())
    )

    provider.bind({"slots": [], "candidates": []})

    assert provider.client.chat.completions.create.call_args.kwargs["max_tokens"] == 1536


def test_api_provider_can_explicitly_disable_deepseek_thinking() -> None:
    provider = _make_provider(enable_thinking=False)
    provider.client.chat.completions.create.return_value = _make_response(
        json.dumps(_valid_payload())
    )

    provider.bind({"slots": [], "candidates": []})

    assert provider.client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"},
    }


def test_api_provider_omits_thinking_extension_for_generic_endpoints() -> None:
    provider = _make_provider()
    provider.client.chat.completions.create.return_value = _make_response(
        json.dumps(_valid_payload())
    )

    provider.bind({"slots": [], "candidates": []})

    assert "extra_body" not in provider.client.chat.completions.create.call_args.kwargs

def test_bind_discards_reasoning_content_and_reads_only_content() -> None:
    """The provider must read only message.content; reasoning_content is never accessed."""
    provider = _make_provider()
    payload = _valid_payload()
    response = _make_response(json.dumps(payload))
    # Verify reasoning_content attribute exists on mock (simulates DeepSeek API).
    assert response.choices[0].message.reasoning_content == "private chain of thought"

    provider.client.chat.completions.create.return_value = response
    result = provider.bind({"slots": [], "candidates": []})

    # Result must be built purely from content, not reasoning_content.
    assert result.binding is not None
    assert result.binding.status == "BOUND"


def test_close_releases_underlying_client() -> None:
    provider = _make_provider()
    provider.close()
    provider.client.close.assert_called_once()


# ---------------------------------------------------------------------------
# Error: empty response
# ---------------------------------------------------------------------------


def test_bind_raises_on_empty_content() -> None:
    provider = _make_provider()
    response = _make_response("")
    provider.client.chat.completions.create.return_value = response

    with pytest.raises(BinderProviderError, match="empty"):
        provider.bind({"slots": [], "candidates": []})

    assert provider.last_call is not None
    assert provider.last_call.structured_output_success is False


def test_bind_raises_on_none_content() -> None:
    provider = _make_provider()
    response = _make_response(None)
    provider.client.chat.completions.create.return_value = response

    with pytest.raises(BinderProviderError, match="empty"):
        provider.bind({"slots": [], "candidates": []})


# ---------------------------------------------------------------------------
# Error: invalid JSON
# ---------------------------------------------------------------------------


def test_bind_raises_on_non_json_response() -> None:
    provider = _make_provider()
    response = _make_response("not json at all")
    provider.client.chat.completions.create.return_value = response

    with pytest.raises(BinderProviderError, match="strict JSON"):
        provider.bind({"slots": [], "candidates": []})


# ---------------------------------------------------------------------------
# Error: schema violations
# ---------------------------------------------------------------------------


def test_binding_from_payload_raises_on_non_object() -> None:
    with pytest.raises(BinderProviderError, match="object"):
        _binding_from_payload([])


def test_binding_from_payload_raises_on_extra_keys() -> None:
    payload = _valid_payload()
    payload["extra_field"] = "unexpected"
    with pytest.raises(BinderProviderError, match="frozen schema"):
        _binding_from_payload(payload)


def test_binding_from_payload_raises_on_missing_keys() -> None:
    payload = _valid_payload()
    del payload["missing_slots"]
    with pytest.raises(BinderProviderError, match="frozen schema"):
        _binding_from_payload(payload)


def test_binding_from_payload_raises_on_slot_bindings_not_dict() -> None:
    payload = _valid_payload()
    payload["slot_bindings"] = "not a dict"
    with pytest.raises(BinderProviderError, match="slot_bindings"):
        _binding_from_payload(payload)


def test_binding_from_payload_raises_when_slot_bindings_value_not_list() -> None:
    payload = _valid_payload()
    payload["slot_bindings"] = {"revenue": "evidence-1"}  # should be a list
    with pytest.raises(BinderProviderError, match="slot_bindings values"):
        _binding_from_payload(payload)


def test_binding_from_payload_raises_when_array_field_has_non_string() -> None:
    payload = _valid_payload()
    payload["missing_slots"] = [1, 2]  # must be list of strings
    with pytest.raises(BinderProviderError, match="missing_slots"):
        _binding_from_payload(payload)


# ---------------------------------------------------------------------------
# Error: network / timeout
# ---------------------------------------------------------------------------


def test_bind_wraps_connection_error_as_binder_provider_error() -> None:
    provider = _make_provider()
    provider.client.chat.completions.create.side_effect = ConnectionError("network down")

    with pytest.raises(BinderProviderError, match="API binder call failed"):
        provider.bind({"slots": [], "candidates": []})

    assert provider.last_call is not None
    assert provider.last_call.provider_response_success is False
    assert provider.last_call.exception_type == "ConnectionError"
    # Error message must be redacted / safe (no raw API key in text).
    assert "sk-" not in (provider.last_call.error or "")


def test_bind_wraps_timeout_error_as_binder_provider_error() -> None:
    provider = _make_provider()
    provider.client.chat.completions.create.side_effect = TimeoutError("timed out")

    with pytest.raises(BinderProviderError, match="API binder call failed"):
        provider.bind({"slots": [], "candidates": []})


def test_bind_metadata_available_after_provider_error() -> None:
    provider = _make_provider()
    response = _make_response("invalid json {{{")
    provider.client.chat.completions.create.return_value = response

    with pytest.raises(BinderProviderError):
        provider.bind({"slots": [], "candidates": []})

    meta = provider.last_call
    assert meta is not None
    assert meta.structured_output_success is False
    assert meta.provider_response_success is True  # provider returned something


# ---------------------------------------------------------------------------
# Metadata correctness
# ---------------------------------------------------------------------------


def test_metadata_captures_finish_reason_and_token_counts() -> None:
    provider = _make_provider()
    payload = _valid_payload()
    response = _make_response(json.dumps(payload), finish_reason="stop")
    provider.client.chat.completions.create.return_value = response

    result = provider.bind({"slots": [], "candidates": []})

    assert result.metadata.finish_reason == "stop"
    assert result.metadata.total_tokens == 30


def test_metadata_provider_name_is_api() -> None:
    provider = _make_provider()
    payload = _valid_payload()
    response = _make_response(json.dumps(payload))
    provider.client.chat.completions.create.return_value = response

    result = provider.bind({"slots": [], "candidates": []})
    assert result.metadata.provider == "api"
    assert result.metadata.provider_role == "evidence_binder"
