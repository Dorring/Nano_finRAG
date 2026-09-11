"""Focused tests for optional DeepSeek thinking control in APIProvider."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_v2.supervisor.api_provider import APIProvider


def _response() -> SimpleNamespace:
    payload = {
        "intent": "DIRECT_FACT",
        "required_slots": [{
            "slot_id": "revenue_fy2024",
            "metric": "Revenue",
            "period": "FY2024",
            "role": "value",
            "value_type": "numeric",
            "unit": None,
        }],
        "operation": None,
        "next_action": "RETRIEVE",
    }
    message = SimpleNamespace(content=json.dumps(payload), reasoning_content="private")
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _provider(*, enable_thinking: bool | None) -> APIProvider:
    with patch("rag_v2.supervisor.api_provider.OpenAI") as openai:
        openai.return_value = MagicMock()
        return APIProvider(
            base_url="https://api.deepseek.com/v1",
            api_key="sk-test",
            model_name="deepseek-flash",
            structured_output=True,
            enable_thinking=enable_thinking,
        )


def test_api_supervisor_disables_deepseek_thinking_for_strict_json() -> None:
    provider = _provider(enable_thinking=False)
    provider.client.chat.completions.create.return_value = _response()

    plan = provider.plan("What was Apple FY2024 revenue?")

    assert plan.intent.value == "DIRECT_FACT"
    assert provider.client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"},
    }


def test_api_supervisor_keeps_generic_endpoint_extension_free() -> None:
    provider = _provider(enable_thinking=None)
    provider.client.chat.completions.create.return_value = _response()

    provider.plan("What was Apple FY2024 revenue?")

    assert "extra_body" not in provider.client.chat.completions.create.call_args.kwargs