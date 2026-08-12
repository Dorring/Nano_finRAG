from __future__ import annotations

from rag_v2.supervisor.bailian_provider import BAILIAN_SUPERVISOR_RESPONSE_FORMAT


def test_bailian_provider_identity_and_structured_schema() -> None:
    assert BAILIAN_SUPERVISOR_RESPONSE_FORMAT["type"] == "json_schema"
    schema = BAILIAN_SUPERVISOR_RESPONSE_FORMAT["json_schema"]
    assert schema["strict"] is True
    assert schema["schema"]["additionalProperties"] is False


def test_bailian_path_does_not_use_json_object() -> None:
    assert BAILIAN_SUPERVISOR_RESPONSE_FORMAT["type"] != "json_object"


def test_bailian_role_is_not_financial_domain_generator() -> None:
    from rag_v2.supervisor.bailian_provider import BailianProvider

    assert BailianProvider.provider_name == "bailian"
    assert BailianProvider.provider_role == "supervisor"
    assert BailianProvider.model_role == "strong_general_llm"


def test_schema_contains_only_frozen_supervisor_fields() -> None:
    properties = BAILIAN_SUPERVISOR_RESPONSE_FORMAT["json_schema"]["schema"]["properties"]
    assert set(properties) == {"intent", "required_slots", "operation", "next_action"}
    assert "answer" not in properties
    assert "value" not in properties
