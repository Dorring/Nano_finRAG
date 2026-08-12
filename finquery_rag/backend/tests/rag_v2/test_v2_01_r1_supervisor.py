from __future__ import annotations

import pytest

from rag_v2.contracts import Action, Intent, RequiredSlot, SupervisorPlan
from rag_v2.contracts.errors import PlanValidationError
from rag_v2.supervisor import StrongGeneralAPIProvider, validate_plan_v2_01


def test_strong_general_provider_records_explicit_model_role() -> None:
    provider = object.__new__(StrongGeneralAPIProvider)
    provider.provider_role = "supervisor"
    provider.model_role = "strong_general_llm"
    provider.model_name = "configured-strong-model"
    assert provider.model_role == "strong_general_llm"
    assert provider.provider_role == "supervisor"
    assert "finquery-finance" not in provider.model_name


def test_financial_sft_model_is_not_an_r1_role() -> None:
    from scripts.evaluation.run_nf_v2_01_r1_strong_general_llm_supervisor import is_financial_sft_model

    assert is_financial_sft_model("finquery-finance-v2-lr010-150") is True
    assert is_financial_sft_model("configured-strong-model") is False


def test_r1_plan_contains_no_answer_or_numeric_fields() -> None:
    slot = RequiredSlot("slot_1", "revenue", "FY2025", "value", "numeric", None)
    plan = SupervisorPlan(Intent.DIRECT_FACT, (slot,), None, Action.RETRIEVE)
    payload = plan.to_dict()
    assert set(payload) == {"intent", "required_slots", "operation", "next_action"}
    assert "answer" not in payload
    assert "value" not in payload


def test_plan_validator_semantics_remain_frozen_for_r1() -> None:
    slot = RequiredSlot("slot_1", "revenue", "FY2025", "value", "numeric", None)
    with pytest.raises(PlanValidationError):
        validate_plan_v2_01(SupervisorPlan(Intent.DIRECT_FACT, (slot,), None, Action.GENERATE))


def test_no_silent_financial_sft_fallback_is_encoded_in_provider_contract() -> None:
    provider = object.__new__(StrongGeneralAPIProvider)
    provider.provider_role = "supervisor"
    provider.model_role = "strong_general_llm"
    assert provider.model_role != "financial_sft_lm"
