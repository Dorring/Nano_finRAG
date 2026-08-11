import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/evaluation/run_nf_opt_23_r1_query_requirement_serialization.py"
SPEC = importlib.util.spec_from_file_location("nf23_r1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_original_question_is_extracted_without_query_plan():
    view = "[QUESTION]\nWhat was revenue in FY2024?\n\n[QUERY PLAN]\nTask: table_single_fact"
    assert MODULE.extract_original_question(view) == "What was revenue in FY2024?"


def test_requirement_fields_are_frozen_and_question_only():
    requirement = MODULE.requirement_from_plan(
        "case",
        {
            "raw_question": "What was revenue in FY2024?",
            "metric_phrases": ["revenue"],
            "periods": ["FY2024"],
            "operation": None,
            "operand_slots": [{"raw_metric_phrase": "revenue", "period": "FY2024", "role": "value"}],
        },
        "What was revenue in FY2024?",
    )
    assert set(requirement) == {"original_question", "target_terms", "explicit_periods", "operation", "required_slots"}
    assert requirement["original_question"] == "What was revenue in FY2024?"
    assert requirement["required_slots"] == [{"target": "revenue", "period": "FY2024", "role": "value"}]


def test_requirement_serializer_is_deterministic_and_candidate_independent():
    requirement = {
        "original_question": "What was revenue in FY2024?",
        "target_terms": ["revenue"],
        "explicit_periods": ["FY2024"],
        "operation": None,
        "required_slots": [{"target": "revenue", "period": "FY2024", "role": "value"}],
    }
    first = MODULE.serialize_requirement(requirement)
    second = MODULE.serialize_requirement(dict(requirement))
    assert first == second
    assert "Gold" not in first
    assert "expected" not in first.lower()


def test_period_and_operation_fail_closed():
    requirement = MODULE.requirement_from_plan(
        "case",
        {"raw_question": "How did it change?", "metric_phrases": [], "periods": [], "operation": None, "operand_slots": []},
        "How did it change?",
    )
    assert requirement["explicit_periods"] == []
    assert requirement["operation"] is None
    assert requirement["required_slots"] == []
    assert MODULE.requirement_category(requirement) == "original_question_only"


def test_requirement_categories_are_fixed():
    base = {"original_question": "q", "target_terms": ["metric"], "explicit_periods": ["FY2024"], "operation": "growth_rate", "required_slots": []}
    assert MODULE.requirement_category(base) == "target_period_operation_or_slot"
    assert MODULE.requirement_category({**base, "operation": None}) == "target_period"
    assert MODULE.requirement_category({**base, "target_terms": []}) == "period_only"
    assert MODULE.requirement_category({**base, "explicit_periods": [], "operation": None}) == "target_only"


def test_frozen_contract_constants():
    assert MODULE.MODEL_ID == "Qwen/Qwen3-Reranker-4B"
    assert MODULE.REVISION == "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
    assert MODULE.MAX_LENGTH == 8192
    assert MODULE.TOP100_SADA_REL.endswith("sada-v1-top100-predictions.jsonl.gz")
