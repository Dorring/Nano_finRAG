from src.evaluation.nf_opt_05_r1 import (
    operand_roles,
    score_operands,
    strict_result_correct,
)


def test_growth_rate_contract_uses_previous_then_current():
    assert operand_roles("growth_rate", 2) == ("previous", "current")


def test_difference_contract_preserves_operand_direction():
    assert operand_roles("difference", 2) == ("minuend", "subtrahend")


def test_execution_is_not_operand_correctness():
    checks = score_operands(
        expected=[
            {"role": "previous", "value": "100", "evidence_chunk_id": "a"},
            {"role": "current", "value": "200", "evidence_chunk_id": "b"},
        ],
        actual=[],
    )
    assert checks["operand_count_correct"] is False
    assert checks["operand_value_correct"] is False


def test_operand_roles_are_order_sensitive():
    checks = score_operands(
        expected=[
            {"role": "previous", "value": "100", "evidence_chunk_id": "a"},
            {"role": "current", "value": "200", "evidence_chunk_id": "b"},
        ],
        actual=[
            {"name": "current", "value": "100", "evidence_chunk_id": "a"},
            {"name": "previous", "value": "200", "evidence_chunk_id": "b"},
        ],
    )
    assert checks["operand_role_assignment_correct"] is False


def test_operand_value_requires_strict_decimal_match():
    checks = score_operands(
        expected=[{"role": "part", "value": "100", "evidence_chunk_id": "a"}],
        actual=[{"name": "part", "value": "101", "evidence_chunk_id": "a"}],
    )
    assert checks["operand_value_correct"] is False


def test_result_requires_value_and_unit_match():
    assert strict_result_correct(
        execution_completed=True,
        actual_value="10",
        expected_value="10",
        actual_unit="percentage",
        expected_unit="percentage",
    )
    assert not strict_result_correct(
        execution_completed=True,
        actual_value="10",
        expected_value="10",
        actual_unit=None,
        expected_unit="percentage",
    )
