"""Strict Oracle operand scoring helpers for NF-OPT-05 R1."""

from __future__ import annotations

from decimal import Decimal
from typing import Any


def operand_roles(operation: str, count: int) -> tuple[str, ...]:
    if operation == "growth_rate":
        return ("previous", "current")
    if operation in {"ratio", "percentage_share"}:
        return ("part", "total")
    if operation == "difference":
        return ("minuend", "subtrahend")
    return tuple(f"operand_{index}" for index in range(count))


def score_operands(
    *,
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
) -> dict[str, bool]:
    count_correct = len(expected) == len(actual)
    role_correct = count_correct and all(
        item["role"] == actual[index].get("name") for index, item in enumerate(expected)
    )
    value_correct = count_correct and all(
        Decimal(str(item["value"])) == Decimal(str(actual[index].get("value")))
        for index, item in enumerate(expected)
    )
    evidence_correct = count_correct and all(
        item.get("evidence_chunk_id") == actual[index].get("evidence_chunk_id")
        for index, item in enumerate(expected)
    )
    return {
        "operand_count_correct": count_correct,
        "operand_role_assignment_correct": role_correct,
        "operand_value_correct": value_correct,
        "actual_operands_have_evidence_identity": all(
            operand.get("evidence_chunk_id") for operand in actual
        ),
        "operand_evidence_identity_correct": evidence_correct,
    }


def strict_result_correct(
    *,
    execution_completed: bool,
    actual_value: str | None,
    expected_value: str | None,
    actual_unit: str | None,
    expected_unit: str | None,
) -> bool:
    if not execution_completed or actual_value is None or expected_value is None:
        return False
    if actual_unit != expected_unit:
        return False
    return Decimal(str(actual_value)) == Decimal(str(expected_value))
