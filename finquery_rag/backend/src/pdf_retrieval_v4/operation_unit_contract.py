"""Operation-aware unit compatibility without unit guessing."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

SCALE_INVARIANT = {"growth_rate", "percentage_change", "ratio", "percentage_share"}
SCALE_SENSITIVE = {"difference", "sum", "absolute_change"}


def _decimal(value: Any) -> Decimal | None:
    try:
        return Decimal(str(value)) if value not in (None, "") else None
    except InvalidOperation:
        return None


def evaluate_operation_units(
    operation: str | None,
    operands: list[dict[str, Any]],
    same_row: bool,
    same_table: bool,
) -> dict[str, Any]:
    if not operands or any(_decimal(item.get("value")) is None for item in operands):
        return {"ready": False, "reason": "numeric_value_missing", "normalized_values": []}
    kinds = {str(item.get("measurement_kind") or "unknown") for item in operands}
    known_kinds = kinds - {"unknown"}
    if len(known_kinds) > 1:
        return {"ready": False, "reason": "measurement_kind_conflict", "normalized_values": []}
    scales = [item.get("unit_context", {}).get("scale") for item in operands]
    currencies = [item.get("unit_context", {}).get("currency") for item in operands]
    scale_statuses = [item.get("unit_context", {}).get("scale_status") for item in operands]
    currency_statuses = [item.get("unit_context", {}).get("currency_status") for item in operands]
    if "conflict" in scale_statuses or "conflict" in currency_statuses:
        return {"ready": False, "reason": "unit_context_conflict", "normalized_values": []}
    operation = str(operation or "")
    shared_context = same_row or same_table
    all_scale_known = all(_decimal(value) is not None for value in scales)
    all_scale_unknown = all(value in (None, "") for value in scales)
    all_currency_known = all(value not in (None, "") for value in currencies)
    all_currency_unknown = all(value in (None, "") for value in currencies)
    known_currency_compatible = all_currency_known and len(set(currencies)) == 1
    unknown_currency_compatible = all_currency_unknown and shared_context
    if operation in SCALE_INVARIANT:
        scale_compatible = all_scale_known or (all_scale_unknown and shared_context)
        currency_compatible = known_currency_compatible or unknown_currency_compatible
        if not scale_compatible:
            return {"ready": False, "reason": "scale_not_shared_or_resolved", "normalized_values": []}
        if not currency_compatible:
            return {"ready": False, "reason": "currency_not_shared_or_resolved", "normalized_values": []}
        normalized = [
            str((_decimal(item["value"]) * (_decimal(scale) or Decimal(1))).normalize())
            for item, scale in zip(operands, scales, strict=True)
        ]
        return {
            "ready": True,
            "reason": None,
            "normalized_values": normalized,
            "scale_contract": "resolved" if all_scale_known else "unresolved_shared_cancels",
            "currency_contract": "resolved" if all_currency_known else "unresolved_shared_cancels",
        }
    kind = next(iter(known_kinds), "unknown")
    if kind in {"percentage", "ratio", "dimensionless"}:
        return {
            "ready": True,
            "reason": None,
            "normalized_values": [str(_decimal(item["value"]).normalize()) for item in operands],
            "scale_contract": "dimensionless",
            "currency_contract": "not_required",
        }
    if operation in SCALE_SENSITIVE or operation:
        if not all_scale_known:
            return {"ready": False, "reason": "scale_required", "normalized_values": []}
        if not known_currency_compatible:
            return {"ready": False, "reason": "currency_required_or_conflict", "normalized_values": []}
        return {
            "ready": True,
            "reason": None,
            "normalized_values": [
                str((_decimal(item["value"]) * _decimal(scale)).normalize())
                for item, scale in zip(operands, scales, strict=True)
            ],
            "scale_contract": "resolved",
            "currency_contract": "resolved",
        }
    return {"ready": False, "reason": "operation_contract_missing", "normalized_values": []}
