from __future__ import annotations

from typing import Any


def project_operands(
    plan: dict[str, Any], result: dict[str, Any], evidence: list[dict[str, Any]]
) -> dict[str, Any]:
    if result["primary_status"] != "unique":
        return {
            "operation": plan.get("operation"),
            "operands": {},
            "typed_calculation_ready": False,
            "blocked_reason": "primary_not_unique",
        }
    primary = next(
        item
        for item in result["sets"]
        if item["evidence_set_id"] == result["primary_set_id"]
    )
    by_id = {item["evidence_id"]: item for item in evidence}
    operands = {}
    ready = True
    for slot in plan.get("operand_slots") or []:
        match = primary["slot_mapping"].get(slot["slot_id"])
        if not match:
            ready = False
            continue
        source = by_id[match["evidence_id"]]
        payload = source.get("payload") or {}
        value = next(
            (
                payload.get(field)
                for field in ("parsed_value", "value", "raw_value", "values")
                if payload.get(field) not in (None, "", [])
            ),
            None,
        )
        scale = payload.get("scale")
        currency = payload.get("currency")
        conflict = (
            payload.get("scale_status") == "conflict"
            or payload.get("currency_status") == "conflict"
        )
        slot_ready = (
            match["typed"]
            and value is not None
            and bool(source.get("source_traceback"))
            and not conflict
        )
        ready = ready and slot_ready
        operands[slot["slot_id"]] = {
            "evidence_id": match["evidence_id"],
            "value": value,
            "scale": scale,
            "currency": currency,
            "ready": slot_ready,
        }
    ready = ready and len(operands) == len(plan.get("operand_slots") or [])
    return {
        "operation": plan.get("operation"),
        "operands": operands,
        "typed_calculation_ready": ready,
        "blocked_reason": None if ready else "operand_contract_incomplete",
    }
