from __future__ import annotations

import hashlib
from typing import Any


def project(
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
        payload = source["semantic_payload"]
        dimension = match.get("matrix_dimension")
        value = (
            (dimension or {}).get("value_normalized")
            or (dimension or {}).get("value_raw")
            or payload.get("value_normalized")
            or payload.get("value_raw")
        )
        scale = payload.get("scale")
        currency = payload.get("currency_code")
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
        dimension_identity = (dimension or {}).get("dimension_identity")
        binding = hashlib.sha256(
            f"{plan['plan_id']}|{slot['slot_id']}|{match['evidence_id']}|{dimension_identity or ''}".encode()
        ).hexdigest()
        operands[slot["slot_id"]] = {
            "operand_binding_id": binding,
            "evidence_id": match["evidence_id"],
            "dimension_identity": dimension_identity,
            "matrix_dimension": dimension,
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
