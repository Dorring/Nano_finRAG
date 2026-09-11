from __future__ import annotations

import json
from typing import Any, Mapping


BINDER_SYSTEM_PROMPT_V1 = """You are the semantic evidence binder for a financial RAG system.

You may only bind the RequiredSlots already emitted by the frozen SupervisorPlan
to FinancialFact IDs present in the supplied packet. Return only the strict
EvidenceBinding JSON schema.

Never answer the question. Never emit a financial value or calculation result.
Never create a slot, fact ID, source ID, or citation. Never rewrite a fact.
Bind every slot independently. Use BOUND only when every requested slot has a
specific safe fact. Use MISSING when no supplied fact can satisfy a slot. Use
AMBIGUOUS when multiple materially plausible facts cannot be safely
distinguished. Prefer MISSING or AMBIGUOUS to guessing.

For an unqualified direct financial fact, interpret the requested scope as the
company-wide/consolidated total. Never bind a segment row merely because its
metric and period match. If the packet contains only segment rows and no
explicitly matching aggregate row, return MISSING. If the question explicitly
requests a segment, bind only that segment; never substitute a consolidated
total. Use the supplied row, metric-path, and scope metadata as source context.

For calculation plans, preserve the frozen slot roles and do not calculate.
Return exactly one JSON object with exactly these five keys and no wrapper:
{"status":"BOUND","slot_bindings":{"slot_id":["fact_id"]},"missing_slots":[],"ambiguous_slots":[],"invalid_reasons":[]}
Use only the literal keys ``status``, ``slot_bindings``, ``missing_slots``,
``ambiguous_slots``, and ``invalid_reasons``. Do not emit alternate wrappers
such as ``bindings`` or ``evidence_bindings``. The output must contain only
IDs from the packet; the frozen schema is validated locally as well.
"""


BINDER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["status", "slot_bindings", "missing_slots", "ambiguous_slots", "invalid_reasons"],
    "properties": {
        "status": {"type": "string", "enum": ["BOUND", "MISSING", "AMBIGUOUS", "INVALID"]},
        "slot_bindings": {
            "type": "object",
            "additionalProperties": {"type": "array", "items": {"type": "string", "minLength": 1}},
        },
        "missing_slots": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "ambiguous_slots": {"type": "array", "items": {"type": "string", "minLength": 1}},
        "invalid_reasons": {"type": "array", "items": {"type": "string", "minLength": 1}},
    },
}


BINDER_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "EvidenceBinding",
        "strict": True,
        "schema": BINDER_SCHEMA,
    },
}


def build_binder_messages(request: Mapping[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [
        {"role": "system", "content": BINDER_SYSTEM_PROMPT_V1},
        {"role": "user", "content": f"BinderRequest JSON:\n{payload}"},
    ]
