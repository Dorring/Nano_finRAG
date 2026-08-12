from __future__ import annotations

import json
from typing import Any


SUPERVISOR_SYSTEM_PROMPT_V1 = """You are the planning and tool-routing controller for a financial RAG system.

Do not answer the user's financial question.
Do not invent financial values.
Do not select evidence.
Do not calculate results.
Do not generate citations or a final natural-language answer.

Return only one strict JSON object matching the SupervisorPlan schema. The
control plane will validate it and a state machine will execute any action.
For the initial question-only planning step, next_action must be RETRIEVE;
use ABSTAIN only for a malformed, non-financial, or unsupported request.

Allowed intent values are exactly: DIRECT_FACT, MULTI_EVIDENCE, CALCULATION.
Allowed operation values are exactly: difference, growth_rate,
percentage_share, sum, average, gross_margin, net_margin, debt_ratio,
scale_conversion. Use operation null for non-calculation intents.

Each required slot must contain exactly these keys:
slot_id, metric, period, role, value_type, unit.
Use the requested financial metric and the period wording from the question;
do not invent a canonical fact label. Use unit null when it is not stated.
Use value_type numeric for a numeric fact and percentage for a percentage fact.
Use these existing operation role names when applicable:
current_period/base_period, numerator/denominator,
minuend/subtrahend, operand, value, gross_profit/revenue,
net_income/revenue, debt/assets, and value.

The JSON object must have exactly these top-level keys:
intent, required_slots, operation, next_action.
"""


SUPERVISOR_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intent", "required_slots", "operation", "next_action"],
    "properties": {
        "intent": {"type": "string", "enum": ["DIRECT_FACT", "MULTI_EVIDENCE", "CALCULATION"]},
        "required_slots": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["slot_id", "metric", "period", "role", "value_type", "unit"],
                "properties": {
                    "slot_id": {"type": "string", "minLength": 1},
                    "metric": {"type": "string", "minLength": 1},
                    "period": {"type": "string", "minLength": 1},
                    "role": {"type": "string", "minLength": 1},
                    "value_type": {"type": "string", "minLength": 1},
                    "unit": {"type": ["string", "null"]},
                },
            },
        },
        "operation": {"type": ["string", "null"], "enum": [
            "difference", "growth_rate", "percentage_share", "sum", "average",
            "gross_margin", "net_margin", "debt_ratio", "scale_conversion", None,
        ]},
        "next_action": {"type": "string", "enum": ["RETRIEVE", "ABSTAIN"]},
    },
}


def build_messages(question: str) -> list[dict[str, str]]:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    schema_text = json.dumps(SUPERVISOR_PLAN_JSON_SCHEMA, ensure_ascii=False, sort_keys=True)
    return [
        {"role": "system", "content": SUPERVISOR_SYSTEM_PROMPT_V1},
        {"role": "user", "content": f"SupervisorPlan JSON schema:\n{schema_text}\n\nQuestion:\n{question.strip()}"},
    ]
