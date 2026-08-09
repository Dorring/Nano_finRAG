from __future__ import annotations

from typing import Any


def validate_prediction(record: dict[str, Any]) -> list[str]:
    errors = []
    pool = record.get("candidate_pool") or []
    keys = [item["candidate_key"] for item in pool]
    if len(keys) != len(set(keys)):
        errors.append("duplicate_candidate_key")
    allowed = set(keys)
    for evidence in record.get("canonical_evidence") or []:
        if evidence["candidate_key"] not in allowed:
            errors.append("evidence_outside_candidate_pool")
        if evidence["evidence_type"] != "raw_candidate" and not evidence.get(
            "source_traceback"
        ):
            errors.append("typed_source_traceback_missing")
    for evidence_set in (record.get("evidence_set_result") or {}).get("sets") or []:
        if any(
            value["candidate_key"] not in allowed
            for value in evidence_set["slot_mapping"].values()
        ):
            errors.append("set_outside_candidate_pool")
    return sorted(set(errors))
