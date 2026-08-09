"""Deterministic Gate 08 R5 field projections for frozen Grade-A views."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

FIELD_SCHEMA_VERSION = "candidate-field-v1"
FIELD_NAMES = ("metric", "axis", "context", "evidence")


def _clean(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _values(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({_clean(value) for value in values if _clean(value)}))


def field_view_id(candidate_key: str, field_name: str) -> str:
    if field_name not in FIELD_NAMES:
        raise ValueError(f"unknown_field:{field_name}")
    raw = f"{candidate_key}|{FIELD_SCHEMA_VERSION}|{field_name}"
    return "cfv:" + hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class CandidateFieldView:
    candidate_key: str
    field_name: str
    field_view_id: str
    document_id: str
    retrieval_text: str

    def to_dict(self) -> dict[str, str]:
        return {
            "candidate_key": self.candidate_key,
            "field_name": self.field_name,
            "field_view_id": self.field_view_id,
            "document_id": self.document_id,
            "retrieval_text": self.retrieval_text,
        }


def _metric_text(record: dict[str, Any]) -> str:
    values: set[str] = set()
    for path in record.get("metric_paths") or []:
        path_text = _clean(path)
        if not path_text:
            continue
        values.add(_clean(path_text.replace("/", " ")))
        values.update(_clean(part) for part in path_text.split("/") if _clean(part))
    return "METRIC:\n" + "\n".join(sorted(values)) if values else ""


def _axis_text(record: dict[str, Any]) -> str:
    groups = (
        ("PERIOD", _values(record.get("periods") or [])),
        ("TEMPORAL", _values(record.get("temporal_types") or [])),
        ("SEGMENT", _values(record.get("segments") or [])),
        ("BUCKET", _values(record.get("buckets") or [])),
    )
    return "\n\n".join(f"{name}:\n" + "\n".join(values) for name, values in groups if values)


def _context_text(record: dict[str, Any]) -> str:
    groups = (
        ("SECTION", _values(record.get("section_path") or [])),
        ("TABLE", _values([record.get("table_title")])),
        ("TYPE", _values([record.get("candidate_type")])),
    )
    return "\n\n".join(f"{name}:\n" + "\n".join(values) for name, values in groups if values)


def _evidence_text(record: dict[str, Any]) -> str:
    lines: set[str] = set()
    for fact in record.get("facts") or []:
        if not isinstance(fact, dict):
            continue
        parts = [
            _clean(fact.get("type")),
            _clean(fact.get("metric")),
            _clean(fact.get("period")),
            *list(_values(fact.get("periods") or [])),
            _clean(fact.get("bucket")),
        ]
        line = " ".join(part for part in parts if part)
        if line:
            lines.add(line)
    matrix = record.get("row_matrix")
    if isinstance(matrix, dict):
        line = " ".join(["row_matrix", *_values(matrix.get("periods") or [])])
        lines.add(line)
    return "EVIDENCE:\n" + "\n".join(sorted(lines)) if lines else ""


def project_candidate_fields(record: dict[str, Any]) -> dict[str, CandidateFieldView]:
    key = _clean(record.get("candidate_key"))
    if not key:
        raise ValueError("missing_candidate_key")
    document_id = _clean(record.get("document_id"))
    texts = {
        "metric": _metric_text(record),
        "axis": _axis_text(record),
        "context": _context_text(record),
        "evidence": _evidence_text(record),
    }
    return {
        field: CandidateFieldView(key, field, field_view_id(key, field), document_id, text)
        for field, text in texts.items()
    }


def canonical_projection_hash(records: list[dict[str, Any]]) -> str:
    payload = [
        project_candidate_fields(record)[field].to_dict()
        for record in sorted(records, key=lambda item: str(item.get("candidate_key") or ""))
        for field in FIELD_NAMES
    ]
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()
