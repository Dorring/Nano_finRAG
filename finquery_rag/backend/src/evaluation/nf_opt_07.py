"""Pure recoverability rules for NF-OPT-07 table-fact extraction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any


class Recoverability(str, Enum):
    SELF_CONTAINED_TABLE = "self_contained_table"
    RECOVERABLE_FROM_PARENT = "recoverable_from_parent"
    RECOVERABLE_FROM_SAME_TABLE_HEADER = "recoverable_from_same_table_header"
    RECOVERABLE_FROM_ADJACENT_TABLE_FRAGMENT = (
        "recoverable_from_adjacent_table_fragment"
    )
    PLAIN_TEXT_FACT = "plain_text_fact"
    NOT_RECOVERABLE = "not_recoverable"
    AMBIGUOUS_TABLE_CONTEXT = "ambiguous_table_context"


_YEAR = re.compile(r"\b(?:FY\s*)?20\d{2}\b", re.IGNORECASE)
_NUMBER = re.compile(r"(?<![A-Za-z])\(?-?\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?")
_SCALE = re.compile(r"\b(?:thousand|million|billion|trillion)s?\b", re.IGNORECASE)


@dataclass(frozen=True)
class AuditInput:
    candidate_key: str
    document_id: str
    page: int | None
    content_hash: str
    content: str
    metadata: dict[str, Any]
    parent: "AuditInput | None" = None


def is_table_candidate(item: AuditInput) -> bool:
    return item.metadata.get("type") in {"table", "table_row"} or "|" in item.content


def has_headers(item: AuditInput) -> bool:
    return len(set(_YEAR.findall(item.content))) >= 2


def has_numeric_cells(item: AuditInput) -> bool:
    return len(_NUMBER.findall(item.content)) >= 1


def has_scale(item: AuditInput) -> bool:
    return bool(_SCALE.search(item.content))


def verified_parent_relation(item: AuditInput) -> bool:
    parent = item.parent
    return bool(
        parent
        and item.metadata.get("parent_id") == parent.metadata.get("evidence_id")
        and item.document_id == parent.document_id
        and item.page == parent.page
    )


def classify_recoverability(item: AuditInput) -> tuple[Recoverability, str | None]:
    """Classify only structure visible in the candidate or explicit parent."""
    if not is_table_candidate(item):
        if (
            len(set(_YEAR.findall(item.content))) == 1
            and len(_NUMBER.findall(item.content)) == 1
        ):
            return Recoverability.PLAIN_TEXT_FACT, None
        return Recoverability.NOT_RECOVERABLE, "not_a_self_describing_fact"
    row_like = bool(item.content.splitlines()[0].strip("| ").strip())
    if has_headers(item) and row_like and has_numeric_cells(item):
        return Recoverability.SELF_CONTAINED_TABLE, None
    if verified_parent_relation(item) and item.parent and has_headers(item.parent):
        return Recoverability.RECOVERABLE_FROM_PARENT, None
    if item.metadata.get("parent_id") and item.parent is None:
        return Recoverability.NOT_RECOVERABLE, "explicit_parent_not_available"
    return Recoverability.NOT_RECOVERABLE, "missing_verified_header_or_parent"


def recoverability_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    recoverable = {
        Recoverability.SELF_CONTAINED_TABLE,
        Recoverability.RECOVERABLE_FROM_PARENT,
        Recoverability.RECOVERABLE_FROM_SAME_TABLE_HEADER,
        Recoverability.RECOVERABLE_FROM_ADJACENT_TABLE_FRAGMENT,
        Recoverability.PLAIN_TEXT_FACT,
    }
    recoverable_count = sum(
        record["recoverability"] in recoverable for record in records
    )
    ambiguous = sum(
        record["recoverability"] == Recoverability.AMBIGUOUS_TABLE_CONTEXT
        for record in records
    )
    unsupported = sum(
        record["recoverability"] == Recoverability.NOT_RECOVERABLE for record in records
    )
    passed = (
        len(records) == 22
        and recoverable_count >= 18
        and ambiguous == 0
        and unsupported <= 4
    )
    return {
        "source_count": len(records),
        "recoverable_source_count": recoverable_count,
        "ambiguous_same_page_join_count": ambiguous,
        "unsupported_source_count": unsupported,
        "gate_passed": passed,
    }
