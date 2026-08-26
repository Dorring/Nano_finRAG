"""Compatibility mapping from I3 runtime results to the legacy API dict."""

from __future__ import annotations

import copy
from collections.abc import Mapping

from .runtime_contract import (
    FinancialQueryResult,
    RuntimeStatus,
)


class LegacyResponseMappingError(RuntimeError):
    """Raised when a runtime result cannot preserve the legacy API shape."""


def to_legacy_query_dict(result: FinancialQueryResult) -> dict:
    """Return the exact public V1 payload expected by the existing endpoint.

    The adapter stores non-answer/source fields in a small compatibility
    payload containing fields that already existed in the V1 response. This
    function restores that payload and replaces only the answer and sources
    with their typed contract equivalents. Runtime status and release status
    remain internal in I3; the public QueryResponse schema is unchanged.
    """

    if not isinstance(result, FinancialQueryResult):
        raise TypeError("result must be FinancialQueryResult")
    if result.status is RuntimeStatus.ERROR:
        raise LegacyResponseMappingError(
            "Financial runtime execution failed before a response was produced.",
        )
    v2_fail_closed = (
        result.runtime_version.value == "V2"
        and result.status is RuntimeStatus.FAIL_CLOSED
    )
    if result.answer is None and not v2_fail_closed:
        raise LegacyResponseMappingError(
            "Financial runtime returned no answer.",
        )

    legacy_payload = result.debug_metadata.get("legacy_response")
    if not isinstance(legacy_payload, Mapping):
        if result.runtime_version.value != "V2":
            raise LegacyResponseMappingError(
                "Legacy V1 response compatibility payload is missing.",
            )
        # V2 owns a structured runtime contract rather than the historical
        # RAGEngine payload. Keep the public transport shape stable without
        # inferring provenance from answer text.
        fallback_answer = result.answer
        if not fallback_answer and v2_fail_closed:
            fallback_answer = (
                "The available evidence is insufficient to release a reliable "
                "financial answer."
            )
        if not fallback_answer:
            raise LegacyResponseMappingError(
                "Trusted V2 runtime returned no user-visible response.",
            )
        legacy_payload = {
            "answer": fallback_answer,
            "sources": copy.deepcopy(result.citations),
            "searched_docs": [],
            "retrieved_chunks": [],
            "retrieval_debug": {},
            "calculations": [],
        }
    payload = copy.deepcopy(dict(legacy_payload))
    if "searched_docs" not in payload:
        raise LegacyResponseMappingError(
            "Legacy V1 response is missing searched_docs.",
        )
    payload["answer"] = (
        result.answer
        if result.answer is not None
        else (
            "The available evidence is insufficient to release a reliable "
            "financial answer."
            if v2_fail_closed
            else None
        )
    )
    payload["sources"] = copy.deepcopy(result.citations)
    return payload
