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
            "Legacy V1 runtime execution failed before a response was produced.",
        )
    if result.answer is None:
        raise LegacyResponseMappingError(
            "Legacy V1 runtime returned no answer.",
        )

    legacy_payload = result.debug_metadata.get("legacy_response")
    if not isinstance(legacy_payload, Mapping):
        raise LegacyResponseMappingError(
            "Legacy V1 response compatibility payload is missing.",
        )
    payload = copy.deepcopy(dict(legacy_payload))
    if "searched_docs" not in payload:
        raise LegacyResponseMappingError(
            "Legacy V1 response is missing searched_docs.",
        )
    payload["answer"] = result.answer
    payload["sources"] = copy.deepcopy(result.citations)
    return payload
