"""Thin adapters from existing financial runtimes to the I1 contract.

I2 deliberately does not route production endpoints through this module.
It wraps an already-created V1 RAGEngine instance and preserves the current
V1 execution dependencies and lifecycle.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable, Mapping
from typing import Any, TYPE_CHECKING, Protocol

from .runtime_contract import (
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeMetadata,
    RuntimeRouterMode,
    RuntimeStatus,
    RuntimeVersion,
)

if TYPE_CHECKING:
    from src.services.rag_engine import RAGEngine


class LegacyFinancialRuntimeAdapterError(RuntimeError):
    """Base error for invalid input to the legacy V1 adapter."""


class UnsupportedResolvedQueryError(LegacyFinancialRuntimeAdapterError):
    """Raised until the legacy query rewrite bypass is implemented."""


class _LegacyEngine(Protocol):
    """Small duck-typed surface needed from an existing RAGEngine instance."""

    async def query(
        self,
        question: str,
        doc_names: list[str] | None = None,
        user_id: int | None = None,
        n_results: int = 3,
        conversation_history: list[dict[str, Any]] | None = None,
        memory_profile: dict[str, Any] | None = None,
    ) -> Mapping[str, Any]: ...


_LEGACY_RESPONSE_FIELDS = (
    "searched_docs",
    "rewritten_question",
    "confidence",
    "context_sufficient",
    "intent",
    "intent_confidence",
    "trace_id",
    "retrieved_chunks",
    "retrieval_debug",
    "calculations",
    "answerability",
    "validation",
    "repair",
)


class LegacyFinancialRuntimeAdapter(FinancialQARuntime):
    """Expose an existing V1 RAGEngine through the I1 runtime port.

    The adapter is intentionally thin. It never constructs a retriever,
    gateway, orchestrator, validator, or calculator. It also owns no session
    lifecycle; session_id is preserved on the request contract for the future
    execution service, while the current endpoint remains responsible for
    loading and persisting conversation messages.

    During I2, pre-resolved queries fail fast because V1 still performs its
    own legacy rewrite when conversation history is supplied. This prevents
    an advanced resolver from being silently followed by a second rewrite.
    """

    def __init__(self, engine: RAGEngine | _LegacyEngine) -> None:
        self._engine = engine

    async def execute(self, request: FinancialQueryRequest) -> FinancialQueryResult:
        """Execute one request using the already-created V1 engine.

        Engine exceptions are converted to an explicit ERROR result with a
        non-release status. Exception text is never copied into the result.
        Invalid adapter input and the unsupported resolved-query mode fail
        fast with a typed exception before the engine is called.
        """
        self._validate_request(request)
        user_id = self._legacy_user_id(request.user_id)
        doc_names, n_results, conversation_history, memory_profile = (
            self._request_options(request)
        )

        try:
            raw_result = await self._engine.query(
                question=request.standalone_query,
                doc_names=doc_names,
                user_id=user_id,
                n_results=n_results,
                conversation_history=conversation_history,
                memory_profile=memory_profile,
            )
        except Exception as exc:
            return self._error_result(
                reason_code="LEGACY_RUNTIME_EXCEPTION",
                exception=exc,
            )

        if not isinstance(raw_result, Mapping):
            return self._error_result(
                reason_code="LEGACY_RESULT_INVALID",
                exception=TypeError("V1 engine result must be a mapping"),
            )
        try:
            return self._map_result(raw_result)
        except (TypeError, ValueError, KeyError) as exc:
            return self._error_result(
                reason_code="LEGACY_RESULT_INVALID",
                exception=exc,
            )

    @staticmethod
    def _validate_request(request: FinancialQueryRequest) -> None:
        if not isinstance(request, FinancialQueryRequest):
            raise TypeError("request must be FinancialQueryRequest")
        if request.query_as_resolved:
            raise UnsupportedResolvedQueryError(
                "Legacy V1 adapter does not yet support pre-resolved queries. "
                "Legacy rewrite bypass must be integrated before "
                "query_as_resolved=True is allowed.",
            )
        if request.standalone_query != request.original_query:
            raise UnsupportedResolvedQueryError(
                "standalone_query differs from original_query while "
                "query_as_resolved is false; the legacy rewrite bypass is "
                "not available in I2.",
            )

    @staticmethod
    def _legacy_user_id(user_id: str) -> int:
        try:
            return int(user_id)
        except (TypeError, ValueError) as exc:
            raise LegacyFinancialRuntimeAdapterError(
                "Legacy V1 requires a numeric user_id because its current "
                "retrieval and vector-store scope uses integer user IDs.",
            ) from exc

    @classmethod
    def _request_options(
        cls,
        request: FinancialQueryRequest,
    ) -> tuple[
        list[str] | None,
        int,
        list[dict[str, Any]],
        dict[str, Any] | None,
    ]:
        metadata = request.request_metadata
        doc_names = cls._optional_string_list(
            metadata.get("document_names"),
            "request_metadata.document_names",
        )
        n_results = metadata.get("n_results", 3)
        if isinstance(n_results, bool) or not isinstance(n_results, int):
            raise LegacyFinancialRuntimeAdapterError(
                "request_metadata.n_results must be a positive integer",
            )
        if n_results <= 0:
            raise LegacyFinancialRuntimeAdapterError(
                "request_metadata.n_results must be a positive integer",
            )

        history_value = metadata.get("conversation_history", [])
        conversation_history = cls._history_list(history_value)
        memory_profile_value = metadata.get("memory_profile")
        if memory_profile_value is not None and not isinstance(
            memory_profile_value,
            Mapping,
        ):
            raise LegacyFinancialRuntimeAdapterError(
                "request_metadata.memory_profile must be a mapping or None",
            )
        memory_profile = (
            None
            if memory_profile_value is None
            else copy.deepcopy(dict(memory_profile_value))
        )
        return doc_names, n_results, conversation_history, memory_profile

    @staticmethod
    def _optional_string_list(
        value: Iterable[Any] | None,
        field_name: str,
    ) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise LegacyFinancialRuntimeAdapterError(
                f"{field_name} must be a list of strings or None",
            )
        result: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise LegacyFinancialRuntimeAdapterError(
                    f"{field_name} must contain non-empty strings",
                )
            result.append(item.strip())
        return result

    @staticmethod
    def _history_list(value: Any) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise LegacyFinancialRuntimeAdapterError(
                "request_metadata.conversation_history must be a list",
            )
        history: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise LegacyFinancialRuntimeAdapterError(
                    "conversation_history entries must be mappings",
                )
            history.append(copy.deepcopy(dict(item)))
        return history

    @classmethod
    def _map_result(cls, raw_result: Mapping[str, Any]) -> FinancialQueryResult:
        answer = raw_result.get("answer")
        if not isinstance(answer, str):
            raise TypeError("V1 result answer must be a string")

        sources = cls._mapping_list(raw_result.get("sources"), "sources")
        calculations = cls._mapping_list(
            raw_result.get("calculations"),
            "calculations",
        )
        answerability = cls._optional_mapping(
            raw_result.get("answerability"),
            "answerability",
        )
        validation = cls._optional_mapping(
            raw_result.get("validation"),
            "validation",
        )
        repair = cls._optional_mapping(raw_result.get("repair"), "repair")

        status, release_status = cls._runtime_status(
            answerability=answerability,
            validation=validation,
            repair=repair,
            calculations=calculations,
        )
        reason_codes = cls._reason_codes(
            answerability=answerability,
            validation=validation,
            calculations=calculations,
        )
        evidence_ids, citation_ids, calculation_ids = cls._provenance(
            sources,
            calculations,
        )

        debug_metadata: dict[str, Any] = {
            "adapter": "legacy_v1",
            # The response mapper consumes this compatibility payload to
            # preserve the existing QueryResponse shape. It contains only
            # already-public V1 fields; answer and sources are carried by the
            # typed contract fields above.
            "legacy_response": {
                key: copy.deepcopy(raw_result[key])
                for key in _LEGACY_RESPONSE_FIELDS
                if key in raw_result
            },
        }
        for key in (
            "trace_id",
            "intent",
            "intent_confidence",
            "context_sufficient",
        ):
            if key in raw_result:
                debug_metadata[key] = copy.deepcopy(raw_result[key])
        if answerability is not None and "status" in answerability:
            debug_metadata["v1_answerability_status"] = answerability["status"]
        if validation is not None and "status" in validation:
            debug_metadata["v1_validation_status"] = validation["status"]
        if calculations:
            debug_metadata["v1_calculation_statuses"] = [
                item.get("status") for item in calculations
            ]

        return FinancialQueryResult(
            status=status,
            answer=answer,
            citations=sources,
            evidence_ids=evidence_ids,
            citation_ids=citation_ids,
            calculation_ids=calculation_ids,
            reason_codes=reason_codes,
            runtime_version=RuntimeVersion.V1,
            router_mode=RuntimeRouterMode.ACTIVE,
            release_status=release_status,
            latency_metadata={},
            debug_metadata=debug_metadata,
            runtime_metadata=RuntimeMetadata(
                implementation="legacy_v1",
                attributes={"adapter": "legacy_v1"},
            ),
        )

    @staticmethod
    def _mapping_list(value: Any, field_name: str) -> list[dict[str, Any]]:
        if value is None:
            return []
        if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
            raise TypeError(f"V1 {field_name} must be a list")
        result: list[dict[str, Any]] = []
        for item in value:
            if not isinstance(item, Mapping):
                raise TypeError(f"V1 {field_name} entries must be mappings")
            result.append(copy.deepcopy(dict(item)))
        return result

    @staticmethod
    def _optional_mapping(
        value: Any,
        field_name: str,
    ) -> dict[str, Any] | None:
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError(f"V1 {field_name} must be a mapping or None")
        return copy.deepcopy(dict(value))

    @staticmethod
    def _runtime_status(
        *,
        answerability: Mapping[str, Any] | None,
        validation: Mapping[str, Any] | None,
        repair: Mapping[str, Any] | None,
        calculations: Iterable[Mapping[str, Any]],
    ) -> tuple[RuntimeStatus, ReleaseStatus]:
        answerability_status = str(
            answerability.get("status", "") if answerability is not None else "",
        ).lower()
        if answerability_status in {"not_answerable", "calculation_blocked"}:
            return RuntimeStatus.FAIL_CLOSED, ReleaseStatus.NOT_RELEASED

        calculation_statuses = {
            str(item.get("status", "")).lower() for item in calculations
        }
        if calculation_statuses & {"blocked", "failed"}:
            return RuntimeStatus.FAIL_CLOSED, ReleaseStatus.NOT_RELEASED

        if repair is not None and bool(repair.get("fallback_used")):
            return RuntimeStatus.FAIL_CLOSED, ReleaseStatus.NOT_RELEASED

        if validation is None:
            # V1 can be constructed with validation disabled in tests. The
            # adapter must not claim a trusted release without a validation
            # verdict that actually exists.
            return RuntimeStatus.ANSWER, ReleaseStatus.NOT_APPLICABLE

        validation_status = str(validation.get("status", "")).lower()
        if validation_status == "passed":
            return RuntimeStatus.ANSWER, ReleaseStatus.RELEASED
        if validation_status == "not_applicable":
            return RuntimeStatus.ANSWER, ReleaseStatus.NOT_APPLICABLE
        if validation_status in {"blocked", "failed", "repairable"}:
            return RuntimeStatus.FAIL_CLOSED, ReleaseStatus.NOT_RELEASED
        return RuntimeStatus.ANSWER, ReleaseStatus.NOT_APPLICABLE

    @staticmethod
    def _reason_codes(
        *,
        answerability: Mapping[str, Any] | None,
        validation: Mapping[str, Any] | None,
        calculations: Iterable[Mapping[str, Any]],
    ) -> list[str]:
        codes: list[str] = []

        def add(value: Any) -> None:
            if isinstance(value, str) and value and value not in codes:
                codes.append(value)

        if answerability is not None:
            reason_values = answerability.get("reason_codes", ())
            if isinstance(reason_values, Iterable) and not isinstance(
                reason_values,
                (str, bytes),
            ):
                for value in reason_values:
                    add(value)
        if validation is not None:
            issues = validation.get("issues", ())
            if isinstance(issues, Iterable) and not isinstance(issues, (str, bytes)):
                for issue in issues:
                    if isinstance(issue, Mapping):
                        add(issue.get("code"))
        for calculation in calculations:
            add(calculation.get("error_code"))
        return codes

    @staticmethod
    def _provenance(
        sources: Iterable[Mapping[str, Any]],
        calculations: Iterable[Mapping[str, Any]],
    ) -> tuple[list[str], list[str], list[str]]:
        evidence_ids: list[str] = []
        citation_ids: list[str] = []
        calculation_ids: list[str] = []

        def add_unique(target: list[str], value: Any) -> None:
            if isinstance(value, str) and value and value not in target:
                target.append(value)

        for source in sources:
            add_unique(evidence_ids, source.get("evidence_id"))
            add_unique(evidence_ids, source.get("chunk_id"))
            add_unique(citation_ids, source.get("citation_id"))

        for calculation in calculations:
            add_unique(calculation_ids, calculation.get("calculation_id"))
            operands = calculation.get("operands", ())
            if not isinstance(operands, Iterable) or isinstance(
                operands,
                (str, bytes),
            ):
                continue
            for operand in operands:
                if not isinstance(operand, Mapping):
                    continue
                add_unique(evidence_ids, operand.get("evidence_chunk_id"))
                add_unique(evidence_ids, operand.get("evidence_id"))
                add_unique(citation_ids, operand.get("citation_id"))

        return evidence_ids, citation_ids, calculation_ids

    @staticmethod
    def _error_result(
        *,
        reason_code: str,
        exception: BaseException,
    ) -> FinancialQueryResult:
        return FinancialQueryResult(
            status=RuntimeStatus.ERROR,
            answer=None,
            reason_codes=[reason_code],
            runtime_version=RuntimeVersion.V1,
            router_mode=RuntimeRouterMode.ACTIVE,
            release_status=ReleaseStatus.NOT_RELEASED,
            latency_metadata={},
            debug_metadata={
                "adapter": "legacy_v1",
                "exception_type": type(exception).__name__,
            },
            runtime_metadata=RuntimeMetadata(
                implementation="legacy_v1",
                attributes={"adapter": "legacy_v1"},
            ),
        )
