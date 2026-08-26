"""TV2-07 frozen integrated runtime readiness evaluation.

The runner sees only query rows.  Gold labels are loaded by the scorer after
execution, so expected answers and evidence cannot enter a V1/V2 request.
This module is evaluation-only and does not change the production router,
Conversation layer, or Trusted V2 algorithms.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import platform
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.runtime.runtime_contract import (
    FinancialQARuntime,
    FinancialQueryRequest,
    FinancialQueryResult,
    ReleaseStatus,
    RuntimeStatus,
)


class TV2ReadinessDecision(str, Enum):
    READY_FOR_CANARY = "READY_FOR_CANARY"
    HOLD_FOR_QUALITY = "HOLD_FOR_QUALITY"
    BLOCKED_FOR_SAFETY = "BLOCKED_FOR_SAFETY"


class TV2ReadinessOutcome(str, Enum):
    CORRECT_RELEASE = "CORRECT_RELEASE"
    CORRECT_RELEASE_AFTER_REPAIR = "CORRECT_RELEASE_AFTER_REPAIR"
    CORRECT_FAIL_CLOSED = "CORRECT_FAIL_CLOSED"
    OVER_CONSERVATIVE_FAIL_CLOSED = "OVER_CONSERVATIVE_FAIL_CLOSED"
    UNSAFE_INCORRECT_RELEASE = "UNSAFE_INCORRECT_RELEASE"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    TIMEOUT = "TIMEOUT"


HARD_GATE_NAMES = (
    "UNSAFE_RELEASES",
    "RELEASED_INCORRECT",
    "FALSE_BINDING",
    "FALSE_CALCULATION_EXECUTION",
    "CALCULATION_RELEASED_INCORRECT",
    "UNKNOWN_CITATION_RELEASE",
    "UNSUPPORTED_CLAIM_RELEASE",
    "WRONG_PERIOD_RELEASE",
    "WRONG_UNIT_SCALE_RELEASE",
    "ASSISTANT_HISTORY_FACT_LEAK",
    "UNVALIDATED_RELEASE",
    "REPAIR_ATTEMPTS_GT_1",
    "V2_INTERNAL_V1_FALLBACK",
    "UNEXPECTED_RUNTIME_ERROR",
    "UNEXPECTED_TIMEOUT",
    "GOLD_EVIDENCE_INJECTION",
    "GOLD_RUNTIME_LEAK",
)

_RAW_CONTEXT_KEYS = frozenset({
    "conversation_history",
    "raw_history",
    "raw_turns",
    "recent_turns",
    "messages",
    "memory_profile",
})
_LABEL_KEY_PREFIXES = (
    "expected",
    "gold",
    "answerable",
    "release",
    "evidence_ids",
    "citation_ids",
    "calculation_ids",
    "label",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(value: Any) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


def _unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            result.append(item)
            seen.add(item)
    return tuple(result)


def _hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping")
    return copy.deepcopy(dict(value))


_RUNTIME_GOLD_KEYS = frozenset({
    "gold",
    "gold_answer",
    "gold_evidence",
    "gold_evidence_ids",
    "answerability",
    "expected_release",
    "expected_route",
    "expected_calculation",
    "review_result",
    "reference_answer",
})


def _assert_runtime_request_is_blind(
    value: Any,
    *,
    path: str = "request",
) -> None:
    """Reject labels/reference data before either runtime is invoked."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).strip().lower()
            if name in _RUNTIME_GOLD_KEYS:
                raise ValueError(
                    f"Gold field is not allowed in runtime request: {path}.{key}"
                )
            if name in _RAW_CONTEXT_KEYS:
                raise ValueError(
                    f"raw context is not allowed in runtime request: {path}.{key}"
                )
            _assert_runtime_request_is_blind(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_runtime_request_is_blind(child, path=f"{path}[{index}]")


def _assert_query_metadata_is_blind(value: Any, *, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).strip().lower()
            if name in _RAW_CONTEXT_KEYS:
                raise ValueError(
                    f"raw context is not allowed in TV2-07 query metadata: {path}.{key}"
                )
            if name.startswith(_LABEL_KEY_PREFIXES):
                raise ValueError(
                    f"Gold field is not allowed in TV2-07 query metadata: {path}.{key}"
                )
            _assert_query_metadata_is_blind(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_query_metadata_is_blind(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class TV2ReadinessQuery:
    """The only case data visible during execution."""

    case_id: str
    question: str
    category: str
    tags: tuple[str, ...] = ()
    original_query: str | None = None
    standalone_query: str | None = None
    query_as_resolved: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    input_turns: tuple[dict[str, str], ...] = ()
    dataset_provenance: str = "tv2_07_wiring_fixture"

    def __post_init__(self) -> None:
        for name in ("case_id", "question", "category"):
            item = str(getattr(self, name)).strip()
            if not item:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, item)
        object.__setattr__(self, "tags", _unique(self.tags))
        original = str(self.original_query or self.question).strip()
        standalone = str(self.standalone_query or original).strip()
        if not original or not standalone:
            raise ValueError("original_query and standalone_query must be non-empty")
        object.__setattr__(self, "original_query", original)
        object.__setattr__(self, "standalone_query", standalone)
        if not isinstance(self.query_as_resolved, bool):
            raise TypeError("query_as_resolved must be bool")
        turns: list[dict[str, str]] = []
        for index, turn in enumerate(self.input_turns):
            if not isinstance(turn, Mapping):
                raise TypeError(f"input_turns[{index}] must be a mapping")
            role = str(turn.get("role", "")).strip().lower()
            text = str(turn.get("text", "")).strip()
            if role not in {"user", "assistant"} or not text:
                raise ValueError(
                    f"input_turns[{index}] requires role=user|assistant and non-empty text"
                )
            turn_id = str(turn.get("turn_id") or f"turn-{index + 1}").strip()
            if not turn_id:
                raise ValueError(f"input_turns[{index}].turn_id must be non-empty")
            turns.append({"turn_id": turn_id, "role": role, "text": text})
        object.__setattr__(self, "input_turns", tuple(turns))
        provenance = str(self.dataset_provenance).strip()
        if not provenance:
            raise ValueError("dataset_provenance must be non-empty")
        object.__setattr__(self, "dataset_provenance", provenance)
        metadata = _mapping(self.metadata)
        _assert_query_metadata_is_blind(metadata)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TV2ReadinessQuery":
        return cls(
            case_id=value.get("case_id"),
            question=value.get("question"),
            category=value.get("category"),
            tags=tuple(value.get("tags", ())),
            original_query=value.get("original_query"),
            standalone_query=value.get("standalone_query"),
            query_as_resolved=bool(value.get("query_as_resolved", False)),
            metadata=value.get("metadata"),
            input_turns=tuple(value.get("input_turns", value.get("turns", ())) or ()),
            dataset_provenance=value.get("dataset_provenance", "tv2_07_wiring_fixture"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "question": self.question,
            "category": self.category,
            "tags": list(self.tags),
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "query_as_resolved": self.query_as_resolved,
            "metadata": copy.deepcopy(self.metadata),
            "input_turns": [dict(turn) for turn in self.input_turns],
            "dataset_provenance": self.dataset_provenance,
        }

    def to_request(self) -> FinancialQueryRequest:
        metadata = copy.deepcopy(self.metadata)
        metadata["readiness_case_id"] = self.case_id
        return FinancialQueryRequest(
            request_id=f"tv2-07-{self.case_id}",
            user_id="tv2-07-evaluation-user",
            session_id=f"tv2-07-{self.case_id}",
            original_query=self.original_query or self.question,
            standalone_query=self.standalone_query or self.question,
            query_as_resolved=self.query_as_resolved,
            conversation_metadata={},
            request_metadata=metadata,
        )


@dataclass(frozen=True)
class TV2ReadinessLabel:
    """Gold data visible only after blind execution completes."""

    case_id: str
    category: str
    answerable: bool
    expected_release: bool
    expected_route: str | None = None
    expected_evidence_ids: tuple[str, ...] = ()
    expected_citation_ids: tuple[str, ...] = ()
    expected_calculation: dict[str, Any] | None = None
    expected_reason_codes: tuple[str, ...] = ()
    required_answer_terms: tuple[str, ...] = ()
    forbidden_answer_terms: tuple[str, ...] = ()
    forbidden_evidence_prefixes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    annotation: dict[str, Any] = field(default_factory=dict)
    dataset_provenance: str = "tv2_07_wiring_fixture"

    def __post_init__(self) -> None:
        for name in ("case_id", "category"):
            item = str(getattr(self, name)).strip()
            if not item:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, item)
        if not isinstance(self.answerable, bool) or not isinstance(self.expected_release, bool):
            raise TypeError("answerable and expected_release must be bool")
        route = self.expected_route
        object.__setattr__(
            self,
            "expected_route",
            str(route).strip() if route is not None and str(route).strip() else None,
        )
        for name in (
            "expected_evidence_ids",
            "expected_citation_ids",
            "expected_reason_codes",
            "required_answer_terms",
            "forbidden_answer_terms",
            "forbidden_evidence_prefixes",
            "tags",
        ):
            object.__setattr__(self, name, _unique(getattr(self, name)))
        if self.expected_calculation is not None:
            if not isinstance(self.expected_calculation, Mapping):
                raise TypeError("expected_calculation must be a mapping")
            object.__setattr__(
                self,
                "expected_calculation",
                copy.deepcopy(dict(self.expected_calculation)),
            )
        object.__setattr__(self, "annotation", _mapping(self.annotation))
        provenance = str(self.dataset_provenance).strip()
        if not provenance:
            raise ValueError("dataset_provenance must be non-empty")
        object.__setattr__(self, "dataset_provenance", provenance)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TV2ReadinessLabel":
        return cls(
            case_id=value.get("case_id"),
            category=value.get("category"),
            answerable=bool(value.get("answerable")),
            expected_release=bool(value.get("expected_release")),
            expected_route=value.get("expected_route"),
            expected_evidence_ids=tuple(value.get("expected_evidence_ids", ())),
            expected_citation_ids=tuple(value.get("expected_citation_ids", ())),
            expected_calculation=value.get("expected_calculation"),
            expected_reason_codes=tuple(value.get("expected_reason_codes", ())),
            required_answer_terms=tuple(value.get("required_answer_terms", ())),
            forbidden_answer_terms=tuple(value.get("forbidden_answer_terms", ())),
            forbidden_evidence_prefixes=tuple(
                value.get("forbidden_evidence_prefixes", ())
            ),
            tags=tuple(value.get("tags", ())),
            annotation=value.get("annotation"),
            dataset_provenance=value.get("dataset_provenance", "tv2_07_wiring_fixture"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "category": self.category,
            "answerable": self.answerable,
            "expected_release": self.expected_release,
            "expected_route": self.expected_route,
            "expected_evidence_ids": list(self.expected_evidence_ids),
            "expected_citation_ids": list(self.expected_citation_ids),
            "expected_calculation": copy.deepcopy(self.expected_calculation),
            "expected_reason_codes": list(self.expected_reason_codes),
            "required_answer_terms": list(self.required_answer_terms),
            "forbidden_answer_terms": list(self.forbidden_answer_terms),
            "forbidden_evidence_prefixes": list(self.forbidden_evidence_prefixes),
            "tags": list(self.tags),
            "annotation": copy.deepcopy(self.annotation),
            "dataset_provenance": self.dataset_provenance,
        }


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A serializable, structured snapshot of one runtime result."""

    runtime_version: str
    status: str
    release_status: str
    answer: str | None
    evidence_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]
    calculation_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    route: str | None
    latency_ms: float
    debug_metadata: dict[str, Any]
    runtime_metadata: dict[str, Any]
    error_code: str | None = None

    @classmethod
    def from_result(cls, result: FinancialQueryResult, *, latency_ms: float) -> "RuntimeSnapshot":
        if not isinstance(result, FinancialQueryResult):
            raise TypeError("runtime must return FinancialQueryResult")
        metadata = result.runtime_metadata.to_dict() if result.runtime_metadata else {}
        route = None
        attrs = metadata.get("attributes")
        if isinstance(attrs, Mapping) and attrs.get("route") is not None:
            route = str(attrs["route"])
        debug = copy.deepcopy(result.debug_metadata)
        trace = debug.get("trace") if isinstance(debug, Mapping) else None
        if route is None and isinstance(trace, Mapping) and trace.get("generation_route"):
            route = str(trace["generation_route"])
        return cls(
            runtime_version=_value(result.runtime_version) or "UNKNOWN",
            status=_value(result.status) or RuntimeStatus.ERROR.value,
            release_status=_value(result.release_status) or ReleaseStatus.NOT_RELEASED.value,
            answer=result.answer,
            evidence_ids=_unique(result.evidence_ids),
            citation_ids=_unique(result.citation_ids),
            calculation_ids=_unique(result.calculation_ids),
            reason_codes=_unique(result.reason_codes),
            route=route,
            latency_ms=float(latency_ms),
            debug_metadata=debug,
            runtime_metadata=metadata,
        )

    @classmethod
    def error(cls, runtime_version: str, code: str, latency_ms: float) -> "RuntimeSnapshot":
        return cls(
            runtime_version=runtime_version,
            status=RuntimeStatus.ERROR.value,
            release_status=ReleaseStatus.NOT_RELEASED.value,
            answer=None,
            evidence_ids=(),
            citation_ids=(),
            calculation_ids=(),
            reason_codes=(code,),
            route=None,
            latency_ms=float(latency_ms),
            debug_metadata={},
            runtime_metadata={},
            error_code=code,
        )

    @property
    def released(self) -> bool:
        return (
            self.status == RuntimeStatus.ANSWER.value
            and self.release_status == ReleaseStatus.RELEASED.value
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "runtime_version": self.runtime_version,
            "status": self.status,
            "release_status": self.release_status,
            "answer": self.answer,
            "evidence_ids": list(self.evidence_ids),
            "citation_ids": list(self.citation_ids),
            "calculation_ids": list(self.calculation_ids),
            "reason_codes": list(self.reason_codes),
            "route": self.route,
            "latency_ms": self.latency_ms,
            "debug_metadata": copy.deepcopy(self.debug_metadata),
            "runtime_metadata": copy.deepcopy(self.runtime_metadata),
            "error_code": self.error_code,
        }


def _trace(snapshot: RuntimeSnapshot) -> dict[str, Any]:
    debug = snapshot.debug_metadata
    trace = debug.get("trace") if isinstance(debug, Mapping) else None
    return copy.deepcopy(dict(trace)) if isinstance(trace, Mapping) else {}


def _attrs(snapshot: RuntimeSnapshot) -> dict[str, Any]:
    attrs = snapshot.runtime_metadata.get("attributes")
    return dict(attrs) if isinstance(attrs, Mapping) else {}


@dataclass(frozen=True)
class TV2ReadinessPrediction:
    """Blind prediction.  It deliberately contains no label."""

    query: TV2ReadinessQuery
    request: dict[str, Any]
    v1: RuntimeSnapshot
    v2: RuntimeSnapshot
    gold_injection_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "request": copy.deepcopy(self.request),
            "v1": self.v1.to_dict(),
            "v2": self.v2.to_dict(),
            "gold_injection_detected": self.gold_injection_detected,
        }


async def _invoke(
    runtime: Any,
    request: FinancialQueryRequest,
    runtime_version: str,
    timeout_seconds: float,
) -> RuntimeSnapshot:
    started = time.perf_counter()
    try:
        execute = runtime.execute
        if inspect.iscoroutinefunction(execute):
            result = await asyncio.wait_for(
                execute(request),
                timeout=timeout_seconds,
            )
        else:
            # Keep synchronous runtimes bounded as well.  A direct call here
            # would block the event loop before wait_for could take effect.
            result = await asyncio.wait_for(
                asyncio.to_thread(execute, request),
                timeout=timeout_seconds,
            )
            if inspect.isawaitable(result):
                result = await asyncio.wait_for(result, timeout=timeout_seconds)
        return RuntimeSnapshot.from_result(
            result,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
    except asyncio.TimeoutError:
        return RuntimeSnapshot.error(
            runtime_version,
            "TIMEOUT",
            (time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # noqa: BLE001 - per-case error is scored.
        return RuntimeSnapshot.error(
            runtime_version,
            type(exc).__name__,
            (time.perf_counter() - started) * 1000,
        )


RuntimeFactory = Callable[[], FinancialQARuntime]
RequestFactory = Callable[[TV2ReadinessQuery], FinancialQueryRequest]


class TV2IntegratedEvaluationRunner:
    """Run the same request through isolated V1 and V2 runtime factories."""

    def __init__(
        self,
        v1_factory: RuntimeFactory,
        v2_factory: RuntimeFactory,
        *,
        timeout_seconds: float = 120.0,
        request_factory: RequestFactory | None = None,
    ) -> None:
        if not callable(v1_factory) or not callable(v2_factory):
            raise TypeError("runtime factories must be callable")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.v1_factory = v1_factory
        self.v2_factory = v2_factory
        self.timeout_seconds = float(timeout_seconds)
        if request_factory is not None and not callable(request_factory):
            raise TypeError("request_factory must be callable")
        self.request_factory = request_factory

    async def run(self, queries: Sequence[TV2ReadinessQuery]) -> list[TV2ReadinessPrediction]:
        predictions: list[TV2ReadinessPrediction] = []
        seen: set[str] = set()
        for query in queries:
            if query.case_id in seen:
                raise ValueError(f"duplicate query case_id: {query.case_id}")
            seen.add(query.case_id)
            if query.input_turns and self.request_factory is None:
                raise ValueError(
                    "multi-turn readiness cases require an injected request_factory"
                )
            request = (
                self.request_factory(query)
                if self.request_factory is not None
                else query.to_request()
            )
            if not isinstance(request, FinancialQueryRequest):
                raise TypeError("request_factory must return FinancialQueryRequest")
            _assert_runtime_request_is_blind(request.to_dict())
            v1 = self.v1_factory()
            v2 = self.v2_factory()
            if not callable(getattr(v1, "execute", None)):
                raise TypeError("v1 factory did not return a runtime")
            if not callable(getattr(v2, "execute", None)):
                raise TypeError("v2 factory did not return a runtime")
            # Both runtimes receive the identical immutable logical request.
            v1_snapshot = await _invoke(v1, request, "V1", self.timeout_seconds)
            v2_snapshot = await _invoke(v2, request, "V2", self.timeout_seconds)
            predictions.append(
                TV2ReadinessPrediction(
                    query=query,
                    request=request.to_dict(),
                    v1=v1_snapshot,
                    v2=v2_snapshot,
                )
            )
        return predictions


def _ids_match(actual: Sequence[str], expected: Sequence[str]) -> bool:
    return not expected or set(actual) == set(expected)


def _contains(text: str | None, terms: Sequence[str]) -> bool:
    value = (text or "").casefold()
    return any(str(term).casefold() in value for term in terms)


def _find_structured_field(value: Any, names: Sequence[str]) -> Any:
    """Find a scalar field in structured metadata only, never answer text."""

    if isinstance(value, Mapping):
        for name in names:
            candidate = value.get(name)
            if isinstance(candidate, (str, int, float, bool)) or candidate is None:
                if name in value:
                    return candidate
        for child in value.values():
            found = _find_structured_field(child, names)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _find_structured_field(child, names)
            if found is not None:
                return found
    return None


def _normalized_field(value: Any) -> str | None:
    if value is None:
        return None
    return str(value).strip().casefold()


def _calculation_matches(snapshot: RuntimeSnapshot, expected: Mapping[str, Any]) -> bool:
    if not expected:
        return True
    expected_id = expected.get("calculation_id") or expected.get("id")
    if expected_id is not None and expected_id not in snapshot.calculation_ids:
        return False
    trace = _trace(snapshot)
    if expected.get("route") is not None and trace.get("generation_route") != expected["route"]:
        return False
    if expected.get("operand_evidence_ids"):
        if set(trace.get("bound_evidence_ids", ())) != set(expected["operand_evidence_ids"]):
            return False
    # Operation/result details are checked only when the structured runtime
    # exposes them; the scorer never treats answer text as provenance.
    structured = _attrs(snapshot).get("calculation_result")
    if isinstance(structured, Mapping):
        for key in ("operation", "result", "unit", "currency", "scale"):
            if expected.get(key) is not None and structured.get(key) != expected[key]:
                return False
    return True


def score_readiness_case(
    prediction: TV2ReadinessPrediction,
    label: TV2ReadinessLabel,
) -> dict[str, Any]:
    """Pure post-run scorer; no runtime call occurs here."""

    if prediction.query.case_id != label.case_id:
        raise ValueError("prediction/label case_id mismatch")
    v2 = prediction.v2
    trace = _trace(v2)
    attrs = _attrs(v2)
    violations: list[str] = []
    if prediction.gold_injection_detected:
        violations.extend(("GOLD_EVIDENCE_INJECTION", "GOLD_RUNTIME_LEAK"))
    if v2.error_code == "TIMEOUT":
        violations.append("UNEXPECTED_TIMEOUT")
    if v2.status == RuntimeStatus.ERROR.value:
        violations.append("UNEXPECTED_RUNTIME_ERROR")
    if int(trace.get("repair_count", 0) or 0) > 1:
        violations.append("REPAIR_ATTEMPTS_GT_1")
    if any(attrs.get(name) for name in ("v1_fallback", "fallback_runtime", "internal_v1_fallback")):
        violations.append("V2_INTERNAL_V1_FALLBACK")
    if any("V1_FALLBACK" in code for code in v2.reason_codes):
        violations.append("V2_INTERNAL_V1_FALLBACK")

    checks: dict[str, bool] = {}
    released = v2.released
    calculator_invoked = bool(trace.get("calculator_invoked", False))
    calculation_forbidden = bool(
        label.annotation.get("calculation_must_not_execute", False)
    )
    if calculation_forbidden and calculator_invoked:
        violations.append("FALSE_CALCULATION_EXECUTION")
    reason_codes_ok = (
        not label.expected_reason_codes
        or set(label.expected_reason_codes).issubset(v2.reason_codes)
    )
    checks["reason_codes"] = reason_codes_ok
    if released:
        validation_codes = {
            str(code).upper()
            for code in (
                tuple(trace.get("validation_reason_codes", ()))
                + tuple(trace.get("failed_checks", ()))
                + tuple(v2.reason_codes)
            )
        }
        if any(
            "WRONG_PERIOD" in code or "PERIOD_MISMATCH" in code
            for code in validation_codes
        ):
            violations.extend(("WRONG_PERIOD_RELEASE", "UNSAFE_RELEASES"))
        if any(
            (
                ("UNIT" in code or "SCALE" in code or "CURRENCY" in code)
                and ("WRONG" in code or "MISMATCH" in code or "INVALID" in code)
            )
            for code in validation_codes
        ):
            violations.extend(("WRONG_UNIT_SCALE_RELEASE", "UNSAFE_RELEASES"))
        validated = bool(
            trace.get("validation_passed")
            and trace.get("release_decision") == ReleaseStatus.RELEASED.value
            and trace.get("release_status") == ReleaseStatus.RELEASED.value
        )
        checks["validated_release"] = validated
        if not validated:
            violations.append("UNVALIDATED_RELEASE")
        route_ok = label.expected_route is None or v2.route == label.expected_route
        checks["route"] = route_ok
        evidence_ok = _ids_match(v2.evidence_ids, label.expected_evidence_ids)
        checks["evidence"] = evidence_ok
        if label.expected_evidence_ids and not evidence_ok:
            violations.extend(("FALSE_BINDING", "RELEASED_INCORRECT", "UNSAFE_RELEASES"))
        citation_ok = _ids_match(v2.citation_ids, label.expected_citation_ids)
        checks["citations"] = citation_ok
        if label.expected_citation_ids and not citation_ok:
            violations.extend(
                ("UNKNOWN_CITATION_RELEASE", "RELEASED_INCORRECT", "UNSAFE_RELEASES")
            )
        history_ok = not any(
            evidence_id.casefold().startswith(prefix.casefold())
            for evidence_id in v2.evidence_ids
            for prefix in label.forbidden_evidence_prefixes
        )
        checks["assistant_history_evidence"] = history_ok
        if not history_ok:
            violations.extend(
                ("ASSISTANT_HISTORY_FACT_LEAK", "RELEASED_INCORRECT", "UNSAFE_RELEASES")
            )
        required_ok = all(
            str(term).casefold() in (v2.answer or "").casefold()
            for term in label.required_answer_terms
        )
        forbidden_ok = not _contains(v2.answer, label.forbidden_answer_terms)
        checks["required_answer_terms"] = required_ok
        checks["forbidden_answer_terms"] = forbidden_ok
        if not forbidden_ok:
            violations.extend(
                ("UNSUPPORTED_CLAIM_RELEASE", "RELEASED_INCORRECT", "UNSAFE_RELEASES")
            )
        calc_ok = _calculation_matches(v2, label.expected_calculation or {})
        checks["calculation"] = calc_ok
        structured_sources = (_trace(v2), _attrs(v2))
        for field_name, aliases, violation in (
            ("period", ("period", "normalized_period", "period_end"), "WRONG_PERIOD_RELEASE"),
            (
                "unit",
                ("unit", "normalized_unit"),
                "WRONG_UNIT_SCALE_RELEASE",
            ),
            (
                "currency",
                ("currency", "normalized_currency"),
                "WRONG_UNIT_SCALE_RELEASE",
            ),
            ("scale", ("scale", "normalized_scale"), "WRONG_UNIT_SCALE_RELEASE"),
        ):
            expected_value = label.annotation.get(f"gold_{field_name}")
            if expected_value is None:
                continue
            actual_value = None
            for source in structured_sources:
                actual_value = _find_structured_field(source, aliases)
                if actual_value is not None:
                    break
            field_ok = (
                actual_value is not None
                and _normalized_field(actual_value) == _normalized_field(expected_value)
            )
            checks[field_name] = field_ok
            if not field_ok:
                violations.extend((violation, "RELEASED_INCORRECT", "UNSAFE_RELEASES"))
        if label.expected_calculation and not calc_ok:
            violations.extend(
                (
                    "CALCULATION_RELEASED_INCORRECT",
                    "RELEASED_INCORRECT",
                    "UNSAFE_RELEASES",
                )
            )
        if not route_ok or not required_ok or not validated or not reason_codes_ok:
            violations.extend(("RELEASED_INCORRECT", "UNSAFE_RELEASES"))
        if not label.answerable or not label.expected_release:
            violations.extend(("UNSAFE_RELEASES", "RELEASED_INCORRECT"))
        correct = (
            label.answerable
            and label.expected_release
            and validated
            and route_ok
            and evidence_ok
            and citation_ok
            and history_ok
            and required_ok
            and forbidden_ok
            and calc_ok
            and reason_codes_ok
            and not violations
        )
        outcome = (
            TV2ReadinessOutcome.CORRECT_RELEASE_AFTER_REPAIR
            if correct and int(trace.get("repair_count", 0) or 0) == 1
            else TV2ReadinessOutcome.CORRECT_RELEASE
            if correct
            else TV2ReadinessOutcome.UNSAFE_INCORRECT_RELEASE
        )
    elif v2.error_code == "TIMEOUT":
        outcome = TV2ReadinessOutcome.TIMEOUT
    elif v2.status == RuntimeStatus.ERROR.value:
        outcome = TV2ReadinessOutcome.EXECUTION_ERROR
    elif label.answerable and label.expected_release:
        outcome = TV2ReadinessOutcome.OVER_CONSERVATIVE_FAIL_CLOSED
    elif not label.answerable and not label.expected_release and reason_codes_ok:
        outcome = TV2ReadinessOutcome.CORRECT_FAIL_CLOSED
    else:
        outcome = TV2ReadinessOutcome.OVER_CONSERVATIVE_FAIL_CLOSED

    if outcome is TV2ReadinessOutcome.UNSAFE_INCORRECT_RELEASE:
        violations.extend(("UNSAFE_RELEASES", "RELEASED_INCORRECT"))

    violations = list(_unique(violations))
    return {
        "case_id": label.case_id,
        "category": label.category,
        "outcome": outcome.value,
        "safety_class": "BLOCKED_FOR_SAFETY" if violations else (
            "SAFE_RELEASE"
            if outcome in (
                TV2ReadinessOutcome.CORRECT_RELEASE,
                TV2ReadinessOutcome.CORRECT_RELEASE_AFTER_REPAIR,
            )
            else "SAFE_NON_RELEASE"
        ),
        "hard_gate_violations": violations,
        "correctness_checks": checks,
        "v2_released": released,
        "v2_status": v2.status,
        "v2_release_status": v2.release_status,
        "v2_route": v2.route,
        "v2_reason_codes": list(v2.reason_codes),
        "v2_evidence_ids": list(v2.evidence_ids),
        "v2_citation_ids": list(v2.citation_ids),
        "v2_calculation_ids": list(v2.calculation_ids),
        "v2_repair_count": int(trace.get("repair_count", 0) or 0),
        "v2_retrieval_rounds": len(trace.get("retrieval_rounds", ()) or ()),
        "v1": prediction.v1.to_dict(),
        "v2": prediction.v2.to_dict(),
        "request": copy.deepcopy(prediction.request),
        "query": prediction.query.to_dict(),
    }


def _pair(v1: Mapping[str, Any], v2: Mapping[str, Any]) -> str:
    left = "RELEASE" if (
        v1.get("status") == RuntimeStatus.ANSWER.value
        and v1.get("release_status") == ReleaseStatus.RELEASED.value
    ) else str(v1.get("status"))
    right = "RELEASE" if (
        v2.get("status") == RuntimeStatus.ANSWER.value
        and v2.get("release_status") == ReleaseStatus.RELEASED.value
    ) else str(v2.get("status"))
    return f"V1_{left}_V2_{right}"


@dataclass(frozen=True)
class TV2ReadinessMetrics:
    total_cases: int
    answerable_cases: int
    unanswerable_cases: int
    correct_release: int
    correct_release_after_repair: int
    correct_fail_closed: int
    over_conservative_fail_closed: int
    unsafe_incorrect_release: int
    execution_error: int
    timeout: int
    release_count: int
    release_precision: float
    answerable_release_rate: float
    safe_decision_rate: float
    strict_e2e_success: float
    runtime_completion_rate: float
    hard_gate_counts: dict[str, int]
    outcome_counts: dict[str, int]
    route_breakdown: dict[str, dict[str, int]]
    recovery_breakdown: dict[str, int]
    repair_breakdown: dict[str, int]
    v1_v2_comparison: dict[str, int]
    latency_summary: dict[str, float]
    quality_hold_reasons: tuple[str, ...]
    decision: TV2ReadinessDecision

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["decision"] = self.decision.value
        result["quality_hold_reasons"] = list(self.quality_hold_reasons)
        return copy.deepcopy(result)


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def aggregate_readiness_metrics(
    scored_cases: Sequence[Mapping[str, Any]],
    labels: Mapping[str, TV2ReadinessLabel],
    *,
    evaluation_scope_complete: bool = False,
    corpus_verified: bool = False,
    canonical_model_verified: bool = False,
) -> TV2ReadinessMetrics:
    total = len(scored_cases)
    answerable = sum(1 for label in labels.values() if label.answerable)
    outcomes = Counter(str(case["outcome"]) for case in scored_cases)
    correct_release = sum(
        1 for case in scored_cases
        if case["outcome"] in (
            TV2ReadinessOutcome.CORRECT_RELEASE.value,
            TV2ReadinessOutcome.CORRECT_RELEASE_AFTER_REPAIR.value,
        )
    )
    after_repair = outcomes[TV2ReadinessOutcome.CORRECT_RELEASE_AFTER_REPAIR.value]
    correct_fc = outcomes[TV2ReadinessOutcome.CORRECT_FAIL_CLOSED.value]
    over = outcomes[TV2ReadinessOutcome.OVER_CONSERVATIVE_FAIL_CLOSED.value]
    unsafe = outcomes[TV2ReadinessOutcome.UNSAFE_INCORRECT_RELEASE.value]
    errors = outcomes[TV2ReadinessOutcome.EXECUTION_ERROR.value]
    timeouts = outcomes[TV2ReadinessOutcome.TIMEOUT.value]
    release_count = sum(1 for case in scored_cases if case["v2_released"])
    hard = Counter()
    route: dict[str, Counter[str]] = defaultdict(Counter)
    recovery = Counter()
    repair = Counter()
    comparison = Counter()
    latencies: list[float] = []
    for case in scored_cases:
        hard.update(case["hard_gate_violations"])
        route[str(case.get("v2_route") or "UNKNOWN")][case["outcome"]] += 1
        reasons = {str(code) for code in case.get("v2_reason_codes", ())}
        rounds = int(case.get("v2_retrieval_rounds", 0) or 0)
        if "NO_PROGRESS" in reasons:
            recovery["NO_PROGRESS"] += 1
        elif any("BUDGET_EXHAUSTED" in code for code in reasons):
            recovery["BUDGET_EXHAUSTED"] += 1
        else:
            recovery[
                "INITIAL_EVIDENCE_READY"
                if rounds <= 1
                else "RECOVERED_AFTER_1_ROUND"
                if rounds == 2
                else "RECOVERED_AFTER_2_PLUS_ROUNDS"
            ] += 1
        trace = case["v2"].get("debug_metadata", {}).get("trace", {})
        if isinstance(trace, Mapping) and trace.get("repair_eligible"):
            repair["REPAIR_ELIGIBLE"] += 1
        if case.get("v2_repair_count", 0):
            repair["REPAIR_ATTEMPTED"] += 1
            repair[
                "REPAIR_SUCCESS"
                if case["outcome"] == TV2ReadinessOutcome.CORRECT_RELEASE_AFTER_REPAIR.value
                else "REPAIR_FAILURE_OR_BLOCKED"
            ] += 1
        else:
            repair["NO_REPAIR"] += 1
        comparison[_pair(case["v1"], case["v2"])] += 1
        latencies.append(float(case["v2"].get("latency_ms", 0)))
    hard = Counter({name: hard.get(name, 0) for name in HARD_GATE_NAMES})
    quality_reasons: list[str] = []
    if not evaluation_scope_complete:
        quality_reasons.append("EVALUATION_SCOPE_NOT_COMPLETE")
    if not corpus_verified:
        quality_reasons.append("CANONICAL_CORPUS_NOT_VERIFIED")
    if not canonical_model_verified:
        quality_reasons.append("CANONICAL_MODEL_NOT_VERIFIED")
    decision = (
        TV2ReadinessDecision.BLOCKED_FOR_SAFETY
        if any(hard.values())
        else TV2ReadinessDecision.HOLD_FOR_QUALITY
        if quality_reasons
        else TV2ReadinessDecision.READY_FOR_CANARY
    )
    correct = correct_release + correct_fc
    return TV2ReadinessMetrics(
        total_cases=total,
        answerable_cases=answerable,
        unanswerable_cases=total - answerable,
        correct_release=correct_release,
        correct_release_after_repair=after_repair,
        correct_fail_closed=correct_fc,
        over_conservative_fail_closed=over,
        unsafe_incorrect_release=unsafe,
        execution_error=errors,
        timeout=timeouts,
        release_count=release_count,
        release_precision=correct_release / release_count if release_count else 1.0,
        answerable_release_rate=correct_release / answerable if answerable else 1.0,
        safe_decision_rate=correct / total if total else 0.0,
        strict_e2e_success=correct / total if total else 0.0,
        runtime_completion_rate=(total - errors - timeouts) / total if total else 0.0,
        hard_gate_counts=dict(hard),
        outcome_counts=dict(outcomes),
        route_breakdown={key: dict(value) for key, value in route.items()},
        recovery_breakdown=dict(recovery),
        repair_breakdown=dict(repair),
        v1_v2_comparison=dict(comparison),
        latency_summary={
            "v2_p50_ms": _percentile(latencies, 50),
            "v2_p95_ms": _percentile(latencies, 95),
            "v2_p99_ms": _percentile(latencies, 99),
        },
        quality_hold_reasons=tuple(quality_reasons),
        decision=decision,
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row {path}:{line_no} must be an object")
            rows.append(row)
    return rows


def load_tv2_07_dataset(
    queries_path: str | Path,
    labels_path: str | Path,
) -> tuple[list[TV2ReadinessQuery], list[TV2ReadinessLabel]]:
    queries = [TV2ReadinessQuery.from_dict(row) for row in _read_jsonl(queries_path)]
    labels = [TV2ReadinessLabel.from_dict(row) for row in _read_jsonl(labels_path)]
    query_ids = [query.case_id for query in queries]
    label_ids = [label.case_id for label in labels]
    if len(query_ids) != len(set(query_ids)) or len(label_ids) != len(set(label_ids)):
        raise ValueError("duplicate TV2-07 case_id")
    if set(query_ids) != set(label_ids):
        raise ValueError("TV2-07 query/label case_id sets differ")
    by_id = {label.case_id: label for label in labels}
    for query in queries:
        if query.category != by_id[query.case_id].category:
            raise ValueError(f"TV2-07 category mismatch: {query.case_id}")
    return queries, labels


def score_predictions(
    predictions: Sequence[TV2ReadinessPrediction],
    labels: Sequence[TV2ReadinessLabel],
    *,
    evaluation_scope_complete: bool = False,
    corpus_verified: bool = False,
    canonical_model_verified: bool = False,
) -> tuple[list[dict[str, Any]], TV2ReadinessMetrics]:
    by_id = {label.case_id: label for label in labels}
    ids = [prediction.query.case_id for prediction in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(by_id):
        raise ValueError("TV2-07 prediction/label case_id sets differ")
    scored = [score_readiness_case(item, by_id[item.query.case_id]) for item in predictions]
    metrics = aggregate_readiness_metrics(
        scored,
        by_id,
        evaluation_scope_complete=evaluation_scope_complete,
        corpus_verified=corpus_verified,
        canonical_model_verified=canonical_model_verified,
    )
    return scored, metrics


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _git_sha(repo_path: str | Path) -> str | None:
    try:
        process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        return process.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _file_sha(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_tv2_07_manifest(
    *,
    repo_path: str | Path,
    queries: Sequence[TV2ReadinessQuery],
    labels: Sequence[TV2ReadinessLabel] | None = None,
    runtime_config: Mapping[str, Any] | None = None,
    corpus_hash: str | None = None,
    model_checkpoint: str | None = None,
    start_time: str | None = None,
) -> dict[str, Any]:
    query_data = [query.to_dict() for query in queries]
    label_data = [label.to_dict() for label in labels or ()]
    checkpoint_sha = _file_sha(model_checkpoint) if model_checkpoint else None
    return {
        "evaluation_name": "TV2-07 Production Readiness",
        "git_sha": _git_sha(repo_path),
        "runtime_config_hash": _hash(runtime_config or {}),
        "runtime_config": copy.deepcopy(dict(runtime_config or {})),
        "queries_sha256": _hash(query_data),
        "labels_sha256": _hash(label_data) if labels is not None else None,
        "evaluation_set_sha256": _hash({"queries": query_data, "labels": label_data}),
        "corpus_sha256": corpus_hash,
        "model_checkpoint": model_checkpoint,
        "model_checkpoint_sha256": checkpoint_sha,
        "python_version": platform.python_version(),
        "dependency_lock_sha256": _file_sha(Path(repo_path) / "uv.lock"),
        "case_count": len(queries),
        "start_time": start_time or _now(),
        "end_time": None,
        "production_runtime": "V1",
        "v2_authority": "OFF",
        "canary": "NOT_STARTED",
    }


def finalize_tv2_07_manifest(
    manifest: Mapping[str, Any],
    *,
    repo_path: str | Path,
    end_time: str | None = None,
) -> dict[str, Any]:
    result = copy.deepcopy(dict(manifest))
    result["end_time"] = end_time or _now()
    current = _git_sha(repo_path)
    result["git_sha_after_run"] = current
    result["code_sha_unchanged"] = current == manifest.get("git_sha")
    if not result["code_sha_unchanged"]:
        raise RuntimeError("TV2-07 code SHA changed during frozen evaluation")
    return result


def write_tv2_07_artifacts(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    queries: Sequence[TV2ReadinessQuery],
    labels: Sequence[TV2ReadinessLabel],
    scored_cases: Sequence[Mapping[str, Any]],
    metrics: TV2ReadinessMetrics,
    runtime_manifest: Mapping[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    _write_json(output / "manifest.json", manifest)
    query_data = [query.to_dict() for query in queries]
    label_data = [label.to_dict() for label in labels]
    _write_json(
        output / "dataset-manifest.json",
        {
            "case_count": len(queries),
            "queries_sha256": _hash(query_data),
            "labels_sha256": _hash(label_data),
            "evaluation_set_sha256": _hash({"queries": query_data, "labels": label_data}),
            "categories": dict(Counter(query.category for query in queries)),
            "tags": sorted({tag for query in queries for tag in query.tags}),
        },
    )
    _write_json(
        output / "runtime-manifest.json",
        dict(runtime_manifest or {
            "production_runtime": "V1",
            "evaluation_runtime": "TrustedFinancialRuntimeV2",
            "v2_factory": "real_tv2_05_factory_required",
            "gold_evidence_injection": False,
        }),
    )
    _write_jsonl(output / "case-results.jsonl", scored_cases)
    _write_json(output / "overall-metrics.json", metrics.to_dict())
    _write_json(
        output / "safety-gates.json",
        {
            "hard_gates": list(HARD_GATE_NAMES),
            "counts": metrics.hard_gate_counts,
            "passed": not any(metrics.hard_gate_counts.values()),
        },
    )
    _write_json(output / "route-breakdown.json", metrics.route_breakdown)
    _write_json(
        output / "calculation-breakdown.json",
        {
            "cases": [
                case
                for case in scored_cases
                if "calculation" in case["category"].casefold()
            ]
        },
    )
    _write_json(output / "recovery-breakdown.json", metrics.recovery_breakdown)
    _write_json(output / "repair-breakdown.json", metrics.repair_breakdown)
    _write_json(output / "v1-v2-comparison.json", metrics.v1_v2_comparison)
    _write_json(output / "latency-summary.json", metrics.latency_summary)
    _write_json(
        output / "error-summary.json",
        {
            "execution_error": metrics.execution_error,
            "timeout": metrics.timeout,
            "unexpected_runtime_error": metrics.hard_gate_counts["UNEXPECTED_RUNTIME_ERROR"],
        },
    )
    _write_json(
        output / "decision.json",
        {
            "decision": metrics.decision.value,
            "safety_passed": not any(metrics.hard_gate_counts.values()),
            "quality_hold_reasons": list(metrics.quality_hold_reasons),
            "production_runtime": "V1",
            "v2_production_authority": "OFF",
            "v2_canary": "NOT_STARTED",
        },
    )


__all__ = [
    "HARD_GATE_NAMES",
    "RuntimeSnapshot",
    "TV2IntegratedEvaluationRunner",
    "TV2ReadinessDecision",
    "TV2ReadinessLabel",
    "TV2ReadinessMetrics",
    "TV2ReadinessOutcome",
    "TV2ReadinessPrediction",
    "TV2ReadinessQuery",
    "RequestFactory",
    "aggregate_readiness_metrics",
    "build_tv2_07_manifest",
    "finalize_tv2_07_manifest",
    "load_tv2_07_dataset",
    "score_predictions",
    "score_readiness_case",
    "write_tv2_07_artifacts",
]
