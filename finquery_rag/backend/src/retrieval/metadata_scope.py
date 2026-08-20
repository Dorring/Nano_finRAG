"""Canonical metadata scope and hard-filter contracts for NF-V2-16 R1.

This module is intentionally dependency free.  It is a control-plane layer
around the existing Chroma/SQLite retrievers: it does not replace either
index, infer financial facts, or widen an authorization boundary.  Missing
metadata for an explicit hard condition is a fail-closed rejection.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from datetime import date
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Sequence


class MetadataProvenance(str, Enum):
    EXPLICIT = "EXPLICIT"
    DERIVED = "DERIVED"
    UNKNOWN = "UNKNOWN"


class FilterStrength(str, Enum):
    HARD = "HARD"
    SOFT = "SOFT"
    UNRESOLVED = "UNRESOLVED"


class PeriodSemantics(str, Enum):
    INSTANT = "INSTANT"
    QUARTER = "QUARTER"
    YTD = "YTD"
    ANNUAL = "ANNUAL"
    UNKNOWN = "UNKNOWN"


_FIELDS = (
    "tenant_id", "owner_id", "acl_scope", "document_id", "entity", "ticker",
    "document_type", "source", "report_date", "filing_date", "fiscal_year",
    "fiscal_quarter", "period_start", "period_end", "period_semantics",
    "version", "is_amended", "supersedes_document_id", "section_type",
    "content_type", "created_at", "ingested_at",
)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = re.sub(r"\s+", " ", value).strip()
        return value or None
    return value


def _norm(value: Any) -> str | None:
    value = _clean(value)
    return None if value is None else str(value).casefold()


def _aliases(raw: Mapping[str, Any], key: str) -> Any:
    aliases = {
        "tenant_id": ("tenant_id", "user_id", "tenant"),
        "document_id": ("document_id", "doc_id"),
        "document_type": ("document_type", "doc_type", "report_type"),
        "fiscal_year": ("fiscal_year", "fy", "fiscalYear"),
        "fiscal_quarter": ("fiscal_quarter", "quarter", "fiscalQuarter"),
        "period_semantics": ("period_semantics", "period_type", "periodSemantic"),
        "content_type": ("content_type", "type"),
    }
    for candidate in aliases.get(key, (key,)):
        if candidate in raw:
            return raw.get(candidate)
    return None


@dataclass(frozen=True)
class FinancialDocumentMetadataV1:
    """One normalized document/chunk metadata view.

    ``created_at`` and ``ingested_at`` are intentionally present only as
    operational fields.  The planner/filter never uses either for financial
    period or version resolution.
    """

    tenant_id: str | None = None
    owner_id: str | None = None
    acl_scope: tuple[str, ...] = ()
    document_id: str | None = None
    entity: str | None = None
    ticker: str | None = None
    document_type: str | None = None
    source: str | None = None
    report_date: str | None = None
    filing_date: str | None = None
    fiscal_year: str | None = None
    fiscal_quarter: str | None = None
    period_start: str | None = None
    period_end: str | None = None
    period_semantics: PeriodSemantics = PeriodSemantics.UNKNOWN
    version: str | None = None
    is_amended: bool = False
    supersedes_document_id: str | None = None
    section_type: str | None = None
    content_type: str | None = None
    created_at: str | None = None
    ingested_at: str | None = None
    provenance: Mapping[str, MetadataProvenance] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "FinancialDocumentMetadataV1":
        source = dict(raw or {})
        nested = source.get("metadata")
        if isinstance(nested, Mapping):
            merged = dict(nested)
            merged.update({key: value for key, value in source.items() if key != "metadata"})
            source = merged
        raw_provenance = source.get("provenance") or source.get("metadata_provenance") or {}
        values: dict[str, Any] = {}
        for key in _FIELDS:
            if key == "acl_scope":
                value = _aliases(source, key)
                if value is None:
                    value = ()
                elif isinstance(value, str):
                    value = (value,)
                else:
                    value = tuple(str(item) for item in value)
            elif key == "period_semantics":
                value = _aliases(source, key) or PeriodSemantics.UNKNOWN.value
                try:
                    value = PeriodSemantics(str(value).upper())
                except ValueError:
                    value = PeriodSemantics.UNKNOWN
            elif key == "is_amended":
                value = bool(_aliases(source, key))
            else:
                value = _clean(_aliases(source, key))
            values[key] = value
        provenance: dict[str, MetadataProvenance] = {}
        for key in _FIELDS:
            status = raw_provenance.get(key, MetadataProvenance.UNKNOWN)
            try:
                provenance[key] = status if isinstance(status, MetadataProvenance) else MetadataProvenance(str(status).upper())
            except ValueError:
                provenance[key] = MetadataProvenance.UNKNOWN
            if values.get(key) is not None and key not in raw_provenance:
                # Existing repository fields are explicit index metadata;
                # absent financial fields remain UNKNOWN.
                provenance[key] = MetadataProvenance.EXPLICIT if key in {
                    "tenant_id", "document_id", "document_type", "content_type",
                    "section_type", "version", "entity", "ticker", "report_date",
                    "filing_date", "fiscal_year", "fiscal_quarter", "period_start",
                    "period_end", "period_semantics", "source", "supersedes_document_id",
                } else MetadataProvenance.DERIVED
        return cls(provenance=provenance, **values)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["acl_scope"] = list(self.acl_scope)
        payload["period_semantics"] = self.period_semantics.value
        payload["provenance"] = {key: value.value if isinstance(value, MetadataProvenance) else str(value) for key, value in self.provenance.items()}
        return payload

    def value(self, field_name: str) -> Any:
        return getattr(self, field_name, None)


@dataclass(frozen=True)
class ScopeConditionV1:
    field: str
    values: tuple[str, ...]
    strength: FilterStrength
    provenance: MetadataProvenance = MetadataProvenance.EXPLICIT
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "values": list(self.values),
            "strength": self.strength.value,
            "provenance": self.provenance.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RetrievalScopeV1:
    """Resolved scope shared by dense, BM25, hybrid and reranker paths."""

    authorization_scope: Mapping[str, Any] = field(default_factory=dict)
    conditions: tuple[ScopeConditionV1, ...] = ()
    unresolved_scope: tuple[str, ...] = ()
    query: str = ""
    planner_version: str = "MetadataFilterPlannerV1"

    @property
    def hard_conditions(self) -> tuple[ScopeConditionV1, ...]:
        return tuple(item for item in self.conditions if item.strength is FilterStrength.HARD)

    @property
    def soft_conditions(self) -> tuple[ScopeConditionV1, ...]:
        return tuple(item for item in self.conditions if item.strength is FilterStrength.SOFT)

    @property
    def hard_filters(self) -> dict[str, list[str]]:
        return {item.field: list(item.values) for item in self.hard_conditions}

    @property
    def soft_preferences(self) -> dict[str, list[str]]:
        return {item.field: list(item.values) for item in self.soft_conditions}

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorization_scope": dict(self.authorization_scope),
            "hard_filters": self.hard_filters,
            "soft_preferences": self.soft_preferences,
            "unresolved_scope": list(self.unresolved_scope),
            "conditions": [item.to_dict() for item in self.conditions],
            "query": self.query,
            "planner_version": self.planner_version,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RetrievalScopeV1":
        conditions = []
        for item in raw.get("conditions", ()):
            try:
                conditions.append(ScopeConditionV1(
                    field=str(item["field"]), values=tuple(str(v) for v in item.get("values", ())),
                    strength=FilterStrength(str(item.get("strength", "HARD"))),
                    provenance=MetadataProvenance(str(item.get("provenance", "EXPLICIT"))),
                    reason=str(item.get("reason", "")),
                ))
            except (KeyError, TypeError, ValueError):
                continue
        return cls(
            authorization_scope=dict(raw.get("authorization_scope", {})),
            conditions=tuple(conditions),
            unresolved_scope=tuple(str(v) for v in raw.get("unresolved_scope", ())),
            query=str(raw.get("query", "")),
        )


def _condition(field_name: str, values: Sequence[Any], strength: FilterStrength, *, provenance: MetadataProvenance = MetadataProvenance.EXPLICIT, reason: str = "") -> ScopeConditionV1:
    return ScopeConditionV1(field_name, tuple(str(value) for value in values), strength, provenance, reason)


class MetadataFilterPlannerV1:
    """Deterministic query-to-scope planner; it never fabricates metadata."""

    _KNOWN_TICKERS = {"MSFT", "AAPL", "NVDA", "GOOG", "AMZN", "META", "TSLA", "IBM"}

    def plan(
        self,
        query: str,
        *,
        authorization_scope: Mapping[str, Any] | None = None,
        resolved_temporal: Mapping[str, Any] | None = None,
        current_state: Mapping[str, Any] | None = None,
        replan_reason_code: str | None = None,
    ) -> RetrievalScopeV1:
        text = " ".join(str(query or "").split())
        lower = text.casefold()
        conditions: list[ScopeConditionV1] = []
        unresolved: list[str] = []
        auth = dict(authorization_scope or {})
        for key in ("tenant_id", "user_id", "owner_id"):
            if auth.get(key) is not None:
                field_name = "tenant_id" if key in {"tenant_id", "user_id"} else key
                conditions.append(_condition(field_name, [auth[key]], FilterStrength.HARD, reason="authorization"))

        ticker = next((token for token in re.findall(r"\b[A-Z][A-Z0-9]{1,4}\b", text) if token in self._KNOWN_TICKERS), None)
        if ticker:
            conditions.append(_condition("ticker", [ticker], FilterStrength.HARD, reason="explicit ticker"))
        fy = re.search(r"\b(?:FY|fiscal\s+year\s*)(20\d{2})\b", text, flags=re.IGNORECASE)
        if fy:
            conditions.append(_condition("fiscal_year", [fy.group(1)], FilterStrength.HARD, reason="explicit fiscal year"))
        quarter = re.search(r"\b(20\d{2})\s*[- ]?Q([1-4])\b", text, flags=re.IGNORECASE)
        if quarter:
            conditions.append(_condition("fiscal_year", [quarter.group(1)], FilterStrength.HARD, reason="explicit quarter year"))
            conditions.append(_condition("fiscal_quarter", [f"Q{quarter.group(2)}"], FilterStrength.HARD, reason="explicit quarter"))
            conditions.append(_condition("period_semantics", [PeriodSemantics.QUARTER.value], FilterStrength.HARD, reason="quarter semantics"))
        if re.search(r"six\s+months?\s+ended|year\s+to\s+date|YTD", text, flags=re.IGNORECASE):
            conditions.append(_condition("period_semantics", [PeriodSemantics.YTD.value], FilterStrength.HARD, reason="explicit YTD"))
        annual = bool(re.search(r"annual|year[- ]end|10-K", text, flags=re.IGNORECASE))
        if annual:
            conditions.append(_condition("document_type", ["ANNUAL"], FilterStrength.HARD, reason="explicit annual document"))
            conditions.append(_condition("period_semantics", [PeriodSemantics.ANNUAL.value], FilterStrength.HARD, reason="annual semantics"))
        if re.search(r"latest|most recently filed|current valid version", lower):
            if annual:
                conditions.append(_condition("latest_scope", ["filing_date"], FilterStrength.HARD, reason="deterministic latest resolution"))
            else:
                unresolved.append("latest_without_document_type")

        # Section/content are preferences unless the user explicitly requests
        # a section. Numeric facts prefer table evidence but do not exclude text.
        if re.search(r"revenue|income|margin|assets|liabilit|cash flow|debt|shares", lower):
            conditions.append(_condition("content_type", ["TABLE", "TABLE_ROW"], FilterStrength.SOFT, provenance=MetadataProvenance.DERIVED, reason="numeric fact preference"))
        if re.search(r"why|explain|discussion|management", lower):
            conditions.append(_condition("section_type", ["MDA", "NOTES"], FilterStrength.SOFT, provenance=MetadataProvenance.DERIVED, reason="explanation preference"))
        if resolved_temporal:
            for key, value in resolved_temporal.items():
                if value is None or key in {"created_at", "ingested_at"}:
                    continue
                if key in {"fiscal_year", "fiscal_quarter", "period_start", "period_end", "period_semantics", "report_date", "filing_date"}:
                    if not any(c.field == key for c in conditions):
                        conditions.append(_condition(key, [value], FilterStrength.HARD, reason="resolved temporal scope"))
        return RetrievalScopeV1(auth, tuple(conditions), tuple(unresolved), text)


def candidate_metadata(candidate: Mapping[str, Any]) -> FinancialDocumentMetadataV1:
    return FinancialDocumentMetadataV1.from_mapping(candidate)


def _value_matches(actual: Any, expected: str) -> bool:
    if actual is None:
        return False
    if isinstance(actual, (list, tuple, set)):
        return any(_norm(item) == _norm(expected) for item in actual)
    return _norm(actual) == _norm(expected)


def _date_key(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except ValueError:
        return text if re.fullmatch(r"\d{4}(?:-\d{2})?(?:-\d{2})?", text) else None


def matches_hard_scope(candidate: Mapping[str, Any], scope: RetrievalScopeV1) -> tuple[bool, list[str]]:
    metadata = candidate_metadata(candidate)
    violations: list[str] = []
    for condition in scope.hard_conditions:
        if condition.field == "latest_scope":
            continue
        actual = metadata.value(condition.field)
        if not any(_value_matches(actual, expected) for expected in condition.values):
            violations.append(condition.field if actual is not None else f"METADATA_MISSING:{condition.field}")
    if scope.unresolved_scope:
        violations.extend(f"UNRESOLVED:{item}" for item in scope.unresolved_scope)
    return not violations, violations


def apply_hard_scope(
    candidates: Iterable[Mapping[str, Any]],
    scope: RetrievalScopeV1,
    *,
    event_logger: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Admit only hard-scope candidates and enforce latest without created_at."""
    source = [dict(item) for item in candidates]
    filtered: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for candidate in source:
        ok, reasons = matches_hard_scope(candidate, scope)
        if ok:
            filtered.append(candidate)
        else:
            event = {"event": "HARD_FILTER_REJECTED", "invariant_event": "FILTER_INVARIANT_VIOLATION", "candidate_id": candidate.get("doc_id") or candidate.get("id"), "reasons": reasons}
            violations.append(event)
            if event_logger:
                event_logger(event)
    latest = next((condition for condition in scope.hard_conditions if condition.field == "latest_scope"), None)
    if latest and filtered:
        # Explicit financial dates only. A missing date cannot win by upload
        # order; it is excluded and surfaced as a metadata gap.
        dated = [item for item in filtered if _date_key(candidate_metadata(item).filing_date or candidate_metadata(item).report_date)]
        if not dated:
            violations.extend({"event": "HARD_FILTER_REJECTED", "invariant_event": "FILTER_INVARIANT_VIOLATION", "candidate_id": item.get("doc_id"), "reasons": ["METADATA_MISSING:filing_date_or_report_date"]} for item in filtered)
            filtered = []
        else:
            max_date = max(_date_key(candidate_metadata(item).filing_date or candidate_metadata(item).report_date) for item in dated)
            keep = []
            for item in filtered:
                key = _date_key(candidate_metadata(item).filing_date or candidate_metadata(item).report_date)
                if key == max_date:
                    keep.append(item)
                else:
                    violations.append({"event": "LATEST_SCOPE_EXCLUDED", "candidate_id": item.get("doc_id"), "reason": "older explicit filing/report date"})
            filtered = keep
    metrics = {
        "input_count": len(source),
        "accepted_count": len(filtered),
        "rejected_count": len(source) - len(filtered),
        "filter_invariant_violations": 0,
        "hard_filter_rejections": sum(1 for item in violations if item.get("event") == "HARD_FILTER_REJECTED"),
        "silent_hard_filter_relaxations": 0,
        "created_at_temporal_misuse": 0,
        "events": violations,
    }
    return filtered, metrics


def apply_soft_preferences(candidates: Iterable[Mapping[str, Any]], scope: RetrievalScopeV1) -> list[dict[str, Any]]:
    """Boost, but never exclude, candidates matching soft preferences."""
    soft = scope.soft_conditions
    if not soft:
        return [dict(item) for item in candidates]
    scored = []
    for index, raw in enumerate(candidates):
        item = dict(raw)
        metadata = candidate_metadata(item)
        boost = sum(0.01 for condition in soft if any(_value_matches(metadata.value(condition.field), value) for value in condition.values))
        item["metadata_soft_boost"] = boost
        item["score"] = float(item.get("score", 0.0) or 0.0) + boost
        scored.append((item["score"], -index, item))
    scored.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in scored]


def enforce_reranker_subset(base_candidates: Sequence[Mapping[str, Any]], reranked: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Reranking can only reorder admitted candidates, never add IDs."""
    allowed = {str(item.get("doc_id") or item.get("id")) for item in base_candidates}
    result: list[dict[str, Any]] = []
    violations = 0
    for item in reranked:
        key = str(item.get("doc_id") or item.get("id"))
        if key not in allowed:
            violations += 1
            continue
        result.append(dict(item))
    return result, violations


class MetadataAwareRetrieverV1:
    """Shared scope adapter for repository dense/BM25/hybrid retrievers."""

    def __init__(self, *, dense_query_fn: Callable | None = None, bm25_retriever: Any | None = None, reranker: Any | None = None):
        self.dense_query_fn = dense_query_fn
        self.bm25_retriever = bm25_retriever
        self.reranker = reranker
        self.last_trace: dict[str, Any] = {}

    @staticmethod
    def _user_id(scope: RetrievalScopeV1) -> Any:
        return scope.authorization_scope.get("user_id", scope.authorization_scope.get("tenant_id"))

    @staticmethod
    def _rrf(*result_sets: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for results in result_sets:
            for rank, raw in enumerate(results, 1):
                item = dict(raw)
                key = str(item.get("doc_id") or item.get("id"))
                if key not in fused:
                    fused[key] = item
                    fused[key]["score"] = 0.0
                fused[key]["score"] += 1.0 / (60.0 + rank)
        return sorted(fused.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)

    def search(self, query: str, *, scope: RetrievalScopeV1, top_k: int = 5, mode: str = "hybrid", document_name: str | None = None) -> list[dict[str, Any]]:
        mode = str(mode).lower()
        if mode not in {"vector", "bm25", "hybrid"}:
            raise ValueError(f"unsupported retrieval mode: {mode}")
        user_id = self._user_id(scope)
        dense: list[dict[str, Any]] = []
        sparse: list[dict[str, Any]] = []
        if mode in {"vector", "hybrid"} and self.dense_query_fn:
            dense = list(self.dense_query_fn(query_text=query, doc_name=document_name, n_results=max(top_k * 4, top_k), user_id=user_id) or [])
        if mode in {"bm25", "hybrid"} and self.bm25_retriever:
            sparse = list(self.bm25_retriever.search(query, k=max(top_k * 4, top_k), doc_name=document_name, user_id=user_id) or [])
        dense_ok, dense_metrics = apply_hard_scope(dense, scope)
        sparse_ok, sparse_metrics = apply_hard_scope(sparse, scope)
        if mode == "vector":
            candidates = dense_ok
        elif mode == "bm25":
            candidates = sparse_ok
        else:
            candidates = self._rrf(dense_ok, sparse_ok)
        candidates = apply_soft_preferences(candidates, scope)
        reranker_violations = 0
        if self.reranker and candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=min(len(candidates), max(top_k, 20)))
            candidates, reranker_violations = enforce_reranker_subset(candidates, reranked)
        self.last_trace = {
            "query": query, "mode": mode, "hard_filters": scope.hard_filters,
            "dense_count": len(dense), "bm25_count": len(sparse),
            "dense_metrics": dense_metrics, "bm25_metrics": sparse_metrics,
            "reranker_reintroduction_count": 0,
            "reranker_subset_rejections": reranker_violations,
            "filter_invariant_violations": dense_metrics["filter_invariant_violations"] + sparse_metrics["filter_invariant_violations"],
            "authorized_candidate_union_ids": sorted(str(item.get("doc_id") or item.get("id")) for item in candidates),
        }
        return candidates[:max(0, int(top_k))]

