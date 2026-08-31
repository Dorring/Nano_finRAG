"""Deterministic query-to-plan semantic alignment.

The SupervisorPlan contract deliberately validates structure, while the
evidence Binder validates a plan against retrieved facts.  Neither layer can
detect a plan which is internally valid but answers a different metric from
the user's question.  This module provides the small, deterministic gate
between those concerns.

The gate is intentionally conservative and finite: it only recognizes
explicit phrases from the production vocabulary and never uses fuzzy matching,
retrieval results, answer text, or another model call to repair a plan.  An
unrecognized query remains ``UNKNOWN`` for backwards compatibility unless the
caller selects the risk-based strict policy for a direct-fact plan; an explicit
recognized mismatch or ambiguity is rejected before retrieval starts.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from collections.abc import Iterable, Mapping
from typing import Any

from rag_v2.contracts.plan import Intent, SupervisorPlan


class SemanticAlignmentStatus(str, Enum):
    """Decision returned by the query-to-plan alignment gate."""

    ALIGNED = "ALIGNED"
    MISMATCH = "MISMATCH"
    AMBIGUOUS = "AMBIGUOUS"
    UNKNOWN = "UNKNOWN"


class BoundEvidenceAlignmentStatus(str, Enum):
    """Decision returned by the independent plan-to-evidence cross-check."""

    ALIGNED = "ALIGNED"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"


class UnknownSemanticPolicy(str, Enum):
    """How to handle a query whose explicit semantics are incomplete.

    ``COMPATIBILITY`` preserves the pre-gate behavior for generic queries
    such as ``Compare the two years``.  ``STRICT_DIRECT_FACT`` is a
    risk-based option for direct fact plans: without an explicit metric, the
    plan is not allowed to proceed merely because its schema is valid.
    """

    COMPATIBILITY = "compatibility"
    STRICT_DIRECT_FACT = "strict_direct_fact"


@dataclass(frozen=True)
class MetricMention:
    """One explicit, canonically recognized metric phrase in a query."""

    metric_id: str
    surface_form: str


@dataclass(frozen=True)
class MetricDefinition:
    """One canonical metric concept and its explicitly equivalent aliases.

    Keeping the ontology explicit is intentional. A schema can validate that
    ``net income`` is a legal value, but only this registry can state that it
    is not equivalent to ``operating income``. The same registry is used by
    query extraction and plan normalization so they cannot silently drift.
    """

    metric_id: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class PeriodMention:
    """One explicit period expression normalized to a stable period ID."""

    period_id: str
    surface_form: str


@dataclass(frozen=True)
class EntityMention:
    """One explicitly recognized issuer/entity in a query."""

    entity_id: str
    surface_form: str


@dataclass(frozen=True)
class OperationMention:
    """One explicitly recognized calculation/comparison operation."""

    operation_id: str
    surface_form: str


@dataclass(frozen=True)
class ScopeMention:
    """One explicitly recognized reporting scope qualifier."""

    scope_id: str
    surface_form: str


@dataclass(frozen=True)
class QuerySemanticFrame:
    """The deterministic semantic facts used by the alignment gate."""

    query: str
    metric_mentions: tuple[MetricMention, ...] = ()
    period_mentions: tuple[PeriodMention, ...] = ()
    entity_mentions: tuple[EntityMention, ...] = ()
    operation_mentions: tuple[OperationMention, ...] = ()
    scope_mentions: tuple[ScopeMention, ...] = ()

    @property
    def metric_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.metric_id for item in self.metric_mentions))

    @property
    def period_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.period_id for item in self.period_mentions))

    @property
    def entity_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.entity_id for item in self.entity_mentions))

    @property
    def operation_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(item.operation_id for item in self.operation_mentions)
        )

    @property
    def scope_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.scope_id for item in self.scope_mentions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "metric_ids": list(self.metric_ids),
            "metric_mentions": [
                {"metric_id": item.metric_id, "surface_form": item.surface_form}
                for item in self.metric_mentions
            ],
            "period_ids": list(self.period_ids),
            "period_mentions": [
                {
                    "period_id": item.period_id,
                    "surface_form": item.surface_form,
                }
                for item in self.period_mentions
            ],
            "entity_ids": list(self.entity_ids),
            "entity_mentions": [
                {"entity_id": item.entity_id, "surface_form": item.surface_form}
                for item in self.entity_mentions
            ],
            "operation_ids": list(self.operation_ids),
            "operation_mentions": [
                {
                    "operation_id": item.operation_id,
                    "surface_form": item.surface_form,
                }
                for item in self.operation_mentions
            ],
            "scope_ids": list(self.scope_ids),
            "scope_mentions": [
                {"scope_id": item.scope_id, "surface_form": item.surface_form}
                for item in self.scope_mentions
            ],
        }


@dataclass(frozen=True)
class PlanSemanticAlignment:
    """Auditable result of comparing a query frame with a SupervisorPlan."""

    status: SemanticAlignmentStatus
    query_metric_ids: tuple[str, ...] = ()
    plan_metric_ids: tuple[str, ...] = ()
    unknown_plan_metrics: tuple[str, ...] = ()
    mismatches: tuple[str, ...] = ()
    query_period_ids: tuple[str, ...] = ()
    plan_period_ids: tuple[str, ...] = ()
    unknown_plan_periods: tuple[str, ...] = ()
    unknown_query_fields: tuple[str, ...] = ()
    query_entity_ids: tuple[str, ...] = ()
    query_operation_ids: tuple[str, ...] = ()
    query_scope_ids: tuple[str, ...] = ()
    expected_metric_ids: tuple[str, ...] = ()
    expected_period_ids: tuple[str, ...] = ()
    expected_entity_ids: tuple[str, ...] = ()
    expected_scope_ids: tuple[str, ...] = ()
    ambiguous_query_fields: tuple[str, ...] = ()
    query_frame: Mapping[str, Any] | None = None
    unknown_policy: UnknownSemanticPolicy = UnknownSemanticPolicy.COMPATIBILITY
    plan_intent: Intent | None = None

    def __post_init__(self) -> None:
        # Keep the immutable dataclass boundary ergonomic for callers that
        # deserialize a trace or construct a result from JSON-like values.
        if not isinstance(self.status, SemanticAlignmentStatus):
            object.__setattr__(
                self,
                "status",
                SemanticAlignmentStatus(self.status),
            )
        object.__setattr__(
            self,
            "unknown_policy",
            coerce_unknown_semantic_policy(self.unknown_policy),
        )
        if self.plan_intent is not None and not isinstance(self.plan_intent, Intent):
            object.__setattr__(self, "plan_intent", Intent(self.plan_intent))

    @property
    def allowed(self) -> bool:
        if self.status in {
            SemanticAlignmentStatus.MISMATCH,
            SemanticAlignmentStatus.AMBIGUOUS,
        }:
            return False
        if (
            self.status is SemanticAlignmentStatus.UNKNOWN
            and self.unknown_policy is UnknownSemanticPolicy.STRICT_DIRECT_FACT
            and self.plan_intent is Intent.DIRECT_FACT
            and "metric" in self.unknown_query_fields
        ):
            return False
        return True

    @property
    def unknown_blocked(self) -> bool:
        """Whether policy, rather than a deterministic contradiction, blocks."""

        return self.status is SemanticAlignmentStatus.UNKNOWN and not self.allowed

    @property
    def ambiguity_blocked(self) -> bool:
        return self.status is SemanticAlignmentStatus.AMBIGUOUS and not self.allowed

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "allowed": self.allowed,
            "query_metric_ids": list(self.query_metric_ids),
            "plan_metric_ids": list(self.plan_metric_ids),
            "unknown_plan_metrics": list(self.unknown_plan_metrics),
            "mismatches": list(self.mismatches),
            "query_period_ids": list(self.query_period_ids),
            "plan_period_ids": list(self.plan_period_ids),
            "unknown_plan_periods": list(self.unknown_plan_periods),
            "unknown_query_fields": list(self.unknown_query_fields),
            "query_entity_ids": list(self.query_entity_ids),
            "query_operation_ids": list(self.query_operation_ids),
            "query_scope_ids": list(self.query_scope_ids),
            "expected_metric_ids": list(self.expected_metric_ids),
            "expected_period_ids": list(self.expected_period_ids),
            "expected_entity_ids": list(self.expected_entity_ids),
            "expected_scope_ids": list(self.expected_scope_ids),
            "ambiguous_query_fields": list(self.ambiguous_query_fields),
            "query_frame": (
                dict(self.query_frame) if isinstance(self.query_frame, Mapping) else None
            ),
            "unknown_policy": self.unknown_policy.value,
            "plan_intent": (
                self.plan_intent.value if self.plan_intent is not None else None
            ),
            "unknown_blocked": self.unknown_blocked,
            "ambiguity_blocked": self.ambiguity_blocked,
        }


@dataclass(frozen=True)
class BoundEvidenceSemanticCheck:
    """Independent deterministic check of admitted facts against the query."""

    status: BoundEvidenceAlignmentStatus
    mismatches: tuple[str, ...] = ()
    checked_fact_ids: tuple[str, ...] = ()
    query_metric_ids: tuple[str, ...] = ()
    query_entity_ids: tuple[str, ...] = ()
    query_period_ids: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.status is not BoundEvidenceAlignmentStatus.MISMATCH

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "allowed": self.allowed,
            "mismatches": list(self.mismatches),
            "checked_fact_ids": list(self.checked_fact_ids),
            "query_metric_ids": list(self.query_metric_ids),
            "query_entity_ids": list(self.query_entity_ids),
            "query_period_ids": list(self.query_period_ids),
        }


# Keep this vocabulary intentionally small and explicit.  Aliases are only
# added when the business meaning is equivalent; notably, operating income
# and net income are separate metrics and must never be aliases.
_METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    MetricDefinition(
        "operating_income",
        (
            "operating income",
            "operating profit",
            "income from operations",
            "operating earnings",
            "营业利润",
            "经营利润",
            "营业收益",
        ),
    ),
    MetricDefinition(
        "net_income",
        (
            "net income",
            "net profit",
            "profit after tax",
            "净利润",
            "净利",
            "归母净利润",
        ),
    ),
    MetricDefinition(
        "revenue",
        (
            "total revenue",
            "net sales",
            "sales revenue",
            "revenue",
            "revenues",
            "sales",
            "营业收入",
            "销售收入",
            "营收",
            "净销售额",
        ),
    ),
    MetricDefinition(
        "operating_margin",
        (
            "operating margin",
            "operating profit margin",
            "营业利润率",
            "经营利润率",
        ),
    ),
    MetricDefinition(
        "net_margin",
        (
            "net margin",
            "net profit margin",
            "净利率",
            "净利润率",
        ),
    ),
    MetricDefinition(
        "gross_profit",
        ("gross profit", "毛利润", "毛利"),
    ),
    MetricDefinition(
        "gross_margin",
        ("gross margin", "gross profit margin", "毛利率"),
    ),
    MetricDefinition(
        "ebitda",
        ("ebitda", "息税折旧摊销前利润"),
    ),
    MetricDefinition(
        "assets",
        ("total assets", "assets", "总资产", "资产"),
    ),
    MetricDefinition(
        "liabilities",
        ("total liabilities", "liabilities", "总负债", "负债"),
    ),
    MetricDefinition(
        "cost_of_revenue",
        (
            "cost of revenue",
            "cost of sales",
            "costs of revenue",
            "cogs",
            "营业成本",
            "销售成本",
        ),
    ),
)


@dataclass(frozen=True)
class _VocabularyDefinition:
    canonical_id: str
    aliases: tuple[str, ...]


# Entity extraction is deliberately an observation aid and an optional
# cross-check.  It is not used to invent a company when the query omits one.
_ENTITY_DEFINITIONS: tuple[_VocabularyDefinition, ...] = (
    _VocabularyDefinition("aapl", ("apple", "aapl", "苹果")),
    _VocabularyDefinition("msft", ("microsoft", "msft", "微软")),
    _VocabularyDefinition("tsla", ("tesla", "tsla", "特斯拉")),
    _VocabularyDefinition("googl", ("google", "alphabet", "googl", "谷歌")),
    _VocabularyDefinition("amzn", ("amazon", "amzn", "亚马逊")),
    _VocabularyDefinition("orcl", ("oracle", "orcl", "甲骨文")),
    _VocabularyDefinition("nvda", ("nvidia", "nvda", "英伟达")),
    _VocabularyDefinition("meta", ("meta", "facebook", "脸书")),
    _VocabularyDefinition("ko", ("coca cola", "coca-cola", "ko")),
    _VocabularyDefinition("ford", ("ford", "福特")),
)

_OPERATION_DEFINITIONS: tuple[_VocabularyDefinition, ...] = (
    _VocabularyDefinition(
        "growth_rate",
        (
            "growth rate",
            "year over year",
            "year-over-year",
            "yoy",
            "grew",
            "growth",
            "同比",
            "增长率",
        ),
    ),
    _VocabularyDefinition(
        "difference",
        ("difference", "change in", "change", "差额", "变化"),
    ),
    _VocabularyDefinition(
        "percentage_share",
        ("percentage of", "percent of", "share of", "占比", "百分比"),
    ),
    _VocabularyDefinition("sum", ("sum of", "total of", "合计")),
    _VocabularyDefinition("average", ("average", "mean", "平均")),
    _VocabularyDefinition(
        "scale_conversion",
        ("convert to", "converted to", "换算为", "转换为"),
    ),
    _VocabularyDefinition(
        "gross_margin",
        ("calculate gross margin", "gross margin percentage", "毛利率"),
    ),
    _VocabularyDefinition(
        "net_margin",
        ("calculate net margin", "net margin percentage", "净利率"),
    ),
    _VocabularyDefinition("debt_ratio", ("debt ratio", "负债率")),
)

_SCOPE_DEFINITIONS: tuple[_VocabularyDefinition, ...] = (
    _VocabularyDefinition(
        "consolidated",
        ("consolidated", "consolidated basis", "合并", "合并口径"),
    ),
    _VocabularyDefinition(
        "company_total",
        ("company-wide", "total company", "全公司", "公司整体"),
    ),
    _VocabularyDefinition("segment", ("segment", "分部", "业务线")),
)


def _normalize_surface(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("’", "'")
    # Retain Unicode word characters (including Chinese), while making
    # punctuation-separated phrases comparable to the prompt's plain text.
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _is_cjk_phrase(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in value)


def _alias_matches(text: str, alias: str) -> bool:
    if _is_cjk_phrase(alias):
        return alias in text
    return re.search(
        rf"(?<!\w){re.escape(alias)}(?!\w)",
        text,
        flags=re.UNICODE,
    ) is not None


def _extract_vocabulary_mentions(
    normalized: str,
    index: tuple[tuple[str, str], ...],
    mention_type: type[Any],
    id_field: str,
) -> tuple[Any, ...]:
    """Extract non-overlapping explicit mentions from a finite vocabulary."""

    matches: list[tuple[int, Any]] = []
    occupied: list[tuple[int, int]] = []
    for alias, canonical_id in index:
        if not _alias_matches(normalized, alias):
            continue
        start = normalized.find(alias)
        end = start + len(alias)
        if start < 0 or any(
            start < old_end and old_start < end
            for old_start, old_end in occupied
        ):
            continue
        occupied.append((start, end))
        matches.append(
            (
                start,
                mention_type(
                    **{id_field: canonical_id, "surface_form": alias},
                ),
            )
        )
    matches.sort(key=lambda item: item[0])
    return tuple(item[1] for item in matches)


_ALIAS_INDEX: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            (_normalize_surface(alias), definition.metric_id)
            for definition in _METRIC_DEFINITIONS
            for alias in definition.aliases
        ),
        key=lambda item: (-len(item[0]), item[0], item[1]),
    )
)


def _vocabulary_index(
    definitions: tuple[_VocabularyDefinition, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                (_normalize_surface(alias), definition.canonical_id)
                for definition in definitions
                for alias in definition.aliases
            ),
            key=lambda item: (-len(item[0]), item[0], item[1]),
        )
    )

_CANONICAL_IDS = frozenset(item.metric_id for item in _METRIC_DEFINITIONS)
_ENTITY_INDEX = _vocabulary_index(_ENTITY_DEFINITIONS)
_OPERATION_INDEX = _vocabulary_index(_OPERATION_DEFINITIONS)
_SCOPE_INDEX = _vocabulary_index(_SCOPE_DEFINITIONS)
_CANONICAL_ENTITY_IDS = frozenset(item.canonical_id for item in _ENTITY_DEFINITIONS)
_CANONICAL_OPERATION_IDS = frozenset(item.canonical_id for item in _OPERATION_DEFINITIONS)
_CANONICAL_SCOPE_IDS = frozenset(item.canonical_id for item in _SCOPE_DEFINITIONS)

_YEAR = r"(?:19|20)\d{2}"
# The query pattern accepts annual and quarterly expressions commonly emitted
# by the resolver (FY2024, fiscal year 2024, Q1 2024, and 2024 Q1).  Relative
# phrases such as "previous year" deliberately remain unresolved here.
_PERIOD_QUERY_RE = re.compile(
    rf"(?<!\d)(?:(?:(?:fiscal\s+year|year\s+ended|fiscal)\s+)?"
    rf"(?:fy\s*)?(?:q[1-4]\s*)?{_YEAR}(?:\s*q[1-4])?"
    rf"|q[1-4]\s*(?:(?:fy|fiscal\s+year)\s*)?{_YEAR})(?!\d)",
    flags=re.IGNORECASE,
)
_PERIOD_FULL_RE = re.compile(
    rf"^(?:(?:fiscal\s+year|year\s+ended|fiscal)\s+)?"
    rf"(?:(?P<q_before>q[1-4])\s*)?(?:fy\s*)?"
    rf"(?P<year>{_YEAR})(?:\s*(?P<q_after>q[1-4]))?$",
    flags=re.IGNORECASE,
)


def coerce_unknown_semantic_policy(
    value: UnknownSemanticPolicy | str,
) -> UnknownSemanticPolicy:
    if isinstance(value, UnknownSemanticPolicy):
        return value
    aliases = {
        "compat": UnknownSemanticPolicy.COMPATIBILITY,
        "compatibility": UnknownSemanticPolicy.COMPATIBILITY,
        "allow": UnknownSemanticPolicy.COMPATIBILITY,
        "strict": UnknownSemanticPolicy.STRICT_DIRECT_FACT,
        "strict_direct_fact": UnknownSemanticPolicy.STRICT_DIRECT_FACT,
    }
    normalized = str(value).strip().casefold()
    try:
        return aliases[normalized]
    except KeyError as exc:
        allowed = ", ".join(item.value for item in UnknownSemanticPolicy)
        raise ValueError(
            f"unknown semantic policy must be one of: {allowed}",
        ) from exc


def canonical_metric_id(value: Any) -> str | None:
    """Resolve one complete plan metric phrase to a canonical metric ID."""

    normalized = _normalize_surface(value)
    if not normalized:
        return None
    if normalized in _CANONICAL_IDS:
        return normalized
    for alias, metric_id in _ALIAS_INDEX:
        if normalized == alias:
            return metric_id
    return None


def metric_alias_registry() -> Mapping[str, tuple[str, ...]]:
    """Return a copy of the explicit metric ontology used by the gate.

    Callers may use this for audits or contract fixtures, but mutating the
    returned mapping cannot change runtime behavior.  New aliases must be
    reviewed as semantic equivalences; they are never learned from retrieval
    results or generated answer text.
    """

    return {
        definition.metric_id: tuple(definition.aliases)
        for definition in _METRIC_DEFINITIONS
    }


def _canonical_vocabulary_id(
    value: Any,
    index: tuple[tuple[str, str], ...],
) -> str | None:
    normalized = _normalize_surface(value)
    if not normalized:
        return None
    for alias, canonical_id in index:
        if normalized == alias:
            return canonical_id
    return None


def canonical_entity_id(value: Any) -> str | None:
    """Resolve a known issuer alias without guessing unknown companies."""

    return _canonical_vocabulary_id(value, _ENTITY_INDEX)


def canonical_operation_id(value: Any) -> str | None:
    """Resolve an explicit operation phrase to a plan operation ID."""

    normalized = _normalize_surface(value)
    if normalized in _CANONICAL_OPERATION_IDS:
        return normalized
    return _canonical_vocabulary_id(value, _OPERATION_INDEX)


def canonical_scope_id(value: Any) -> str | None:
    """Resolve a known reporting-scope qualifier."""

    return _canonical_vocabulary_id(value, _SCOPE_INDEX)


def canonical_period_id(value: Any) -> str | None:
    """Normalize one explicit annual or quarterly period expression."""

    normalized = _normalize_surface(value)
    if not normalized:
        return None
    match = _PERIOD_FULL_RE.fullmatch(normalized)
    if match is None:
        return None
    year = match.group("year")
    quarter = match.group("q_before") or match.group("q_after")
    if quarter:
        quarter_number = quarter[1:] if quarter[:1].casefold() == "q" else quarter
        return f"FY{year}-Q{quarter_number}"
    return f"FY{year}"


def extract_query_semantic_frame(query: str) -> QuerySemanticFrame:
    """Extract explicit semantic mentions without model or retrieval.

    This is an alignment signal, not a general-purpose NLU system.  Unknown
    companies, scopes, qualifiers, and relative periods remain unknown rather
    than being guessed.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = _normalize_surface(query)
    metric_matches: list[tuple[int, MetricMention]] = []
    occupied: list[tuple[int, int]] = []
    for alias, metric_id in _ALIAS_INDEX:
        # Chinese aliases are allowed to match inside the normalized text;
        # English aliases still use word boundaries to avoid partial matches.
        if not _alias_matches(normalized, alias):
            continue
        start = normalized.find(alias)
        end = start + len(alias)
        if start < 0 or any(start < old_end and old_start < end for old_start, old_end in occupied):
            continue
        occupied.append((start, end))
        metric_matches.append(
            (start, MetricMention(metric_id=metric_id, surface_form=alias)),
        )
    metric_matches.sort(key=lambda item: item[0])

    period_matches: list[tuple[int, PeriodMention]] = []
    period_occupied: list[tuple[int, int]] = []
    for match in _PERIOD_QUERY_RE.finditer(normalized):
        period_id = canonical_period_id(match.group(0))
        if period_id is None:
            continue
        start, end = match.span()
        if any(
            start < old_end and old_start < end
            for old_start, old_end in period_occupied
        ):
            continue
        period_occupied.append((start, end))
        period_matches.append(
            (
                start,
                PeriodMention(
                    period_id=period_id,
                    surface_form=match.group(0),
                ),
            ),
        )
    period_matches.sort(key=lambda item: item[0])
    return QuerySemanticFrame(
        query=query.strip(),
        metric_mentions=tuple(item[1] for item in metric_matches),
        period_mentions=tuple(item[1] for item in period_matches),
        entity_mentions=_extract_vocabulary_mentions(
            normalized,
            _ENTITY_INDEX,
            EntityMention,
            "entity_id",
        ),
        operation_mentions=_extract_vocabulary_mentions(
            normalized,
            _OPERATION_INDEX,
            OperationMention,
            "operation_id",
        ),
        scope_mentions=_extract_vocabulary_mentions(
            normalized,
            _SCOPE_INDEX,
            ScopeMention,
            "scope_id",
        ),
    )


def align_query_to_plan(
    query: str,
    plan: SupervisorPlan,
    *,
    unknown_policy: UnknownSemanticPolicy | str = UnknownSemanticPolicy.COMPATIBILITY,
    semantic_context: Mapping[str, Any] | None = None,
) -> PlanSemanticAlignment:
    """Compare explicit query semantics with all plan slot semantics.

    The gate rejects only deterministic contradictions.  If no recognized
    metric is present in the query, it returns ``UNKNOWN`` and leaves existing
    generic/operation-only queries backwards compatible.  Optional semantic
    context may provide already-authorized expected metric, period, entity, or
    scope expectations; it is never inferred from a candidate or from the
    answer.
    """

    if not isinstance(plan, SupervisorPlan):
        raise TypeError("plan must be a SupervisorPlan")
    policy = coerce_unknown_semantic_policy(unknown_policy)
    frame = extract_query_semantic_frame(query)
    query_metric_ids = frame.metric_ids
    query_period_ids = frame.period_ids
    query_entity_ids = frame.entity_ids
    query_operation_ids = frame.operation_ids
    query_scope_ids = frame.scope_ids
    plan_metric_ids: list[str] = []
    unknown_plan_metrics: list[str] = []
    plan_period_ids: list[str] = []
    unknown_plan_periods: list[str] = []
    mismatches: list[str] = []
    ambiguous_query_fields: list[str] = []
    for slot in plan.required_slots:
        metric_id = canonical_metric_id(slot.metric)
        if metric_id is None:
            unknown_plan_metrics.append(slot.metric)
        else:
            plan_metric_ids.append(metric_id)
        period_id = canonical_period_id(slot.period)
        if period_id is None:
            unknown_plan_periods.append(slot.period)
        else:
            plan_period_ids.append(period_id)

    plan_metric_ids_tuple = tuple(dict.fromkeys(plan_metric_ids))
    plan_period_ids_tuple = tuple(dict.fromkeys(plan_period_ids))
    expected_entity_ids = _expected_vocabulary_ids(
        semantic_context,
        keys=("entity", "resolved_entity", "active_entity"),
        canonicalizer=canonical_entity_id,
    )
    expected_metric_ids = _expected_vocabulary_ids(
        semantic_context,
        keys=("metric", "resolved_metric", "active_metric"),
        canonicalizer=canonical_metric_id,
    )
    expected_period_ids = _expected_vocabulary_ids(
        semantic_context,
        keys=("period", "resolved_period", "active_period"),
        canonicalizer=canonical_period_id,
    )
    expected_scope_ids = _expected_vocabulary_ids(
        semantic_context,
        keys=("scope", "resolved_scope", "active_scope"),
        canonicalizer=canonical_scope_id,
    )
    unknown_query_fields = ("metric",) if not query_metric_ids else ()
    # Preserve compatibility for generic or operation-only prompts without
    # allowing a missing metric to hide contradictions in fields the user did
    # state explicitly.  In particular, ``Compare the two years`` remains
    # UNKNOWN, while ``Compare FY2024`` still has its period checked.
    if query_metric_ids:
        query_set = set(query_metric_ids)
        plan_set = set(plan_metric_ids_tuple)
        result_metric = _result_metric_for_operation(plan.operation)
        for metric_id in query_metric_ids:
            if metric_id in plan_set or metric_id == result_metric:
                continue
            mismatches.append(f"query_metric_not_planned:{metric_id}")
        # A direct-fact plan has no hidden operand vocabulary: any additional
        # metric is a different answer target.  Calculation plans may
        # legitimately bind operand metrics (for example gross_profit +
        # revenue for gross_margin).
        if plan.intent is Intent.DIRECT_FACT:
            for metric_id in plan_metric_ids_tuple:
                if metric_id not in query_set:
                    mismatches.append(f"planned_metric_not_in_query:{metric_id}")

    # A caller may provide an already-authorized semantic expectation from the
    # Conversation layer.  This is deliberately a cross-check, not a source
    # of inferred values: explicit query semantics must agree with the
    # expectation, and the Supervisor plan must target the same concept.  The
    # intersection rule still permits a calculation plan to include a derived
    # result metric alongside its operand metrics.
    if expected_metric_ids:
        expected_metric_set = set(expected_metric_ids)
        if query_metric_ids and not (set(query_metric_ids) & expected_metric_set):
            mismatches.extend(
                f"query_metric_not_expected:{metric_id}"
                for metric_id in query_metric_ids
            )
        if plan_metric_ids_tuple and not (
            set(plan_metric_ids_tuple) & expected_metric_set
        ):
            mismatches.extend(
                f"planned_metric_not_expected:{metric_id}"
                for metric_id in plan_metric_ids_tuple
            )
    if len(query_metric_ids) > 1 and query_set != plan_set:
        ambiguous_query_fields.append("metric")
    if unknown_plan_metrics:
        mismatches.extend(
            f"unrecognized_plan_metric:{metric}"
            for metric in dict.fromkeys(unknown_plan_metrics)
        )

    # A comparison or growth plan may include an implicit prior period not
    # named by the query.  Require every explicit query period to be present,
    # but do not reject those derived extra periods.
    if query_period_ids:
        plan_period_set = set(plan_period_ids_tuple)
        mismatches.extend(
            f"query_period_not_planned:{period_id}"
            for period_id in query_period_ids
            if period_id not in plan_period_set
        )
        mismatches.extend(
            f"unrecognized_plan_period:{period}"
            for period in dict.fromkeys(unknown_plan_periods)
        )

    if expected_period_ids:
        expected_period_set = set(expected_period_ids)
        if query_period_ids and not (set(query_period_ids) & expected_period_set):
            mismatches.extend(
                f"query_period_not_expected:{period_id}"
                for period_id in query_period_ids
            )
        # Comparison/growth plans can contain an inferred prior period.  Only
        # require that at least one plan period matches the authorized
        # expectation; do not reject the derived companion period.
        if plan_period_ids_tuple and not (
            set(plan_period_ids_tuple) & expected_period_set
        ):
            mismatches.extend(
                f"planned_period_not_expected:{period_id}"
                for period_id in plan_period_ids_tuple
            )

    if len(query_operation_ids) > 1:
        ambiguous_query_fields.append("operation")
    elif query_operation_ids:
        query_operation = query_operation_ids[0]
        plan_operation = canonical_operation_id(plan.operation)
        if plan_operation is None:
            mismatches.append("query_operation_not_planned:" + query_operation)
        elif query_operation != plan_operation:
            mismatches.append(
                f"query_operation_not_planned:{query_operation}"
            )

    if len(query_entity_ids) > 1 and expected_entity_ids:
        ambiguous_query_fields.append("entity")
    elif query_entity_ids and expected_entity_ids and not (
        set(query_entity_ids) & set(expected_entity_ids)
    ):
        mismatches.extend(
            f"query_entity_not_expected:{entity_id}"
            for entity_id in query_entity_ids
        )
    if len(query_scope_ids) > 1 and expected_scope_ids:
        ambiguous_query_fields.append("scope")
    elif query_scope_ids and expected_scope_ids and not (
        set(query_scope_ids) & set(expected_scope_ids)
    ):
        mismatches.extend(
            f"query_scope_not_expected:{scope_id}"
            for scope_id in query_scope_ids
        )

    if ambiguous_query_fields:
        status = SemanticAlignmentStatus.AMBIGUOUS
    elif mismatches:
        status = SemanticAlignmentStatus.MISMATCH
    elif not query_metric_ids:
        status = SemanticAlignmentStatus.UNKNOWN
    else:
        status = SemanticAlignmentStatus.ALIGNED

    return PlanSemanticAlignment(
        status=status,
        query_metric_ids=query_metric_ids,
        plan_metric_ids=plan_metric_ids_tuple,
        unknown_plan_metrics=tuple(dict.fromkeys(unknown_plan_metrics)),
        mismatches=tuple(dict.fromkeys(mismatches)),
        query_period_ids=query_period_ids,
        plan_period_ids=plan_period_ids_tuple,
        unknown_plan_periods=tuple(dict.fromkeys(unknown_plan_periods)),
        unknown_query_fields=unknown_query_fields,
        query_entity_ids=query_entity_ids,
        query_operation_ids=query_operation_ids,
        query_scope_ids=query_scope_ids,
        expected_metric_ids=expected_metric_ids,
        expected_period_ids=expected_period_ids,
        expected_entity_ids=expected_entity_ids,
        expected_scope_ids=expected_scope_ids,
        ambiguous_query_fields=tuple(dict.fromkeys(ambiguous_query_fields)),
        query_frame=frame.to_dict(),
        unknown_policy=policy,
        plan_intent=plan.intent,
    )


def _expected_vocabulary_ids(
    semantic_context: Mapping[str, Any] | None,
    *,
    keys: tuple[str, ...],
    canonicalizer: Any,
) -> tuple[str, ...]:
    """Normalize optional caller-owned expectations without inventing values."""

    if not isinstance(semantic_context, Mapping):
        return ()
    values: list[Any] = []
    for key in keys:
        value = semantic_context.get(key)
        if value is None:
            continue
        if isinstance(value, (str, bytes)):
            values.append(value)
        elif isinstance(value, Iterable):
            values.extend(value)
        else:
            values.append(value)
    result: list[str] = []
    for value in values:
        canonical = canonicalizer(value)
        if canonical and canonical not in result:
            result.append(canonical)
    return tuple(result)


def _result_metric_for_operation(operation: str | None) -> str | None:
    normalized = canonical_operation_id(operation)
    return {
        "gross_margin": "gross_margin",
        "net_margin": "net_margin",
    }.get(normalized)


def _fact_value(fact: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = fact.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def align_bound_evidence_to_query(
    query: str,
    plan: SupervisorPlan,
    facts: Iterable[Mapping[str, Any]],
    slot_bindings: Mapping[str, Iterable[str]],
    *,
    selected_fact_ids: Iterable[str] | None = None,
) -> BoundEvidenceSemanticCheck:
    """Cross-check Binder-admitted facts against both plan and query.

    Binder providers remain responsible for selecting evidence.  This second,
    deterministic check is an independent firewall: a provider cannot turn a
    structurally valid but semantically wrong plan into an admitted packet.
    """

    if not isinstance(plan, SupervisorPlan):
        raise TypeError("plan must be a SupervisorPlan")
    frame = extract_query_semantic_frame(query)
    fact_map: dict[str, Mapping[str, Any]] = {}
    for fact in facts:
        if not isinstance(fact, Mapping):
            continue
        fact_id = _fact_value(fact, "fact_id", "evidence_id", "candidate_id")
        if fact_id is not None:
            fact_map[str(fact_id)] = fact
    ordered_ids: list[str] = []
    for slot_id, fact_ids in slot_bindings.items():
        del slot_id
        for fact_id in fact_ids:
            normalized = str(fact_id).strip()
            if normalized and normalized not in ordered_ids:
                ordered_ids.append(normalized)
    for fact_id in selected_fact_ids or ():
        normalized = str(fact_id).strip()
        if normalized and normalized not in ordered_ids:
            ordered_ids.append(normalized)
    if not ordered_ids:
        return BoundEvidenceSemanticCheck(
            status=BoundEvidenceAlignmentStatus.UNKNOWN,
            query_metric_ids=frame.metric_ids,
            query_entity_ids=frame.entity_ids,
            query_period_ids=frame.period_ids,
        )

    slot_map = {slot.slot_id: slot for slot in plan.required_slots}
    plan_metric_ids = {
        metric_id
        for metric_id in (canonical_metric_id(slot.metric) for slot in plan.required_slots)
        if metric_id
    }
    mismatches: list[str] = []
    for slot_id, fact_ids in slot_bindings.items():
        slot = slot_map.get(str(slot_id))
        if slot is None:
            mismatches.append(f"unknown_slot_binding:{slot_id}")
            continue
        expected_metric = canonical_metric_id(slot.metric)
        expected_period = canonical_period_id(slot.period)
        for fact_id in fact_ids:
            normalized_id = str(fact_id).strip()
            fact = fact_map.get(normalized_id)
            if fact is None:
                mismatches.append(f"bound_fact_not_supplied:{normalized_id}")
                continue
            fact_metric = canonical_metric_id(
                _fact_value(fact, "metric", "normalized_metric", "raw_metric")
            )
            fact_period = canonical_period_id(
                _fact_value(fact, "period", "normalized_period", "raw_period")
            )
            if expected_metric and fact_metric != expected_metric:
                mismatches.append(
                    f"fact_metric_not_matching_slot:{normalized_id}:{slot.slot_id}"
                )
            if expected_period and fact_period != expected_period:
                mismatches.append(
                    f"fact_period_not_matching_slot:{normalized_id}:{slot.slot_id}"
                )
            if frame.metric_ids and plan.intent is Intent.DIRECT_FACT:
                if fact_metric is None:
                    mismatches.append(f"fact_metric_unverifiable:{normalized_id}")
                elif fact_metric not in set(frame.metric_ids):
                    mismatches.append(f"fact_metric_not_in_query:{normalized_id}")
            elif frame.metric_ids and fact_metric is not None:
                if fact_metric not in plan_metric_ids:
                    mismatches.append(f"fact_metric_not_in_plan:{normalized_id}")

            if frame.entity_ids:
                fact_entity = canonical_entity_id(
                    _fact_value(fact, "entity", "company", "ticker")
                )
                if fact_entity is None:
                    mismatches.append(f"fact_entity_unverifiable:{normalized_id}")
                elif fact_entity not in set(frame.entity_ids):
                    mismatches.append(f"fact_entity_not_in_query:{normalized_id}")
            if frame.period_ids:
                if fact_period is None:
                    mismatches.append(f"fact_period_unverifiable:{normalized_id}")
                elif fact_period not in set(
                    canonical_period_id(slot.period) for slot in plan.required_slots
                ):
                    mismatches.append(f"fact_period_not_in_plan:{normalized_id}")

    return BoundEvidenceSemanticCheck(
        status=(
            BoundEvidenceAlignmentStatus.MISMATCH
            if mismatches
            else BoundEvidenceAlignmentStatus.ALIGNED
        ),
        mismatches=tuple(dict.fromkeys(mismatches)),
        checked_fact_ids=tuple(ordered_ids),
        query_metric_ids=frame.metric_ids,
        query_entity_ids=frame.entity_ids,
        query_period_ids=frame.period_ids,
    )


__all__ = [
    "BoundEvidenceAlignmentStatus",
    "BoundEvidenceSemanticCheck",
    "EntityMention",
    "MetricMention",
    "MetricDefinition",
    "OperationMention",
    "PeriodMention",
    "PlanSemanticAlignment",
    "QuerySemanticFrame",
    "SemanticAlignmentStatus",
    "UnknownSemanticPolicy",
    "align_query_to_plan",
    "align_bound_evidence_to_query",
    "canonical_entity_id",
    "canonical_metric_id",
    "metric_alias_registry",
    "canonical_operation_id",
    "canonical_period_id",
    "canonical_scope_id",
    "coerce_unknown_semantic_policy",
    "extract_query_semantic_frame",
]
