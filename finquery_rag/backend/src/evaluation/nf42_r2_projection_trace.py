"""NF42 R2 projection-to-selection attribution infrastructure.

This module provides the trace data structures, enums, and classification
functions needed to attribute where structured facts are lost between
extraction and the final released answer.

It is evaluation-only: production code paths are never altered.
"""
from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# ---------------------------------------------------------------------------
# Stable hashing helpers
# ---------------------------------------------------------------------------

def stable_json_hash(payload: Any) -> str:
    """SHA-256 of a canonical JSON representation."""
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_text(value: str) -> str:
    """SHA-256 of a UTF-8 string."""
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Integrity error
# ---------------------------------------------------------------------------

class EvaluationIntegrityError(RuntimeError):
    """Raised when evaluation integrity cannot be guaranteed."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StructuredFactLossStage(str, Enum):
    """First-loss stage for a correct structured fact."""

    NOT_EXTRACTED = "not_extracted"
    EXTRACTED_NOT_PROJECTION_ELIGIBLE = "extracted_not_projection_eligible"
    DROPPED_DURING_PROJECTION = "dropped_during_projection"
    RANKED_BELOW_SELECTOR_INPUT = "ranked_below_selector_input"
    ENTERED_SELECTOR_NOT_SELECTED = "entered_selector_not_selected"
    SELECTED_VALUE_NOT_USED = "selected_value_not_used"
    VALUE_USED_RAW_ANSWER_WRONG = "value_used_raw_answer_wrong"
    RAW_CORRECT_VALIDATION_REGRESSION = "raw_correct_validation_regression"
    RELEASED_CORRECT = "released_correct"


class ProjectionExclusionReason(str, Enum):
    """Reason a structured fact was excluded before becoming a candidate."""

    MISSING_RAW_VALUE = "missing_raw_value"
    METRIC_PERIOD_CONFLICT = "metric_period_conflict"
    ANCHOR_CONFLICT = "anchor_conflict"
    REQUIRED_ANCHOR_MISSING = "required_anchor_missing"
    NON_POSITIVE_SCORE = "non_positive_score"
    DUPLICATE_PROJECTED_CANDIDATE = "duplicate_projected_candidate"


class RegressionCause(str, Enum):
    """Cause of a regression in an existing correct case."""

    LEGACY_CORRECT_FACT_NOT_EXTRACTED = "legacy_correct_fact_not_extracted"
    LEGACY_CORRECT_FACT_NOT_PROJECTED = "legacy_correct_fact_not_projected"
    LEGACY_CORRECT_CANDIDATE_DISPLACED = "legacy_correct_candidate_displaced"
    WRONG_PERIOD_RANKED_HIGHER = "wrong_period_ranked_higher"
    WRONG_METRIC_RANKED_HIGHER = "wrong_metric_ranked_higher"
    WRONG_SCALE_RANKED_HIGHER = "wrong_scale_ranked_higher"
    DUPLICATE_FACT_CROWDING = "duplicate_fact_crowding"
    VALUE_SELECTION_CHANGED = "value_selection_changed"
    CITATION_SOURCE_CHANGED = "citation_source_changed"
    VALIDATION_ONLY_REGRESSION = "validation_only_regression"
    REGRESSION_TRACE_INSUFFICIENT = "regression_trace_insufficient"
    UNCLASSIFIED = "unclassified"


class GoldMatchGranularity(str, Enum):
    """Granularity at which a fact matched the gold answer source."""

    CANDIDATE_KEY = "candidate_key"
    CHUNK_ID = "chunk_id"
    FILENAME_PAGE = "filename_page"


# ---------------------------------------------------------------------------
# Expected baseline (from verified NF42 R1 artifact)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NF42ExpectedBaseline:
    """Expected R1 baseline metrics that R2 must reproduce."""

    all_gold_case_count: int
    any_gold_case_count: int
    partial_gold_case_count: int

    current_correct_fact_cases: int
    structured_correct_fact_cases: int

    current_all_gold_raw_correct: int
    structured_all_gold_raw_correct: int

    current_all_gold_released_correct: int
    structured_all_gold_released_correct: int

    current_any_gold_released_correct: int
    structured_any_gold_released_correct: int

    regression_case_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_gold_case_count": self.all_gold_case_count,
            "any_gold_case_count": self.any_gold_case_count,
            "partial_gold_case_count": self.partial_gold_case_count,
            "current_correct_fact_cases": self.current_correct_fact_cases,
            "structured_correct_fact_cases": self.structured_correct_fact_cases,
            "current_all_gold_raw_correct": self.current_all_gold_raw_correct,
            "structured_all_gold_raw_correct": self.structured_all_gold_raw_correct,
            "current_all_gold_released_correct": self.current_all_gold_released_correct,
            "structured_all_gold_released_correct": self.structured_all_gold_released_correct,
            "current_any_gold_released_correct": self.current_any_gold_released_correct,
            "structured_any_gold_released_correct": self.structured_any_gold_released_correct,
            "regression_case_count": self.regression_case_count,
        }

    @classmethod
    def from_metrics_dict(cls, metrics: dict[str, Any]) -> NF42ExpectedBaseline:
        """Build from the ``metrics`` block of an NF42 R1 baseline artifact."""
        return cls(
            all_gold_case_count=int(metrics["all_gold_case_count"]),
            any_gold_case_count=int(metrics["any_gold_case_count"]),
            partial_gold_case_count=int(metrics["partial_gold_case_count"]),
            current_correct_fact_cases=int(metrics["current_correct_fact_cases"]),
            structured_correct_fact_cases=int(metrics["structured_correct_fact_cases"]),
            current_all_gold_raw_correct=int(metrics["current_all_gold_raw_correct"]),
            structured_all_gold_raw_correct=int(metrics["structured_all_gold_raw_correct"]),
            current_all_gold_released_correct=int(metrics["current_all_gold_released_correct"]),
            structured_all_gold_released_correct=int(metrics["structured_all_gold_released_correct"]),
            current_any_gold_released_correct=int(metrics["current_any_gold_released_correct"]),
            structured_any_gold_released_correct=int(metrics["structured_any_gold_released_correct"]),
            regression_case_count=int(metrics["regression_case_count"]),
        )


# ---------------------------------------------------------------------------
# Baseline field groups and comparison (fail-closed on missing fields)
# ---------------------------------------------------------------------------

CURRENT_BASELINE_FIELDS: tuple[str, ...] = (
    "all_gold_case_count",
    "any_gold_case_count",
    "current_correct_fact_cases",
    "current_all_gold_raw_correct",
    "current_all_gold_released_correct",
    "current_any_gold_released_correct",
)

STRUCTURED_BASELINE_FIELDS: tuple[str, ...] = (
    "all_gold_case_count",
    "any_gold_case_count",
    "structured_correct_fact_cases",
    "structured_all_gold_raw_correct",
    "structured_all_gold_released_correct",
    "structured_any_gold_released_correct",
)

CROSS_VARIANT_FIELDS: tuple[str, ...] = (
    "regression_case_count",
)


def baseline_fields_match(
    *,
    actual: dict[str, Any],
    expected: dict[str, Any],
    fields: tuple[str, ...],
) -> bool:
    """Check that ``actual`` matches ``expected`` for exactly the given fields.

    Fails closed (raises ``EvaluationIntegrityError``) if any field is missing
    from either dict, so a partial baseline cannot silently pass.
    """
    missing_actual = [f for f in fields if f not in actual]
    missing_expected = [f for f in fields if f not in expected]
    if missing_actual or missing_expected:
        raise EvaluationIntegrityError(
            "Baseline fields missing: "
            f"actual={missing_actual}, expected={missing_expected}"
        )
    return all(actual[f] == expected[f] for f in fields)


# ---------------------------------------------------------------------------
# Observed side effects (real boundary observation, not inferred)
# ---------------------------------------------------------------------------

@dataclass
class ObservedSideEffects:
    """Records real observations of side-effect boundaries.

    Boundaries that are genuinely not installed on the engine are recorded in
    ``unavailable_boundaries`` with a ``not_installed`` status — they are NOT
    silently treated as zero.  Boundaries that exist and were observed are
    listed in ``observed_boundaries``.
    """

    retrieval_calls: int = 0
    model_chat_completion_requests: int = 0

    memory_write_calls: int = 0
    feedback_write_calls: int = 0
    session_write_calls: int = 0
    document_state_write_calls: int = 0

    observed_boundaries: tuple[str, ...] = ()
    unavailable_boundaries: tuple[str, ...] = ()

    def all_boundaries_accounted_for(self) -> bool:
        """True when all six boundaries are either observed or confirmed not_installed."""
        all_boundaries = {
            "retrieval", "model", "memory",
            "feedback", "session", "document_state",
        }
        accounted = set(self.observed_boundaries) | set(self.unavailable_boundaries)
        return accounted == all_boundaries

    def all_observed_zero(self) -> bool:
        """True only when every observed boundary has zero calls."""
        return (
            self.retrieval_calls == 0
            and self.model_chat_completion_requests == 0
            and self.memory_write_calls == 0
            and self.feedback_write_calls == 0
            and self.session_write_calls == 0
            and self.document_state_write_calls == 0
        )

    @property
    def passed(self) -> bool:
        """True when all boundaries are accounted for and all observed calls are zero."""
        return self.all_boundaries_accounted_for() and self.all_observed_zero()

    def to_dict(self) -> dict[str, Any]:
        def _status(name: str, calls: int) -> dict[str, Any]:
            if name in self.unavailable_boundaries:
                return {"status": "not_installed"}
            return {"status": "observed", "calls": calls}

        def _boundary_observed(name: str) -> bool:
            return name in self.observed_boundaries or name in self.unavailable_boundaries

        return {
            "retrieval_boundary_observed": _boundary_observed("retrieval"),
            "model_boundary_observed": _boundary_observed("model"),
            "memory_boundary_observed": _boundary_observed("memory"),
            "feedback_boundary_observed": _boundary_observed("feedback"),
            "session_boundary_observed": _boundary_observed("session"),
            "document_state_boundary_observed": _boundary_observed("document_state"),
            "retrieval_calls": self.retrieval_calls,
            "model_calls": self.model_chat_completion_requests,
            "memory_write_calls": self.memory_write_calls,
            "feedback_write_calls": self.feedback_write_calls,
            "session_write_calls": self.session_write_calls,
            "document_state_write_calls": self.document_state_write_calls,
            "unavailable_boundaries": list(self.unavailable_boundaries),
            "per_boundary": {
                "retrieval": _status("retrieval", self.retrieval_calls),
                "model": _status("model", self.model_chat_completion_requests),
                "memory": _status("memory", self.memory_write_calls),
                "feedback": _status("feedback", self.feedback_write_calls),
                "session": _status("session", self.session_write_calls),
                "document_state": _status("document_state", self.document_state_write_calls),
            },
            "passed": self.passed,
        }


# ---------------------------------------------------------------------------
# Frozen context verification report
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenContextVerificationReport:
    """Result of loading and verifying frozen contexts.

    ``content_hash_verified_count`` is the number of per-candidate content
    hashes verified (135 = 5 candidates × 27 cases).  ``final_context_hash_verified_count``
    is the number of per-case final context hashes verified (27).
    """

    content_hash_verified_count: int
    final_context_hash_verified_count: int
    failed_cases: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return (
            self.content_hash_verified_count > 0
            and self.final_context_hash_verified_count > 0
            and not self.failed_cases
        )


# ---------------------------------------------------------------------------
# Frozen document identity mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrozenDocumentIdentity:
    """Explicit mapping from internal document_id to filename."""

    document_id: str
    filename: str


# ---------------------------------------------------------------------------
# Fact semantic identity (provider-independent comparison)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FactSemanticIdentity:
    """Provider-independent identity for comparing facts across variants.

    Two facts from different providers (current vs structured_shadow) are
    semantically equivalent when these fields all match.  This avoids
    comparing provider-specific ``fact_id`` values.
    """

    candidate_key: str
    canonical_value: str
    currency: str | None
    unit: str | None
    period: str | None
    metric_identity_hash: str | None


def fact_semantic_key(fact: Any) -> str:
    """Compute a provider-independent semantic key for a fact.

    The key is a SHA-256 of the canonical tuple of semantic identity fields,
    so two facts from different providers with the same candidate_key, value,
    currency, unit, and period produce the same key.
    """
    identity = FactSemanticIdentity(
        candidate_key=getattr(fact, "candidate_key", None) or "",
        canonical_value=str(getattr(fact, "canonical_value", None) or ""),
        currency=getattr(fact, "currency", None),
        unit=getattr(fact, "unit", None),
        period=getattr(fact, "period", None),
        metric_identity_hash=getattr(fact, "source_span_hash", None),
    )
    payload = (
        identity.candidate_key,
        identity.canonical_value,
        identity.currency or "",
        identity.unit or "",
        identity.period or "",
        identity.metric_identity_hash or "",
    )
    return sha256_text("|".join(payload))


# ---------------------------------------------------------------------------
# Trace data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericEvidenceCandidateTrace:
    """Unified trace for a projected numeric evidence candidate (P2 onward)."""

    projected_candidate_id: str

    provider: str
    candidate_key: str | None
    candidate_rank: int | None

    source_fact_ids: tuple[str, ...]
    source_span_hash: str

    document_id: str | None
    page: int | None

    projected_text_hash: str
    projected_value_hashes: tuple[str, ...]

    metric: str | None
    period: str | None
    currency: str | None
    unit: str | None

    base_evidence_score: float
    anchor_match_count: int
    anchor_conflict_count: int
    relation_score: float
    value_granularity_score: float
    component_pair_score: float
    retrieval_score: float
    final_pre_selector_score: float

    pre_selector_rank: int | None
    selector_input: bool
    selector_output_rank: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "projected_candidate_id": self.projected_candidate_id,
            "provider": self.provider,
            "candidate_key": self.candidate_key,
            "candidate_rank": self.candidate_rank,
            "source_fact_ids": list(self.source_fact_ids),
            "source_span_hash": self.source_span_hash,
            "document_id": self.document_id,
            "page": self.page,
            "projected_text_hash": self.projected_text_hash,
            "projected_value_hashes": list(self.projected_value_hashes),
            "metric": self.metric,
            "period": self.period,
            "currency": self.currency,
            "unit": self.unit,
            "base_evidence_score": self.base_evidence_score,
            "anchor_match_count": self.anchor_match_count,
            "anchor_conflict_count": self.anchor_conflict_count,
            "relation_score": self.relation_score,
            "value_granularity_score": self.value_granularity_score,
            "component_pair_score": self.component_pair_score,
            "retrieval_score": self.retrieval_score,
            "final_pre_selector_score": self.final_pre_selector_score,
            "pre_selector_rank": self.pre_selector_rank,
            "selector_input": self.selector_input,
            "selector_output_rank": self.selector_output_rank,
        }


@dataclass(frozen=True)
class FactProjectionExclusionTrace:
    """Trace for a fact that was excluded during projection."""

    fact_id: str
    candidate_key: str | None
    provider: str
    reason: ProjectionExclusionReason
    source_span_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "candidate_key": self.candidate_key,
            "provider": self.provider,
            "reason": self.reason.value,
            "source_span_hash": self.source_span_hash,
        }


@dataclass
class NewFactFunnelTrace:
    """Per-fact funnel trace for a newly correct structured fact."""

    case_id: str
    fact_id: str
    candidate_key: str | None

    correct_fact_extracted: bool
    projection_eligible: bool
    projected_candidate_id: str | None
    pre_selector_rank: int | None
    entered_selector_input: bool
    selected_by_selector: bool
    value_selected: bool
    raw_answer_correct: bool
    released_answer_correct: bool

    first_loss_stage: StructuredFactLossStage = StructuredFactLossStage.NOT_EXTRACTED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fact_id": self.fact_id,
            "candidate_key": self.candidate_key,
            "correct_fact_extracted": self.correct_fact_extracted,
            "projection_eligible": self.projection_eligible,
            "projected_candidate_id": self.projected_candidate_id,
            "pre_selector_rank": self.pre_selector_rank,
            "entered_selector_input": self.entered_selector_input,
            "selected_by_selector": self.selected_by_selector,
            "value_selected": self.value_selected,
            "raw_answer_correct": self.raw_answer_correct,
            "released_answer_correct": self.released_answer_correct,
            "first_loss_stage": self.first_loss_stage.value,
        }


@dataclass
class RegressionCaseTrace:
    """Trace for a case that regressed from correct to incorrect.

    Uses provider-independent semantic keys (not provider-specific fact_ids)
    so that the same conceptual fact can be tracked across variants.
    """

    case_id: str

    current_supporting_gold_fact_keys: list[str] = field(default_factory=list)
    current_selected_values_hash: list[str] = field(default_factory=list)
    current_raw_correct: bool = False
    current_released_correct: bool = False

    structured_extracted_semantic_keys: list[str] = field(default_factory=list)
    structured_projected_semantic_keys: list[str] = field(default_factory=list)
    structured_selected_semantic_keys: list[str] = field(default_factory=list)
    structured_value_semantic_keys: list[str] = field(default_factory=list)
    structured_selected_values_hash: list[str] = field(default_factory=list)
    structured_raw_correct: bool = False
    structured_released_correct: bool = False

    first_divergence_stage: str = "unclassified"
    regression_cause: RegressionCause = RegressionCause.UNCLASSIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "current": {
                "supporting_gold_fact_keys": self.current_supporting_gold_fact_keys,
                "selected_values_hash": self.current_selected_values_hash,
                "raw_correct": self.current_raw_correct,
                "released_correct": self.current_released_correct,
            },
            "structured": {
                "extracted_semantic_keys": self.structured_extracted_semantic_keys,
                "projected_semantic_keys": self.structured_projected_semantic_keys,
                "selected_semantic_keys": self.structured_selected_semantic_keys,
                "value_semantic_keys": self.structured_value_semantic_keys,
                "selected_values_hash": self.structured_selected_values_hash,
                "raw_correct": self.structured_raw_correct,
                "released_correct": self.structured_released_correct,
            },
            "first_divergence_stage": self.first_divergence_stage,
            "regression_cause": self.regression_cause.value,
        }


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_new_fact_loss(trace: NewFactFunnelTrace) -> StructuredFactLossStage:
    """Determine the first stage at which a correct fact was lost."""
    if not trace.correct_fact_extracted:
        return StructuredFactLossStage.NOT_EXTRACTED

    if not trace.projection_eligible:
        return StructuredFactLossStage.EXTRACTED_NOT_PROJECTION_ELIGIBLE

    if trace.projected_candidate_id is None:
        return StructuredFactLossStage.DROPPED_DURING_PROJECTION

    if not trace.entered_selector_input:
        return StructuredFactLossStage.RANKED_BELOW_SELECTOR_INPUT

    if not trace.selected_by_selector:
        return StructuredFactLossStage.ENTERED_SELECTOR_NOT_SELECTED

    if not trace.value_selected:
        return StructuredFactLossStage.SELECTED_VALUE_NOT_USED

    if not trace.raw_answer_correct:
        return StructuredFactLossStage.VALUE_USED_RAW_ANSWER_WRONG

    if not trace.released_answer_correct:
        return StructuredFactLossStage.RAW_CORRECT_VALIDATION_REGRESSION

    return StructuredFactLossStage.RELEASED_CORRECT


def classify_regression_cause(
    *,
    current_supporting_gold_fact_keys: set[str],
    structured_extracted_semantic_keys: set[str],
    structured_projected_semantic_keys: set[str],
    structured_selected_semantic_keys: set[str],
    structured_value_semantic_keys: set[str],
    current_raw_correct: bool,
    structured_raw_correct: bool,
    current_released_correct: bool,
    structured_released_correct: bool,
) -> tuple[str, RegressionCause]:
    """Determine the first divergence stage and regression cause.

    Uses provider-independent semantic keys, not provider-specific fact_ids.
    The ``current_supporting_gold_fact_keys`` are the semantic keys of facts
    that supported the correct Current answer.  We check whether equivalent
    facts survive in the Structured path at each stage.

    Attribution order:
        0. regression_trace_insufficient — no supporting facts identified
        1. fact_extraction   — supporting fact not in structured extracted
        2. fact_projection   — extracted but not projected
        3. pre_selector_ranking_or_selection — projected but not selected
        4. value_selection   — selected but not in value set
        5. answer_rendering  — values same but raw answer wrong
        6. validation        — raw correct but released wrong
    """
    # Stage 0: if we cannot identify supporting facts, attribution is impossible
    if not current_supporting_gold_fact_keys:
        return ("regression_trace_insufficient", RegressionCause.REGRESSION_TRACE_INSUFFICIENT)

    # Stage 1: extraction divergence — supporting fact not extracted in structured
    if not current_supporting_gold_fact_keys <= structured_extracted_semantic_keys:
        return ("fact_extraction", RegressionCause.LEGACY_CORRECT_FACT_NOT_EXTRACTED)

    # Stage 2: projection divergence — extracted but not projected
    if not current_supporting_gold_fact_keys <= structured_projected_semantic_keys:
        return ("fact_projection", RegressionCause.LEGACY_CORRECT_FACT_NOT_PROJECTED)

    # Stage 3: selection/ranking divergence — projected but not selected
    if not current_supporting_gold_fact_keys <= structured_selected_semantic_keys:
        return ("pre_selector_ranking_or_selection", RegressionCause.LEGACY_CORRECT_CANDIDATE_DISPLACED)

    # Stage 4: value selection divergence — selected but not in value set
    if not current_supporting_gold_fact_keys <= structured_value_semantic_keys:
        return ("value_selection", RegressionCause.VALUE_SELECTION_CHANGED)

    # Stage 5: raw answer divergence — values same but raw answer wrong
    if current_raw_correct and not structured_raw_correct:
        return ("answer_rendering", RegressionCause.VALUE_SELECTION_CHANGED)

    # Stage 6: validation-only regression — raw correct but released wrong
    if current_raw_correct == structured_raw_correct and current_released_correct and not structured_released_correct:
        return ("validation", RegressionCause.VALIDATION_ONLY_REGRESSION)

    return ("unclassified", RegressionCause.UNCLASSIFIED)


# ---------------------------------------------------------------------------
# Function identity (fail-closed)
# ---------------------------------------------------------------------------

def function_identity(fn: Any) -> dict[str, Any]:
    """Record the source-level identity of a function.

    Uses ``inspect.getsource`` so that a fixed description string cannot
    masquerade as an unchanged implementation.  Fails closed: if the
    source cannot be retrieved the function raises
    ``EvaluationIntegrityError`` rather than hashing an empty string.
    """
    try:
        source = inspect.getsource(fn)
    except (TypeError, OSError) as exc:
        raise EvaluationIntegrityError(
            f"Cannot fingerprint {fn!r}"
        ) from exc

    if not source.strip():
        raise EvaluationIntegrityError(
            f"Empty source identity for {fn!r}"
        )

    return {
        "module": getattr(fn, "__module__", None),
        "qualname": getattr(fn, "__qualname__", None),
        "source_sha256": sha256_text(source),
    }
