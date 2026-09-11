"""Gate 08 R2 candidate-aligned query construction.

Builds retrieval queries from a Gate 07 QueryPlan for the 4-lane
candidate-aligned direct retrieval.  The Raw Metric Phrase is always
preserved as-is; concept features are appended, never replacing the
metric phrase.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

from rag_v2.supervisor.semantic_alignment import extract_query_semantic_frame
from src.pdf_retrieval_v4.query_plan_models import OperandSlot, QueryPlan


# Retrieval-only aliases for terms that have the same financial meaning in
# the covered filing schemas.  This list intentionally stays narrower than
# the Supervisor vocabulary: retrieval may broaden a candidate pool, while
# Binder/semantic alignment remains the authority for admission.  In
# particular, operating income and net income are not aliases.
_RETRIEVAL_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "revenue": ("net sales", "total net sales", "total revenue"),
    "revenues": ("net sales", "total net sales", "total revenue"),
    "net sales": ("revenue", "total net sales", "total revenue"),
    "total net sales": ("revenue", "net sales", "total revenue"),
    "total revenue": ("revenue", "net sales", "total net sales"),
}

# A planner metric phrase can contain an explicit business segment before the
# metric suffix (``Productivity and Business Processes revenue``). The index
# reader tokenizes terms as an OR expression, so searching that whole phrase
# together with ``revenue`` can still crowd the segment row out. Extracting a
# bounded scope-only variant preserves the user's concrete label without
# inventing a new ontology or changing Binder admission semantics.
_METRIC_SUFFIXES = tuple(
    sorted(
        {
            "operating income",
            "operating margin",
            "net income",
            "net margin",
            "gross profit",
            "gross margin",
            "total assets",
            "total debt",
            "revenue",
            "revenues",
            "sales",
            "net sales",
            "operating expenses",
            "debt",
        },
        key=len,
        reverse=True,
    )
)
_SCOPE_PREFIX_STOPWORDS = frozenset(
    {
        "total",
        "net",
        "gross",
        "operating",
        "income",
        "margin",
        "assets",
        "debt",
        "expenses",
        "expense",
        "value",
        "amount",
    }
)


def _normalize_phrase(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w\s]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def _phrase_present(text: Any, phrase: Any) -> bool:
    normalized_text = _normalize_phrase(text)
    normalized_phrase = _normalize_phrase(phrase)
    if not normalized_text or not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


def _metric_aliases(raw_phrase: Any) -> tuple[str, ...]:
    """Return safe retrieval aliases not already present in the phrase."""

    normalized = _normalize_phrase(raw_phrase)
    aliases = _RETRIEVAL_METRIC_ALIASES.get(normalized, ())
    return tuple(alias for alias in aliases if not _phrase_present(raw_phrase, alias))


def extract_scope_phrase(raw_phrase: Any) -> str | None:
    """Extract a concrete scope prefix from a planner metric phrase.

    This is retrieval query shaping only. A prefix is emitted when a known
    metric suffix is present and the remaining phrase is not a generic
    financial qualifier. Binder and the semantic alignment firewall remain
    responsible for deciding whether a returned row is admissible.
    """

    phrase = " ".join(str(raw_phrase or "").split()).strip()
    normalized = _normalize_phrase(phrase)
    if not normalized:
        return None
    for suffix in _METRIC_SUFFIXES:
        match = re.search(rf"(?:^|\s){re.escape(suffix)}$", normalized)
        if match is None:
            continue
        prefix = normalized[: match.start()].strip()
        if not prefix:
            return None
        tokens = prefix.split()
        if len(tokens) == 1 and tokens[0] in _SCOPE_PREFIX_STOPWORDS:
            return None
        if all(token in _SCOPE_PREFIX_STOPWORDS for token in tokens):
            return None
        # Return a normalized phrase so variant deduplication is stable while
        # retaining all concrete words from the user's scope label.
        return prefix
    return None


def _meaningful_temporal_kind(value: Any) -> bool:
    """Keep only temporal qualifiers that carry retrieval signal.

    ``unspecified`` and ``unknown`` are planner bookkeeping values, not
    filing vocabulary.  Adding them to a lexical/dense query introduces
    noise and can crowd a correctly labelled aggregate row out of the
    bounded candidate pool.
    """

    normalized = _normalize_phrase(value)
    return normalized not in {"", "unspecified", "unknown"}


def _entity_terms(plan: QueryPlan) -> tuple[str, ...]:
    """Return explicit canonical issuer terms for retrieval-only narrowing.

    Query plans created without an explicit document filter intentionally keep
    ``issuer`` unset.  The question may still name a supported company,
    however, and the candidate views index document identities using stable
    ticker tokens (for example ``aapl_fy2025``).  Reusing the deterministic
    semantic frame here adds that token to slot queries without inventing an
    issuer or changing Binder authority.  Unknown entities remain unknown.
    """

    # A document_scope is already enforced by the index reader. Repeating
    # the document/ticker identifiers in every dense query adds lexical
    # noise without increasing isolation, and can demote the relevant row
    # inside the bounded lane. Keep entity terms for unscoped retrieval,
    # where they are the only deterministic narrowing signal.
    if plan.document_scope:
        return ()
    values: list[str] = []
    if plan.issuer:
        values.append(str(plan.issuer))
    try:
        frame = extract_query_semantic_frame(plan.raw_question)
    except (TypeError, ValueError):
        frame = None
    if frame is not None:
        values.extend(str(item) for item in frame.entity_ids)
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _normalize_phrase(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(value)
    return tuple(result)


def append_missing_query_terms(query: Any, terms: Iterable[Any]) -> str:
    """Append only terms absent from a query, preserving first-seen order.

    Targeted recovery receives the original standalone query plus slot
    metadata.  This helper keeps the metadata useful without producing
    artificial repetitions such as ``revenue Revenue FY2024``.  It is a
    lexical construction helper only; Binder remains responsible for
    semantic admission.
    """

    base = " ".join(str(query or "").split()).strip()
    parts: list[str] = [base] if base else []
    for term in terms:
        text = " ".join(str(term or "").split()).strip()
        if text and not _phrase_present(base, text):
            parts.append(text)
    return " ".join(parts)


def build_raw_question_query(plan: QueryPlan) -> str:
    """Build the raw-question query for 4-lane search.

    Combines: raw_question + issuer + metric_phrases + periods.
    """
    parts: list[str] = [plan.raw_question]
    if plan.issuer:
        parts.append(str(plan.issuer))
    parts.extend(str(p) for p in plan.metric_phrases if p)
    parts.extend(str(p) for p in plan.periods if p)
    return " | ".join(p for p in parts if p.strip())


def build_slot_query(plan: QueryPlan, slot: dict[str, Any]) -> str:
    """Build a per-slot query for 4-lane search.

    Combines: raw_metric_phrase + normalized_period + temporal_kind +
    top-3 concept features.

    The Raw Metric Phrase is preserved as-is.  Concept features are
    appended, not replacing the metric phrase.
    """
    parts: list[str] = []
    raw_phrase = slot.get("raw_metric_phrase")
    if raw_phrase:
        parts.append(str(raw_phrase))
        # Keep the user's/planner's metric phrase first, then add only the
        # small set of deterministic filing-label aliases.  The Binder still
        # decides whether a returned candidate satisfies the RequiredSlot.
        parts.extend(_metric_aliases(raw_phrase))
    period = slot.get("period")
    if period:
        parts.append(str(period))
    temporal_kind = slot.get("temporal_kind")
    if _meaningful_temporal_kind(temporal_kind):
        parts.append(str(temporal_kind))
    concepts = list(slot.get("concept_candidates") or ())
    parts.extend(str(c) for c in concepts[:3] if c)
    parts.extend(_entity_terms(plan))
    return " | ".join(p for p in parts if p.strip())


def build_slot_query_variants(plan: QueryPlan, slot: dict[str, Any]) -> list[str]:
    """Build bounded per-slot query variants for retrieval alias expansion.

    The index reader tokenizes a query into an OR expression.  Putting
    ``revenue``, ``net sales`` and ``total net sales`` into one query therefore
    makes the broadest term dominate ranking and can bury the canonical filing
    row.  Each approved alias is consequently searched as its own bounded
    variant, then fused by :class:`CandidateDirectRetriever`.  This broadens
    recall without allowing an unsafe metric equivalence such as operating
    income == net income.
    """

    raw_phrase = slot.get("raw_metric_phrase")
    phrases: list[str] = []
    if raw_phrase:
        phrases.append(str(raw_phrase))
        scope_phrase = extract_scope_phrase(raw_phrase)
        if scope_phrase and not _phrase_present(scope_phrase, raw_phrase):
            phrases.append(scope_phrase)
        phrases.extend(_metric_aliases(raw_phrase))

    variants: list[str] = []
    seen: set[str] = set()
    for phrase in phrases or [""]:
        parts: list[str] = []
        if phrase:
            parts.append(phrase)
        period = slot.get("period")
        if period:
            parts.append(str(period))
        temporal_kind = slot.get("temporal_kind")
        if _meaningful_temporal_kind(temporal_kind):
            parts.append(str(temporal_kind))
        concepts = list(slot.get("concept_candidates") or ())
        parts.extend(str(c) for c in concepts[:3] if c)
        parts.extend(_entity_terms(plan))
        query = " | ".join(p for p in parts if p.strip())
        normalized = _normalize_phrase(query)
        if query and normalized not in seen:
            seen.add(normalized)
            variants.append(query)
    return variants


def _slot_to_dict(slot: OperandSlot) -> dict[str, Any]:
    return {
        "raw_metric_phrase": slot.raw_metric_phrase,
        "period": slot.period,
        "temporal_kind": slot.temporal_kind,
        "concept_candidates": list(slot.concept_candidates),
        "role": slot.role,
        "bucket_label": slot.bucket_label,
        "segment_label": slot.segment_label,
    }


def build_all_queries(plan: QueryPlan) -> dict[str, Any]:
    """Build all queries for a plan.

    Returns::

        {
            "raw_question": [query],
            "slots": {slot_id: [query]},
        }
    """
    raw_query = build_raw_question_query(plan)
    slot_queries: dict[str, list[str]] = {}
    for slot in plan.operand_slots:
        slot_dict = _slot_to_dict(slot)
        slot_queries[slot.slot_id] = build_slot_query_variants(plan, slot_dict)
    return {"raw_question": [raw_query], "slots": slot_queries}


__all__ = [
    "append_missing_query_terms",
    "build_all_queries",
    "build_raw_question_query",
    "build_slot_query",
    "build_slot_query_variants",
    "extract_scope_phrase",
]
