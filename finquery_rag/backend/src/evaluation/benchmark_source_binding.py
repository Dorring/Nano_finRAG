"""Strict source-to-index binding helpers for NF-EVAL-02.

The module is evaluation-only.  It deliberately fails closed when a source
cannot be uniquely identified from the benchmark document, PDF page, metric,
period, value and unit/scale.  It never changes retrieval or production
index state.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any, Mapping


class SourceBindingError(ValueError):
    """Raised when an input source or candidate is structurally invalid."""


@dataclass(frozen=True)
class BindingDecision:
    """A non-authoritative decision for one expected source."""

    status: str
    candidate: Mapping[str, Any] | None
    reasons: tuple[str, ...]
    scores: tuple[tuple[str, int], ...] = ()


_TOKEN_RE = re.compile(r"[a-z0-9]+", flags=re.IGNORECASE)
_GENERIC_METRIC_WORDS = {
    "and", "by", "of", "the", "for", "from", "in", "net", "total",
    "income", "revenue", "revenues", "sales", "amount", "value",
}


def normalize_text(value: Any) -> str:
    """Normalize PDF extraction whitespace and punctuation for matching."""

    if value is None:
        return ""
    text = str(value).casefold().replace("&", " and ")
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return " ".join(text.split())


def metric_tokens(value: Any) -> tuple[str, ...]:
    tokens = tuple(token.casefold() for token in _TOKEN_RE.findall(str(value or "")))
    return tuple(token for token in tokens if len(token) >= 3 and token not in {"the"})


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return None


def source_expected_value(label: Mapping[str, Any], source_index: int) -> str | None:
    """Return the value belonging to this source, not the whole answer.

    Calculation operands and composite components are source-ordered in the
    frozen reference labels.  Direct answers use the answer value.
    """

    calculation = label.get("calculation")
    if isinstance(calculation, Mapping):
        operands = calculation.get("operands")
        if isinstance(operands, list) and source_index < len(operands):
            value = operands[source_index].get("value")
            return str(value) if value is not None else None
    answer = label.get("expected_answer") or {}
    components = answer.get("component_values")
    if isinstance(components, list) and source_index < len(components):
        value = components[source_index].get("canonical_value")
        return str(value) if value is not None else None
    value = answer.get("canonical_value")
    return str(value) if value is not None else None


def source_expected_metric(label: Mapping[str, Any], source: Mapping[str, Any], source_index: int) -> str:
    calculation = label.get("calculation")
    if isinstance(calculation, Mapping):
        operands = calculation.get("operands")
        if isinstance(operands, list) and source_index < len(operands):
            metric = operands[source_index].get("metric")
            if metric:
                return str(metric)
    answer = label.get("expected_answer") or {}
    components = answer.get("component_values")
    if isinstance(components, list) and source_index < len(components):
        metric = components[source_index].get("metric")
        if metric:
            return str(metric)
    return str(source.get("row_label") or source.get("section") or "")


def expected_value_variants(value: Any, source: Mapping[str, Any]) -> tuple[str, ...]:
    """Generate scale-aware strings which must occur in candidate context."""

    number = _decimal(value)
    if number is None:
        return ()
    unit = str(source.get("unit") or "").casefold()
    values: list[Decimal] = [number]
    if unit in {"currency", "volume", "count", "numeric", "financial_volume"}:
        if unit == "currency":
            values.insert(0, number / Decimal(1_000_000))
        elif unit == "volume":
            values.insert(0, number / Decimal(1_000_000_000_000))
        elif unit == "count":
            values.insert(0, number / Decimal(1_000_000_000))
    variants: list[str] = []
    for item in values:
        plain = format(item, "f")
        if "." in plain:
            plain = plain.rstrip("0").rstrip(".")
        comma = f"{item:,.6f}".rstrip("0").rstrip(".")
        for rendered in (plain, comma):
            if rendered and rendered not in variants:
                variants.append(rendered)
        if unit == "volume":
            variants.append(plain + "t")
            variants.append(plain + " trillion")
        if unit == "count":
            variants.append(plain + "b")
            variants.append(plain + " billion")
    return tuple(dict.fromkeys(variants))


def _value_matches(context: str, variants: tuple[str, ...]) -> bool:
    normalized = normalize_text(context)
    compact = re.sub(r"[^a-z0-9%-]", "", context.casefold())
    digit_compact = re.sub(r"[^0-9-]", "", context)
    numeric_tokens = [
        _decimal(token)
        for token in re.findall(
            r"(?<![a-z0-9])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![a-z0-9])",
            context.casefold(),
        )
    ]
    for variant in variants:
        expected_number = _decimal(variant)
        if expected_number is not None:
            if expected_number in numeric_tokens:
                return True
            digit_variant = re.sub(r"[^0-9-]", "", str(variant))
            # Some converted tables split a grouped number across a line
            # break (for example ``281,72`` + ``4``).  Only use compact-digit
            # matching for values with at least three digits so that ``1``
            # cannot match the beginning of ``1.2`` or ``10``.
            if len(digit_variant) >= 3 and digit_variant in digit_compact:
                return True
            # Numeric variants must not fall through to substring matching:
            # ``1`` is not a match for ``1.2`` or ``10``.
            continue
        candidate = normalize_text(variant)
        if candidate and re.search(rf"(?<![a-z0-9]){re.escape(candidate)}(?![a-z0-9])", normalized):
            return True
        compact_variant = re.sub(r"[^a-z0-9%-]", "", str(variant).casefold())
        if compact_variant and compact_variant in compact:
            return True
        digit_variant = re.sub(r"[^0-9-]", "", str(variant))
        if len(digit_variant) >= 3 and digit_variant in digit_compact:
            return True
    return False


def _metric_matches(context: str, metric: str, row_label: str | None) -> bool:
    normalized = normalize_text(context)
    values = [metric, row_label]
    for raw in values:
        tokens = metric_tokens(raw)
        if not tokens:
            continue
        # Require the distinctive words; generic words alone are insufficient.
        distinctive = tuple(token for token in tokens if token not in _GENERIC_METRIC_WORDS)
        required = distinctive or tokens
        if all(re.search(rf"\b{re.escape(token)}\d*\b", normalized) for token in required):
            return True
    return False


def _period_matches(context: str, source: Mapping[str, Any]) -> bool:
    expected = str(source.get("column_header") or source.get("period") or "")
    years = tuple(dict.fromkeys(re.findall(r"\b(?:19|20)\d{2}\b", expected)))
    if years:
        normalized = normalize_text(context)
        # Converted PDFs occasionally concatenate adjacent year columns
        # (``202520242023``); requiring word boundaries would reject a valid
        # table header.  The expected year itself remains an exact 4-digit
        # token in the source contract.
        return all(year in normalized for year in years)
    return True


def _unit_matches(context: str, source: Mapping[str, Any], variants: tuple[str, ...]) -> bool:
    unit = str(source.get("unit") or "").casefold()
    normalized = normalize_text(context)
    if unit == "percentage":
        return "%" in context or "percent" in normalized or "margin" in normalized
    if unit == "volume":
        return bool(
            re.search(r"\b(?:\d+(?:\.\d+)?)\s*[tb]\b", context, flags=re.IGNORECASE)
            or any(suffix in normalized for suffix in (" trillion", " billion"))
        )
    if unit == "count":
        return "transaction" in normalized or bool(
            re.search(r"\b(?:\d+(?:\.\d+)?)\s*b\b", context, flags=re.IGNORECASE)
        )
    if unit == "currency":
        return any(token in normalized for token in ("million", "billion", "trillion", " usd", "$")) or bool(variants)
    return True


def candidate_context(candidate: Mapping[str, Any], by_id: Mapping[str, Mapping[str, Any]]) -> str:
    content = str(candidate.get("content") or "")
    metadata = candidate.get("metadata")
    metadata_text = " "
    if isinstance(metadata, Mapping):
        metadata_text = " ".join(str(metadata.get(key) or "") for key in (
            "table_header_context", "section_path", "section_title", "table_title",
            "row_label", "column_header", "unit", "scale",
        ))
    parent_id = metadata.get("parent_id") if isinstance(metadata, Mapping) else None
    parent = by_id.get(str(parent_id)) if parent_id else None
    page_table = by_id.get(
        f"__page_table__:{candidate.get('benchmark_document_id')}:{candidate.get('page')}"
    )
    page_text = by_id.get(
        f"__page_text__:{candidate.get('benchmark_document_id')}:{candidate.get('page')}"
    )
    parent_text = str(parent.get("content") or "") if parent else ""
    if page_table is not None and page_table is not parent:
        parent_text += "\n" + str(page_table.get("content") or "")
    if page_text is not None:
        parent_text += "\n" + str(page_text.get("content") or "")
    return "\n".join((content, metadata_text, parent_text))


def binding_reasons(
    *,
    label: Mapping[str, Any],
    source: Mapping[str, Any],
    source_index: int,
    candidate: Mapping[str, Any],
    by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    reasons: list[str] = []
    expected_document = str(source.get("document_id") or "")
    if candidate.get("benchmark_document_id") != expected_document:
        reasons.append("document_mismatch")
    if int(candidate.get("page") or -1) != int(source.get("page") or -2):
        reasons.append("page_mismatch")
    context = candidate_context(candidate, by_id)
    metadata = candidate.get("metadata")
    own_context = "\n".join(
        (
            str(candidate.get("content") or ""),
            " ".join(
                str(metadata.get(key) or "")
                for key in ("section_path", "section_title", "table_title", "row_label", "column_header", "unit", "scale")
            ) if isinstance(metadata, Mapping) else "",
        )
    )
    metric = source_expected_metric(label, source, source_index)
    # The parent table is used for headers and units, but not metric/value
    # matching: otherwise every row in a table would appear to match every
    # source on that page.
    if not _metric_matches(own_context, metric, source.get("row_label")):
        reasons.append("metric_mismatch")
    value = source_expected_value(label, source_index)
    variants = expected_value_variants(value, source)
    if not variants or not _value_matches(own_context, variants):
        reasons.append("value_mismatch")
    if not _period_matches(context, source):
        reasons.append("period_mismatch")
    if not _unit_matches(context, source, variants):
        reasons.append("unit_scale_mismatch")
    return tuple(reasons)


def choose_candidate(
    *,
    label: Mapping[str, Any],
    source: Mapping[str, Any],
    source_index: int,
    candidates: list[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
    top20_keys: set[str],
) -> BindingDecision:
    matches: list[tuple[int, Mapping[str, Any]]] = []
    for candidate in candidates:
        reasons = binding_reasons(
            label=label, source=source, source_index=source_index,
            candidate=candidate, by_id=by_id,
        )
        if reasons:
            continue
        block_type = str(candidate.get("block_type") or "")
        specificity = {"table_row": 40, "table": 30, "text": 20, "front_matter": 10}.get(block_type, 0)
        metadata = candidate.get("metadata")
        if isinstance(metadata, Mapping) and metadata.get("table_layout_extracted"):
            specificity += 10
        matches.append((specificity, candidate))
    if not matches:
        return BindingDecision("missing_from_index", None, ("no_unique_strict_match",))
    best_score = max(score for score, _ in matches)
    best = [candidate for score, candidate in matches if score == best_score]
    if len(best) != 1:
        return BindingDecision(
            "ambiguous", None, ("multiple_strict_matches",),
            tuple((str(item.get("candidate_key")), best_score) for item in best),
        )
    candidate = best[0]
    return BindingDecision(
        "bound", candidate, ("document", "page", "metric", "value", "period", "unit_scale"),
        ((str(candidate.get("candidate_key")), best_score),),
    )


def binding_passes(decision: BindingDecision) -> bool:
    return decision.status == "bound" and decision.candidate is not None


def golden_promotion_gate(
    *,
    question_count: int,
    answerable_count: int,
    expected_source_count: int,
    bound_candidate_identity_count: int,
    negative_evidence_verified_count: int,
    ambiguous_identity_count: int,
    missing_from_index_count: int,
    out_of_scope_identity_count: int,
    unresolved_anomaly_count: int,
    all_cases_ready: bool,
) -> bool:
    """Return whether the benchmark can be promoted from draft to Golden.

    This is an evaluation-only gate.  It intentionally uses record counts
    rather than case counts for source identity and fails closed on any
    ambiguity, missing index record, scope violation, or unresolved anomaly.
    """

    return all(
        (
            question_count == 72,
            answerable_count == 64,
            expected_source_count == 80,
            bound_candidate_identity_count == 80,
            negative_evidence_verified_count == 8,
            ambiguous_identity_count == 0,
            missing_from_index_count == 0,
            out_of_scope_identity_count == 0,
            unresolved_anomaly_count == 0,
            all_cases_ready,
        )
    )


def negative_review_passes(manual_review: Mapping[str, Any] | None) -> bool:
    """Validate an explicit negative-evidence review attestation.

    Raw keyword hits are retained for audit, but qualitative mentions are not
    treated as a disclosed answer.  The reviewer must explicitly assess the
    number of quantitative matches as zero.
    """

    if not isinstance(manual_review, Mapping):
        return False
    return bool(
        manual_review.get("human_negative_evidence_reviewed") is True
        and manual_review.get("negative_evidence_reviewed") is True
        and manual_review.get("full_document_search_completed") is True
        and str(manual_review.get("reviewer") or "").strip()
        and str(manual_review.get("review_notes") or "").strip()
        and manual_review.get("positive_quantitative_match_count") == 0
    )
