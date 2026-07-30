"""Pluggable, deterministic financial fact extractors.

The production default remains the legacy extractor in
``DeterministicAnswerExtractor``.  The structured implementation is an
explicit, default-off provider used by the NF42 frozen-context shadow run.
It emits facts with the same source metadata required by the existing answer
selection, citation, calculation, and validation stages.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Protocol



class FactExtractorConfigurationError(ValueError):
    """Raised when an unsupported deterministic fact provider is requested."""


@dataclass(frozen=True)
class ExtractedFinancialFact:
    fact_id: str
    candidate_key: str | None
    candidate_rank: int | None
    document_id: str | None
    page: int | None
    metric: str | None
    canonical_value: Decimal | None
    raw_value: str | None
    currency: str | None
    unit: str | None
    scale: str | None
    period: str | None
    row_identity: str | None
    column_identity: str | None
    extraction_confidence: float | None
    source_text: str
    chunk: dict


@dataclass(frozen=True)
class FactExtractionResult:
    facts: tuple[ExtractedFinancialFact, ...]


class DeterministicFactExtractor(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def revision(self) -> str: ...

    def extract(self, *, question: str, evidence: tuple[dict, ...], observer=None) -> FactExtractionResult: ...


@dataclass(frozen=True)
class CurrentProductionFactExtractor:
    """Marker for the unchanged legacy production extraction path."""

    @property
    def name(self) -> str:
        return "current"

    @property
    def revision(self) -> str:
        return "legacy-production/v1"

    def extract(self, *, question: str, evidence: tuple[dict, ...], observer=None) -> FactExtractionResult:
        # The current implementation intentionally remains in the legacy
        # method to preserve its byte-level output compatibility.
        return FactExtractionResult(())


_NUMBER = re.compile(
    r"(?<![\w.])(?:[$€£]|(?:usd|chf|rs\.?)\s*)?\(?\d[\d,]*(?:\.\d+)?\)?"
    r"\s*(?:%|per\s+cent|million|billion|thousand(?:s)?(?:\s+of\s+swiss\s+francs)?|swiss\s+francs|francs)?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _fact_id(*, candidate_key: str | None, stage: str, ordinal: int, raw_value: str) -> str:
    payload = f"{candidate_key or 'unresolved'}|{stage}|{ordinal}|{raw_value}".encode("utf-8")
    return f"production-fact:v1:{hashlib.sha256(payload).hexdigest()}"


def _cells(line: str) -> list[str]:
    if "|" not in line:
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _expression(value: str) -> str | None:
    match = _NUMBER.search(value or "")
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else None


def _currency(value: str) -> str | None:
    lowered = value.lower()
    if "$" in lowered or "usd" in lowered:
        return "USD"
    if "swiss franc" in lowered or "chf" in lowered:
        return "CHF"
    return None


def _unit(value: str) -> str | None:
    lowered = value.lower()
    if "%" in lowered or "per cent" in lowered:
        return "percentage"
    if "billion" in lowered:
        return "billion"
    if "million" in lowered:
        return "million"
    if "thousand" in lowered:
        return "thousand"
    return "currency" if _currency(value) else None


def _scale(value: str) -> str:
    lowered = value.lower()
    if "billion" in lowered:
        return "1000000000"
    if "million" in lowered:
        return "1000000"
    if "thousand" in lowered:
        return "1000"
    return "1"


def _canonical(value: str) -> Decimal | None:
    match = re.search(r"\(?\s*([-+]?\d[\d,]*(?:\.\d+)?)", value)
    if not match:
        return None
    try:
        number = Decimal(match.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    if value.strip().startswith("("):
        number = -number
    if "%" in value or "per cent" in value.lower():
        return number / Decimal("100")
    return number * Decimal(_scale(value))


@dataclass(frozen=True)
class StructuredShadowFactExtractor:
    """Generic line/table extractor retaining row, column and source links."""

    @property
    def name(self) -> str:
        return "structured_shadow"

    @property
    def revision(self) -> str:
        return "structured-shadow/v1"

    def extract(self, *, question: str, evidence: tuple[dict, ...], observer=None) -> FactExtractionResult:
        # Deferred import avoids ``src.generation.__init__`` importing this
        # provider while it is still being initialized.
        from src.generation.deterministic_observer import ProductionFactTrace

        facts: list[ExtractedFinancialFact] = []
        for chunk in evidence:
            metadata = chunk.get("metadata") or {}
            lines = [re.sub(r"\s+", " ", line).strip() for line in (chunk.get("content") or "").splitlines() if line.strip()]
            headers: list[str] = []
            for line in lines:
                if line.lower().startswith("table column context:"):
                    headers = _YEAR.findall(line)
                    break
            ordinal = 0
            for line in lines:
                if line.lower().startswith("table column context:"):
                    continue
                cells = _cells(line)
                values: list[tuple[str, str, str | None, str | None]] = []
                if cells:
                    labels = [cell for cell in cells if _expression(cell) is None]
                    metric = " | ".join(labels[:2]) or line
                    numeric_index = 0
                    for cell in cells:
                        expression = _expression(cell)
                        if expression is None:
                            continue
                        period = headers[numeric_index] if numeric_index < len(headers) else None
                        values.append((expression, metric, period, str(numeric_index)))
                        numeric_index += 1
                else:
                    matches = list(_NUMBER.finditer(line))
                    years = _YEAR.findall(line)
                    period = years[0] if len(years) == 1 else None
                    values = [(re.sub(r"\s+", " ", match.group(0)).strip(), line, period, None) for match in matches]
                for expression, metric, period, column in values:
                    candidate_key = metadata.get("candidate_key")
                    observed_id = _fact_id(candidate_key=candidate_key, stage="structured_shadow", ordinal=ordinal, raw_value=expression)
                    fact = ExtractedFinancialFact(
                        fact_id=observed_id,
                        candidate_key=candidate_key,
                        candidate_rank=metadata.get("candidate_rank"),
                        document_id=metadata.get("document_id") or metadata.get("filename") or metadata.get("doc_name"),
                        page=metadata.get("page"),
                        metric=metric,
                        canonical_value=_canonical(expression),
                        raw_value=expression,
                        currency=_currency(expression),
                        unit=_unit(expression),
                        scale=_scale(expression),
                        period=period,
                        row_identity=metadata.get("row_id") or metadata.get("parent_row_id"),
                        column_identity=column,
                        extraction_confidence=1.0,
                        source_text=line,
                        chunk=chunk,
                    )
                    facts.append(fact)
                    callback = getattr(observer, "on_fact_candidate_extracted", None)
                    if callable(callback):
                        try:
                            callback(candidate=ProductionFactTrace(
                                fact_id=observed_id, candidate_key=fact.candidate_key,
                                candidate_rank=fact.candidate_rank, document_id=fact.document_id,
                                page=fact.page, extraction_stage="structured_shadow",
                                source_span_hash=_hash(line), raw_value=expression,
                                canonical_value=str(fact.canonical_value) if fact.canonical_value is not None else None,
                                currency=fact.currency, unit=fact.unit, scale=fact.scale,
                                period=fact.period, evaluation_text=line,
                            ))
                        except Exception:
                            pass
                    ordinal += 1
                if not values and len(line) >= 12:
                    candidate_key = metadata.get("candidate_key")
                    observed_id = _fact_id(
                        candidate_key=candidate_key,
                        stage="structured_shadow_text",
                        ordinal=ordinal,
                        raw_value=line,
                    )
                    fact = ExtractedFinancialFact(
                        fact_id=observed_id,
                        candidate_key=candidate_key,
                        candidate_rank=metadata.get("candidate_rank"),
                        document_id=metadata.get("document_id") or metadata.get("filename") or metadata.get("doc_name"),
                        page=metadata.get("page"),
                        metric=line,
                        canonical_value=None,
                        raw_value=None,
                        currency=None,
                        unit=None,
                        scale=None,
                        period=None,
                        row_identity=metadata.get("row_id") or metadata.get("parent_row_id"),
                        column_identity=None,
                        extraction_confidence=1.0,
                        source_text=line,
                        chunk=chunk,
                    )
                    facts.append(fact)
                    callback = getattr(observer, "on_fact_candidate_extracted", None)
                    if callable(callback):
                        try:
                            callback(candidate=ProductionFactTrace(
                                fact_id=observed_id, candidate_key=fact.candidate_key,
                                candidate_rank=fact.candidate_rank, document_id=fact.document_id,
                                page=fact.page, extraction_stage="structured_shadow_text",
                                source_span_hash=_hash(line), raw_value=None,
                                canonical_value=None, currency=None, unit=None, scale=None,
                                period=None, evaluation_text=line,
                            ))
                        except Exception:
                            pass
                    ordinal += 1
        return FactExtractionResult(tuple(facts))


def build_fact_extractor_provider(name: str | None = None) -> DeterministicFactExtractor:
    selected = (name or "current").strip().lower()
    if selected == "current":
        return CurrentProductionFactExtractor()
    if selected == "structured_shadow":
        return StructuredShadowFactExtractor()
    raise FactExtractorConfigurationError(
        "DETERMINISTIC_FACT_EXTRACTOR must be one of: current, structured_shadow"
    )
