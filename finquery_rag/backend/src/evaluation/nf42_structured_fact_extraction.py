"""Generic, offline structured-fact extraction for the NF42 A/B gate.

The extractor only reads the verified NF39 R2 frozen payload.  It is not
registered in the production answer path; this phase measures candidate fact
coverage before any behaviour switch is considered.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.evaluation.nf40_frozen_context import FrozenCaseContext
from src.evaluation.nf41_numeric_identity import normalize_numeric_identity


_NUMBER = re.compile(
    r"(?<![\w.])(?:[$€£]|(?:usd|chf|rs\.?)\s*)?\(?\d[\d,]*(?:\.\d+)?\)?"
    r"\s*(?:%|per\s+cent|million|billion|thousand(?:s)?(?:\s+of\s+swiss\s+francs)?|swiss\s+francs|francs)?",
    re.IGNORECASE,
)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class StructuredExtractedFact:
    fact_id: str
    candidate_key: str
    candidate_rank: int
    document_id: str
    page: int | None
    value_expression: str | None
    canonical_value: str | None
    value_type: str | None
    currency: str | None
    period: str | None
    metric_text: str
    extraction_kind: str


def _line_cells(line: str) -> list[str]:
    if "|" not in line:
        return []
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _period_headers(lines: list[str]) -> list[str]:
    for line in lines:
        if line.lower().startswith("table column context:"):
            return _YEAR.findall(line)
    return []


def _numeric_expression(cell: str) -> str | None:
    match = _NUMBER.search(cell or "")
    return re.sub(r"\s+", " ", match.group(0)).strip() if match else None


def _number_fact(
    *, candidate, ordinal: int, expression: str, metric_text: str, period: str | None, kind: str
) -> StructuredExtractedFact:
    identity = normalize_numeric_identity(expression, period=period)
    return StructuredExtractedFact(
        fact_id=f"{candidate.candidate_key}:nf42:{kind}:{ordinal}",
        candidate_key=candidate.candidate_key,
        candidate_rank=candidate.rank,
        document_id=candidate.document_id,
        page=candidate.page,
        value_expression=expression,
        canonical_value=str(identity.canonical_value) if identity else None,
        value_type=identity.value_type if identity else None,
        currency=identity.currency if identity else None,
        period=period,
        metric_text=metric_text[:700],
        extraction_kind=kind,
    )


def extract_structured_facts(context: FrozenCaseContext) -> list[StructuredExtractedFact]:
    """Extract all line, narrative, and table-row facts with stable IDs."""
    facts: list[StructuredExtractedFact] = []
    for candidate in context.candidates:
        body = candidate.rendered_content.partition("\n")[2]
        lines = [re.sub(r"\s+", " ", line).strip() for line in body.splitlines() if line.strip()]
        headers = _period_headers(lines)
        ordinal = 0
        for line in lines:
            if line.lower().startswith("table column context:"):
                continue
            cells = _line_cells(line)
            if cells:
                label_cells = [cell for cell in cells if not _numeric_expression(cell)]
                row_label = " | ".join(label_cells[:2]) or line
                numeric_cells = [cell for cell in cells if _numeric_expression(cell)]
                for value_index, cell in enumerate(numeric_cells):
                    expression = _numeric_expression(cell)
                    if expression is None:
                        continue
                    period = headers[value_index] if value_index < len(headers) else None
                    facts.append(_number_fact(
                        candidate=candidate, ordinal=ordinal, expression=expression,
                        metric_text=row_label, period=period, kind="table_cell",
                    ))
                    ordinal += 1
                continue
            matches = list(_NUMBER.finditer(line))
            if matches:
                years = _YEAR.findall(line)
                period = years[0] if len(years) == 1 else None
                for match in matches:
                    expression = re.sub(r"\s+", " ", match.group(0)).strip()
                    facts.append(_number_fact(
                        candidate=candidate, ordinal=ordinal, expression=expression,
                        metric_text=line, period=period, kind="narrative_numeric",
                    ))
                    ordinal += 1
            elif len(line) >= 12:
                facts.append(StructuredExtractedFact(
                    fact_id=f"{candidate.candidate_key}:nf42:text:{ordinal}",
                    candidate_key=candidate.candidate_key,
                    candidate_rank=candidate.rank,
                    document_id=candidate.document_id,
                    page=candidate.page,
                    value_expression=None,
                    canonical_value=None,
                    value_type=None,
                    currency=None,
                    period=None,
                    metric_text=line[:700],
                    extraction_kind="text_line",
                ))
                ordinal += 1
    return facts


def fact_matches_case(fact: StructuredExtractedFact, case) -> bool:
    if not any(
        source.filename == fact.document_id and (source.page is None or source.page == fact.page)
        for source in case.expected_sources
    ):
        return False
    if case.expected_numbers:
        actual = normalize_numeric_identity(fact.value_expression, period=fact.period)
        if actual is None:
            return False
        for expected in case.expected_numbers:
            target = normalize_numeric_identity(expected)
            if target and actual.value_type == target.value_type and actual.canonical_value == target.canonical_value:
                return True
            raw = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", fact.value_expression or "")
            if raw and target and actual.value_type == "amount" and target.value_type == "amount":
                try:
                    if Decimal(raw.group(0).replace(",", "")) == target.canonical_value:
                        return True
                except InvalidOperation:
                    pass
        return False
    text = fact.metric_text.lower()
    return bool(text and any(expected.lower() in text for expected in case.expected_answer_contains))
