"""Contracts for an isolated native Inline XBRL financial-fact retriever."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import re
from typing import Any


EXCLUDED_ISSUER_CIKS = frozenset(
    {
        "320193", "19617", "21344", "789019", "1045810", "78003", "1318605", "1403161",
        "1652044", "1018724", "1326801", "1065280",
    }
)


@dataclass(frozen=True)
class StructuredFactSource:
    issuer: str
    cik: str
    filing_date: str
    report_date: str
    accession_number: str
    primary_document: str
    split: str

    @property
    def archive_url(self) -> str:
        accession = self.accession_number.replace("-", "")
        return f"https://www.sec.gov/Archives/edgar/data/{int(self.cik)}/{accession}/{self.primary_document}"


PINNED_V2_SOURCES = (
    StructuredFactSource("Walmart Inc.", "104169", "2026-03-13", "2026-01-31", "0000104169-26-000055", "wmt-20260131.htm", "development"),
    StructuredFactSource("ADOBE INC.", "796343", "2026-01-15", "2025-11-28", "0000796343-26-000003", "adbe-20251128.htm", "development"),
    StructuredFactSource("Salesforce, Inc.", "1108524", "2026-03-02", "2026-01-31", "0001108524-26-000060", "crm-20260131.htm", "development"),
    StructuredFactSource("COSTCO WHOLESALE CORP /NEW", "909832", "2025-10-08", "2025-08-31", "0000909832-25-000101", "cost-20250831.htm", "holdout"),
    StructuredFactSource("HOME DEPOT, INC.", "354950", "2026-03-18", "2026-02-01", "0001628280-26-019436", "hd-20260201.htm", "holdout"),
    StructuredFactSource("PEPSICO INC", "77476", "2026-02-03", "2025-12-27", "0000077476-26-000007", "pep-20251227.htm", "holdout"),
)


@dataclass(frozen=True)
class NativeFinancialFact:
    fact_identity: str
    document_id: str
    issuer: str
    cik: str
    accession_number: str
    concept: str
    label: str
    statement: str | None
    context_id: str
    period_start: str | None
    period_end: str
    period_kind: str
    unit_ref: str | None
    unit_measure: str | None
    currency: str | None
    scale_power: int | None
    raw_value: str
    normalized_value: str | None
    source_fact_id: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def validate_v2_sources(sources: tuple[StructuredFactSource, ...]) -> None:
    if len(sources) != 6 or len({source.cik for source in sources}) != 6:
        raise ValueError("V2 requires exactly six unique issuers")
    if {source.split for source in sources} != {"development", "holdout"}:
        raise ValueError("V2 requires development and holdout splits")
    if sum(source.split == "development" for source in sources) != 3:
        raise ValueError("V2 requires three development issuers")
    if any(source.cik in EXCLUDED_ISSUER_CIKS for source in sources):
        raise ValueError("V2 issuer overlaps a frozen benchmark or prior development corpus")


def normalize_concept_label(concept: str) -> str:
    local = concept.split(":", 1)[-1]
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local)
    words = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", words)
    return " ".join(words.casefold().split())


def parse_numeric_value(raw: str, *, sign: str | None, scale: str | None) -> tuple[str | None, int | None]:
    cleaned = " ".join(raw.split()).replace(",", "").replace("$", "").strip()
    negative_parentheses = cleaned.startswith("(") and cleaned.endswith(")")
    cleaned = cleaned.strip("() ")
    if not cleaned or cleaned in {"—", "-", "N/A"}:
        return None, int(scale) if scale not in (None, "") else None
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None, int(scale) if scale not in (None, "") else None
    if negative_parentheses or sign == "-":
        value = -abs(value)
    scale_power = int(scale) if scale not in (None, "") else 0
    normalized = value * (Decimal(10) ** scale_power)
    return format(normalized, "f"), scale_power


def fact_identity(*, document_id: str, concept: str, context_id: str, unit_ref: str | None, source_fact_id: str) -> str:
    payload = "|".join(("structured-fact-v2/v1", document_id, concept, context_id, unit_ref or "", source_fact_id))
    return f"fact:v2:{sha256(payload.encode('utf-8')).hexdigest()}"


def metric_tokens(value: str) -> frozenset[str]:
    """Tokenize a concept label without issuer- or benchmark-specific aliases."""
    return frozenset(re.findall(r"[a-z0-9]+", value.casefold()))


def structured_fact_score(
    *,
    query_issuer: str,
    query_metric: str,
    query_periods: tuple[str, ...],
    fact: dict[str, Any],
) -> float | None:
    """Apply the single frozen V2 document/metric/period scoring contract."""
    if str(fact["issuer"]).casefold() != query_issuer.casefold():
        return None
    query_metric_tokens = metric_tokens(query_metric)
    fact_metric_tokens = metric_tokens(str(fact["label"]))
    if not query_metric_tokens or query_metric_tokens != fact_metric_tokens:
        return None
    period_match = str(fact["period_end"]) in query_periods
    return 9.0 + (3.0 if period_match else 0.0)


def parse_structured_fact_query(question: str) -> tuple[str, str, tuple[str, ...]] | None:
    """Parse only the frozen public benchmark question template."""
    match = re.fullmatch(
        r"According to (.+?)'s Form 10-K, what was (.+?) for the periods? ended "
        r"(\d{4}-\d{2}-\d{2})(?: and (\d{4}-\d{2}-\d{2}))?\?",
        question,
    )
    if match is None:
        return None
    issuer, metric, first_period, second_period = match.groups()
    periods = (first_period,) if second_period is None else (first_period, second_period)
    return issuer, metric, periods
