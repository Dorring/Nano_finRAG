"""Isolation guards for the NF-OPT-17 financial hard-negative dev corpus."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from hashlib import sha256
import re
from typing import Any


EXCLUDED_ANNOTATION_FIELDS = frozenset(
    {
        "case_id",
        "expected_answer",
        "expected_sources",
        "expected_value",
        "gold_source",
        "reference_answer",
        "source_index",
    }
)


@dataclass(frozen=True)
class SecFilingSource:
    """Pinned SEC primary-document identity for a non-benchmark issuer."""

    issuer: str
    cik: str
    filing_date: str
    accession_number: str
    primary_document: str
    form: str = "10-K"

    @property
    def archive_url(self) -> str:
        accession = self.accession_number.replace("-", "")
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{int(self.cik)}/{accession}/{self.primary_document}"
        )


# These issuers are intentionally disjoint from the eight frozen evaluation
# reports. The primary-document identity is from SEC submissions metadata.
PINNED_DEVELOPMENT_SOURCES = (
    SecFilingSource(
        issuer="Alphabet Inc.",
        cik="1652044",
        filing_date="2026-02-05",
        accession_number="0001652044-26-000018",
        primary_document="goog-20251231.htm",
    ),
    SecFilingSource(
        issuer="AMAZON COM INC",
        cik="1018724",
        filing_date="2026-02-06",
        accession_number="0001018724-26-000004",
        primary_document="amzn-20251231.htm",
    ),
    SecFilingSource(
        issuer="Meta Platforms, Inc.",
        cik="1326801",
        filing_date="2026-01-29",
        accession_number="0001628280-26-003942",
        primary_document="meta-20251231.htm",
    ),
    SecFilingSource(
        issuer="NETFLIX INC",
        cik="1065280",
        filing_date="2026-01-23",
        accession_number="0001065280-26-000034",
        primary_document="nflx-20251231.htm",
    ),
)


# Generic US-GAAP labels only; no issuer- or benchmark-specific aliases.
FINANCIAL_CONCEPTS = {
    "us-gaap:Assets": "total assets",
    "us-gaap:CashAndCashEquivalentsAtCarryingValue": "cash and cash equivalents",
    "us-gaap:CostOfRevenue": "cost of revenue",
    "us-gaap:EarningsPerShareBasic": "basic earnings per share",
    "us-gaap:EarningsPerShareDiluted": "diluted earnings per share",
    "us-gaap:GrossProfit": "gross profit",
    "us-gaap:IncomeTaxExpenseBenefit": "income tax expense",
    "us-gaap:Liabilities": "total liabilities",
    "us-gaap:NetCashProvidedByUsedInOperatingActivities": "net cash from operating activities",
    "us-gaap:NetIncomeLoss": "net income",
    "us-gaap:OperatingIncomeLoss": "operating income",
    "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment": "capital expenditures",
    "us-gaap:ResearchAndDevelopmentExpense": "research and development expense",
    "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "us-gaap:Revenues": "revenue",
    "us-gaap:SellingGeneralAndAdministrativeExpense": "selling, general and administrative expense",
    "us-gaap:StockholdersEquity": "stockholders' equity",
    "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding": "weighted-average diluted shares",
    "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic": "weighted-average basic shares",
}

CORE_NEGATIVE_TYPES = frozenset({"same_row_wrong_period", "same_table_wrong_metric"})

_CONCEPT_EXCLUDED_TOKENS = (
    "Abstract",
    "Axis",
    "Disclosure",
    "Domain",
    "Member",
    "Policy",
    "Reconciliation",
    "Schedule",
    "Table",
    "TextBlock",
)
_FINANCIAL_CONCEPT_MARKERS = (
    "Amortization",
    "Asset",
    "Borrowing",
    "Cash",
    "Compensation",
    "Cost",
    "Debt",
    "Depreciation",
    "Dividend",
    "Earnings",
    "Equity",
    "Expense",
    "Goodwill",
    "Income",
    "Interest",
    "Inventory",
    "Investment",
    "Lease",
    "Liabilit",
    "Loss",
    "Margin",
    "Payable",
    "Payment",
    "Proceeds",
    "Profit",
    "Property",
    "Receivable",
    "Revenue",
    "Securities",
    "Share",
    "Stock",
    "Tax",
)


@dataclass(frozen=True)
class FinancialFact:
    """A table-backed iXBRL fact with resolved period and source lineage."""

    source_cik: str
    accession_number: str
    primary_document: str
    issuer: str
    fact_id: str
    concept: str
    metric: str
    context_id: str
    period_end: str
    period_kind: str
    table_index: int
    row_index: int
    evidence_excerpt: str

    @property
    def candidate_key(self) -> str:
        return f"dev:sec:{self.source_cik}:{self.accession_number}:{self.fact_id}"

    @property
    def content_hash(self) -> str:
        return sha256(self.evidence_excerpt.encode("utf-8")).hexdigest()


def metric_label_for_concept(concept: str) -> str | None:
    """Normalize an ordinary US-GAAP numeric concept without issuer aliases."""
    if concept in FINANCIAL_CONCEPTS:
        return FINANCIAL_CONCEPTS[concept]
    if not concept.startswith("us-gaap:"):
        return None
    local_name = concept.removeprefix("us-gaap:")
    if any(token in local_name for token in _CONCEPT_EXCLUDED_TOKENS):
        return None
    if not any(token in local_name for token in _FINANCIAL_CONCEPT_MARKERS):
        return None
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", local_name)
    words = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", words)
    return words.replace("And", "and").lower()


def normalize_excerpt(value: str, *, limit: int = 600) -> str:
    """Create a bounded, reproducible source excerpt."""
    return " ".join(value.split())[:limit]


def parse_context_period(context: Mapping[str, str]) -> tuple[str, str] | None:
    """Return a validated (period end, period kind) pair from a context."""
    instant = str(context.get("instant") or "").strip()
    end = str(context.get("end") or "").strip()
    start = str(context.get("start") or "").strip()
    if instant:
        date.fromisoformat(instant)
        return instant, "instant"
    if start and end:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
        if end_date <= start_date:
            return None
        return end, "duration"
    return None


def build_direct_fact_question(fact: FinancialFact) -> str:
    """Build a value-free direct-fact query from source-side metadata."""
    timing = (
        f"as of {fact.period_end}"
        if fact.period_kind == "instant"
        else f"for the fiscal year ended {fact.period_end}"
    )
    return f"According to {fact.issuer}'s Form 10-K, what was {fact.metric} {timing}?"


def fact_annotation_view(fact: FinancialFact) -> dict[str, Any]:
    """Expose source lineage without serializing a label value."""
    return {
        "candidate_key": fact.candidate_key,
        "fact_id": fact.fact_id,
        "xbrl_concept": fact.concept,
        "context_id": fact.context_id,
        "period_end": fact.period_end,
        "period_kind": fact.period_kind,
        "table_index": fact.table_index,
        "row_index": fact.row_index,
        "evidence_excerpt": fact.evidence_excerpt,
        "candidate_content_sha256": fact.content_hash,
    }


def build_hard_negative_annotation(
    *,
    positive: FinancialFact,
    wrong_period: FinancialFact,
    wrong_metric: FinancialFact,
) -> dict[str, Any]:
    """Create a table- and context-derived hard-negative annotation."""
    if positive.fact_id in {wrong_period.fact_id, wrong_metric.fact_id}:
        raise ValueError("hard negative cannot reuse the positive fact")
    if (positive.table_index, positive.row_index) != (wrong_period.table_index, wrong_period.row_index):
        raise ValueError("wrong-period negative must come from the same table row")
    if positive.period_end == wrong_period.period_end:
        raise ValueError("wrong-period negative must have a different period")
    if positive.table_index != wrong_metric.table_index:
        raise ValueError("wrong-metric negative must come from the same table")
    if positive.metric == wrong_metric.metric:
        raise ValueError("wrong-metric negative must have a different metric")
    annotation_hash = sha256(
        "|".join((positive.candidate_key, wrong_period.candidate_key, wrong_metric.candidate_key)).encode("utf-8")
    ).hexdigest()
    return {
        "annotation_id": f"dev-annotation:{annotation_hash}",
        "query_id": f"dev-query:{annotation_hash}",
        "question": build_direct_fact_question(positive),
        "issuer": positive.issuer,
        "source_document": {
            "cik": positive.source_cik,
            "accession_number": positive.accession_number,
            "primary_document": positive.primary_document,
        },
        "positive_candidate": fact_annotation_view(positive),
        "hard_negatives": [
            {"negative_type": "same_row_wrong_period", "candidate": fact_annotation_view(wrong_period)},
            {"negative_type": "same_table_wrong_metric", "candidate": fact_annotation_view(wrong_metric)},
        ],
        "label_provenance": "deterministic_ixbrl_table_and_context_relation",
        "annotation_mode": "authorized_ai_assisted_development_annotation",
        "human_review_status": "not_reviewed",
        "expected_answer_stored": False,
    }


def validate_generated_annotation(record: Mapping[str, Any]) -> None:
    """Fail closed on leaked Gold fields or invalid source identities."""
    forbidden = EXCLUDED_ANNOTATION_FIELDS.intersection(record)
    if forbidden:
        raise ValueError(f"generated annotation includes forbidden fields: {sorted(forbidden)}")
    if record.get("expected_answer_stored") is not False:
        raise ValueError("development annotations may not store an expected answer")
    positive = record.get("positive_candidate")
    negatives = record.get("hard_negatives")
    if not isinstance(positive, Mapping) or not isinstance(negatives, Sequence):
        raise ValueError("generated annotation has invalid candidate structure")
    if len(negatives) != 2:
        raise ValueError("generated annotation must contain exactly two hard negatives")
    negative_types = {str(item.get("negative_type")) for item in negatives if isinstance(item, Mapping)}
    if negative_types != CORE_NEGATIVE_TYPES:
        raise ValueError("generated annotation is missing a required core negative type")
    positive_key = str(positive.get("candidate_key") or "")
    negative_keys = {
        str(item.get("candidate", {}).get("candidate_key") or "")
        for item in negatives
        if isinstance(item, Mapping) and isinstance(item.get("candidate"), Mapping)
    }
    if not positive_key or positive_key in negative_keys or len(negative_keys) != 2:
        raise ValueError("generated annotation candidate identities are invalid")


def source_manifest_hash(sources: Sequence[SecFilingSource]) -> str:
    """Return a stable identity hash without downloading filing content."""
    serialized = "\n".join(
        "|".join(
            (
                source.issuer,
                source.cik,
                source.form,
                source.filing_date,
                source.accession_number,
                source.primary_document,
            )
        )
        for source in sources
    )
    return sha256(serialized.encode("utf-8")).hexdigest()


def validate_development_sources(
    sources: Sequence[SecFilingSource],
    *,
    frozen_filenames: Iterable[str],
) -> None:
    """Fail closed on duplicate issuers, non-10-K sources, or name leakage."""
    frozen = {str(name).casefold() for name in frozen_filenames}
    if len(sources) < 4:
        raise ValueError("NF-OPT-17 requires at least four independent development filings")
    if len({source.cik for source in sources}) != len(sources):
        raise ValueError("development filing CIKs must be unique")
    for source in sources:
        if source.form != "10-K":
            raise ValueError(f"development source must be a 10-K: {source.issuer}")
        if not source.primary_document.endswith((".htm", ".html")):
            raise ValueError(f"development source must identify an SEC HTML primary document: {source.issuer}")
        if source.primary_document.casefold() in frozen:
            raise ValueError(f"development source filename leaks frozen corpus identity: {source.primary_document}")


def source_record(source: SecFilingSource) -> dict[str, Any]:
    """Create a JSON-safe source record without answer or benchmark fields."""
    record = asdict(source)
    record["archive_url"] = source.archive_url
    record["source_identity_sha256"] = sha256(
        f"{source.cik}|{source.accession_number}|{source.primary_document}".encode("utf-8")
    ).hexdigest()
    return record


def build_annotation_contract() -> dict[str, Any]:
    """Define the independent annotation surface before any labels are created."""
    return {
        "schema": "nf-opt-17/hard-negative-annotation/v1",
        "development_only": True,
        "frozen_benchmark_read_allowed": False,
        "allowed_query_fields": ["query_id", "question", "issuer", "form", "fiscal_period"],
        "allowed_candidate_fields": [
            "candidate_key",
            "document_id",
            "page",
            "evidence_type",
            "content",
            "metadata",
        ],
        "required_negative_taxonomy": [
            "same_page_wrong_row",
            "same_table_wrong_metric",
            "same_row_wrong_period",
            "same_value_wrong_metric",
            "same_metric_wrong_statement",
            "same_issuer_wrong_fiscal_period",
        ],
        "forbidden_fields": sorted(EXCLUDED_ANNOTATION_FIELDS),
        "training_allowed": False,
        "production_index_writes_allowed": False,
    }


def validate_annotation_record(record: Mapping[str, Any]) -> None:
    """Prevent frozen labels from entering a future hard-negative annotation set."""
    forbidden = EXCLUDED_ANNOTATION_FIELDS.intersection(record)
    if forbidden:
        raise ValueError(f"annotation record includes forbidden frozen-benchmark fields: {sorted(forbidden)}")
    if not str(record.get("question") or "").strip():
        raise ValueError("annotation record requires a question")
    if not str(record.get("candidate_key") or "").strip():
        raise ValueError("annotation record requires a candidate_key")
    if not str(record.get("negative_type") or "").strip():
        raise ValueError("annotation record requires a negative_type")
