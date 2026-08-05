"""Isolation guards for the NF-OPT-17 financial hard-negative dev corpus."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
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
