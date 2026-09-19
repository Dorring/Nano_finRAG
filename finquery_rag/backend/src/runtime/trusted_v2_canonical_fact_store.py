"""Build strict V2 fact-store records from canonical parsed financial documents.

The builder consumes only source-derived parser output and the existing
``html_semantic_adapter``. It never reads questions, Gold, answers,
conversation state, or model output. Each emitted record is one numeric cell
with metric, period, value, and a physical provenance chain.
"""

from __future__ import annotations

import copy
import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.pdf_retrieval_v4.html_semantic_adapter import build_semantic_corpus


CANONICAL_FACT_STORE_SCHEMA_VERSION = "trusted-v2-canonical-fact-store/v1"
_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _stable_unique(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            result.append(text)
            seen.add(text)
    return result


def _canonical_period(
    *, normalized_period: Any, period_end: Any, period_semantics: Any
) -> str | None:
    """Render an FY label only for an explicitly annual source cell."""

    source = _text(normalized_period) or _text(period_end)
    if not source:
        return None
    if _text(period_semantics).upper() == "ANNUAL":
        match = _YEAR_RE.search(source)
        if match:
            return f"FY{match.group(1)}"
    return source


def _record_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(_text(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class CanonicalFactStoreBuildSummary:
    documents_seen: int
    atomic_facts_seen: int
    emitted_facts: int
    skipped_missing_metric: int
    skipped_missing_period: int
    skipped_missing_value: int
    skipped_missing_provenance: int
    annual_fy_periods: int
    exact_source_periods: int
    currency_populated: int
    scale_populated: int
    unit_populated: int
    statement_type_populated: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": CANONICAL_FACT_STORE_SCHEMA_VERSION,
            "documents_seen": self.documents_seen,
            "atomic_facts_seen": self.atomic_facts_seen,
            "emitted_facts": self.emitted_facts,
            "skipped_missing_metric": self.skipped_missing_metric,
            "skipped_missing_period": self.skipped_missing_period,
            "skipped_missing_value": self.skipped_missing_value,
            "skipped_missing_provenance": self.skipped_missing_provenance,
            "annual_fy_periods": self.annual_fy_periods,
            "exact_source_periods": self.exact_source_periods,
            "currency_populated": self.currency_populated,
            "scale_populated": self.scale_populated,
            "unit_populated": self.unit_populated,
            "statement_type_populated": self.statement_type_populated,
        }


def build_canonical_fact_store(
    parsed_documents: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], CanonicalFactStoreBuildSummary]:
    """Emit immutable Binder-ready cell facts from canonical parsed documents."""

    documents = [copy.deepcopy(dict(item)) for item in parsed_documents]
    # The semantic adapter builds a complete graph for its input. Process a
    # corpus document-by-document so a full asset build stays bounded; the
    # resulting record contract remains physical-cell deterministic.
    if len(documents) > 1:
        combined: list[dict[str, Any]] = []
        summaries: list[CanonicalFactStoreBuildSummary] = []
        for document in documents:
            records, summary = build_canonical_fact_store([document])
            combined.extend(records)
            summaries.append(summary)
        keys = [str(record["candidate_key"]) for record in combined]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate canonical candidate key")
        return combined, CanonicalFactStoreBuildSummary(
            documents_seen=sum(item.documents_seen for item in summaries),
            atomic_facts_seen=sum(item.atomic_facts_seen for item in summaries),
            emitted_facts=sum(item.emitted_facts for item in summaries),
            skipped_missing_metric=sum(item.skipped_missing_metric for item in summaries),
            skipped_missing_period=sum(item.skipped_missing_period for item in summaries),
            skipped_missing_value=sum(item.skipped_missing_value for item in summaries),
            skipped_missing_provenance=sum(item.skipped_missing_provenance for item in summaries),
            annual_fy_periods=sum(item.annual_fy_periods for item in summaries),
            exact_source_periods=sum(item.exact_source_periods for item in summaries),
            currency_populated=sum(item.currency_populated for item in summaries),
            scale_populated=sum(item.scale_populated for item in summaries),
            unit_populated=sum(item.unit_populated for item in summaries),
            statement_type_populated=sum(item.statement_type_populated for item in summaries),
        )
    # The canonical parser stores document metadata below ``document`` while
    # the shared semantic adapter expects it at the immediate input level.
    # Flatten only that existing source object; tables and iXBRL arrays remain
    # the exact parser artifacts.
    adapter_documents: list[dict[str, Any]] = []
    for item in documents:
        metadata = item.get("document")
        if not isinstance(metadata, Mapping):
            raise ValueError("parsed document is missing its document metadata object")
        adapter_documents.append({**item, **copy.deepcopy(dict(metadata))})
    corpus = build_semantic_corpus(adapter_documents)
    docs_by_id: dict[str, Mapping[str, Any]] = {
        _text(item.get("document", {}).get("document_id")): item
        for item in documents
        if isinstance(item.get("document"), Mapping)
        and _text(item.get("document", {}).get("document_id"))
    }
    tables_by_id: dict[tuple[str, str], Mapping[str, Any]] = {}
    for item in documents:
        document = item.get("document")
        if not isinstance(document, Mapping):
            continue
        document_id = _text(document.get("document_id"))
        for table in item.get("tables") or ():
            if isinstance(table, Mapping) and _text(table.get("table_id")):
                tables_by_id[(document_id, _text(table.get("table_id")))] = table

    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    populated: Counter[str] = Counter()
    annual_fy_periods = 0
    exact_source_periods = 0
    for atomic in corpus["atomic_facts"]:
        payload = atomic.to_dict()
        document_id = _text(payload.get("document_id"))
        table_id = _text(payload.get("table_fragment_id"))
        row_id = _text(payload.get("row_id"))
        cell_id = _text(payload.get("cell_id"))
        metric = _text(payload.get("metric_path")) or _text(payload.get("leaf_metric"))
        fact_meta = corpus["fact_meta"].get(payload.get("semantic_fact_id"), {})
        period_semantics = _text(fact_meta.get("period_semantics"))
        period = _canonical_period(
            normalized_period=payload.get("normalized_period"),
            period_end=payload.get("period_end"),
            period_semantics=period_semantics,
        )
        source_period = _text(payload.get("normalized_period")) or _text(payload.get("period_end"))
        value = _text(payload.get("value_normalized"))
        if not metric:
            skipped["missing_metric"] += 1
            continue
        if not period:
            skipped["missing_period"] += 1
            continue
        if not value:
            skipped["missing_value"] += 1
            continue
        if not all((document_id, table_id, row_id, cell_id)):
            skipped["missing_provenance"] += 1
            continue
        document = docs_by_id.get(document_id, {})
        document_meta = document.get("document") if isinstance(document, Mapping) else {}
        document_meta = document_meta if isinstance(document_meta, Mapping) else {}
        table = tables_by_id.get((document_id, table_id), {})
        table = table if isinstance(table, Mapping) else {}
        traceback = payload.get("source_traceback")
        traceback = traceback if isinstance(traceback, Mapping) else {}
        source_parts = [
            _text(table.get("table_title")),
            _text(payload.get("leaf_metric")) or metric,
            *_stable_unique(fact_meta.get("header_path") or ()),
            _text(payload.get("value_raw")),
        ]
        content = " | ".join(part for part in source_parts if part)
        if not content:
            skipped["missing_provenance"] += 1
            continue
        currency = _text(payload.get("currency_code")) or _text(table.get("currency")) or None
        scale = (_text(payload.get("scale")) or _text(payload.get("scale_unit")) or _text(table.get("scale")) or None)
        unit = _text(table.get("unit")) or None
        statement_type = _text(table.get("section_type")) or None
        for name, field in (("currency", currency), ("scale", scale), ("unit", unit), ("statement_type", statement_type)):
            if field:
                populated[name] += 1
        if period.startswith("FY"):
            annual_fy_periods += 1
        elif source_period:
            exact_source_periods += 1
        physical_id = _record_id("physical", document_id, table_id, row_id, cell_id)
        evidence_id = _record_id("evidence", document_id, table_id, row_id, cell_id)
        candidate_key = _record_id("candidate", document_id, table_id, row_id, cell_id)
        citation_id = _record_id("citation", document_id, table_id, row_id, cell_id)
        records.append({
            "candidate_key": candidate_key,
            "candidate_id": candidate_key,
            "fact_id": evidence_id,
            "evidence_id": evidence_id,
            "citation_id": citation_id,
            "citation_ids": [citation_id],
            "provenance_complete": True,
            "fact_type": "atomic_financial_cell",
            "entity": _text(document_meta.get("company")) or _text(document_meta.get("ticker")) or None,
            "ticker": _text(document_meta.get("ticker")) or None,
            "metric": metric,
            "normalized_metric": metric,
            "period": period,
            "normalized_period": source_period or period,
            "period_start": _text(payload.get("period_start")) or None,
            "period_end": _text(payload.get("period_end")) or None,
            "period_semantics": period_semantics or "UNKNOWN",
            "value": value,
            "parsed_numeric_value": value,
            "raw_value": _text(payload.get("value_raw")) or None,
            "currency": currency,
            "scale": scale,
            "unit": unit,
            "statement_type": statement_type,
            "document_id": document_id,
            "physical_source_id": physical_id,
            "table_fragment_id": table_id,
            "table_id": table_id,
            "row_id": row_id,
            "cell_id": cell_id,
            "pdf_page": traceback.get("pdf_page"),
            "content": content,
            "source_text": content,
            "source_traceback": {"document_id": document_id, "table_fragment_id": table_id, "row_id": row_id, "cell_id": cell_id, "pdf_page": traceback.get("pdf_page")},
            "metadata_origin": "canonical_parsed_corpus",
        })
    keys = [str(record["candidate_key"]) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate canonical candidate key")
    return records, CanonicalFactStoreBuildSummary(
        documents_seen=len(documents), atomic_facts_seen=len(corpus["atomic_facts"]), emitted_facts=len(records),
        skipped_missing_metric=skipped["missing_metric"], skipped_missing_period=skipped["missing_period"],
        skipped_missing_value=skipped["missing_value"], skipped_missing_provenance=skipped["missing_provenance"],
        annual_fy_periods=annual_fy_periods, exact_source_periods=exact_source_periods,
        currency_populated=populated["currency"], scale_populated=populated["scale"],
        unit_populated=populated["unit"], statement_type_populated=populated["statement_type"],
    )


__all__ = ["CANONICAL_FACT_STORE_SCHEMA_VERSION", "CanonicalFactStoreBuildSummary", "build_canonical_fact_store"]
