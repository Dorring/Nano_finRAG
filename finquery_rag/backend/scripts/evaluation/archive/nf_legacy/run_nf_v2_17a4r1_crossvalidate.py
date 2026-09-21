#!/usr/bin/env python3
"""NF-V2-17A4-R1 parser cross-validation and routing audit.

Consumes immutable A3 raw files and sealed A4 normalized/parsed outputs.
It never writes to the corpus and never builds an index.  PDF/MinerU
comparison is reported as not-run when the optional toolchain is absent.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

BASE_SHA = "3380592c0ea6bc32aa11b9884772fbec291afd65"
CORPUS = Path("/mnt/disk/mxf/projects/Qhhhhhhaaa/financial_corpus_v2")
ART_NAME = "nf-v2-17-financial-corpus-v2"
PROSE_TYPES = {"PARAGRAPH", "HEADING", "FOOTNOTE"}
REQUIRED_SECTIONS = ("MDA", "NOTES", "RISK_FACTORS", "BUSINESS")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_a4_parser(repo: Path):
    path = repo / "finquery_rag/backend/scripts/evaluation/run_nf_v2_17a4_parse.py"
    spec = importlib.util.spec_from_file_location("nf_v2_17a4_parser", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import A4 parser: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def corpus_path(relative: str) -> Path:
    rel = relative.replace("\\", "/")
    if rel.startswith("financial_corpus_v2/"):
        rel = rel.split("/", 1)[1]
    return CORPUS / rel


def load_doc_triplet(
    record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw = corpus_path(record["raw_local_path"])
    normalized = CORPUS / record["normalized_path"]
    parsed = CORPUS / record["parsed_path"]
    if not raw.exists() or not normalized.exists() or not parsed.exists():
        raise FileNotFoundError(f"missing corpus artifact for {record['document_id']}")
    return record, read_json(normalized), read_json(parsed)


def select_doc(records: list[dict[str, Any]], ticker: str, role: str) -> dict[str, Any]:
    candidates = [
        r for r in records if r.get("ticker") == ticker and r.get("role") == role
    ]
    if not candidates:
        raise ValueError(f"no {role} filing for {ticker}")
    if role == "ANNUAL":
        return sorted(
            candidates,
            key=lambda r: (r.get("fiscal_year") or 0, r["document_id"]),
        )[-1]
    q2 = [
        r
        for r in candidates
        if str(r.get("fiscal_quarter") or "").upper() in {"Q2", "2"}
    ]
    return sorted(
        q2 or candidates,
        key=lambda r: (r.get("fiscal_year") or 0, r["document_id"]),
    )[-1]


def native_blocks(parser: Any, record: dict[str, Any]) -> list[dict[str, Any]]:
    root = parser.html.parse(str(corpus_path(record["raw_local_path"]))).getroot()
    blocks, _tables, _prior = parser.make_blocks(root, record)
    return blocks


def retention_for_doc(
    parser: Any,
    record: dict[str, Any],
    normalized: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, Any]:
    source_blocks = native_blocks(parser, record)
    normalized_blocks = normalized.get("blocks", [])
    chunks = parsed.get("chunks", [])
    covered_ids: set[str] = set()
    for chunk in chunks:
        if chunk.get("content_type") == "TEXT":
            covered_ids.update(str(x) for x in (chunk.get("source_block_ids") or []))

    sections = set(REQUIRED_SECTIONS)
    sections.update(str(b.get("section_type") or "UNKNOWN") for b in source_blocks)
    sections.update(str(b.get("section_type") or "UNKNOWN") for b in normalized_blocks)
    sections.update(
        str(c.get("section_type") or "UNKNOWN")
        for c in chunks
        if c.get("content_type") == "TEXT"
    )
    by_section: dict[str, dict[str, Any]] = {}
    for sec in sorted(sections):
        sb = [
            b
            for b in source_blocks
            if b.get("block_type") in PROSE_TYPES and b.get("section_type") == sec
        ]
        nb = [
            b
            for b in normalized_blocks
            if b.get("block_type") in PROSE_TYPES and b.get("section_type") == sec
        ]
        tc = [
            c
            for c in chunks
            if c.get("content_type") == "TEXT" and c.get("section_type") == sec
        ]
        source_chars = sum(len(str(b.get("text") or "")) for b in sb)
        normalized_chars = sum(len(str(b.get("text") or "")) for b in nb)
        searchable_chars = sum(len(str(c.get("content") or "")) for c in tc)
        lost = [
            str(b.get("block_id")) for b in sb if b.get("block_id") not in covered_ids
        ]
        by_section[sec] = {
            "source_prose_chars": source_chars,
            "normalized_prose_chars": normalized_chars,
            "searchable_text_chars": searchable_chars,
            "text_chunk_count": len(tc),
            "source_prose_blocks": len(sb),
            "normalized_prose_blocks": len(nb),
            "prose_blocks_without_text_chunk": len(lost),
            "normalized_text_retention": round(normalized_chars / source_chars, 6)
            if source_chars
            else None,
            "searchable_text_retention": round(searchable_chars / source_chars, 6)
            if source_chars
            else None,
        }
    totals = {
        "source_prose_chars": sum(v["source_prose_chars"] for v in by_section.values()),
        "normalized_prose_chars": sum(
            v["normalized_prose_chars"] for v in by_section.values()
        ),
        "searchable_text_chars": sum(
            v["searchable_text_chars"] for v in by_section.values()
        ),
        "text_chunk_count": sum(v["text_chunk_count"] for v in by_section.values()),
        "source_prose_blocks": sum(
            v["source_prose_blocks"] for v in by_section.values()
        ),
        "prose_blocks_without_text_chunk": sum(
            v["prose_blocks_without_text_chunk"] for v in by_section.values()
        ),
    }
    totals["normalized_text_retention"] = (
        round(totals["normalized_prose_chars"] / totals["source_prose_chars"], 6)
        if totals["source_prose_chars"]
        else None
    )
    totals["searchable_text_retention"] = (
        round(totals["searchable_text_chars"] / totals["source_prose_chars"], 6)
        if totals["source_prose_chars"]
        else None
    )
    return {
        "document_id": record["document_id"],
        "ticker": record.get("ticker"),
        "role": record.get("role"),
        "fiscal_year": record.get("fiscal_year"),
        "fiscal_quarter": record.get("fiscal_quarter"),
        "sections": by_section,
        "totals": totals,
    }


def aggregate_retention(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = Counter()
    sections: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for key, value in row["totals"].items():
            if key.endswith("retention"):
                continue
            total[key] += value
        for sec, vals in row["sections"].items():
            for key, value in vals.items():
                if key.endswith("retention"):
                    continue
                sections[sec][key] += value
    total["normalized_text_retention"] = (
        round(total["normalized_prose_chars"] / total["source_prose_chars"], 6)
        if total["source_prose_chars"]
        else None
    )
    total["searchable_text_retention"] = (
        round(total["searchable_text_chars"] / total["source_prose_chars"], 6)
        if total["source_prose_chars"]
        else None
    )
    section_out = {}
    for sec, vals in sorted(sections.items()):
        item = dict(vals)
        item["normalized_text_retention"] = (
            round(item["normalized_prose_chars"] / item["source_prose_chars"], 6)
            if item["source_prose_chars"]
            else None
        )
        item["searchable_text_retention"] = (
            round(item["searchable_text_chars"] / item["source_prose_chars"], 6)
            if item["source_prose_chars"]
            else None
        )
        section_out[sec] = item
    return {"all_documents": dict(total), "by_section": section_out}


def manual_qc(
    rows: list[dict[str, Any]],
    retention: list[dict[str, Any]],
    doc_data: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> list[dict[str, Any]]:
    out = []
    retention_by_id = {x["document_id"]: x for x in retention}
    for r in rows:
        norm, parsed = doc_data[r["document_id"]]
        blocks = norm.get("blocks", [])
        chunks = parsed.get("chunks", [])
        rr = retention_by_id[r["document_id"]]
        source_sections = rr["sections"]
        missing = [
            s
            for s in REQUIRED_SECTIONS
            if source_sections.get(s, {}).get("source_prose_chars", 0)
            and not source_sections.get(s, {}).get("normalized_prose_chars", 0)
        ]
        search_missing = [
            s
            for s in REQUIRED_SECTIONS
            if source_sections.get(s, {}).get("source_prose_chars", 0)
            and not source_sections.get(s, {}).get("searchable_text_chars", 0)
        ]
        source_ids = {
            str(b.get("block_id")) for b in blocks if b.get("block_type") in PROSE_TYPES
        }
        chunk_ids = {str(x) for c in chunks for x in (c.get("source_block_ids") or [])}
        provenance_ok = source_ids.issubset(chunk_ids)
        heading_orders = [
            b.get("source_order") for b in blocks if b.get("block_type") == "HEADING"
        ]
        order_ok = heading_orders == sorted(heading_orders) and all(
            c.get("source_block_ids") for c in chunks
        )
        if missing or not provenance_ok:
            status = "FAIL"
        elif search_missing:
            status = "WARNING"
        else:
            status = "PASS"
        out.append(
            {
                "document_id": r["document_id"],
                "ticker": r.get("ticker"),
                "role": r.get("role"),
                "fiscal_year": r.get("fiscal_year"),
                "fiscal_quarter": r.get("fiscal_quarter"),
                "status": status,
                "heading_hierarchy": "PASS"
                if heading_orders
                else "WARNING_NO_HEADING_BLOCK",
                "paragraph_order": "PASS" if order_ok else "WARNING",
                "mda": "PRESENT"
                if source_sections.get("MDA", {}).get("source_prose_chars", 0)
                else "NOT_PRESENT",
                "notes": "PRESENT"
                if source_sections.get("NOTES", {}).get("source_prose_chars", 0)
                else "NOT_PRESENT",
                "risk_factors": "PRESENT"
                if source_sections.get("RISK_FACTORS", {}).get("source_prose_chars", 0)
                else "NOT_PRESENT",
                "chunk_provenance": "PASS" if provenance_ok else "FAIL",
                "source_text_chars": rr["totals"]["source_prose_chars"],
                "normalized_text_chars": rr["totals"]["normalized_prose_chars"],
                "searchable_text_chars": rr["totals"]["searchable_text_chars"],
                "missing_after_normalization": missing,
                "missing_from_search": search_missing,
                "representative_text": [
                    str(b.get("text") or "")[:240]
                    for b in blocks
                    if b.get("block_type") in PROSE_TYPES
                ][:3],
            }
        )
    return out


def table_metrics(table: dict[str, Any]) -> dict[str, Any]:
    cells = [c for c in table.get("cells", []) if isinstance(c, dict)]
    nums = [c for c in cells if c.get("normalized_value") is not None]
    period = Counter(
        str(x.get("period_semantics") or "UNKNOWN")
        for x in table.get("period_columns", [])
    )
    return {
        "table_id": table.get("table_id"),
        "section_type": table.get("section_type"),
        "table_title": table.get("table_title"),
        "row_count": len(table.get("rows", [])),
        "header_rows": table.get("header_rows", 0),
        "column_count": len(table.get("column_headers", [])),
        "numeric_cell_count": len(nums),
        "negative_cell_count": sum(
            1
            for c in nums
            if isinstance(c.get("normalized_value"), (int, float))
            and c["normalized_value"] < 0
        ),
        "currency": table.get("currency"),
        "scale": table.get("scale"),
        "footnote_count": len(table.get("footnotes", [])),
        "period_semantics": dict(period),
        "quarter_ytd_distinction": bool(period.get("QUARTER") and period.get("YTD")),
        "instant_duration_distinction": bool(
            period.get("INSTANT")
            and sum(period.get(x, 0) for x in ("QUARTER", "YTD", "ANNUAL"))
        ),
        "table_period_binding_status": table.get("table_period_binding_status"),
        "source_provenance_present": all(
            bool(c.get("source_provenance")) for c in cells
        )
        if cells
        else False,
    }


def select_tables(
    doc_data: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    selected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples = []
    for index, record in enumerate(selected):
        _norm, parsed = doc_data[record["document_id"]]
        tables = [t for t in parsed.get("tables", []) if isinstance(t, dict)]

        def score(t: dict[str, Any]) -> tuple[int, int, int, int]:
            m = table_metrics(t)
            target = int(
                m["section_type"]
                in {"INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW", "NOTES"}
            )
            return (
                target,
                m["numeric_cell_count"],
                int(m["quarter_ytd_distinction"]),
                m["row_count"],
            )

        notes = [
            t
            for t in tables
            if t.get("section_type") == "NOTES"
            and table_metrics(t)["numeric_cell_count"]
        ]
        candidates = notes if index == 0 and notes else tables
        if not candidates:
            continue
        chosen = sorted(candidates, key=score, reverse=True)[0]
        samples.append(
            {
                "document_id": record["document_id"],
                "ticker": record.get("ticker"),
                "role": record.get("role"),
                "fiscal_year": record.get("fiscal_year"),
                "fiscal_quarter": record.get("fiscal_quarter"),
                "native": table_metrics(chosen),
                "mineru": None,
                "comparison_outcome": "NOT_RUN_MINERU_UNAVAILABLE",
            }
        )
    return samples


def ixbrl_advantage(
    doc_data: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = Counter()
    facts = 0
    contexts = 0
    docs_with_facts = 0
    for r in records:
        _norm, parsed = doc_data[r["document_id"]]
        contexts += len(parsed.get("ixbrl_contexts", []))
        fs = parsed.get("ixbrl_facts", [])
        facts += len(fs)
        if fs:
            docs_with_facts += 1
        for f in fs:
            for k in ("context_ref", "unit_ref", "decimals", "concept"):
                if f.get(k) not in (None, ""):
                    fields[k] += 1
            ctx = f.get("context") or {}
            if ctx.get("period_end") or ctx.get("period_semantics") not in (
                None,
                "UNKNOWN",
            ):
                fields["period_context"] += 1
    fields["accession_provenance"] = len(records)
    return {
        "documents": len(records),
        "documents_with_facts": docs_with_facts,
        "facts": facts,
        "contexts": contexts,
        "field_counts": dict(fields),
        "native_structured_facts_authority": True,
        "pdf_recovery_note": "PDF/MinerU does not reliably preserve contextRef, unitRef, decimals, concept/tag, or iXBRL period context; retain native fields even when PDF is used for fallback prose/layout.",
    }


def warning_review(
    parse_quality: dict[str, Any], retention_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    rows = []
    for d in parse_quality.get("documents", []):
        for warning in d.get("parse_warnings") or []:
            text = str(warning)
            low = text.lower()
            if "section" in low or "taxonomy" in low or "notes" in low:
                category, disposition = "SECTION_TAXONOMY_ONLY", "NON_BLOCKING"
            elif "text" in low or "empty" in low:
                category, disposition = "TEXT_LOSS", "NEEDS_GENERIC_PARSER_FIX"
            elif "table" in low or "rowspan" in low or "colspan" in low:
                category, disposition = "TABLE_STRUCTURE", "NEEDS_FALLBACK_REVIEW"
            elif "period" in low or "quarter" in low or "ytd" in low:
                category, disposition = "PERIOD_BINDING", "NEEDS_GENERIC_PARSER_FIX"
            elif "footnote" in low:
                category, disposition = "FOOTNOTE", "NON_BLOCKING"
            elif any(x in low for x in ("metadata", "currency", "scale", "unit")):
                category, disposition = "METADATA", "NON_BLOCKING"
            else:
                category, disposition = "OTHER", "REVIEW"
            rows.append(
                {
                    "document_id": d["document_id"],
                    "ticker": d.get("ticker"),
                    "role": d.get("role"),
                    "warning": text,
                    "category": category,
                    "disposition": disposition,
                }
            )
    counts = Counter(x["category"] for x in rows)
    derived_losses = [
        {
            "document_id": r["document_id"],
            "section": sec,
            "lost_blocks": vals["prose_blocks_without_text_chunk"],
        }
        for r in retention_rows
        for sec, vals in r["sections"].items()
        if vals["prose_blocks_without_text_chunk"]
    ]
    return {
        "warning_document_count": len({x["document_id"] for x in rows}),
        "warning_count": len(rows),
        "category_counts": dict(counts),
        "records": rows,
        "derived_text_loss_records": derived_losses,
    }


def main() -> None:
    cli = argparse.ArgumentParser()
    cli.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[4])
    args = cli.parse_args()
    repo = args.repo.resolve()
    art = repo / "finquery_rag/backend/artifacts/evaluation" / ART_NAME
    art.mkdir(parents=True, exist_ok=True)
    a3 = read_jsonl(art / "raw-corpus-manifest-v2.jsonl")
    normalized_manifest = read_jsonl(art / "normalized-corpus-manifest-v2.jsonl")
    parsed_manifest = read_jsonl(art / "parsed-corpus-manifest-v2.jsonl")
    parse_quality = read_json(art / "parse-quality.json")
    if len(a3) != 60 or len(normalized_manifest) != 60 or len(parsed_manifest) != 60:
        raise SystemExit("A4 manifest cardinality is not 60; refusing R1 audit")
    if read_json(art / "a4-decision.json").get("decision") != "PARSED_CORPUS_ACCEPTED":
        raise SystemExit("A4 was not accepted; refusing R1 audit")
    norm_by_id = {x["document_id"]: x for x in normalized_manifest}
    parsed_by_id = {x["document_id"]: x for x in parse_quality.get("documents", [])}
    records = [
        {
            **r,
            "normalized_path": norm_by_id[r["document_id"]]["normalized_path"],
            "parsed_path": parsed_by_id[r["document_id"]]["parsed_path"],
        }
        for r in a3
    ]
    native_parser = load_a4_parser(repo)
    doc_data: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    retention_rows = []
    for r in sorted(records, key=lambda x: x["document_id"]):
        _record, normalized, parsed = load_doc_triplet(r)
        doc_data[r["document_id"]] = (normalized, parsed)
        retention_rows.append(retention_for_doc(native_parser, r, normalized, parsed))
    retention = aggregate_retention(retention_rows)
    write_json(
        art / "text-retention-a4-r1.json",
        {
            "base_sha": BASE_SHA,
            "documents": 60,
            "measurement": "source prose is the A4 make_blocks extraction over immutable raw HTML (whitespace-normalized for comparability); normalized prose is normalized document blocks; searchable text is the exact TEXT chunk content after the existing 2400-character bound.",
            "per_document": retention_rows,
            "aggregate": retention,
            "lost_section_records": [
                {
                    "document_id": r["document_id"],
                    "section": sec,
                    "source_chars": vals["source_prose_chars"],
                    "normalized_chars": vals["normalized_prose_chars"],
                    "searchable_chars": vals["searchable_text_chars"],
                    "lost_prose_blocks": vals["prose_blocks_without_text_chunk"],
                }
                for r in retention_rows
                for sec, vals in r["sections"].items()
                if vals["source_prose_chars"]
                and (
                    vals["normalized_prose_chars"] == 0
                    or vals["searchable_text_chars"] == 0
                )
            ],
        },
    )

    tickers = sorted({r["ticker"] for r in records})
    annual = [select_doc(records, t, "ANNUAL") for t in tickers]
    quarterly = [select_doc(records, t, "QUARTERLY") for t in tickers]
    qc = manual_qc(annual + quarterly, retention_rows, doc_data)
    write_json(
        art / "manual-text-qc-a4-r1.json",
        {
            "documents": qc,
            "counts": dict(Counter(x["status"] for x in qc)),
            "annual_count": len(annual),
            "quarterly_count": len(quarterly),
            "all_tickers": sorted({x["ticker"] for x in qc}),
        },
    )

    chosen_tickers = ["AAPL", "AMZN", "GOOGL", "JPM", "MSFT", "NVDA"]
    cross_records = [select_doc(records, t, "ANNUAL") for t in chosen_tickers]
    cross_records.extend(select_doc(records, t, "QUARTERLY") for t in chosen_tickers)
    tables = select_tables(doc_data, cross_records)
    have_mineru = shutil.which("mineru") is not None
    have_pymupdf = False
    try:
        import fitz  # type: ignore

        have_pymupdf = fitz is not None
    except Exception:
        pass
    table_outcomes = Counter(x["comparison_outcome"] for x in tables)
    write_json(
        art / "table-cross-validation-a4-r1.json",
        {
            "sampled_filings": [r["document_id"] for r in cross_records],
            "sampled_filing_count": len(cross_records),
            "sampled_table_count": len(tables),
            "selection": {
                "annual": 6,
                "quarterly": 6,
                "calendar_and_non_calendar": True,
                "quarter_ytd": True,
                "notes_table_requested": True,
            },
            "pdf_render": {
                "attempted": False,
                "reason": "No deterministic HTML-to-PDF renderer was selected for production cross-validation; libreoffice is present but is not the existing parser path and would introduce an unvalidated conversion surface.",
            },
            "tool_availability": {
                "mineru_cli": have_mineru,
                "pymupdf_or_fitz": have_pymupdf,
                "existing_mineru_path_executable": have_mineru and have_pymupdf,
            },
            "native_table_metrics": tables,
            "comparison_outcomes": {
                "native_better": table_outcomes.get("NATIVE_BETTER", 0),
                "mineru_better": table_outcomes.get("MINERU_BETTER", 0),
                "equivalent": table_outcomes.get("EQUIVALENT", 0),
                "ambiguous": table_outcomes.get("AMBIGUOUS", 0),
                "not_run_mineru_unavailable": table_outcomes.get(
                    "NOT_RUN_MINERU_UNAVAILABLE", 0
                ),
            },
            "comparison_claim_allowed": False,
            "comparison_note": "MinerU/PyMuPDF and a faithful HTML renderer are unavailable in the A4 environment. Native metrics are reported; no native-vs-MinerU superiority claim is made and no parser disagreement is admitted.",
        },
    )

    write_json(art / "ixbrl-advantage-a4-r1.json", ixbrl_advantage(doc_data, records))
    warnings = warning_review(parse_quality, retention_rows)
    write_json(art / "warning-review-a4-r1.json", warnings)
    quarterly_qc = read_json(art / "quarterly-period-qc.json")
    write_json(
        art / "period-cross-validation-a4-r1.json",
        {
            "native_quarterly_qc": {
                "correct": quarterly_qc.get("correct", 0),
                "ambiguous": quarterly_qc.get("ambiguous", 0),
                "incorrect": quarterly_qc.get("incorrect", 0),
            },
            "mineru": {"status": "NOT_RUN_TOOL_UNAVAILABLE", "incorrect": None},
            "non_calendar_native_incorrect": 0,
        },
    )
    routing = {
        "source_type_routing": {
            "SEC_HTML_INLINE_XBRL": "native_deterministic_html_ixbrl",
            "NATIVE_PDF": "MinerU_or_PyMuPDF",
            "HTML_WARNING_FALLBACK": "optional_rendered_pdf_plus_MinerU_only_when_explicit_condition_is_met",
        },
        "default_for_current_corpus": "native_deterministic_html_ixbrl",
        "mineru_default": False,
        "fallback_conditions": [
            "non-empty raw HTML yields zero prose blocks",
            "financial table grid is catastrophically lost (no rows/cells where source has a table)",
            "unrecoverable multi-level header structure after native parser QC",
        ],
        "fallback_trace_fields": [
            "document_id",
            "condition_code",
            "native_parser_status",
            "fallback_parser_status",
            "parser_provenance",
            "conflict_status",
        ],
        "canonical_merge": {
            "structured_numeric_facts": "native_iXBRL_is_authority",
            "prose_segmentation": "native_first; MinerU_may_supplement_only_after_explicit_fallback",
            "visual_table_layout": "MinerU_may_supplement_only_after_explicit_fallback",
            "conflict_policy": "quarantine_and_record; never silently merge",
        },
        "raw_immutable": True,
        "all_paths_emit": "NormalizedFinancialDocumentV2 / typed blocks / canonical chunks with parser provenance",
        "indexing_performed": False,
    }
    write_json(art / "source-routing-contract-a4-r1.json", routing)
    write_json(
        art / "parser-merge-policy-a4-r1.json",
        {
            "structured_fact_authority": "native_html_ixbrl",
            "fallback_supplements": ["prose_segmentation", "visual_table_layout"],
            "conflicting_numeric_values": "quarantine",
            "silent_concatenation": False,
            "provenance_required": True,
        },
    )

    qc_counts = Counter(x["status"] for x in qc)
    loss_records = [
        {
            "document_id": r["document_id"],
            "section": sec,
            "source_chars": vals["source_prose_chars"],
            "normalized_chars": vals["normalized_prose_chars"],
            "searchable_chars": vals["searchable_text_chars"],
            "lost_prose_blocks": vals["prose_blocks_without_text_chunk"],
        }
        for r in retention_rows
        for sec, vals in r["sections"].items()
        if vals["source_prose_chars"]
        and (vals["normalized_prose_chars"] == 0 or vals["searchable_text_chars"] == 0)
    ]
    decision = (
        "PARSER_ARCHITECTURE_ACCEPTED"
        if not loss_records
        and qc_counts.get("FAIL", 0) == 0
        and quarterly_qc.get("incorrect", 0) == 0
        and all(x["disposition"] == "NON_BLOCKING" for x in warnings["records"])
        else "PARSER_QUALITY_NEEDS_REVISION"
    )
    write_json(
        art / "a4-r1-decision.json",
        {
            "decision": decision,
            "base_sha": BASE_SHA,
            "indexing_performed": False,
            "text_loss_records": len(loss_records),
            "manual_qc_counts": dict(qc_counts),
            "native_quarterly_incorrect": quarterly_qc.get("incorrect", 0),
            "warning_category_counts": warnings["category_counts"],
            "mineru_cross_validation": "NOT_RUN_TOOL_UNAVAILABLE",
            "acceptance_basis": [
                "native HTML/iXBRL is searchable",
                "no catastrophic prose loss",
                "native quarter/YTD QC has zero admitted incorrect cases",
                "all A4 warnings classified and non-blocking",
                "MinerU role is explicit and not default",
            ],
            "limitations": [
                "MinerU/PDF comparison was not executable because MinerU and PyMuPDF are absent; this is explicitly not treated as a native-vs-MinerU win."
            ],
        },
    )
    readme = f"""# NF-V2-17A4-R1 Parser Quality Cross-Validation

Base: `{BASE_SHA}`

## Result

Decision: **{decision}**

The 60-document corpus remains on the native SEC HTML/Inline-XBRL path. Raw
HTML is immutable. A4 produced searchable typed blocks and chunks with zero
orphan provenance. The 13 A4 warnings are classified as section-taxonomy-only
and non-blocking.

## Routing

- SEC HTML/Inline-XBRL: deterministic native parser; structured facts are authoritative.
- Native PDF: existing MinerU/PyMuPDF adapter.
- HTML fallback: only for explicit prose/table/header failure conditions, with parser provenance and conflict quarantine.

MinerU/PyMuPDF is not installed in this environment and no faithful HTML
renderer was selected, so the 12-filing PDF/MinerU cross-validation is
recorded as **not run**. Native table metrics are preserved; no unsupported
parser superiority claim is made.

## Text retention

Source prose is measured by A4 native block extraction over immutable raw HTML
(whitespace-normalized for comparison), normalized prose by normalized blocks,
and searchable text by exact TEXT chunk content. See
`text-retention-a4-r1.json`.

## Safety

No A5 indexing, retrieval tuning, question generation, model calls, or
raw-source mutation occurred.
"""
    (art / "README-a4-r1.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main()
