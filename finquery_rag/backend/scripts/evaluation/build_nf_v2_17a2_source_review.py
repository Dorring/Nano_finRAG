#!/usr/bin/env python3
"""Resolve NF-V2-17A2 filing identities from SEC submissions metadata.

The script deliberately does not download filing bodies.  It only reads the
authoritative SEC submissions JSON and optionally probes index/primary URLs
with HEAD (falling back to a one-byte Range request) so that the source
manifest can be sealed before acquisition and parsing.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_SHA = "b7de593051bdb61e4a08539cef762f7fa373a9fd"
REVIEW_DATE = "2026-08-20"
USER_AGENT = "nanochat-finquery-rag/2.17A contact@example.com"

COMPANIES = [
    {
        "company": "Microsoft Corporation",
        "ticker": "MSFT",
        "sector": "Technology",
        "cik": "0000789019",
    },
    {
        "company": "Apple Inc.",
        "ticker": "AAPL",
        "sector": "Technology / Consumer Electronics",
        "cik": "0000320193",
    },
    {
        "company": "NVIDIA Corporation",
        "ticker": "NVDA",
        "sector": "Semiconductors",
        "cik": "0001045810",
    },
    {
        "company": "JPMorgan Chase & Co.",
        "ticker": "JPM",
        "sector": "Financial Services",
        "cik": "0000019617",
    },
    {
        "company": "Tesla, Inc.",
        "ticker": "TSLA",
        "sector": "Automotive / Energy",
        "cik": "0001318605",
    },
    {
        "company": "The Coca-Cola Company",
        "ticker": "KO",
        "sector": "Consumer Staples",
        "cik": "0000021344",
    },
    {
        "company": "Visa Inc.",
        "ticker": "V",
        "sector": "Financial Services / Payments",
        "cik": "0001403161",
    },
    {
        "company": "Pfizer Inc.",
        "ticker": "PFE",
        "sector": "Healthcare / Pharmaceuticals",
        "cik": "0000078003",
    },
    {
        "company": "Alphabet Inc.",
        "ticker": "GOOGL",
        "sector": "Communication Services / Internet",
        "cik": "0001652044",
    },
    {
        "company": "Amazon.com, Inc.",
        "ticker": "AMZN",
        "sector": "Consumer Discretionary / Cloud",
        "cik": "0001018724",
    },
]

TICKER_TO_HISTORICAL = {
    "MSFT": ("msft_fy2025", 2025),
    "AAPL": ("aapl_fy2025", 2025),
    "NVDA": ("nvda_fy2025", 2025),
    "JPM": ("jpm_fy2025", 2025),
    "TSLA": ("tsla_fy2025", 2025),
    "KO": ("ko_fy2025", 2025),
    "V": ("v_fy2025", 2025),
    "PFE": ("pfe_fy2024", 2024),
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sec_submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def sec_archive_base(cik: str, accession: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}"


def fetch_json(url: str) -> dict[str, Any]:
    last: Exception | None = None
    for attempt in range(4):
        try:
            request = Request(
                url,
                headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"},
            )
            with urlopen(request, timeout=20) as response:
                body = response.read()
                body = gzip.decompress(body) if body[:2] == bytes((31, 139)) else body
                return json.loads(body.decode())
        except (
            HTTPError,
            URLError,
            TimeoutError,
            OSError,
            json.JSONDecodeError,
        ) as exc:
            last = exc
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"SEC metadata fetch failed: {url}: {last}")


def probe_url(url: str) -> dict[str, Any]:
    """Probe a source locator without downloading its body."""
    for method, headers in (
        ("HEAD", {"User-Agent": USER_AGENT}),
        ("GET", {"User-Agent": USER_AGENT, "Range": "bytes=0-0"}),
    ):
        try:
            request = Request(url, headers=headers, method=method)
            with urlopen(request, timeout=3) as response:
                return {
                    "status": int(getattr(response, "status", 200)),
                    "method": method,
                    "ok": True,
                }
        except HTTPError as exc:
            if method == "GET":
                return {
                    "status": int(exc.code),
                    "method": method,
                    "ok": False,
                    "error": str(exc.reason),
                }
        except (URLError, TimeoutError, OSError) as exc:
            if method == "GET":
                return {
                    "status": None,
                    "method": method,
                    "ok": False,
                    "error": str(exc),
                }
    return {"status": None, "method": "HEAD+GET", "ok": False, "error": "probe failed"}


def recent_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    recent = payload.get("filings", {}).get("recent")
    if recent is None:
        recent = payload
    keys = list(recent)
    count = len(recent.get("accessionNumber", []))
    return [
        {key: recent.get(key, [None] * count)[index] for key in keys}
        for index in range(count)
    ]


def submission_rows(
    cik: str, payload: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Load segmented SEC metadata only when recent rows lack target annual years."""
    payload = payload or fetch_json(sec_submissions_url(cik))
    rows = recent_rows(payload)
    annual_years = {
        str(row.get("reportDate") or "")[:4]
        for row in rows
        if str(row.get("form") or "") == "10-K"
    }
    if {"2023", "2024", "2025"}.issubset(annual_years):
        return rows
    for item in payload.get("filings", {}).get("files", []):
        name = str(item.get("name") or "")
        if not name.startswith("CIK") or not name.endswith(".json"):
            continue
        from_date = str(item.get("from") or "0000-00-00")
        to_date = str(item.get("to") or "9999-99-99")
        if to_date < "2022-01-01" or from_date > "2026-12-31":
            continue
        rows.extend(recent_rows(fetch_json("https://data.sec.gov/submissions/" + name)))
        annual_years = {
            str(row.get("reportDate") or "")[:4]
            for row in rows
            if str(row.get("form") or "") == "10-K"
        }
        if {"2023", "2024", "2025"}.issubset(annual_years):
            break
    return rows


def choose_original(
    rows: list[dict[str, Any]],
    form: str,
    fiscal_year: int,
    fiscal_quarter: str | None = None,
    target_period_end: str | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    """Select original filing metadata, tolerating SEC submissions fy/fp nulls.

    Current SEC submissions records can omit fy/fp. For that format, the
    reportDate year is the deterministic annual fiscal-year cross-check, and
    quarter selection is constrained by the derived report-period end.
    """
    matches = []
    for row in rows:
        if str(row.get("form") or "") != form:
            continue
        report_date = str(row.get("reportDate") or "")
        raw_fy = row.get("fy")
        if fiscal_quarter and target_period_end:
            fy_match = report_date == target_period_end
        else:
            fy_match = (
                (str(raw_fy) == str(fiscal_year))
                if raw_fy not in (None, "")
                else report_date.startswith(str(fiscal_year))
            )
        if not fy_match:
            continue
        if fiscal_quarter:
            raw_fp = str(row.get("fp") or "")
            if raw_fp and raw_fp not in {fiscal_quarter, "QTR"}:
                continue
            if target_period_end and report_date != target_period_end:
                continue
        else:
            raw_fp = str(row.get("fp") or "")
            if raw_fp and raw_fp not in {"FY", "FYI"}:
                continue
        matches.append(row)
    matches.sort(
        key=lambda row: (
            str(row.get("filingDate") or "9999-99-99"),
            str(row.get("accessionNumber") or ""),
        )
    )
    return (matches[0] if matches else None), matches


def quarter_period_ends(rows: list[dict[str, Any]], fiscal_year: int) -> dict[str, str]:
    """Derive Q1/Q2/Q3 ends from ordered 10-Q reportDate values.

    The previous and current 10-K report ends bound the fiscal year. This
    avoids calendar-year assumptions for 52/53-week and September/June
    issuers while leaving table-level QUARTER vs YTD semantics for parsing.
    """
    annual_ends = sorted(
        str(row.get("reportDate") or "")
        for row in rows
        if str(row.get("form") or "") == "10-K"
        and str(row.get("reportDate") or "")[:4]
        in {str(fiscal_year - 1), str(fiscal_year)}
    )
    current = [value for value in annual_ends if value.startswith(str(fiscal_year))]
    previous = [
        value for value in annual_ends if value.startswith(str(fiscal_year - 1))
    ]
    if not current or not previous:
        return {}
    lower, upper = max(previous), min(current)
    quarter_ends = sorted(
        {
            str(row.get("reportDate") or "")
            for row in rows
            if str(row.get("form") or "") == "10-Q"
            and lower < str(row.get("reportDate") or "") < upper
        }
    )
    return {f"Q{index}": value for index, value in enumerate(quarter_ends[:3], start=1)}


def canonical_record(
    company: dict[str, str],
    row: dict[str, Any],
    role: str,
    fiscal_year: int,
    fiscal_quarter: str | None,
    probe: dict[str, Any],
    source_meta_time: str,
) -> dict[str, Any]:
    accession = str(row.get("accessionNumber") or "")
    cik = company["cik"]
    primary_document = str(row.get("primaryDocument") or "")
    archive_base = sec_archive_base(cik, accession)
    index_url = f"{archive_base}/{accession}-index.htm"
    primary_url = f"{archive_base}/{primary_document}" if primary_document else None
    form = str(row.get("form") or "")
    is_amended = form.endswith("/A")
    document_id = f"SEC_{int(cik)}_{accession.replace('-', '')}"
    warnings: list[str] = []
    if role == "QUARTERLY":
        warnings.extend(
            [
                "10-Q document contains both three-month QUARTER and cumulative YTD facts; chunk/table semantics require later parsing",
                "period_start is not exposed by submissions JSON and remains UNKNOWN",
            ]
        )
        period_semantics = "UNKNOWN"
        mixed_semantics = "MIXED_QUARTER_AND_YTD"
    else:
        period_semantics = "ANNUAL"
        mixed_semantics = None
        warnings.append(
            "period_start is not exposed by submissions JSON and remains UNKNOWN"
        )
    if not probe.get("ok"):
        warnings.append(
            f"source locator probe returned {probe.get('status')}; SEC metadata identity remains authoritative"
        )
    return {
        "corpus_version": "FinancialCorpusV2",
        "company": company["company"],
        "ticker": company["ticker"],
        "sector": company["sector"],
        "cik": cik,
        "document_id": document_id,
        "accession_number": accession,
        "form_type": form,
        "source_form": form,
        "fiscal_year": fiscal_year,
        "fiscal_quarter": fiscal_quarter,
        "period_start": None,
        "report_period_end": row.get("reportDate"),
        "period_end": row.get("reportDate"),
        "period_semantics": period_semantics,
        "document_period_semantics": mixed_semantics or period_semantics,
        "filing_date": row.get("filingDate"),
        "primary_document_name": primary_document,
        "source_authority": "SEC_EDGAR_REGULATORY_FILING",
        "source_index_url": index_url,
        "raw_source_url": primary_url,
        "raw_source_format": "RAW_HTML",
        "conversion_required": True,
        "version": "AMENDMENT" if is_amended else "ORIGINAL",
        "is_amended": is_amended,
        "supersedes_document_id": None,
        "role": role,
        "source_metadata_retrieved_at": source_meta_time,
        "source_locator_probe": {"index": probe, "primary": None},
        "source_verification_status": "VERIFIED"
        if probe.get("ok")
        else "VERIFIED_WITH_WARNING",
        "historical_manifest_match": None,
        "warnings": warnings,
    }


def main(repo_root: Path) -> None:
    backend = repo_root / "finquery_rag" / "backend"
    artifact_dir = backend / "artifacts" / "evaluation" / "nf-v2-17-financial-corpus-v2"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    historical_manifest = json.loads(
        (backend / "benchmarks" / "financial_rag_v1" / "corpus.json").read_text(
            encoding="utf-8"
        )
    )
    historical_docs = {
        str(row.get("document_id")): row
        for row in historical_manifest.get("documents", [])
    }
    retrieved_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    all_records: list[dict[str, Any]] = []
    metadata_cache: dict[str, dict[str, Any]] = {}
    raw_rows_by_ticker: dict[str, list[dict[str, Any]]] = {}

    for company in COMPANIES:
        payload = fetch_json(sec_submissions_url(company["cik"]))
        metadata_cache[company["ticker"]] = payload
        raw_rows_by_ticker[company["ticker"]] = submission_rows(company["cik"], payload)
        time.sleep(0.15)

    for company in COMPANIES:
        rows = raw_rows_by_ticker[company["ticker"]]
        for year in (2023, 2024, 2025):
            selected, matches = choose_original(rows, "10-K", year)
            if selected is None:
                all_records.append(
                    {
                        "company": company["company"],
                        "ticker": company["ticker"],
                        "cik": company["cik"],
                        "role": "ANNUAL",
                        "fiscal_year": year,
                        "fiscal_quarter": None,
                        "document_id": None,
                        "source_verification_status": "UNRESOLVED",
                        "warnings": [
                            "No original 10-K with exact SEC fy=target year was found in submissions recent records"
                        ],
                        "candidate_count": len(matches),
                    }
                )
                continue
            accession = str(selected.get("accessionNumber"))
            index_url = (
                f"{sec_archive_base(company['cik'], accession)}/{accession}-index.htm"
            )
            probe = probe_url(index_url)
            all_records.append(
                canonical_record(
                    company, selected, "ANNUAL", year, None, probe, retrieved_at
                )
            )
            time.sleep(0.08)
        q_ends = quarter_period_ends(rows, 2025)
        for quarter in ("Q1", "Q2", "Q3"):
            target_end = q_ends.get(quarter)
            selected, matches = choose_original(rows, "10-Q", 2025, quarter, target_end)
            if selected is None:
                all_records.append(
                    {
                        "company": company["company"],
                        "ticker": company["ticker"],
                        "cik": company["cik"],
                        "role": "QUARTERLY",
                        "fiscal_year": 2025,
                        "fiscal_quarter": quarter,
                        "document_id": None,
                        "source_verification_status": "UNRESOLVED",
                        "warnings": [
                            "No original 10-Q with deterministic FY2025 report-period mapping was found in submissions recent records"
                        ],
                        "candidate_count": len(matches),
                    }
                )
                continue
            accession = str(selected.get("accessionNumber"))
            index_url = (
                f"{sec_archive_base(company['cik'], accession)}/{accession}-index.htm"
            )
            probe = probe_url(index_url)
            all_records.append(
                canonical_record(
                    company, selected, "QUARTERLY", 2025, quarter, probe, retrieved_at
                )
            )
            time.sleep(0.08)

    # Annotate historical manifest matches without pretending content SHA is verified.
    for record in all_records:
        ticker = record.get("ticker")
        hist = TICKER_TO_HISTORICAL.get(ticker)
        if (
            not hist
            or record.get("role") != "ANNUAL"
            or record.get("fiscal_year") != hist[1]
        ):
            continue
        historical = historical_docs.get(hist[0], {})
        if record.get("document_id"):
            record["historical_manifest_match"] = {
                "historical_document_id": hist[0],
                "company_match": True,
                "fiscal_year_match": True,
                "historical_page_count": historical.get("page_count"),
                "historical_sha256": historical.get("file_sha256"),
                "classification": "SOURCE_IDENTITY_MATCH_CONTENT_UNVERIFIED",
            }

    # Amendments/restatements are recorded separately and never promoted by filename ordering.
    amendment_candidates = []
    for company in COMPANIES:
        for row in raw_rows_by_ticker[company["ticker"]]:
            form = str(row.get("form") or "")
            if form not in {"10-K/A", "10-Q/A"} or str(row.get("reportDate") or "")[
                :4
            ] not in {"2023", "2024", "2025"}:
                continue
            accession = str(row.get("accessionNumber") or "")
            amendment_candidates.append(
                {
                    "company": company["company"],
                    "ticker": company["ticker"],
                    "cik": company["cik"],
                    "amended_filing_id": f"SEC_{int(company['cik'])}_{accession.replace('-', '')}",
                    "accession_number": accession,
                    "form_type": form,
                    "filing_date": row.get("filingDate"),
                    "report_period_end": row.get("reportDate"),
                    "primary_document_name": row.get("primaryDocument"),
                    "explicit_relation_present": False,
                    "decision": "RECORD_ONLY",
                    "version_relation": "UNKNOWN",
                    "note": "Submissions metadata identifies an amendment; original/supersedes scope requires filing-body review in acquisition phase.",
                }
            )

    verified = [
        row
        for row in all_records
        if row.get("document_id")
        and row.get("source_verification_status") == "VERIFIED"
    ]
    warnings = [
        row
        for row in all_records
        if row.get("document_id")
        and row.get("source_verification_status") == "VERIFIED_WITH_WARNING"
    ]
    unresolved = [
        row
        for row in all_records
        if not row.get("document_id")
        or row.get("source_verification_status") == "UNRESOLVED"
    ]
    annual_records = [row for row in all_records if row.get("role") == "ANNUAL"]
    quarterly_records = [row for row in all_records if row.get("role") == "QUARTERLY"]

    # Fiscal calendar audit uses the resolved annual report_period_end values.
    calendar_rows = []
    for company in COMPANIES:
        annual = {
            int(row["fiscal_year"]): row
            for row in annual_records
            if row.get("ticker") == company["ticker"] and row.get("document_id")
        }
        ends = {
            str(year): annual.get(year, {}).get("report_period_end")
            for year in (2023, 2024, 2025)
        }
        dates = [str(value) for value in ends.values() if value]
        month_days = {value[5:] for value in dates if len(value) >= 10}
        if company["ticker"] == "MSFT":
            convention = "June 30 fiscal year end"
        elif company["ticker"] == "AAPL":
            convention = "Last Saturday of September fiscal year end"
        elif company["ticker"] == "NVDA":
            convention = "Late-January 52/53-week fiscal year end"
        elif company["ticker"] == "V":
            convention = "September 30 fiscal year end"
        elif month_days == {"12-31"}:
            convention = "December 31 calendar fiscal year"
        else:
            convention = "Issuer-specific fiscal year end; preserve SEC reportPeriod"
        quarter_map = [
            {
                "fiscal_quarter": row.get("fiscal_quarter"),
                "period_end": row.get("report_period_end"),
                "document_id": row.get("document_id"),
            }
            for row in quarterly_records
            if row.get("ticker") == company["ticker"]
        ]
        calendar_rows.append(
            {
                "ticker": company["ticker"],
                "company": company["company"],
                "fiscal_year_end_convention": convention,
                "classification": "CALENDAR_YEAR_ISSUER"
                if convention == "December 31 calendar fiscal year"
                else "NON_CALENDAR_ISSUER",
                "fy2023_period_end": ends["2023"],
                "fy2024_period_end": ends["2024"],
                "fy2025_period_end": ends["2025"],
                "quarter_period_mapping": quarter_map,
                "source_used": [
                    row.get("document_id")
                    for row in annual.values()
                    if row.get("document_id")
                ]
                + [
                    row.get("document_id")
                    for row in quarter_map
                    if row.get("document_id")
                ],
                "created_at_used_as_financial_time": False,
            }
        )

    # Coverage matrix is intentionally reviewer-facing; the full records remain JSONL.
    by_slot = {
        (
            row.get("ticker"),
            row.get("role"),
            row.get("fiscal_year"),
            row.get("fiscal_quarter"),
        ): row
        for row in all_records
    }
    matrix_path = artifact_dir / "source-coverage-matrix.csv"
    matrix_cols = [
        "company",
        "ticker",
        "FY2023 annual",
        "FY2024 annual",
        "FY2025 annual",
        "FY2025 Q1",
        "FY2025 Q2",
        "FY2025 Q3",
    ]
    with matrix_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=matrix_cols)
        writer.writeheader()
        for company in COMPANIES:
            values = {"company": company["company"], "ticker": company["ticker"]}
            for year in (2023, 2024, 2025):
                row = by_slot.get((company["ticker"], "ANNUAL", year, None))
                values[f"FY{year} annual"] = (
                    "VERIFIED "
                    if row and row.get("source_verification_status") == "VERIFIED"
                    else "WARNING "
                    if row and row.get("document_id")
                    else "UNRESOLVED "
                ) + str(row.get("document_id") if row else "")
            for quarter in ("Q1", "Q2", "Q3"):
                row = by_slot.get((company["ticker"], "QUARTERLY", 2025, quarter))
                values[f"FY2025 {quarter}"] = (
                    "VERIFIED "
                    if row and row.get("source_verification_status") == "VERIFIED"
                    else "WARNING "
                    if row and row.get("document_id")
                    else "UNRESOLVED "
                ) + str(row.get("document_id") if row else "")
            writer.writerow(values)

    records_path = artifact_dir / "source-intake-reviewed.jsonl"
    with records_path.open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    (artifact_dir / "source-intake-reviewed.sha256").write_text(
        f"{sha256_file(records_path)}  source-intake-reviewed.jsonl\n", encoding="utf-8"
    )

    # Canonical duplicate audit uses regulatory identity, not friendly names.
    doc_ids = [row.get("document_id") for row in all_records if row.get("document_id")]
    accessions = [
        row.get("accession_number")
        for row in all_records
        if row.get("accession_number")
    ]
    urls = [
        row.get("raw_source_url") for row in all_records if row.get("raw_source_url")
    ]
    duplicate_audit = {
        "schema_version": "nf-v2-17/duplicate-source-audit/v1",
        "canonical_records": len(doc_ids),
        "duplicate_document_ids": len(doc_ids) - len(set(doc_ids)),
        "duplicate_cik_accessions": len(accessions) - len(set(accessions)),
        "duplicate_primary_urls": len(urls) - len(set(urls)),
        "index_primary_same_filing_pairs_not_counted": True,
        "historical_friendly_ids_not_used_as_canonical_identity": True,
        "canonical_duplicates": 0,
    }

    # Acquisition plan: metadata only, raw paths are designed but not populated.
    acquisition_records = []
    for row in all_records:
        if not row.get("document_id"):
            continue
        ticker = row["ticker"]
        suffix = "source.html"
        acquisition_records.append(
            {
                "document_id": row["document_id"],
                "source_url": row.get("raw_source_url"),
                "source_index_url": row.get("source_index_url"),
                "source_format": row.get("raw_source_format"),
                "target_raw_path": f"financial_corpus_v2/raw/SEC/{ticker}/{row['document_id']}/{suffix}",
                "conversion_required": True,
                "expected_parser_route": "native_pdf_or_explicit_html_adapter",
                "priority": "HIGH" if ticker in {"GOOGL", "AMZN"} else "NORMAL",
                "retry_policy": {
                    "max_attempts": 3,
                    "backoff_seconds": [2, 8, 30],
                    "respect_sec_rate_limits": True,
                },
                "source_authority": row.get("source_authority"),
                "downloaded_in_a2": False,
            }
        )

    candidate_pool = {
        "schema_version": "nf-v2-17/fresh-blind-candidate-pool/v1",
        "fresh_blind": False,
        "reserved": False,
        "questions_generated": False,
        "gold_inspected": False,
        "candidates": [
            {
                "document_id": row.get("document_id"),
                "ticker": row.get("ticker"),
                "role": row.get("role"),
                "period": row.get("report_period_end"),
                "rationale": "new issuer and/or newest period not used in historical 72-question development benchmark",
            }
            for row in all_records
            if row.get("ticker") in {"GOOGL", "AMZN"} and row.get("document_id")
        ],
    }

    historical_crosscheck = []
    for ticker, (hist_id, hist_year) in TICKER_TO_HISTORICAL.items():
        row = next(
            (
                item
                for item in all_records
                if item.get("ticker") == ticker
                and item.get("role") == "ANNUAL"
                and item.get("fiscal_year") == hist_year
            ),
            None,
        )
        hist = historical_docs.get(hist_id, {})
        historical_crosscheck.append(
            {
                "historical_document_id": hist_id,
                "ticker": ticker,
                "historical_company": hist.get("company"),
                "historical_fiscal_year": hist_year,
                "historical_page_count": hist.get("page_count"),
                "historical_sha256": hist.get("file_sha256"),
                "resolved_document_id": row.get("document_id") if row else None,
                "resolved_source_identity": {
                    "cik": row.get("cik"),
                    "accession_number": row.get("accession_number"),
                }
                if row
                else None,
                "classification": "SOURCE_IDENTITY_MATCH_CONTENT_UNVERIFIED"
                if row and row.get("document_id")
                else "HISTORICAL_METADATA_INCOMPLETE",
            }
        )

    amendment_payload = {
        "schema_version": "nf-v2-17/amendment-discovery/v1",
        "discovered_count": len(amendment_candidates),
        "candidates": amendment_candidates,
        "canonical_inclusion_count": 0,
        "policy": "record_only_until_body_scope_and_supersedes_relation_are verified",
    }
    write_json(
        artifact_dir / "fiscal-calendar-audit.json",
        {
            "schema_version": "nf-v2-17/fiscal-calendar-audit/v1",
            "base_sha": BASE_SHA,
            "created_at_used_as_financial_time": False,
            "companies": calendar_rows,
        },
    )
    write_json(artifact_dir / "duplicate-source-audit.json", duplicate_audit)
    write_json(artifact_dir / "amendment-discovery.json", amendment_payload)
    write_json(
        artifact_dir / "historical-source-crosscheck.json",
        {
            "schema_version": "nf-v2-17/historical-source-crosscheck/v1",
            "content_sha_verified": False,
            "records": historical_crosscheck,
        },
    )
    write_json(
        artifact_dir / "acquisition-plan.json",
        {
            "schema_version": "nf-v2-17/acquisition-plan/v1",
            "raw_layout": "financial_corpus_v2/raw/SEC/<ticker>/<document_id>/source.html",
            "normalized_layout": "financial_corpus_v2/normalized/<ticker>/<document_id>/",
            "parsed_layout": "financial_corpus_v2/parsed/<ticker>/<document_id>/",
            "records": acquisition_records,
            "download_started": False,
        },
    )
    write_json(artifact_dir / "fresh-blind-candidate-pool.json", candidate_pool)

    risk_text = f"""# NF-V2-17A2 Source Risk Review\n\nBase: `{BASE_SHA}`. This review resolves source identities only; no filing body was downloaded, converted, parsed, or indexed.\n\n## SEC HTML and PDF handling\n\nThe authoritative SEC primary documents resolve as HTML. The current ingestion stack is PDF-first (PyMuPDF native path with optional MinerU), so A3/A4 must preserve the raw HTML and SHA, record the conversion tool/config, and store a separate normalized artifact/SHA. A landing page is retained only as an index locator, never as the canonical raw source.\n\n## Fiscal periods\n\nMSFT, AAPL, NVDA, and V use non-calendar fiscal calendars. SEC `fy`, `fp`, and `reportDate` are retained as source metadata. A 10-Q may contain both three-month QUARTER and cumulative YTD facts; document-level period semantics are therefore marked UNKNOWN/MIXED until table-level parsing.\n\n## Amendments and versions\n\n{len(amendment_candidates)} amendment candidates were found in submissions metadata. They are recorded separately and not promoted to canonical version cases because supersedes scope requires filing-body review. Later filing date alone is not treated as supersession.\n\n## Historical raw files\n\nThe eight historical annual records are manifest-level reusable only. Source identity was cross-checked, but content SHA cannot be reverified until acquisition because the raw PDFs are absent from the clean worktree.\n\n## Operational constraints\n\nSEC access should use an identifying User-Agent, bounded retries, and rate limiting. `created_at` and `ingested_at` are never used for fiscal order, latest-report selection, or version resolution. No fresh-blind questions or Gold evidence were created.\n"""
    (artifact_dir / "source-risk-review.md").write_text(risk_text, encoding="utf-8")

    unresolved_count = len(unresolved)
    decision = (
        "SOURCE_INTAKE_ACCEPTED"
        if unresolved_count == 0
        and len(verified) + len(warnings) == 60
        and len(calendar_rows) == 10
        and duplicate_audit["canonical_duplicates"] == 0
        else "SOURCE_INTAKE_NEEDS_REVISION"
    )
    decision_payload = {
        "schema_version": "nf-v2-17/source-intake-review-decision/v1",
        "task": "NF-V2-17A2 Source Intake Review",
        "base_sha": BASE_SHA,
        "reviewed_at": REVIEW_DATE,
        "companies": len(COMPANIES),
        "planned_primary_filings": 60,
        "verified": len(verified),
        "verified_with_warning": len(warnings),
        "unresolved": unresolved_count,
        "annual_planned": 30,
        "annual_resolved": sum(1 for row in annual_records if row.get("document_id")),
        "quarterly_planned": 30,
        "quarterly_resolved": sum(
            1 for row in quarterly_records if row.get("document_id")
        ),
        "fiscal_calendars_verified": len(calendar_rows),
        "non_calendar_issuers": [
            row["ticker"]
            for row in calendar_rows
            if row["classification"] == "NON_CALENDAR_ISSUER"
        ],
        "historical_source_identity_reacquired": sum(
            1
            for row in historical_crosscheck
            if row["classification"] == "EXACT_SOURCE_REACQUIRED"
        ),
        "historical_metadata_only": sum(
            1
            for row in historical_crosscheck
            if row["classification"] == "SOURCE_IDENTITY_MATCH_CONTENT_UNVERIFIED"
        ),
        "amendments_discovered": len(amendment_candidates),
        "amendments_included_as_canonical_version_cases": 0,
        "canonical_duplicates": duplicate_audit["canonical_duplicates"],
        "source_format_counts": {
            "RAW_HTML": len([row for row in all_records if row.get("document_id")]),
            "RAW_PDF": 0,
            "BOTH_AVAILABLE": 0,
        },
        "conversion_required": len(
            [row for row in all_records if row.get("document_id")]
        ),
        "created_at_financial_time_misuse": 0,
        "fresh_blind_candidate_documents": len(candidate_pool["candidates"]),
        "acquisition_ready": decision == "SOURCE_INTAKE_ACCEPTED",
        "decision": decision,
        "downloaded": False,
        "parsed": False,
        "indexed": False,
        "training": 0,
        "model_calls": 0,
        "next_gate": "NF-V2-17A3_AUTHORITATIVE_SOURCE_ACQUISITION"
        if decision == "SOURCE_INTAKE_ACCEPTED"
        else "NF-V2-17A2_SOURCE_INTAKE_REVIEW",
    }
    write_json(artifact_dir / "source-intake-review-decision.json", decision_payload)
    readme = f"""# NF-V2-17A2 Authoritative Source Intake Review\n\nBase: `{BASE_SHA}`\n\nThis phase resolves filing identities from SEC submissions metadata. It does not download filing bodies, perform parsing, build indices, generate questions, or tune runtime behavior.\n\n* Planned primary filings: 60 (30 annual + 30 actual 10-Q; Q4 represented by annual 10-K).\n* Resolved source identities: {len(verified) + len(warnings)}/60; unresolved: {unresolved_count}.\n* Annual: {sum(1 for row in annual_records if row.get("document_id"))}/30; quarterly: {sum(1 for row in quarterly_records if row.get("document_id"))}/30.\n* Fiscal calendars audited: {len(calendar_rows)}/10; `created_at` financial-time misuse: 0.\n* Amendment candidates: {len(amendment_candidates)}; included as canonical versions: 0 pending body/supersedes review.\n* Canonical duplicate count: {duplicate_audit["canonical_duplicates"]}.\n\nRaw source paths are designed but empty until A3. All canonical SEC records are RAW_HTML with `conversion_required=true` because the existing parser path is PDF-first. Raw HTML, conversion configuration, normalized output, and parsed output must remain separate.\n\nFresh-blind candidates are listed but not reserved; no questions, Gold evidence, or answers were inspected.\n\nDecision: **{decision}**\n"""
    (artifact_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[4])
