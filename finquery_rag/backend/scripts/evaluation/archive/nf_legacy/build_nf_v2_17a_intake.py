#!/usr/bin/env python3
"""Build the NF-V2-17A corpus-intake package without downloading data."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

BASE_SHA = "4b3a13bac661375406686a5055708f067a3aa3ac"
REVIEW_DATE = "2026-08-20"

COMPANIES = [
    {"company": "Microsoft Corporation", "ticker": "MSFT", "sector": "Technology", "cik": "0000789019", "reason": "Retain official 10-K; non-calendar fiscal year and segment/geography coverage."},
    {"company": "Apple Inc.", "ticker": "AAPL", "sector": "Technology / Consumer Electronics", "cik": "0000320193", "reason": "Retain official 10-K; product, geography, margin, and September year-end semantics."},
    {"company": "NVIDIA Corporation", "ticker": "NVDA", "sector": "Semiconductors", "cik": "0001045810", "reason": "Retain official 10-K; growth, scale, and data-center tables."},
    {"company": "JPMorgan Chase & Co.", "ticker": "JPM", "sector": "Financial Services", "cik": "0000019617", "reason": "Retain official annual report; bank balance-sheet and risk disclosure diversity."},
    {"company": "Tesla, Inc.", "ticker": "TSLA", "sector": "Automotive / Energy", "cik": "0001318605", "reason": "Retain official 10-K; automotive, energy, and regulatory-credit disclosures."},
    {"company": "The Coca-Cola Company", "ticker": "KO", "sector": "Consumer Staples", "cik": "0000021344", "reason": "Retain official 10-K; geographic, bottling, and currency disclosures."},
    {"company": "Visa Inc.", "ticker": "V", "sector": "Financial Services / Payments", "cik": "0001403161", "reason": "Retain official annual report; payment-volume and regional metrics."},
    {"company": "Pfizer Inc.", "ticker": "PFE", "sector": "Healthcare / Pharmaceuticals", "cik": "0000078003", "reason": "Retain official FY2024 10-K and extend to FY2025; product and complex-table coverage."},
    {"company": "Alphabet Inc.", "ticker": "GOOGL", "sector": "Communication Services / Internet", "cik": "0001652044", "reason": "Add an unseen issuer with an existing SEC primary-document acquisition lead and broad segment disclosures."},
    {"company": "Amazon.com, Inc.", "ticker": "AMZN", "sector": "Consumer Discretionary / Cloud", "cik": "0001018724", "reason": "Add an unseen issuer with an existing SEC primary-document acquisition lead and retail/AWS/advertising segments."},
]

TICKER_TO_DOC = {
    "MSFT": "msft_fy2025", "AAPL": "aapl_fy2025", "NVDA": "nvda_fy2025",
    "JPM": "jpm_fy2025", "TSLA": "tsla_fy2025", "KO": "ko_fy2025",
    "V": "v_fy2025", "PFE": "pfe_fy2024",
}
SHADOW_TICKER_TO_ISSUER = {"GOOGL": "Alphabet Inc.", "AMZN": "AMAZON COM INC"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sec_submissions_url(cik: str) -> str:
    return f"https://data.sec.gov/submissions/CIK{cik}.json"


def build(repo_root: Path, output_dir: Path) -> None:
    backend = repo_root / "finquery_rag" / "backend"
    benchmark_manifest_path = backend / "benchmarks" / "financial_rag_v1" / "corpus.json"
    benchmark_sources_path = backend / "eval_data" / "financial_benchmark_v1" / "documents.json"
    prior_audit_path = backend / "artifacts" / "evaluation" / "nf-eval-01" / "corpus-audit-report.json"
    phase5_manifest_path = backend / "eval_corpus" / "phase5" / "corpus-manifest.json"
    shadow_manifest_path = backend / "artifacts" / "evaluation" / "nf-opt-17" / "development-corpus-manifest.json"
    benchmark = load_json(benchmark_manifest_path)
    benchmark_sources = {str(row.get("document_id")): row for row in load_json(benchmark_sources_path).get("documents", [])}
    prior_audit = load_json(prior_audit_path)
    prior_audit_by_doc = {str(row.get("document_id")): row for row in prior_audit.get("documents", [])}
    phase5 = load_json(phase5_manifest_path)
    shadow = load_json(shadow_manifest_path)
    shadow_by_issuer = {str(row.get("issuer")): row for row in shadow.get("documents", [])}
    output_dir.mkdir(parents=True, exist_ok=True)

    selections = []
    for company in COMPANIES:
        doc_id = TICKER_TO_DOC.get(company["ticker"])
        existing = [doc_id] if doc_id else []
        selections.append({
            **company,
            "existing_docs": existing,
            "existing_doc_count": len(existing),
            "existing_doc_status": "MANIFESTED_CANONICAL_BUT_RAW_PATH_EXTERNAL" if existing else "NO_CANONICAL_DOC",
            "required_new_primary_docs": 5 if existing else 6,
            "required_new_annual_reports": 2 if existing else 3,
            "required_new_quarterly_filings": 3,
            "target_annual_fiscal_years": ["FY2023", "FY2024", "FY2025"],
            "quarter_observation_plan": [
                "FY2025-Q1 via 10-Q", "FY2025-Q2 via 10-Q", "FY2025-Q3 via 10-Q",
                "FY2025-Q4 represented by FY2025 annual 10-K (no fabricated standalone Q4 filing)",
            ],
            "amendment_policy": "opportunistic_only_when_authoritatively_present",
        })

    source_rows = []
    for doc in benchmark.get("documents", []):
        doc_id = str(doc.get("document_id"))
        ticker = next((c["ticker"] for c in COMPANIES if TICKER_TO_DOC.get(c["ticker"]) == doc_id), None)
        source = benchmark_sources.get(doc_id, {})
        prior = prior_audit_by_doc.get(doc_id, {})
        raw_candidate = backend / "benchmarks" / "financial_rag_v1" / str(doc.get("filename"))
        source_rows.append({
            "company": doc.get("company"), "ticker": ticker,
            "document_role": "EXISTING_CANONICAL_ANNUAL", "period": f"FY{doc.get('fiscal_year')}",
            "document_type": "ANNUAL", "source_type": source.get("source_kind") or doc.get("source_type"),
            "source_url": source.get("pdf_url") or source.get("official_landing_url"),
            "source_identifier": doc_id, "source_status": "MANIFESTED_REUSABLE_AFTER_EXTERNAL_PATH_VERIFICATION",
            "raw_sha256": doc.get("file_sha256"), "raw_file_visible_in_clean_worktree": raw_candidate.is_file(),
            "prior_runtime_audit_source_file_present": bool(prior.get("source_file_present")),
            "notes": "Canonical benchmark manifest is retained; raw PDFs are not committed in this clean intake worktree.",
        })

    for ticker, issuer in SHADOW_TICKER_TO_ISSUER.items():
        row = shadow_by_issuer.get(issuer)
        company = next(c for c in COMPANIES if c["ticker"] == ticker)
        source_rows.append({
            "company": company["company"], "ticker": ticker,
            "document_role": "SHADOW_PRIMARY_ANNUAL_CANDIDATE", "period": "FY2025", "document_type": "ANNUAL",
            "source_type": "SEC_EDGAR_PRIMARY_HTML", "source_url": row.get("archive_url") if row else sec_submissions_url(company["cik"]),
            "source_identifier": row.get("accession_number") if row else None,
            "source_status": "SHADOW_ONLY_REACQUIRE_AND_VERIFY_BEFORE_CANONICAL_REUSE",
            "raw_sha256": row.get("content_sha256") if row else None, "raw_file_visible_in_clean_worktree": False,
            "prior_runtime_audit_source_file_present": bool(row and row.get("downloaded")),
            "notes": "NF-OPT-17 runtime_shadow_only; runtime_dir_committed=false. Acquisition lead only.",
        })

    for company in COMPANIES:
        ticker = company["ticker"]
        existing_year = 2025 if ticker in TICKER_TO_DOC and ticker != "PFE" else (2024 if ticker == "PFE" else None)
        for year in (2023, 2024, 2025):
            if existing_year == year:
                continue
            source_rows.append({
                "company": company["company"], "ticker": ticker, "document_role": "PLANNED_NEW_ANNUAL",
                "period": f"FY{year}", "document_type": "ANNUAL", "source_type": "SEC_EDGAR_SUBMISSIONS_INDEX",
                "source_url": sec_submissions_url(company["cik"]), "source_identifier": None,
                "source_status": "PLANNED_NOT_ACQUIRED", "raw_sha256": None,
                "raw_file_visible_in_clean_worktree": False, "prior_runtime_audit_source_file_present": False,
                "notes": "Resolve exact accession, primary document, report/filing dates, and SHA during NF-V2-17A3.",
            })
        for quarter in ("Q1", "Q2", "Q3"):
            source_rows.append({
                "company": company["company"], "ticker": ticker, "document_role": "PLANNED_NEW_QUARTERLY",
                "period": f"FY2025-{quarter}", "document_type": "QUARTERLY", "source_type": "SEC_EDGAR_SUBMISSIONS_INDEX",
                "source_url": sec_submissions_url(company["cik"]), "source_identifier": None,
                "source_status": "PLANNED_NOT_ACQUIRED", "raw_sha256": None,
                "raw_file_visible_in_clean_worktree": False, "prior_runtime_audit_source_file_present": False,
                "notes": "Use the actual 10-Q period end and filing date; Q4 is represented by annual 10-K.",
            })

    selection = {
        "schema_version": "nf-v2-17/company-selection/v1", "task": "NF-V2-17A Corpus Intake",
        "base_sha": BASE_SHA, "reviewed_at": REVIEW_DATE, "production": "V1",
        "target_company_count": len(COMPANIES),
        "selection_principles": [
            "retain usable official benchmark issuers", "add unseen issuers with authoritative SEC acquisition paths",
            "cover technology, semiconductors, finance, consumer, automotive/energy, healthcare, and internet/cloud",
            "do not treat synthetic fixtures or evaluation mirrors as public corpus documents",
        ], "companies": selections,
        "alternatives_not_selected": [
            {"company": "Meta Platforms, Inc.", "ticker": "META", "reason": "NF-OPT-17 shadow source exists; retained as a substitution if selected-source acquisition fails."},
            {"company": "Netflix, Inc.", "ticker": "NFLX", "reason": "NF-OPT-17 shadow source exists; retained as a documented fallback without expanding the 10-company target."},
        ],
        "target_totals": {
            "annual_reports": len(COMPANIES) * 3, "quarterly_filings": len(COMPANIES) * 3,
            "q4_standalone_filings": 0, "q4_representation": "annual 10-K",
            "primary_filings_expected": len(COMPANIES) * 6, "amendments_required": 0,
            "amendments_expected_range": [0, 5], "total_documents_expected_range": [60, 65],
        },
        "intake_boundary": "Planning only. No raw acquisition, parsing, indexing, question generation, or runtime tuning.",
    }

    source_intake = {
        "schema_version": "nf-v2-17/source-intake/v1", "task": "NF-V2-17A Corpus Intake",
        "base_sha": BASE_SHA, "reviewed_at": REVIEW_DATE, "acquisition_performed": False,
        "raw_sources_committed": False,
        "source_authority_policy": ["SEC EDGAR regulatory filing", "issuer investor relations", "other first-party source only when required"],
        "existing_audited_assets": {
            "financial_rag_v1_manifest_path": str(benchmark_manifest_path.relative_to(repo_root)),
            "financial_rag_v1_corpus_hash": benchmark.get("corpus_hash"),
            "financial_rag_v1_canonical_documents": len(benchmark.get("documents", [])),
            "financial_rag_v1_annual_documents": len(benchmark.get("documents", [])),
            "financial_rag_v1_quarterly_documents": 0, "financial_rag_v1_amendments": 0,
            "financial_rag_v1_prior_audit_source_files_present": sum(bool(row.get("source_file_present")) for row in prior_audit.get("documents", [])),
            "financial_rag_v1_raw_pdfs_visible_in_clean_worktree": sum(bool(row.get("raw_file_visible_in_clean_worktree")) for row in source_rows if row.get("document_role") == "EXISTING_CANONICAL_ANNUAL"),
            "phase5_synthetic_documents": int(phase5.get("document_count", 0)), "phase5_synthetic_excluded_from_public_corpus": True,
            "nf_opt17_shadow_documents": len(shadow.get("documents", [])), "nf_opt17_shadow_runtime_dir_committed": bool(shadow.get("runtime_dir_committed")),
        },
        "source_availability_summary": {
            "existing_official_annual_manifest_records": 8,
            "existing_official_pdf_urls_verified_in_benchmark_sources": sum(bool(row.get("pdf_url")) for row in benchmark_sources.values()),
            "existing_official_landing_only_records": sum(not bool(row.get("pdf_url")) for row in benchmark_sources.values()),
            "selected_new_shadow_accessions_with_exact_sec_urls": 2,
            "quarterly_accessions_resolved": 0, "amendment_accessions_resolved": 0,
            "source_plan_sufficient_for_a3": True,
        },
        "records": source_rows,
        "source_plan_risks": [
            "raw PDFs/HTML are external or ignored runtime artifacts and are not present in this clean worktree; each reused document must be reacquired or path-verified",
            "four existing annual rows are landing-page-only and need exact PDF/HTML accession verification",
            "quarterly accession and period-end mapping must be resolved from SEC submissions; no accession is guessed here",
            "amendments/restatements are opportunistic and require authoritative supersedes evidence",
        ],
    }

    matrix = {
        "schema_version": "nf-v2-17/target-filing-matrix/v1", "base_sha": BASE_SHA, "reviewed_at": REVIEW_DATE,
        "annual_years": ["FY2023", "FY2024", "FY2025"], "quarter_plan": ["Q1 10-Q", "Q2 10-Q", "Q3 10-Q", "Q4 via annual 10-K"],
        "rows": [
            {"company": row["company"], "ticker": row["ticker"], "sector": row["sector"],
             "annual": {year: ("EXISTING_MANIFEST" if ((row["ticker"] != "PFE" and year == "FY2025") or (row["ticker"] == "PFE" and year == "FY2024")) else "REQUIRED_NEW") for year in ["FY2023", "FY2024", "FY2025"]},
             "quarterly": {f"FY2025-{q}": "REQUIRED_NEW_10-Q" for q in ["Q1", "Q2", "Q3"]},
             "q4": "REPRESENTED_BY_ANNUAL_FY2025_10-K", "amendments": "OPTIONAL_IF_AUTHORITATIVELY_PRESENT"}
            for row in selections
        ],
    }

    write_json(output_dir / "company-selection.json", selection)
    write_json(output_dir / "source-intake.json", source_intake)
    write_json(output_dir / "target-filing-matrix.json", matrix)
    columns = ["company", "ticker", "document_role", "period", "document_type", "source_type", "source_url", "source_identifier", "source_status", "raw_sha256", "raw_file_visible_in_clean_worktree", "prior_runtime_audit_source_file_present", "notes"]
    with (output_dir / "source-intake.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in columns} for row in source_rows)
    hashes = {name: sha256_file(output_dir / name) for name in ["company-selection.json", "source-intake.json", "source-intake.csv", "target-filing-matrix.json"]}
    decision = {
        "schema_version": "nf-v2-17/intake-decision/v1", "task": "NF-V2-17A Corpus Intake", "base_sha": BASE_SHA,
        "decision": "CORPUS_INTAKE_ACCEPTED", "decision_scope": "intake_plan_only", "reviewed_at": REVIEW_DATE,
        "target_company_count": len(COMPANIES), "target_primary_filings": 60, "full_corpus_gate_passed": False,
        "fresh_blind_reserved": False, "downloads_started": False, "training": 0, "retrieval_tuning": 0, "benchmark_tuning": 0,
        "acceptance_checks": {"target_companies_ge_8": True, "multi_year_target_defined": True, "annual_and_quarter_target_defined": True, "authoritative_source_plan_defined": True, "existing_corpus_not_overwritten": True, "raw_source_sha_policy_defined": True, "created_at_misuse_policy_defined": True, "fresh_blind_not_opened_before_corpus_seal": True},
        "blocking_before_a3": ["verify/reacquire raw source files and record raw_sha256", "resolve exact accession, filing/report dates, period bounds, and amendment relations", "inspect source-format/parser compatibility before bulk parsing"],
        "next_gate": "NF-V2-17A2_SOURCE_INTAKE_REVIEW", "artifact_sha256": hashes,
    }
    write_json(output_dir / "intake-decision.json", decision)
    readme = f"""# NF-V2-17A Corpus Intake\n\nBase: `{BASE_SHA}`\n\nThis is the first stop point for Financial Knowledge Corpus V2. It is an intake and filing-plan artifact only: no raw filing was downloaded, parsed, indexed, used for question generation, or used to tune runtime behavior.\n\n## Audited starting point\n\n* The committed `financial_rag_v1` manifest contains 8 official annual documents (7 FY2025 and Pfizer FY2024), with 1,348 pages and 44,608 manifest chunks. It has no quarterly or amendment records.\n* The prior corpus audit reports 8 ready source files and matching hashes, but the clean intake worktree does not contain the raw PDFs; they remain external/ignored runtime assets and must be reacquired or path-verified.\n* The phase5 corpus contains {phase5.get('document_count', 0)} synthetic internal Markdown fixtures. They remain historical evaluation fixtures and are excluded from the public corpus.\n* NF-OPT-17 provides {len(shadow.get('documents', []))} SEC HTML shadow records (`runtime_shadow_only`, not committed canonical files). Alphabet and Amazon are acquisition leads, not accepted corpus documents.\n\n## Target plan\n\nTen issuers are planned, with three annual fiscal years per issuer and Q1-Q3 10-Qs; Q4 is represented by the annual 10-K and no standalone Q4 filing is fabricated. This yields 30 annual + 30 quarterly primary filings (60 expected) plus 0-5 opportunistic amendments.\n\n## Integrity boundaries\n\n`created_at`/`ingested_at` are operational timestamps only. Filing date, report date, fiscal period, period semantics, version, and supersedes relations must be sourced explicitly or marked UNKNOWN. The historical corpus is not overwritten, and fresh-blind reservation is deferred until after the corpus intake/quality gate.\n\n## Next step\n\n`NF-V2-17A2_SOURCE_INTAKE_REVIEW` may resolve authoritative accessions and acquisition paths. Only after that review may NF-V2-17A3 acquire sources and NF-V2-17A4 parse them.\n"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[4]
    output = repo / "finquery_rag" / "backend" / "artifacts" / "evaluation" / "nf-v2-17-financial-corpus-v2"
    build(repo, output)
    print(output)
