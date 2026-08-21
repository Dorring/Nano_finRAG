# NF-V2-17A2 Source Risk Review

Base: `b7de593051bdb61e4a08539cef762f7fa373a9fd`. This review resolves source identities only; no filing body was downloaded, converted, parsed, or indexed.

## SEC HTML and PDF handling

The authoritative SEC primary documents resolve as HTML. The current ingestion stack is PDF-first (PyMuPDF native path with optional MinerU), so A3/A4 must preserve the raw HTML and SHA, record the conversion tool/config, and store a separate normalized artifact/SHA. A landing page is retained only as an index locator, never as the canonical raw source.

## Fiscal periods

MSFT, AAPL, NVDA, and V use non-calendar fiscal calendars. SEC `fy`, `fp`, and `reportDate` are retained as source metadata. A 10-Q may contain both three-month QUARTER and cumulative YTD facts; document-level period semantics are therefore marked UNKNOWN/MIXED until table-level parsing.

## Amendments and versions

3 amendment candidates were found in submissions metadata. They are recorded separately and not promoted to canonical version cases because supersedes scope requires filing-body review. Later filing date alone is not treated as supersession.

## Historical raw files

The eight historical annual records are manifest-level reusable only. Source identity was cross-checked, but content SHA cannot be reverified until acquisition because the raw PDFs are absent from the clean worktree.

## Operational constraints

SEC access should use an identifying User-Agent, bounded retries, and rate limiting. `created_at` and `ingested_at` are never used for fiscal order, latest-report selection, or version resolution. No fresh-blind questions or Gold evidence were created.
