# NF-V2-17A Corpus Intake

Base: `4b3a13bac661375406686a5055708f067a3aa3ac`

This is the first stop point for Financial Knowledge Corpus V2. It is an intake and filing-plan artifact only: no raw filing was downloaded, parsed, indexed, used for question generation, or used to tune runtime behavior.

## Audited starting point

* The committed `financial_rag_v1` manifest contains 8 official annual documents (7 FY2025 and Pfizer FY2024), with 1,348 pages and 44,608 manifest chunks. It has no quarterly or amendment records.
* The prior corpus audit reports 8 ready source files and matching hashes, but the clean intake worktree does not contain the raw PDFs; they remain external/ignored runtime assets and must be reacquired or path-verified.
* The phase5 corpus contains 18 synthetic internal Markdown fixtures. They remain historical evaluation fixtures and are excluded from the public corpus.
* NF-OPT-17 provides 4 SEC HTML shadow records (`runtime_shadow_only`, not committed canonical files). Alphabet and Amazon are acquisition leads, not accepted corpus documents.

## Target plan

Ten issuers are planned, with three annual fiscal years per issuer and Q1-Q3 10-Qs; Q4 is represented by the annual 10-K and no standalone Q4 filing is fabricated. This yields 30 annual + 30 quarterly primary filings (60 expected) plus 0-5 opportunistic amendments.

## Integrity boundaries

`created_at`/`ingested_at` are operational timestamps only. Filing date, report date, fiscal period, period semantics, version, and supersedes relations must be sourced explicitly or marked UNKNOWN. The historical corpus is not overwritten, and fresh-blind reservation is deferred until after the corpus intake/quality gate.

## Next step

`NF-V2-17A2_SOURCE_INTAKE_REVIEW` may resolve authoritative accessions and acquisition paths. Only after that review may NF-V2-17A3 acquire sources and NF-V2-17A4 parse them.
