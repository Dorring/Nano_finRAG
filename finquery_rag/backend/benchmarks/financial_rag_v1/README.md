# Financial RAG Benchmark v1

This directory is the expanded benchmark foundation for eight newly ingested
annual-report documents.  It is intentionally separate from
`benchmarks/financial_rag_legacy_v0`, which preserves the original three
documents and 27 diagnostic questions.

## Scope contract

`corpus.json` is the only source of truth for the v1 document whitelist.
Every Dense/Chroma, BM25, RRF, reranker, final-context, citation, and
evaluation adapter must retain candidates only when
`candidate.document_id` belongs to the eight IDs in that file.  The global
tenant index may legally contain the three legacy documents; their presence
is not benchmark contamination.  A scope-integrity report must still be
zero for every retrieval/ranking/output stage before a v1 baseline is run.

The 72 records in `data/*.draft.jsonl` are draft annotations, not Golden or
Sealed labels.  They require human review of the question, answer, source,
and calculation fields before entering a formal baseline.

Phase 1.1 applies semantic quality gates before that review: duplicate intent,
ambiguous metrics, undefined multi-source output contracts, question/label
period mismatches, generic section/table placeholders, and repeated
no-answer templates must all be zero.  A clean quality audit still does not
promote a record to Golden; `review-status.jsonl` remains the authority for
human approval.

This directory contains the Phase 1 benchmark foundation for eight financial
reports.  The 72 records in `data/` are **Draft only**: they are generated for
review and must not be used as Golden, Sealed, or Baseline evaluation data.

The original 27-question set remains a separate legacy development/diagnostic
set.  It is not mixed into this benchmark's final metrics.

## Corpus

`corpus.json` contains only stable, non-sensitive document identity metadata.
PDFs, indexes, database files, absolute server paths, and chunk text stay in
the ignored server runtime directory.

## Draft workflow

1. Review the question for unambiguous company, metric, period, and answer type.
2. Locate the exact PDF page (1-based physical PDF page) and evidence row.
3. Fill the normalized answer, unit, scale, period, and calculation operands.
4. Set every required review flag to true only after independent verification.
5. Promote to Golden only after `ready_for_golden=true` and a validator pass.

Run the validator from `finquery_rag/backend`:

```bash
python -m scripts.evaluation.validate_financial_rag_benchmark
```

No production RAG code is changed by this benchmark directory.
