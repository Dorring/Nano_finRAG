# NF-V2-16 R1 real retrieval audit

## Current paths at the frozen base

* **Dense/vector:** `finquery_rag/backend/src/services/vector_store.py`
  uses the persistent Chroma collection `rag_global_knowledge_base`, cosine
  distance, the configured embedding provider, and `where` constraints for
  `user_id` plus optional `doc_name`.  Table-cell rows are not dense indexed.
* **BM25/keyword:** `finquery_rag/backend/src/services/retrieval.py`
  (`SqliteBM25Retriever`) uses SQLite FTS5 `fts_index` over the single indexed
  `content` field (unicode61 plus jieba preprocessing).  `chunk_store` keeps
  JSON metadata, `user_id`, and `doc_name`; search requires `user_id` and can
  exact-filter `doc_name`.  Table-cell facts are secondary BM25 evidence.
* **Hybrid:** `finquery_rag/backend/src/retrieval/retrieval_pipeline.py`
  retrieves dense and BM25 candidates, combines them with reciprocal-rank
  fusion, then optionally reranks.  The legacy facade exposes `use_hybrid`.
* **Structured table evidence:** BM25 `get_table_cell_evidence` attaches
  aligned cells to selected table rows; it is not an independent search
  backend.
* **Reranking:** `src/services/reranker.py` provides disabled/no-op,
  heuristic, and optional cross-encoder/BGE implementations.  R1 now checks
  the reranker output is a subset of the hard-filtered candidate universe.
* **No external/web search or Weaviate path** is present in this repository.

## Existing metadata and security boundary

The dense and BM25 indexes expose `doc_id`, `doc_name`, `user_id`, page/type
and parser/table hierarchy metadata.  `document_registry` additionally has
`document_id`, `tenant_id`, filename/hash/counts, integer upload-derived
`version`, status, parser/splitter/embedding versions and operational
`created_at`/`updated_at`.  It does not contain entity/ticker, report/filing
date, fiscal year/quarter, period semantics, amendment or supersedes fields.

Both legacy paths enforce user/tenant scope before returning candidates.  They
did not share a typed fiscal/entity/version scope, and the prior reranker path
did not record a post-rerank subset assertion.  R1 adds a canonical
`FinancialDocumentMetadataV1`/`RetrievalScopeV1` adapter that is applied to
dense, BM25, hybrid union and reranker outputs.  Missing metadata for an
explicit hard condition is rejected; there is no silent relaxation.

## created_at audit

`document_registry.created_at` is assigned by `time.time()` during registry
registration and is used for lifecycle/listing, duplicate/session/feedback
and operational diagnostics.  No runtime financial query path uses it as
report date, filing date, effective financial date, latest annual winner or
version dominance.  R1's latest resolver uses only explicit
`filing_date`/`report_date`; missing dates fail closed.

## Scope and failure semantics

Authorization is immutable and precedes every other condition.  Explicit
entity, fiscal period, document type, period semantics and version relations
are HARD.  Section/content labels are SOFT unless explicitly requested.
Replans preserve the original hard conditions.  Empty hard-filter results
remain empty and are surfaced as a missing slot; the controller can target a
missing slot, but cannot drop the constraint.  Unresolved conflicts,
no-progress, tool errors and budget exhaustion terminate fail-closed.

The R1 integration corpus below is marked `TEST_FIXTURE`; it exercises the
real repository SQLite FTS5 retriever and the same adapter used by the dense
and hybrid branches.  It is not a recall claim about the frozen 72-question
benchmark or production financial corpus.
