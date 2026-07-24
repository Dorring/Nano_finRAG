# FinQuery RAG configuration

Runtime configuration is read from environment variables.

## Optional MinerU PDF parsing

The default parser is `native`: PyMuPDF text/layout reconstruction, Camelot
tables, and a dependency-free PyMuPDF table fallback. It has no extra model or
GPU residency and remains the recommended default for born-digital financial
reports.

MinerU is an opt-in parser backend for scanned PDFs, complicated multi-column
layouts, and tables that the native parser cannot preserve. FinQuery consumes
MinerU's stable flat `*_content_list.json` artifact and maps it into the same
tenant-scoped chunk, page, section, and citation contract as native ingest.

```bash
# Default: no MinerU process, model download, or GPU allocation.
PARSER_BACKEND=native

# Explicit fallback for a known difficult document. MinerU must already be
# installed or reachable as a separately managed service.
PARSER_BACKEND=mineru
MINERU_COMMAND=mineru
MINERU_API_URL=http://127.0.0.1:8001
MINERU_BACKEND=pipeline
MINERU_TIMEOUT_SECONDS=600
```

`PARSER_BACKEND=auto` is deliberately conservative. It remains native unless
`MINERU_AUTO_ENABLED=true` and `MINERU_API_URL` are both set; then documents
whose first few pages have unusually little extractable text are routed to
MinerU. This protects normal upload latency and avoids silently consuming GPU
resources. An explicit `mineru` failure is surfaced as an upload failure rather
than silently mixing parser outputs; retry with `native` or repair the MinerU
service.

After changing parser backend, delete and re-upload the affected document so
all dense/BM25 rows receive consistent parser lineage and page metadata. Run
the existing `run-http`, `retrieval-eval-bundle`, and `failure-analysis` flow
on the same cases before making MinerU the default for any document class.

Authenticated users can inspect non-content parser/index diagnostics after an
upload. The endpoint reports parser/splitter/embedding lineage, text/table
chunk counts, page coverage, and warnings such as `page_1_concentration` or
`missing_sparse_index`:

```bash
curl -H "Authorization: Bearer <token>" \
  http://127.0.0.1:8000/document-registry/quality
```

## Reranking

The FastAPI backend enables the dependency-free heuristic reranker by default.
Embedding and reranker models are configurable, but CI/preflight checks do not
download or load model weights.

```bash
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
RAG_RERANKER=heuristic
RAG_CANDIDATE_MULTIPLIER=2
```

Recommended embedding overrides:

- `BAAI/bge-small-en-v1.5` for English PDF demos.
- `intfloat/multilingual-e5-small` for mixed English/simple Chinese demos.
- `/local/path/to/embedding-model` for offline production-like runs.

Available rerankers:

- unset — uses the FastAPI default `heuristic` reranker.
- `none` / `off` — disables reranking for baseline comparisons.
- `heuristic` — dependency-free lexical reranker; recommended default for local English and simple Chinese demos.
- `cross-encoder` — optional model-backed reranker.

Cross-encoder reranking must also set a model name or local path:

```bash
RAG_RERANKER=cross-encoder
RAG_RERANKER_MODEL=/path/to/local/cross-encoder-model
RAG_CANDIDATE_MULTIPLIER=4
```

Use a local model path for production-like runs. If `RAG_RERANKER_MODEL` is
missing, initialization fails explicitly instead of silently downloading a model
or changing retrieval behavior.

Check retrieval model configuration without loading model weights:

```bash
python -m src.eval_cli doctor --warn-only --out /tmp/finquery_doctor.json
```

For any reranker change, run an eval report before merging:

```bash
python -m src.eval_cli run --cases <cases.jsonl> --out <candidate.jsonl> --user-id <id>
python -m src.eval_cli score --cases <cases.jsonl> --predictions <candidate.jsonl> --out <candidate_report.json>
python -m src.eval_cli compare --baseline <baseline_report.json> --candidate <candidate_report.json>
```

For a compact artifact bundle suitable for baseline/candidate comparison:

```bash
python -m src.eval_cli retrieval-eval-bundle \
  --cases eval/golden_smoke.jsonl \
  --predictions eval/predictions_smoke.jsonl \
  --k 1 --k 3 --k 5 \
  --out-dir /tmp/finquery_retrieval_eval
```

## Hierarchical indexing

FinQuery indexes PDF text as small child chunks for retrieval, while carrying a
bounded parent section/page excerpt in chunk metadata. At answer time,
`build_context` expands matching child chunks back to their shared parent excerpt
and merges sibling child hits from the same parent.

This gives the retriever precise child-level matching while giving the generator
enough local section context for titles, financial line items, and paragraph
continuations. Source records keep the original child `chunk_id` plus
`parent_id`, `section_path`, and `child_hit_count` for debugging.

This metadata is written during ingestion. After changing this behavior or
deploying it to an existing local store, re-upload documents or rebuild the
Chroma/BM25 indexes from freshly ingested chunks so older rows receive
`parent_id`, `section_path`, and `parent_excerpt`.

## Health and readiness probes

The backend exposes two unauthenticated operational probes:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

- `/healthz` is a lightweight liveness check for process availability.
- `/readyz` returns a non-secret RAG dependency snapshot and uses HTTP 503
  when required local stores are unavailable or runtime configuration is
  invalid.

`/readyz` is intentionally cheap: it does not call the LLM, does not request
embeddings, and does not read tenant document content.

Run the same non-secret checks from a shell before deployment or after moving
runtime data paths:

```bash
python -m src.eval_cli doctor   --bm25-db "$BM25_DB_PATH"   --out /tmp/finquery_doctor.json
```

`doctor` returns `0` when required stores are ready, `1` when readiness is
degraded, and still writes a JSON snapshot suitable for support/debugging. Use
`--warn-only` for non-blocking diagnostics in ad-hoc maintenance scripts.

## Intent routing

FinQuery includes a deterministic intent router before retrieval. It is
conservative by design:

- clear greetings, thanks, and product-help questions bypass retrieval;
- clear out-of-scope general questions bypass retrieval with a refusal;
- financial QA, document summaries, and financial calculations continue through
  the RAG pipeline;
- unknown domain-specific wording defaults to retrieval to avoid suppressing
  valid document questions.

The router exposes `intent` and `intent_confidence` in `/query` responses and
the final `/query/stream` event. It does not call an LLM and has no external
dependencies.

## Runtime storage paths

Use explicit paths in production so the app, health checks, eval tooling, and
background scripts read and write the same local stores:

```bash
CHROMA_PATH=/var/lib/finquery/chroma_db
DOCUMENT_REGISTRY_DB_PATH=/var/lib/finquery/document_registry.db
BM25_DB_PATH=/var/lib/finquery/rag_bm25.db
SESSIONS_DB_PATH=/var/lib/finquery/sessions.db
MEMORY_PROFILE_DB_PATH=/var/lib/finquery/memory_profile.db
TRACE_DB_PATH=/var/lib/finquery/trace_log.db
FEEDBACK_DB_PATH=/var/lib/finquery/feedback.db
```

When unset, FinQuery keeps the existing development defaults under the backend
working directory.

## Memory profile

FinQuery separates short-term session history from editable user preference
memory. Session history is used to resolve follow-up questions. The memory
profile stores bounded preferences such as:

- `preferred_language`
- `preferred_currency`
- `preferred_unit`
- `default_period`
- `default_company`
- `watchlist`
- `focus_metrics`

The profile is used only for query planning/rewrite ambiguity, such as resolving
"what about margin?" to the user's default period, company, or unit. It is never
used as a factual source for answers; answer grounding still comes from retrieved
document context and citations.

Authenticated API:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/memory/profile

curl -X PUT -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"preferred_currency":"USD","default_period":"FY2024","focus_metrics":["revenue","gross margin"]}' \
  http://127.0.0.1:8000/memory/profile

curl -X DELETE -H "Authorization: Bearer <token>" http://127.0.0.1:8000/memory/profile
```

## Deployment preflight

Before a server cutover, run the aggregate preflight command. It combines
readiness checks, migration audit, fixture audit, eval gate, baseline comparison,
and retrieval diagnostics into one JSON report without calling LLMs or embedding
models:

```bash
python -m src.eval_cli preflight   --cases eval/golden_smoke.jsonl   --predictions eval/predictions_smoke.jsonl   --baseline eval/baseline_smoke_report.json   --bm25-db "$BM25_DB_PATH"   --registry-db "$DOCUMENT_REGISTRY_DB_PATH"   --chroma-path "$CHROMA_PATH"   --trace-db "$TRACE_DB_PATH"   --feedback-db "$FEEDBACK_DB_PATH"   --out /tmp/finquery_preflight.json
```

Use `--warn-only` for exploratory runs on partially initialized environments.
A normal non-zero exit means at least one section failed and should be reviewed
before serving production traffic.

## Migration readiness audit

Before deploying code that relies on tenant-scoped chunk IDs against existing
local data, run a non-content migration audit. It inspects identifiers, table
shape, and aggregate counts only; it does not read chunk text or document
content:

```bash
python -m src.eval_cli migration-audit   --bm25-db "$BM25_DB_PATH"   --registry-db "$DOCUMENT_REGISTRY_DB_PATH"   --chroma-path "$CHROMA_PATH"   --out /tmp/finquery_migration_audit.json
```

The command returns `1` when high-risk legacy patterns are found, such as
unscoped BM25 `doc_id` values, missing `user_id`, mismatched user prefixes, or
unscoped Chroma embedding IDs. Use `--warn-only` for exploratory audits that
should not fail a maintenance script. High-risk findings usually mean the BM25
and/or Chroma indexes should be rebuilt from tenant-scoped chunks before serving
production queries.

## BM25 maintenance

The sparse index stores canonical chunk rows in `chunk_store` and query rows in
the FTS5 table `fts_index`. Use the maintenance commands after manual database
changes, interrupted writes, or before production cutovers:

```bash
python -m src.eval_cli bm25-check --db "$BM25_DB_PATH"
python -m src.eval_cli bm25-rebuild --db "$BM25_DB_PATH"
```

Both commands accept `--user-id` for tenant-scoped checks/rebuilds. A global
rebuild also removes orphan FTS rows that can no longer be attributed to a
tenant.

## Trace IDs

Non-streaming `/query` responses and `/query/stream` final `done` events include
`trace_id` when structured tracing is successfully written. Use this ID with
trace export/query tooling to locate the exact request path during debugging or
replay preparation. If trace persistence fails, the answer path still succeeds
and `trace_id` is `null`.

Authenticated users can inspect only their own traces through the API:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/traces
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/traces/<trace_id>
```

Trace API responses omit `tenant_id` and decode stored JSON columns such as
sources, candidates, and filter conditions.

## Answer feedback

Authenticated users can submit feedback for their own traced answers. The API
validates that the `trace_id` belongs to the current user before storing
feedback:

```bash
curl -X POST -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
  -d '{"trace_id":"<trace_id>","rating":"down","comment":"missing citation"}' \
  http://127.0.0.1:8000/feedback

curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/feedback
```

Feedback responses omit `tenant_id`. Ratings are constrained to `up` or `down`;
comments are optional and capped at 2000 characters.

Export down-rated traced answers into replay cases for triage and regression
testing:

```bash
python -m src.eval_cli feedback-to-replay \
  --feedback-db "$FEEDBACK_DB_PATH" \
  --trace-db "$TRACE_DB_PATH" \
  --tenant-id 1 \
  --rating down \
  --out eval/feedback_replay.jsonl
```

The output is regular evaluation-case JSONL. Feedback fields are preserved in
case metadata; missing trace rows are skipped.

## Trace retention

Trace logs are stored in SQLite and can grow over time. Use explicit retention
for production or long-running demos:

```bash
TRACE_TTL_SECONDS=2592000
python -m src.eval_cli traces-cleanup --db "$TRACE_DB_PATH" --ttl-seconds "$TRACE_TTL_SECONDS"
```

Pass `--tenant-id` to clean one tenant only. Without `--tenant-id`, cleanup is
global and should be treated as an operator/admin action.

## Document lifecycle registry

The upload path records document lifecycle state in `document_registry`.
Use the authenticated endpoint below to inspect current user's registry rows,
including failed or in-progress documents that may not appear in vector-store
document listings:

```bash
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8000/document-registry
curl -H "Authorization: Bearer <token>" "http://127.0.0.1:8000/document-registry?status=failed"
```

Responses intentionally omit file and content hashes.

## LLM backend: local SFT1147 adapter

FinQuery uses the OpenAI SDK client in `src.main`, so any OpenAI-compatible
chat-completions endpoint can be used as the answer generator. For the current
finance-demo baseline, use the nanochat SFT1147 adapter:

```bash
export LLM_API_BASE_URL=http://127.0.0.1:8500/v1
export LLM_MODEL_NAME=finquery-finance-sft1147
export LLM_API_KEY=not-needed-for-local
```

The corresponding model-side command is documented in
[`docs/SFT1147_RAG_DEMO.md`](docs/SFT1147_RAG_DEMO.md). The adapter provides
`GET /v1/models` and `POST /v1/chat/completions`, including streaming deltas, so
no RAG main-flow code change is required.

Keep the RAG layer responsible for citation/source validation, no-answer gating,
confidence, and traceability. The SFT model is the grounded answer generator,
not the source of truth outside retrieved context.

## Interview/demo runbook

For a complete interview-oriented walkthrough, including the live question
script, SFT adapter setup, evaluation workflow, and resume-safe metric wording,
see [`docs/INTERVIEW_DEMO.md`](docs/INTERVIEW_DEMO.md).
