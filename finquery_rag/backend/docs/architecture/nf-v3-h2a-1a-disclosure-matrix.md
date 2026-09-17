# NF-V3 H2A-1A — Model-Facing Disclosure Matrix

Status: **AUTHORITY IMPLEMENTED FOR THE V2 EVIDENCE BOUNDARIES; FOUR PATHS OUTSIDE
IT ARE NAMED BELOW AND A IS NOT COMPLETE**

Baseline: `1a19a80`. Companion to
`nf-v3-h2a-context-artifact-runtime-plan.md`.

## The question this phase exists to answer

> Which component decides which financial evidence fields a model may see?

**V2 evidence boundaries: `rag_v2/evidence/disclosure.py`.** `project(evidence,
profile=...)`, deny by default. The Binder and the Specialist both route through
it; `binder_fact_view` keeps its own nested-table *extraction* and takes its
*policy* from the authority, so there are not two allowlists.

**System-wide: not yet.** The matrix below records four production-reachable
model boundaries that are outside it. They are listed rather than omitted,
because a disclosure authority that does not know what it does not cover is the
same situation this phase was created to fix.

## Matrix

| # | Call site | Model | Purpose | Raw text? | Projection | Reachability |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `rag_v2/supervisor/bailian_provider.py:155` | `V2_SUPERVISOR_MODEL` | question → plan | no | **none needed** — sees the question only | prod default |
| 2 | `rag_v2/supervisor/api_provider.py:82` | any OpenAI-compat | question → plan | no | none needed | prod (alt) |
| 3 | `rag_v2/evidence/binder_provider.py:405` | `V2_BINDER_MODEL` | slot → fact id | no | **yes** — `build_runtime_binder_fact_views` → `disclosure.BINDER` | prod |
| 4 | `rag_v2/evidence/binder_provider.py:238` | any OpenAI-compat | slot → fact id | no | **yes** — same | prod (alt) |
| 5 | `src/runtime/trusted_v2_generation.py:310` → `LocalSpecialistGenerator` | local torch checkpoint | final answer | **was: whole packet incl. `metadata.source_text`** | **now yes** — `disclosure.SPECIALIST` | prod |
| 6 | `src/runtime/trusted_v2_coordinator.py:1111` | dispatcher to #5 | candidate stage | n/a | n/a | prod |
| 7 | `src/conversation/resolver.py:171` → `bailian_client.py:91` | `qwen3.6-flash` | follow-up → standalone query | no — reads only `user_query` / `standalone_query` | none needed | prod (`MULTITURN_CONTEXT_MODE=on`) |
| **8** | `src/application/rag_orchestrator.py:607` → `llm_gateway.py:72` | nanochat OpenAI cli | **V1 answer** | **YES — the entire assembled retrieval context** | **NONE** | prod when `FINANCIAL_RUNTIME_MODE ∈ {v1, shadow}` |
| 9 | `src/generation/llm_gateway.py:72` | same | same | **YES** | **NONE** | as #8 |
| 10 | `src/retrieval/query_processor.py:252` | same | follow-up → search query | transitively — 160-char-truncated turns | **NONE** | prod (`v1`/`shadow`) |
| 11 | `src/generation/llm_gateway.py:97` | same | streaming answer | YES | **NONE** | legacy; effectively dead |
| **12** | `src/services/process_tables.py:185` | NVIDIA NIM llama-3.1-8b | clean extracted tables | **YES — full page text + table markdown** | **NONE** | prod via `POST /upload` when `NVIDIA_API_KEY` set |
| 13 | `src/services/vector_store.py:16` | MiniLM | embeddings | YES on ingest | n/a (embedding, not generation) | V1 only |
| 14 | `src/services/reranker.py:109` | cross-encoder | rerank | YES | NONE | off by default |
| 15 | `src/services/reranker.py:419` | bge-reranker-v2-m3 | rerank | YES | partial (`build_evidence_bundle`) | off by default |
| 16 | `src/pdf_retrieval_v4/candidate_view_index.py:551` | MiniLM | embed R4 query | no | n/a | V2 |

Non-production (eval/test/never-constructed) sites are listed in the audit and
are not repeated here.

## What A changed, stated precisely

**The Specialist boundary was the real one.** At HEAD it received the *complete
evidence packet*, whose `metadata` carries `source_text` (set from
`evidence_text`/`content`/`raw_content`) and `metadata.retrieval_context.
retrieval_texts` (raw retrieval text). The prompt did not render it — its
`Evidence: {ev['source_text']}` branch reads the top level, and the text is
nested — but the model boundary received it.

So the honest description of the old state is not "no leak". It is: **raw source
text was crossing the model boundary in the payload, and only the prompt's
misreading of the shape kept it out of the rendered text.** Two independent
things had to both be true, and neither was a decision.

The profile is the eleven fields the prompt actually reads. `source_text` is
deliberately excluded, so the dead branch cannot start firing if the packet
shape changes.

## Remaining trust gaps

Recorded, not fixed. Each needs its own decision about *what may cross*, which
is a design question rather than a wiring one.

1. **V1 answer generation (#8/#9) — the largest exposure.** The whole assembled
   context — raw extracted document text, chunks joined by `---` — goes into the
   answer prompt with no projection; the only bound is a token budget and a
   score filter. Reachable whenever `FINANCIAL_RUNTIME_MODE` is `v1` or `shadow`,
   which the production-integration document describes as the supported rollback
   path. Governing it means deciding what a *retrieved chunk* may disclose, which
   is a different object with a different lifecycle from an evidence packet.
2. **V1 query rewrite (#10).** Truncated conversation turns, which may quote
   document text, into a rewrite prompt.
3. **Ingestion table enhancement (#12).** Full page text to a third-party model
   at upload time. Different subsystem, different threat model — this is
   document processing, not evidence disclosure — but it is a model receiving
   raw financial document text and it is not governed by anything.
4. **Rerankers and embeddings (#13–#15).** Raw chunk content to local models.
   Embedding is arguably out of scope (no generative disclosure); rerankers are
   off by default.

## Why A stops here rather than extending

Extending the authority to the V1 context blob is not a wiring change. The V1
path passes a *context string*, not evidence objects — governing it means
designing what a retrieved chunk may disclose, and doing that well requires the
same audit-first treatment this phase got. Doing it as a tail-end addition to A
would produce exactly the kind of unexamined projection the phase exists to
prevent.

The V1 path is also the documented rollback, not the default; `v2` is.

## What is solid

- One authority, one implementation, two profiles, deny by default.
- The Specialist boundary — the one that was actually ungoverned — is closed,
  with a test that drives the real generation capability rather than the helper.
- Future evidence-contract fields are invisible to every profile until named.
- Disclosure is recorded by field name only, so the trace answers "what could
  this model see" without carrying content.
- No behaviour change: 3956 passed, 0 failed; readiness gate unchanged
  (`false_release 0`, `semantic_mismatch 2`, `label_alias 3`, `unexplained 0`).
