# NF-V3 H2A-1A — Model-Facing Disclosure Matrix

Status: **A OWNED BOUNDARIES GOVERNED — V2 evidence, V1 answer, ingestion.
ONE DIALOGUE BOUNDARY IS CLASSIFIED RATHER THAN COVERED (see below).**

Baseline: `1a19a80`. Companion to
`nf-v3-h2a-context-artifact-runtime-plan.md`.

## The question this phase exists to answer

> Which component decides which financial evidence fields a model may see?

**V2 evidence boundaries: `rag_v2/evidence/disclosure.py`.** `project(evidence,
profile=...)`, deny by default. The Binder and the Specialist both route through
it; `binder_fact_view` keeps its own nested-table *extraction* and takes its
*policy* from the authority, so there are not two allowlists.

**System-wide, with one stated exception.** Every production-reachable boundary
that carries evidence or source material is governed; the matrix below is the
record, and the one exception is named in "Remaining" rather than left implicit.
A disclosure authority that does not know what it does not cover is the situation
this phase was created to fix, so the matrix is kept complete either way.

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

## First pass: the V2 evidence boundaries

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

## After the V1 fix

Three boundaries are now governed by one authority:

| Boundary | Profile | What changed |
| --- | --- | --- |
| Binder | `BINDER` | policy moved into the authority; its ~50-field surface is unchanged |
| Binder + Specialist (V2 evidence) | `SPECIALIST` | was ungoverned; received whole packets incl. `metadata.source_text` |
| **V1 answer (#8/#9)** | `V1_ANSWER` | was ungoverned; received the whole assembled context with no projection at all |
| **Ingestion table clean (#12)** | `INGEST_TABLE` | was ungoverned; raw page text to a third-party model |

The V1 projection sits at the *entry* to `ContextBuilder.build` rather than at
each caller, so every caller is covered by construction and the projection
precedes formatting — a finished context string cannot be projected back into
fields.

**The V1 audit was incomplete on the first pass, and that is the useful part.**
Reading `ContextBuilder.build` gives you a field list; it does not give you the
fields the boundary needs, because its output has a *second* consumer.
`last_context_evidence` feeds `EvidenceItem.from_chunk`, whose `document_name`
drives the answerability check — so an allowlist built from the formatter alone
silently added a "could not verify these documents" suffix to every V1 answer.
One existing test caught it. The profile is now the union of what the formatter
and that consumer read, and a test pins the trap by running a projected chunk
through `EvidenceItem.from_chunk` and asserting `document_name` survives.

`INGEST_TABLE` is the one profile that *permits* raw text, because cleaning an
extracted table is impossible without the table. Routing it through the
authority strips nothing; it records the decision, so "who allowed a model to
see raw page text?" has an answer instead of being an omission nobody examined.

## Remaining: one boundary, classified rather than ignored

**`src/retrieval/query_processor.py:252` — the V1 follow-up rewrite.**

It passes conversation turns (truncated to 160 characters each) plus a memory
profile bounded by `ALLOWED_PROFILE_FIELDS`. This is a *dialogue* boundary, not
an evidence boundary: it carries no retrieval object and no evidence packet, so
the evidence disclosure authority has nothing to project there. Assistant turns
can transitively quote document text, so it is not cleanly source-free either.

Whether dialogue needs a disclosure profile of its own is a real question that
this phase does not answer, and it is named here rather than counted as covered.
It is production-reachable on `v1`/`shadow` only.

Not model-facing in the sense this phase governs: embeddings
(`vector_store.py:16`, `candidate_view_index.py:551`) produce no generation, and
the rerankers (`reranker.py:109,419`) score rather than generate and are off by
default.

## Verification

```
full suite                     3963 passed + 143 skipped, 0 failed
V1 regression path (mode=v1)    694 passed, 7 skipped
disclosure tests                 16 passed
readiness gate                 false_release 0, semantic_mismatch 2,
                               label_alias 3, unexplained 0
```

## The principle this phase establishes

> Rollback may degrade capability, but it must not silently degrade model-input
> trust guarantees.

V1 is the documented rollback path and it was the largest ungoverned exposure in
the system. Governing it cost one profile and a projection at one call site; the
alternative would have been a trust guarantee that held only on the path that
happened to be default.

## The question, answered

> Which component decides which financial evidence fields a model may see?

**`rag_v2/evidence/disclosure.py`**, for every production-reachable boundary that
carries evidence or source material: `project(evidence, profile=...)`, deny by
default, four profiles, one implementation. The single exception is stated above
by name.
