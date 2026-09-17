# NF-V3 H2A-3A — Model Context Surface Audit

Status: **OPEN (audit only).** Predecessor: `nf-v3-h2a-2-artifact-authority.md`.
Baseline: `HEAD 9497943`, seal `nf-v3-h2a2-artifact-authority`.

H2A-3A answers one question before anything is built: **what does each model
actually see today, who owns it, and what is unbounded?** It changes no
production code. `AgentContextPack`, `ArtifactStore` and `ContextBudget` do not
exist after this phase — that is the point of doing the audit first.

---

## 0 · Scope, method, and how the classes were assigned

**Method.** Every boundary below was found by tracing an actual network or model
invocation, not by reading a function name. Three sweeps covered ingestion, the
`rag_v2` package, and serialization; each finding was then re-checked against the
source by reading the call site. Claims that were only reachable from
`scripts/evaluation/**` or tests are labelled as such and are **not** counted as
production boundaries.

**Two whole-model families exist and only one of them is live.**

| Runtime | Selected by | Live default? |
| --- | --- | --- |
| Trusted V2 | `FINANCIAL_RUNTIME_MODE` (default `v2`, `.env.example:49`) + `TRUSTED_V2_RUNTIME_BUILDER` (`.env.example:57`) | **yes** |
| V1 legacy | `FINANCIAL_RUNTIME_MODE in {v1, shadow}` | no (rollback + shadow only) |

So the live path is `POST /query` → `QueryLifecycleService.execute_user_turn`
(`src/runtime/query_lifecycle.py:381`) → `_build_financial_runtime`
(`src/main.py:265`) → `build_trusted_v2_runtime_for_request`
(`src/runtime/trusted_v2_production.py:983`) → `BoundedTrustedV2Coordinator.execute`.
V1 is reachable on the same endpoint but is not the default. **Both are audited**,
because a rollback runtime that has drifted is a rollback runtime that does not
roll back.

**Class assignment.** Classes describe *evidence and calculation* governance —
whether an authority decides what may cross:

| Class | Meaning |
| --- | --- |
| **A** | fully governed: one authority decides the field set, applied before assembly, single assembly point |
| **B** | governed, but context assembly is duplicated (≥2 implementations of the same projection policy) |
| **C** | partially governed: the live caller is governed but the boundary itself cannot enforce it |
| **D** | raw/unbounded context path: model-visible content whose field set no authority owns |
| **E** | no evidence context at all (query/instruction only) |

**E is not a defect.** The Supervisor is *supposed* to plan from the question
alone. Classifying it D would be as wrong as classifying the ingest boundary A
and then calling it safe.

---

## 1 · Model boundary matrix

Live boundaries first. Every row was read at the cited line.

| # | Boundary | Call site | Role | Model / provider | Class |
| --- | --- | --- | --- | --- | --- |
| **B1** | V2 Supervisor | `rag_v2/supervisor/bailian_provider.py:155` · `api_provider.py:82` | plan / tool-routing control | `V2_SUPERVISOR_MODEL` on `V2_SUPERVISOR_BASE_URL` (default provider `bailian`) | **E** |
| **B2** | V2 Evidence Binder | `rag_v2/evidence/binder_provider.py:405` (bailian) · `:238` (api) | slot → fact selection, selection only | `V2_BINDER_MODEL` (falls back to `V2_SUPERVISOR_*`) | **B** |
| **B3** | V2 Specialist | `src/generation/local_specialist_generator.py:246`, reached via `src/runtime/trusted_v2_generation.py:326` | candidate answer generation | local nanochat Step-156 checkpoint, SHA256-pinned | **A** |
| **B4** | V1 answer | `src/generation/llm_gateway.py:72` (sync) · `:97` (stream) | answer generation | `LLM_MODEL_NAME` (default `nanochat`) on `LLM_API_BASE_URL` | **C** |
| **B5** | V1 query rewrite | `src/retrieval/query_processor.py:252` | follow-up → standalone search query | same nanochat client | **D** |
| **B6** | Conversation resolver | `src/conversation/bailian_client.py:91` | contextual query resolution | `BAILIAN_CONTEXT_MODEL` (default `qwen3.6-flash`) | **D** |
| **B7** | Ingest table | `src/services/process_tables.py:204` | clean/summarise an extracted table | `NVIDIA_MODEL_NAME` (default `meta/llama-3.1-8b-instruct`) | **A** |
| **B8** | Dense embedding | `src/services/vector_store.py:16` (MiniLM, module-level) | retrieval encode (index write + query read) | `EMBEDDING_MODEL_NAME` (default `all-MiniLM-L6-v2`) | **D** |
| **B9** | Reranker (model-backed) | `src/services/reranker.py:109` · `:419` | candidate reordering | `CrossEncoder` / `BAAI/bge-reranker-v2-m3` | **D** |
| **B10** | MinerU | `src/services/mineru_parser.py:174` | OCR / layout / table extraction | local MinerU deployment, opt-in via `PARSER_BACKEND` | **D** |

**B9 is model-free by default.** `DEFAULT_RERANKER = "heuristic"`
(`src/services/retrieval_config.py:15`) selects `HeuristicReranker`, pure lexical.
The model-backed classes are reachable only when `RERANKER_PROVIDER` /
`RAG_RERANKER` names them. **B10 is opt-in**: `PARSER_BACKEND` defaults away from
MinerU (`src/services/ingest.py:42-68`).

**Defined but not on any production path** (do not appear in the matrix above):
`LocalProvider`, `StrongGeneralAPIProvider`, `DeterministicFallbackProvider`,
the constrained / slotwise / pairwise / shortlist-comparative binder providers,
`MockGeneratorProviderV1`, `ReplayGeneratorProviderV1`, `TrustedRAGRuntimeV2`.
All are reached only from `scripts/evaluation/**` and tests. This matters for one
reason recorded in §3: **four of those binder providers build their own fact
view and bypass the disclosure authority**, so "which fields may the binder see"
has a second answer there. That is evaluation debt, not a live leak.

### Per-boundary detail

| # | Caller chain (production) | Input object(s) | Prompt builder | Evidence source | Calculation source | Conversation source | Metadata source |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | coordinator `:1582` → `SupervisorService.plan` | `standalone_query` only | `rag_v2/supervisor/prompt.py:83` | — | — | — | — |
| B2 | coordinator → `SemanticEvidenceEvaluationCapability.evaluate` (`src/runtime/trusted_v2_binder.py:903`) | `BinderRequest.to_dict()` | `rag_v2/evidence/prompt.py:64` | `state.evidence_packets` → `build_runtime_binder_fact_views` | — | — | `_METADATA_PROMOTED_KEYS` (`trusted_v2_binder.py:84`) |
| B3 | coordinator `_candidate_stage` (`:1125`) → `TrustedV2GenerationCapability._call_specialist` | projected evidence list + projected calc + `normalized_query` | `LocalSpecialistGenerator.render_prompt` (`:159`) | `_bound_items(state)` (`trusted_v2_generation.py:93`) | `state._calculation_result_obj` | — | — |
| B4 | `RAGOrchestrator:607`, `RAGEngine:382/462` | one assembled **string** `context` | `src/generation/prompt_builder.py:14` + f-string at `llm_gateway.py:66` | `ContextBuilder.build` over retrieved chunks | — | — | — |
| B5 | `RAGOrchestrator:188` → `LLMGateway.rewrite_query` | rewrite prompt f-string | `query_processor.py:237` | — | — | `conversation_history[-4:]`, 160 chars each | `memory_profile` |
| B6 | `QueryLifecycleService:400/458` → `ConversationShadowService` → `ConversationContextManager.process_user_turn` → resolver `:217` | `messages` list | `src/conversation/resolver.py:194` | — | — | `turns[-4:]` + `compressed_history` | `dialogue_state` (5 fields) |
| B7 | `POST /upload` → `process_pdf` → `enhance_table_with_context` | `page_text`, `table_md["md"]`, `page_num` | `process_tables.py:163` | raw PDF page + whole table | — | — | — |
| B8 | `POST /upload` (`add_documents`) · query time (`query_collection`) | chunk `content` | — | raw chunk text | — | — | — |
| B9 | `RAGEngine:126` → `RetrievalPipeline` | `(query, chunk["content"])` pairs | `reranker.py:116`; BGE path uses `build_token_budgeted_text` | raw chunk text | — | — | — |
| B10 | `POST /upload` → `process_pdf_with_mineru` | the PDF file | — | whole document | — | — | — |

### Token, budget and retry behaviour

| # | Token counting / budget | Output cap | Timeout | Retry |
| --- | --- | --- | --- | --- |
| B1 | none; provider-reported `usage.prompt_tokens` recorded post-hoc (`bailian_provider.py:206`) | **not sent** (bailian) / 1024 (api) | 180 s / 120 s | `max_retries=0` (bailian, `trusted_v2_production.py:648`) · **SDK default 2** (api — see §6) |
| B2 | none; provider-reported `input_tokens` recorded post-hoc | **not sent** (bailian) / 1024 (api) | 180 s | `max_retries=0` (`:704`, `:714`) |
| B3 | **exact tokenizer, measurement only** — `len(prompt_tokens)` (`local_specialist_generator.py:240,264`) | 128 | n/a (local) | none |
| B4 | `tiktoken cl100k_base` over `max_context_tokens=1100`, `safe_limit=900` (`context_builder.py:96`) | 512 | SDK default | none (`except` → error string) |
| B5 | none; 4 messages × 160 chars (`query_processor.py:218-223`) | 100 | SDK default | none (`except` → return question) |
| B6 | regex word-count ×1.15 (`context_budget.py:38-47`) via `ContextBudgetManager` | 512 | 3 s | **2 retries, exponential + jitter** (`bailian_client.py:88-132`) |
| B7 | **none** | 1024 | 30 s | none |
| B8 | encoder `max_seq_length` (implicit) | — | — | none |
| B9 | none (cross-encoder) / `max_length=1024` word-count budget (BGE) | — | — | none |
| B10 | subprocess `MINERU_TIMEOUT_SECONDS` (600) | — | 600 s | none |

---

## 2 · Context-source matrix

Model-visible inputs by semantic category. Read this as "which categories does
each boundary carry", not as a field list — the field lists are in §3.

| # | Instructions | Conversation | Trusted evidence | Calculation | Runtime / control | Raw source |
| --- | --- | --- | --- | --- | --- | --- |
| B1 | system prompt + `SUPERVISOR_PLAN_JSON_SCHEMA` | — | — | — | the question (as the object to plan over) | — |
| B2 | system prompt + `BINDER_SCHEMA` | — | **yes** — projected fact views | — | intent, operation, required_slots | — |
| B3 | `[ANSWER RULES]` 1–7 inline | — | **yes** — projected evidence | **yes** — projected calc | — | — |
| B4 | `SYSTEM_PROMPT` (FinQuery rules 1–8) | — | **yes** — as formatted string | — | — | **yes** — chunk `content`/`parent_excerpt` are retrieved document text |
| B5 | inline rewrite prompt | **yes** — 4 turns × 160 chars | — | — | — | — |
| B6 | inline resolver system prompt | **yes** — 4 turns + compressed history | — | — | dialogue state (5 fields) | — |
| B7 | inline ingest prompt (rules 1–4) | — | — | — | page number | **yes** — whole page text + whole table markdown |
| B8 | — | — | — | — | — | **yes** — whole chunk text |
| B9 | — | — | — | — | — | **yes** — whole chunk text (cross-encoder) |
| B10 | — | — | — | — | — | **yes** — the whole document |

**Instruction ownership.** Four distinct system prompts exist, three of them
inline constants in the module that calls the model (`resolver.py:194`,
`process_tables.py:163`, `query_processor.py:237`) and one in
`src/generation/prompt_builder.py`. Two are module-level named constants
(`SUPERVISOR_SYSTEM_PROMPT_V1`, `BINDER_SYSTEM_PROMPT_V1`). There is no single
owner of "the instructions a model may receive" — but for H2A-3A this is
**recorded, not classified as a defect**: §2 of the brief is explicit that
duplicated prompt building is not automatically wrong, and the four prompts serve
four genuinely different roles with no shared semantics.

**Runtime/control reaches exactly one boundary** — the Binder, which receives
`intent`, `operation` and `required_slots`. That is necessary: slot IDs are the
selection vocabulary. No boundary receives route, retry reason, repair reason or
budget state.

---

## 3 · Disclosure matrix

One authority: `rag_v2/evidence/disclosure.py`. Four profiles
(`BINDER`, `SPECIALIST`, `V1_ANSWER`, `INGEST_TABLE`), two artifact types
(`EVIDENCE`, `CALCULATION`). Deny by default.

| Boundary | Profile | What enters disclosure | What exits | Projection timing | Field that bypasses? |
| --- | --- | --- | --- | --- | --- |
| B2 Binder | `BINDER` | whole evidence packet dict | allowlisted scalars + lists | **before** assembly | no — but see below |
| B3 Specialist | `SPECIALIST` | evidence packet dict | 11 allowlisted fields (§ probe below) | **before** assembly | no |
| B3 Specialist (calc) | `SPECIALIST` | `CalculationResult.to_dict()` | `operation`, `unit`, `value` | **before** assembly | no |
| B4 V1 answer | `V1_ANSWER` | retrieved chunk dict | 11 fields + governed `metadata` container | **before** formatting | no |
| B7 Ingest | `INGEST_TABLE` | `{page_text, table_markdown, page_num}` | same three | **before** prompt build | no — raw is the role |
| B1/B5/B6/B8/B9/B10 | **none** | — | — | — | n/a — no evidence crosses |

### Verified probe (B3, run against this tree)

```
projected keys : citation_id, currency, document_id, evidence_id, metric,
                 period, scale, scope, unit, value
'page'         : absent
'source_text'  : absent
```

`project()` is applied at `trusted_v2_generation.py:309` per item and
`project_calculation` at `:297`, both **before** `_call_specialist` assembles the
prompt at `:326`. `last_disclosed_fields` records what crossed **by name only**
(`:319-325`).

### The Binder projects by a different mechanism

`build_runtime_binder_fact_view` (`rag_v2/evidence/binder_fact_view.py:152`) does
not call `project()`. It takes its field list from
`allowed_fields(EvidenceDisclosureProfile.BINDER)` (`:90`) and does its own
extraction, because it needs a ladder across `[fact, structural_context,
metadata]`. The *policy* is shared; the *extractor* is a second implementation.
That is why B2 is class **B**, not A: the two cannot drift in field set, but they
can drift in resolution behaviour, and `_CANONICAL_ONLY_FIELDS` (`:99`) exists
precisely because they already did once (H2A-2D-2B, page).

### Four evaluation-only binder providers bypass the authority

`constrained_binder_provider.py`, `slotwise_binder.py`, `pairwise_binder.py`,
`shortlist_comparative_binder.py` call `build_binder_fact_views` /
`build_binder_fact_views_v2` (`binder_fact_view.py:441`, `:398`), which copy
`_FACT_FIELDS` + `_SOURCE_FIELDS` literally rather than reading the allowlist.
They are not on the production path. They are recorded because a future
`AgentContextPack` must not be built on a projector that has two answers to
"which fields".

### Previously-fixed contracts — re-verified, unchanged

| Contract | Verified |
| --- | --- |
| V2 Specialist evidence | `project(..., SPECIALIST)` before the call; probe above |
| V2 Specialist calculation | `project_calculation(...)` reduces `to_dict()` (which carries operand `source_text` and raw `error_message`) to 3 fields |
| V1 answer | `project(chunk, V1_ANSWER)` at `context_builder.py:65`, before formatting |
| Ingestion-table raw source | `project(..., INGEST_TABLE)` at `process_tables.py:151` — permissive by design, recorded rather than omitted |

Regression coverage already exists in `tests/harness/test_model_facing_disclosure.py`
(20 tests across all four boundaries). **H2A-3A adds no disclosure test**; it
changes no contract.

---

## 4 · Raw-source exposure paths

Raw source is not forbidden. What the brief requires is that every raw path has
an **explicit role contract**. Four paths have one; three do not.

| # | Raw content that crosses | Role contract? | Where it is stated |
| --- | --- | --- | --- |
| B7 Ingest | whole page text + whole table markdown | **yes** | `_INGEST_TABLE_FIELDS` + the comment at `disclosure.py:230-239` ("the one boundary whose legitimate input *is* raw document text") |
| B4 V1 answer | retrieved chunk text + `parent_excerpt` | **yes** | `_V1_ANSWER_FIELDS` + `_V1_ANSWER_METADATA_FIELDS` (`disclosure.py:169-228`) |
| B5 Rewrite | prior user/assistant message text (160 chars each) | **no** | bounded by an inline constant, owned by no one |
| B6 Resolver | prior turn text + compressed history | **no** | bounded by `ContextBudgetManager` defaults |
| B8 Embedding | whole chunk text | **no** | embedded raw; no profile exists |
| B9 Reranker | whole chunk text (cross-encoder path) | **partial** | `build_token_budgeted_text` bounds it (`evidence_bundle_serializer.py:58`) but no authority names the fields |
| B10 MinerU | whole document | **no** | opt-in, file-in |

**B8 and B10 are the honest gaps.** Neither is a text-generation boundary, so
neither was in H2A-1's disclosure scope — but both are models consuming raw
document text, and the brief's rule ("every raw-source path must have an explicit
role contract") does not exempt them by modality. Whether they *should* fall
under a disclosure authority is a **3B design decision**, not an audit finding —
the profiles govern *evidence artifacts*, and an embedding call sees a chunk
before any evidence object exists.

### The ingest path's real risk is downstream, not at the boundary

The `INGEST_TABLE` boundary is correctly governed and is the widest exposure in
the system. But its output is written back into the retrieval index unvalidated:
`process_tables.py:211-219` accepts the model's `CLEANED TABLE:` block verbatim
into `content` and `parent_excerpt` (`src/services/ingest.py:684-701`), and those
two fields are permitted into the **V1 answer prompt** by `V1_ANSWER`. The
prompt's rule 3 ("Do NOT change numeric values") has no enforcement. So a model
that alters a number at B7 puts that number in front of a model at B4 as
evidence. Recorded as a **trust-risk**, not fixed here — it is an ingestion
validation decision, and H2A-3A implements nothing.

---

## 5 · Whole-state serialization

**Does `AdaptiveRAGStateV1` or its `asdict()` reach any model prompt today? No.**

`AdaptiveRAGStateV1` (`rag_v2/adaptive/adaptive_contracts.py:375-426`) is a
mutable dataclass with 35 fields and a pure dump serializer
(`to_dict() → asdict(self)`, `:523`). Those fields include `evidence_packets`
(whole packets **including the `metadata` bag and its extracted source text**),
`calculation_result`, `candidate_answer`, `tool_history`, `transitions` and
`turns`. It is exactly the object that must never be serialized into a prompt.

Every occurrence, classified:

| Site | What is serialized | Reaches a model? | Classification |
| --- | --- | --- | --- |
| `adaptive_contracts.py:523` `to_dict()` | whole state, `asdict` | **no** | dump for traces/persistence only |
| `adaptive_state_machine.py:47` `"state": self.state.to_dict()` | whole state inside a run result | **no** | trace artifact; `rag_v2/adaptive/` contains no model call at all |
| `src/conversation/context_budget.py:71` | `dialogue_state.to_dict()` → `json.dumps` | **no** | **counted only** (`:72` feeds `count_tokens`); the repr is never returned — `prepare_context` returns `(turns, compressed, tokens)` |
| `src/conversation/sqlite_store.py:157` | `state.to_dict()` | **no** | persistence |
| `rag_v2/evidence/transport_retry.py:29,34` | `request.to_dict()`, raw `request.facts` | **no** | **SHA-256 only** |
| `rag_v2/generation/providers.py:108` | `json.dumps(packets)` | **no** | packet-set hash |
| `trusted_v2_generation.py:232` | `calculation.to_dict()` | **projected first** | the internal-diagnostics form is what is written at the call site; `project_calculation` reduces it to 3 fields before the call |
| `rag_v2/generation/rendering.py:21` | whole packet into `GenerationInputV1.packet` | **no** | reachable only from `rag_v2/runtime/runtime.py:30`, which is evaluation-only |

**Outside production**, one real whole-packet-to-prompt path exists:
`scripts/evaluation/run_nf_v2_06_r0_verified_generation.py:394` does
`json.dumps(packet)` straight into a user message with no projection. Sealed, not
live — but it is the template a future author would copy.

**Consequence for H2A-3:** the "we can solve this by wrapping the whole-state
serialization in a pack" failure mode the brief warns about **has no precedent to
justify it**. There is nothing today that dumps state into a prompt, so the
compiler is not a repair — it is a new boundary, and it should be argued on its
own merits in 3B.

---

## 6 · Budget matrix — RunBudget vs ContextBudget

**These are different questions and are kept separate.** `RunBudget` bounds how
much *work* the runtime may do; `ContextBudget` bounds how much *information* one
invocation may see. The audit found both, and found the second one fragmented.

### RunBudget — implemented

| Control | Where | Default |
| --- | --- | --- |
| replan rounds | `AdaptiveRAGBudgetV1.max_replan_rounds` (`rag_v2/adaptive/adaptive_budget.py:12`) | 2 |
| total tool calls | `.max_total_tool_calls` | 5 |
| same-tool retries | `.max_same_tool_retry` | 1 |
| identical-query retry | `.max_identical_query_retry` | 0 — **declared, reserved, unenforced** (`RESERVED_FIELDS`, `:27-29`) |
| shadow timeout | `FinancialRuntimeRouter.shadow_timeout_ms` (`runtime_router.py:83`) | 5000 ms |
| provider timeouts | per-provider env (`V2_*_TIMEOUT_SECONDS`) | 120–180 s |
| subprocess timeout | `MINERU_TIMEOUT_SECONDS` | 600 s |

`AdaptiveRAGBudgetV1` is coherent and honest — it even warns at construction when
an operator sets a bound nothing enforces (`trusted_v2_production.py:785-792`).
**`RunBudget` needs no H2A-3 work.**

### ContextBudget — fragmented across five places, none authoritative

| # | Bound | Unit | Where | Governs |
| --- | --- | --- | --- | --- |
| C1 | `max_context_tokens=1100`, `safe_limit=900` | tiktoken `cl100k_base` | `context_builder.py:25,96` | V1 answer evidence |
| C2 | `CONTEXT_TARGET_TOKENS=4096`; `CONTEXT_MAX_TOKENS=8192`; `CONTEXT_SUMMARY_MAX_TOKENS=768` | word-count ×1.15 | `context_budget.py:34-36` | conversation turns |
| C3 | 4 turns × 160 chars | characters | `query_processor.py:218-223` | V1 rewrite history |
| C4 | output caps: 128 / 512 / 1024 / 100 | tokens | per provider | none — output only |
| C5 | `max_length=1024` | `len(text.split())` | `evidence_bundle_serializer.py:66` | BGE reranker bundle |

**Three findings.**

1. **C2's two headline numbers are inert.** `self.max_tokens` and
   `self.summary_max_tokens` are assigned at `context_budget.py:35-36` and **read
   nowhere else in the file or the repository**. The effective bound is
   `self.target_tokens` via `available_budget = max(200, target - base)`
   (`:73`). An operator setting `CONTEXT_MAX_TOKENS` changes nothing.
2. **C1 and C2 are measured in incompatible units.** C1 counts with a real BPE
   tokenizer; C2 counts regex word/punctuation matches × 1.15. They are never
   combined, so nothing is arithmetically wrong today — but there is no single
   place that can answer "how many tokens will this invocation see", which is
   exactly what H2A-4's token measurements will require.
3. **`V2ExecutionRequest.runtime_budget`** (`trusted_v2_contracts.py:166`) is
   declared, copied through `to_dict`/`from_dict` (`:213-214`, `:279`, `:299`),
   and **never populated by a producer nor read by a consumer**. It is carriage,
   not a control. The plan doc already flagged this (`nf-v3-h2a-context-artifact-runtime-plan.md`, Step 4).

**Not implemented, and deliberately not implemented in 3A:** there is no
`ContextBudget` type. The audit's contribution is the separation and the
inventory, not a new class.

---

## 7 · Tokenizer matrix

The rule from the brief: token-budget claims must use the real tokenizer for the
actual model boundary, and a boundary without a reliable mapping is **NOT
DETERMINED** rather than invented.

| # | Model / provider | Tokenizer | Exact? | Approximate fallback in use | Class |
| --- | --- | --- | --- | --- | --- |
| B1 Supervisor | bailian / api | server-side only | **no** | none — provider-reported `usage.prompt_tokens` recorded **after** the call (`bailian_provider.py:206`) | reported post-hoc |
| B2 Binder | bailian / api | server-side only | **no** | same (`binder_provider.py:331`) | reported post-hoc |
| B3 Specialist | local nanochat Step-156 | **the model's own BPE**, from `build_model` | **yes** | none needed | **EXACT** |
| B4 V1 answer | nanochat via OpenAI-compat adapter | `tiktoken cl100k_base` | **no — mismatched** | `len(text)/3` if tiktoken is absent (`context_builder.py:126`) | **MISMATCHED** |
| B5 Rewrite | nanochat | none | **no** | 160-char slice | none |
| B6 Resolver | qwen3.6-flash (DashScope) | none | **no** | regex word-count ×1.15 | approximate |
| B7 Ingest | `meta/llama-3.1-8b-instruct` | none | **no** | none — no measurement at all | none |
| B8 Embedding | `all-MiniLM-L6-v2` | HF tokenizer, internal to the encoder | **yes, but not exposed** | implicit `max_seq_length` | not measured |
| B9 Reranker | cross-encoder / bge-reranker-v2-m3 | none passed | **no** | `len(text.split())` | approximate |
| B10 MinerU | local deployment | n/a (document input) | — | — | n/a |

**The one boundary that gets this right is B3.** `LocalSpecialistGenerator`
encodes the rendered prompt with the checkpoint's own tokenizer and reports
`len(prompt_tokens)` as `rendered_input_length` (`local_specialist_generator.py:238-242, 264`).
It measures exactly and does not gate — the number exists for telemetry, not
control.

**B4 is the sharpest finding in this matrix.** The served model is `nanochat`
(`LLM_MODEL_NAME`, adapter at `http://127.0.0.1:8500/v1`); the budget is measured
with `tiktoken.get_encoding("cl100k_base")`, which is OpenAI's tokenizer for a
different model family. The 1100-token budget is therefore an *estimate of a
different model's tokenization* being used to fit a 2048-token window — and the
comment at `rag_engine.py:63-64` does the arithmetic in exactly those terms. If
`tiktoken` is missing, the fallback is `len(text)/3`
(`context_builder.py:126`), a character heuristic. **No H2A-4 token claim may
reuse C1's numbers as if they were real tokens for the nanochat boundary.**

**NOT DETERMINED:** the tokenizer for B1, B2, B5, B6, B7 and B9 is not available
in-process. For B1/B2 the provider reports true input tokens after the fact,
which is sufficient for measurement but not for pre-call budgeting. For B5–B7 and
B9 there is no mapping at all in this repository. Those must be resolved with the
serving stack before any budget is enforced there — not guessed.

---

## 8 · Ordering and deterministic selection

| # | Selection / ordering today | Semantically contractual? |
| --- | --- | --- |
| B2 Binder | `state.evidence_packets` order, then a **deterministic relevance filter** by canonical metric id (`trusted_v2_binder.py:130-168`); falls back to the full packet when no fact exposes a metric (`:167`) | **yes** — documented as "deterministic context shaping, never slot admission" |
| B2 handles | `F01..Fn` in frozen packet order (`binder_selection.py:35-37`), no filtering or ranking | **yes** |
| B3 Specialist | `state.evidence_packets` order, filtered by `bound_evidence_ids` (`trusted_v2_generation.py:93-107`) | insertion order — deterministic, but stated nowhere as a contract |
| B4 V1 context | retrieval order after RRF/rerank + content dedup (`context_builder.py:74-81`) | **yes** — score order is the retrieval contract |
| B5 Rewrite | `conversation_history[-4:]`, caller's order | recency, implicit |
| B6 Resolver | `ContextRelevanceFilter` score descending, top 4, then **re-sorted to original turn order** (`relevance_filter.py:85-90`); `ContextBudgetManager` re-sorts chronologically by turn index (`context_budget.py:99-100`) | **yes** — the re-sort is deliberate and is what makes the result order-independent of the score sort |
| B7 Ingest | tables in page order, `tables_by_page` insertion order (`ingest.py:674`) | page order |

**Every model-facing ordering today is deterministic.** No boundary uses a
`set` iteration, a dict-hash order, or a random tiebreak. Two orderings are
*documented* as contractual (binder metric filter, `F01..Fn`); the rest are
incidental-but-stable.

The one thing H2A-3 must not lose: `relevance_filter` scores with a stable sort
and then **discards the score order entirely** in favour of original turn order
(`relevance_filter.py:90`). A compiler that selected by score and kept score
order would silently change what the conversation resolver sees. **Selection and
presentation order are separate decisions here, and the code already treats them
separately.**

**Existing selection machinery worth reusing rather than replacing:**
`_facts_for_binding` (`trusted_v2_binder.py:130`) is already a deterministic,
role-specific, explanation-carrying context selector — for one boundary. It is
the shape the compiler should generalize, not a thing it should replace.

---

## 9 · Context duplication

Classified per the brief, with **nothing removed in 3A**. This inventory is
H2A-4's ablation input.

| Duplication | Where | Class |
| --- | --- | --- |
| Evidence as structured value **and** as raw retrieval text | B2/B3 see typed fields; B4 sees the same document text as prose | **required distinct representation** — different roles, different artifacts |
| Calculation value + operands + rendered explanation | `CalculationResult.to_dict()` carries operands (with `source_text`), `render_calculation_result` renders prose | **trust-risk duplication** — `to_dict()` is the internal form; the SPECIALIST profile reduces it, and `render_calculation_result` output goes into the candidate answer, not the prompt |
| System prompt repeated across four boundaries | `prompt_builder.py`, `supervisor/prompt.py`, `evidence/prompt.py`, two inline | **harmless** — different roles, no shared semantics (§2 note) |
| Conversation history in two prompts | B5 rewrite (`history[-4:]`) and B6 resolver (`turns[-4:]`) | **required distinct representation** — different consumers, different projections |
| `parent_excerpt` + matched child snippets | `context_builder.py:242-247` composes both into one content string | **token waste** — the same passage appears as parent text and as child hit; already partially deduped at `:74-81` (first 100 chars) |
| Compressed history **and** recent turns | `prepare_context` returns both (`context_budget.py:111`) | **required** — L1/L3 tiers; but L3 is a hardcoded vocabulary summary (`:136-152`), so it can only ever name 7 companies / 5 metrics / 9 periods |
| `page` in `metadata` and at top level | H2A-2D-2B made `EvidencePacketV1.page` the authority and forbade the metadata read in the binder view | **already resolved** — retained here because V1's chunk `metadata["page"]` is a *different* object on a *different* path |
| Citation metadata in several forms | `citation_id` on the packet, `_structured_citations` entries, public `citations` | **legitimate projection** (H2A-2 taxonomy) |

**The one duplication to watch:** B3's specialist prompt renders
`Source: {document_id}:{page}` with `page = ev.get("page") or 1`
(`local_specialist_generator.py:181,191`). `page` is **not** in
`_SPECIALIST_FIELDS`, so it never crosses, so **the fallback fires on every
call: the model is told the evidence comes from page 1, always.** This is not
token waste — it is a **model-visible fabricated provenance field**. The profile's
own comment (`disclosure.py:150-153`) recorded that page "is added here when it
survives H2A-1C" and it has not; the prompt meanwhile stopped omitting it and
started asserting it. Not fixed in 3A. See §15 F5.

---

## 10 · Traceability — what already exists

The brief asks what provenance H2A-3 can *retain* rather than build. All five
questions have an existing deterministic answer:

| Question | Existing explanation |
| --- | --- |
| Why is this evidence item here? | `bound_evidence_ids` ← Binder `slot_bindings` ← `ClaimProvenance.bound_evidence_ids` |
| Why this slot? | `SupervisorPlan.required_slots` with `role` / `value_type`; `plan` + `normalization` on the run |
| Why this calculation? | `CalculationResult.calculation_id`, `operands[].evidence_chunk_id`, `formula_version` |
| Why this turn? | `DialogueTurn.turn_id` ∈ `DialogueState.referenced_turn_ids`, or `relevance_filter` recency/entity score |
| Why this raw text? | ingest: the table's page + `page_num`; V1: `ContextBuilder._last_context_evidence` |

Plus two purpose-built disclosure traces, **field names only, never values**:
`TrustedV2GenerationCapability.last_disclosed_fields` (namespaced
`evidence.*` / `calculation.*`) and `ContextBuilder.last_disclosed_fields`.

**No new observability subsystem is needed.** The answer to "why was this
model-visible item included?" is already derivable from `EvidenceBinding`,
`ClaimProvenance`, `CalculationResult` and the two disclosure traces. What is
missing is only that nothing *joins* them at the invocation — and that is a
presentation question for 3B, not a provenance gap.

---

## 11 · Proposed compiler boundary — disclosure ordering

The brief's two options, evaluated against the invariants rather than aesthetics.

| | Option A | Option B |
| --- | --- | --- |
| Order | artifacts → **select** → **disclose** → pack → prompt | artifacts → **disclose** → **select** → pack → prompt |

**Neither wholesale. The audit's recommendation is a split, and the split is
forced by a fact in this codebase.**

Selection today operates on *authoritative* fields, not on projected ones:

- `_facts_for_binding` selects by `canonical_metric_id(fact["metric"])` — and
  `metric` **is** in `_BINDER_FIELDS`, so selection would still work after
  disclosure;
- but `classify_evidence_scope` / `query_allows_evidence_scope`
  (`trusted_v2_binder.py:184-204`) read scope and segment labels, and
  `_bound_items` selects by `bound_evidence_ids`;
- and `_merge_parent_context_chunks` (`context_builder.py:167`) **synthesizes new
  content** — `parent_excerpt` + matched child snippets — from fields that are
  individually permitted. Disclosure-then-select would have to admit
  `parent_excerpt` wholesale to preserve current V1 behaviour, which is a wider
  permission than the profile grants today.

**Recommendation:** select on authoritative objects, disclose the selected set,
and never let the pack carry anything but the disclosed view:

```
authoritative artifacts
      ↓  deterministic, role-specific selection   (on authoritative fields)
      ↓  Disclosure Authority projection          (deny-by-default, per role)
   AgentContextPack  ── the disclosed view is the ONLY thing it holds
      ↓  role-specific prompt assembly
   one model invocation
      ↓  discard
```

This is Option A **with a hard constraint on what the pack may contain**: a pack
that holds an authoritative object "for convenience" is Option A with a hole in
it. The invariant the brief names is the acceptance test:

> `AgentContextPack` must never become a container that allows raw ungoverned
> evidence/calculation objects to bypass Disclosure Authority.

Two supporting facts make this cheap rather than speculative. The precedent
already exists in B3: `trusted_v2_generation.py` selects (`_bound_items`), then
projects (`:297`, `:309`), then assembles (`:326`) — in that order, and passing
`last_disclosed_fields` out by name. And the denied-by-default structure means a
pack field that no profile names is invisible rather than leaked, so the compiler
cannot accidentally widen exposure by adding a field.

**What this does not decide:** whether the pack holds the disclosed dicts or
references to them. That is a 3B question (§17 Q3).

---

## 12 · `AgentContextPack` — negative requirements (frozen now)

Frozen as stated in the brief. These are constraints, not aspirations, and 3B
inherits them.

`AgentContextPack` is **NOT**:

- persistent runtime state
- an `ArtifactStore`
- a memory database
- a replacement for `EvidenceBinding`
- a replacement for `ClaimProvenance`
- a replacement for `CalculationResult`
- serialized `RunState`
- a model-generated plan
- a durable source of truth

It **is**: `temporary per-inference governed context projection`. Its lifetime
ends with the model call / trace required for that invocation.

**One correction to the existing plan doc.** `nf-v3-h2a-context-artifact-runtime-plan.md`
Step 5 sketches a pack containing *"goal, current phase, plan summary,
verified/missing slots, evidence refs, calculation refs, recent observations,
failure state, available capabilities, remaining budget, dialogue summary."*
That list is a **narrowed `RunState`** — precisely the shape this brief's second
frozen principle rejects ("each role samples the fields it wants" + we have
reinvented `RunState`). It should be treated as superseded by §13 below, and the
plan doc's `BoundFact` references (`:151`, `:188`, `:343`) are stale — that type
was deleted in H2A-2D-4. Documentation drift, recorded, not corrected here.

---

## 13 · Minimum conceptual shape (proposed, not implemented)

Only categories proven necessary by a live boundary are included. **No class is
defined in 3A.**

| Category | Necessary because | Present in |
| --- | --- | --- |
| `role` | every projection is role-specific; four profiles already prove this | all |
| invocation / task identity | traceability (§10); joins the existing `request_id` / `candidate_generation_id` | all |
| governed instruction payload | every boundary has a system prompt + schema | B1–B7 |
| governed conversation payload | B5, B6 | B5, B6 |
| selected governed evidence projections | B2, B3, B4 | B2–B4 |
| governed calculation projection | B3 | B3 |
| reference metadata for output validation | B2 (`F01..Fn` handles), B3 (`[E#]`/`[C#]` citation vocabulary) | B2, B3 |
| context-budget accounting | §6 C1/C2; needed for H2A-4 | prospective |

**Explicitly excluded** (proven unnecessary by the audit):

- anything B1 does not need — the Supervisor gets the question and nothing else,
  and that is correct;
- **route / retry reason / repair reason / budget state** — no live boundary
  receives any of them;
- `raw EvidencePacket` objects — no role-specific raw-source contract requires
  them at a *generation* boundary, and Disclosure Authority has approved the
  projected representation instead. B7's raw contract is an *ingestion* input,
  not a pack field.

**The role-specificity is not a nicety — it is the whole design.** A single pack
type with every category, sampled per role, is `RunState` with fewer fields. The
correct shape is **one deterministic projection per role**, each with its own
field list, sharing the Disclosure Authority and nothing else.

---

## 14 · Measurement baseline for H2A-4 (surfaces prepared, ablation not run)

The comparison must support these; the audit records where each is already
available and where it must be built.

| Metric | Available today? |
| --- | --- |
| task success | yes — readiness/benchmark harness |
| grounded success | yes |
| false release / binding | yes |
| no-answer correctness | yes |
| calculation correctness | yes |
| **average / P50 / P95 / max input tokens** | **only B3 is exact.** B1/B2 have provider-reported true tokens after the fact; B4's number is `cl100k_base` over a nanochat prompt; B5–B7/B9 have none |
| model-visible evidence count | yes — `bound_evidence_ids`, `last_disclosed_fields` |
| retrieved evidence count | yes — `retrieval_rounds` |
| **raw-source exposure ratio** | **must be built**, and must not use character estimates |
| context-build latency | partially — `BinderCallMetadata.latency_ms`, specialist `latency_seconds`; no compiler-stage timing exists |
| end-to-end latency | yes — `FinancialQueryResult.latency_metadata` |

```
Raw Source Exposure Ratio =
    raw source tokens exposed to model
  ─────────────────────────────────────
    raw source tokens retrieved
```

**This ratio cannot be computed today for any boundary except B3**, because
"raw source tokens" requires a real tokenizer at both ends and five boundaries do
not have one (§7). It is not computed in 3A, and per the brief it must not be
estimated from characters when it is.

---

## 15 · Findings register

| # | Finding | Severity | Class | Sealed contract touched? |
| --- | --- | --- | --- | --- |
| **F1** | B4 measures its 1100-token budget with `tiktoken cl100k_base` for a **nanochat** model; `len/3` fallback if absent | **high** — every V1 context budget is an estimate for the wrong tokenizer | B4 | no |
| **F2** | `ContextBudgetManager.max_tokens` / `summary_max_tokens` are assigned and **never read**; two live budgets use incompatible units | medium — operator-facing config that does nothing | B6 | no |
| **F3** | `V2ExecutionRequest.runtime_budget` declared, round-tripped, never populated or read | low — dead carriage | — | no |
| **F4** | `APIProvider` omits `max_retries`, so the OpenAI SDK default (**2**) applies, while every other V2 provider pins `max_retries=0`; `TransportRetryPolicy` asserts `sdk_max_retries == 0` for the binder only | medium — undeclared retry budget on the supervisor path | B1 | no |
| **F5** | Specialist prompt renders `Source: {doc}:1` **always** — `page` is absent from `_SPECIALIST_FIELDS`, so `ev.get("page") or 1` fires on every call. A **fabricated provenance field** shown to a model | **high** — model-visible false statement | B3 | no (profile unchanged; the prompt is what asserts it) |
| **F6** | Ingest model output is written into the index **unvalidated** and later permitted into the V1 answer prompt; prompt rule "Do NOT change numeric values" is unenforced | **high** — trust-risk: one model's output becomes another's evidence | B7→B4 | no |
| **F7** | `BINDER` is projected by a second extractor (`build_runtime_binder_fact_view`) rather than `project()`; four evaluation-only binder providers bypass the authority entirely | low — policy shared, extractor duplicated | B2 | no |
| **F8** | B5/B6/B8/B9/B10 carry raw or conversation context with **no role contract**; B8/B10 consume raw document text outside any disclosure profile | medium — the brief's rule, applied to non-generation models | B5–B10 | no |
| **F9** | The plan doc's Step-5 `AgentContextPack` sketch is a narrowed `RunState`, and three `BoundFact` references are stale (type deleted in H2A-2D-4) | low — documentation drift | — | no |

**F5 and F6 are the two that are not "audit bookkeeping"** — both are
model-visible trust defects on live paths. Neither is fixed in 3A: F5 is a
disclosure-profile decision (adding `page` to `_SPECIALIST_FIELDS` widens model
exposure, which H2A-1 explicitly declined to do without the provenance question
settled), and F6 is an ingestion-validation decision. Both belong in a 3B
decision, listed in §17.

**No sealed H2A-2 contract regresses.** No production file was modified in this
phase; the only write is this document.

---

## 16 · NOT DETERMINED

Recorded rather than guessed, per the brief.

1. **The tokenizer for B1, B2, B5, B6, B7, B9.** Not available in-process. B1/B2
   report true input tokens post-hoc; B5–B7 and B9 have no mapping at all.
2. **Whether the served nanochat endpoint's tokenizer can be exposed to the
   backend.** B3 reaches it only because the checkpoint is loaded in-process. The
   V1 answer path talks HTTP to an adapter at `LLM_API_BASE_URL`; whether that
   adapter can report prompt tokens like the Bailian providers do is unknown.
3. **The real input-token cost of B2's fact view.** `_facts_for_binding` bounds
   *relevance* but not size; the number of fact views per request was not
   measured, and without a tokenizer it cannot be converted to tokens.
4. **Whether `NVIDIA_API_KEY` is set in any deployed environment.** B7 is
   skipped entirely when it is not (`process_tables.py:140-142`), so the widest
   boundary may not run at all in production. Not resolvable from the repository.
5. **Whether `BgeV2M3Reranker` or `CrossEncoderReranker` is ever selected in
   deployment.** The default is `heuristic`; the model-backed paths are
   config-reachable but their use is unknown.
6. **Whether the V1 rollback runtime is exercised in production.** Default is
   `v2`; `v1`/`shadow` are reachable by env only.

---

## 17 · Questions requiring a 3B design decision

Ordered by what blocks what. Each is a decision H2A-3A deliberately did not make.

**Q1 — Does the compiler own selection, or only assembly?**
`_facts_for_binding` already performs deterministic, role-specific selection for
one boundary. Does 3B generalize it into a compiler, or leave it where it is and
give the compiler only the *assembly* step? Generalizing is cleaner; leaving it
avoids moving a live behaviour that H2A-2's readiness numbers depend on.

**Q2 — What is the compiler's first boundary?**
B3 (Specialist) is the only fully-governed, exactly-tokenized, in-process
boundary, and the only one where H2A-4's raw-source-exposure ratio is computable
today. B2 (Binder) is the highest-volume and has post-hoc true tokens. Choosing
B3 makes H2A-4 measurable sooner; choosing B2 makes it more useful. **Not both
at once** — a compiler that lands on two boundaries at once cannot be attributed.

**Q3 — Does `AgentContextPack` hold disclosed dicts, or references to them?**
§11 recommends select-then-disclose. Holding dicts is simple and makes the pack
self-contained; holding references keeps the pack small but puts a lookup back
between the pack and the prompt, which is where ungoverned objects traditionally
leak. The invariant in §11 holds either way only if the reference is to the
*disclosed* view, never to the authoritative object.

**Q4 — Is F5 fixed by widening the profile, or by removing the assertion?**
The specialist prompt asserts a page it was never given. Widening
`_SPECIALIST_FIELDS` to include `page` makes the assertion true and expands model
exposure; removing the fallback makes the prompt honest and loses a line the
model has always seen. H2A-1 deliberately deferred this to "when it survives
[provenance lineage]". **It now survives** (H2A-2D-2B made `EvidencePacketV1.page`
the authority). The deferral condition is met; the decision is 3B's.

**Q5 — Does a disclosure authority extend to B8 (embedding) and B10 (MinerU)?**
Both consume raw document text with no role contract, and neither handles an
*evidence artifact* — they run before one exists. Extending the profiles to
non-generation models would widen the authority's meaning; not extending it
leaves §4's rule unmet for two boundaries. This is a scoping decision about what
"Disclosure Authority" is *for*.

**Q6 — Does `ContextBudget` become a type, or a discipline?**
§6 found five bounds in three units and no owner. A `ContextBudget` type is the
obvious answer, but it must reference a real tokenizer per boundary (§7), and six
of ten boundaries have none. A type that carries estimates would be worse than
the current explicit fragmentation. Possibly the honest 3B answer is: *no type
until the tokenizer mapping exists, and C2's inert numbers deleted or wired.*

**Q7 — Is the third duplication policy needed?**
`_V1_ANSWER_FIELDS` permits `metadata` as a governed container
(`NESTED_FIELD_POLICY`). §5's select-then-disclose ordering would have the
compiler select on authoritative fields and then project — which for V1 means
`_merge_parent_context_chunks` still runs after projection, synthesizing content.
That is governed today, but a compiler that assumes "project → pack → done" would
silently break it. Deciding whether synthesis is a compiler responsibility or
stays a boundary responsibility is a 3B question.

**Q8 — Does the compiler trace its own selection?**
§10 established that every "why was this included" answer already exists, and
that nothing joins them at the invocation. Whether the pack carries that join
(adding a field) or 3B leaves it to the existing traces is a decision, not an
implementation detail — it determines whether `AgentContextPack` has a
`provenance` member.

---

## Exit gate

| # | Requirement | Status |
| --- | --- | --- |
| 1 | every production model boundary enumerated | ✅ §1 (B1–B10; eval-only providers named and excluded) |
| 2 | every model-visible context source has an authority owner | ✅ §2, §3, §4 (owners found; three gaps recorded as F8, not papered over) |
| 3 | every evidence/calculation path has a disclosure classification | ✅ §3 |
| 4 | all raw-source model exposure explicitly classified | ✅ §4 |
| 5 | whole-state serialization paths known | ✅ §5 — **none reaches a model** |
| 6 | `RunBudget` and `ContextBudget` separated | ✅ §6 |
| 7 | tokenizer availability known per boundary | ✅ §7 (six boundaries NOT DETERMINED, stated as such) |
| 8 | deterministic ordering/selection known | ✅ §8 — all deterministic |
| 9 | duplicated model-visible information catalogued | ✅ §9 |
| 10 | minimum `AgentContextPack` role proposed, not durable | ✅ §12, §13 |
| 11 | unresolved questions marked NOT DETERMINED | ✅ §16 |
| 12 | no sealed H2A-2 contract regresses | ✅ §15 — 0 production diff |

**Stop after H2A-3A.** H2A-3B is not started; `AgentContextPack`, `ArtifactStore`
and `ContextBudget` do not exist. H2A-4 ablation not started.
