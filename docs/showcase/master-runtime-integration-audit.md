# Master Runtime Integration Audit - I0

Audit date: 2026-08-24
Baseline: master @ 2f64c918dd63b4fa8c18faef98a0cb97e8d04369
Audit branch: codex/master-runtime-integration-i0
Scope: read-only. No runtime changes, V2 wiring, benchmark calls, or production switch.

## Executive findings

- Formal master production path remains V1: /query -> RAGEngine -> RAGOrchestrator.
- /query/stream calls the same engine and V1 validator, but duplicates endpoint lifecycle code and emits the completed answer as one SSE token. It is not token streaming.
- ConversationContextManager and all listed V2/R4 components are not imported by the master HTTP path.
- 137/140 is COMPONENT_CONTEXT_EVAL from a standalone runner, not /query E2E and not production Runtime V2.
- SessionManager is SQLite short-term message history. user_id is the current tenant boundary. DialogueState is not persisted to SQLite.
- The only formal rewrite is RAGOrchestrator -> LLMGateway -> QueryProcessor. An explicit resolved-query bypass is required before advanced resolver integration.
- V1 has chunk_id and calculation evidence_chunk_id in parts of the response, but no unified evidence_ids/release_status/runtime_version contract.
- I0 recommendation: Runtime Port + V1 adapter + SQLite ConversationStateStore + one shared query execution service before any V2 connection.

## 1. /query actual call chain

~~~text
POST /query
 -> auth: get_current_user
 -> validate session_id
 -> SessionManager.get_recent_messages(session_id, current_user.id)
 -> UserMemoryStore.get_profile(current_user.id)
 -> resolve authorized document names
 -> get_rag_engine (lazy RAGEngine)
    -> SqliteBM25Retriever + vector query_collection
    -> RetrievalPipeline + RRF + optional V1 reranker
    -> QueryProcessor, ContextBuilder, sufficiency evaluator
    -> CalculationPipeline (default on)
    -> GroundedValidationPipeline (default on)
    -> RAGOrchestrator
 -> RAGEngine.query -> QueryRequest -> RAGOrchestrator.answer
    -> legacy rewrite when history is present
    -> intent/document scope -> retrieval -> context
    -> deterministic calculation/answer or LLM generation
    -> answerability -> post-generation validation/repair -> trace
    -> AnswerResult -> to_legacy_dict
 -> SessionManager user/assistant commits
 -> QueryResponse
~~~

Evidence: main.py:955-1028; services/rag_engine.py:464-489; application/rag_orchestrator.py:157-190, 432-635.

## 2. /query/stream actual call chain

~~~text
POST /query/stream
 -> auth/document scope
 -> StreamingResponse(generate)
 -> load RAGEngine, session history and memory profile
 -> await engine.query (same complete V1 path)
 -> commit user and assistant messages
 -> emit one token event containing the complete answer
 -> emit done event (sources, validation, trace, calculations)
~~~

Evidence: main.py:1037-1170. LLMGateway.generate_stream() exists at generation/llm_gateway.py:87-114 but is unused. Since engine.query returns only after validation, current stream does not release substantive answer before validation; it is nevertheless buffered full execution, not incremental generation.

## 3. Legacy query rewrite

~~~text
RAGOrchestrator.answer
 -> LLMGateway.rewrite_query
 -> QueryProcessor.rewrite
~~~

Files: retrieval/query_processor.py:162-264; generation/llm_gateway.py:41-58; call site application/rag_orchestrator.py:184-190.

Rewrite requires at least two history messages and follow-up markers; it uses up to four recent user/assistant messages plus preference memory. Errors fall back to the original question. MULTITURN_CONTEXT_ENABLED is documented only; master code does not read it.

## 4. SQLite SessionManager

File: services/session_manager.py.

Table conversations: id, session_id, user_id, role, content, metadata_json, created_at. Index: (session_id, user_id, created_at). schema_version is 2. Connections are thread-local SQLite with WAL and busy_timeout=5000.

Lifecycle:

- no explicit create-session row; first message creates it;
- default retention is eight Q/A pairs, pruned after add;
- SESSION_TTL_SECONDS defaults to zero; enabled cleanup is opportunistic;
- clear-one/all, history and summaries are main.py:1174-1227;
- user and assistant messages are separate commits after runtime;
- no session lock or transaction around resolution -> runtime -> turn commit;
- all current message operations filter user_id, so user_id acts as tenant.

Advanced state gap:

- conversation/store.py has only ConversationStateStore and InMemoryConversationStore;
- ConversationContextManager defaults to in-memory LRU and loses state on process restart;
- store API takes only conversation_id, no tenant_id;
- DialogueState supports entity/metric/period/topic, turns, compression and referenced IDs (conversation/contracts.py:68-114), but no SQLite table or lifecycle cascade exists.

## 5. ConversationContextManager insertion point

The direct points are before engine.query(): /query main.py:965-980 and /query/stream main.py:1062-1073. Do not integrate twice. Add one shared QueryExecutionService:

~~~text
auth + tenant/session scope
 -> SQLite conversation state
 -> ConversationContextManager.process_user_turn
 -> clarification/out-of-scope gate
 -> standalone_query -> FinancialRuntimeAdapter
 -> runtime result
 -> assistant commit with verified provenance
 -> HTTP/SSE serializer
~~~

The context layer resolves the user question only; it does not own retrieval, evidence truth, calculation or release authority. process_user_turn is synchronous and BailianClient uses blocking urllib, so an async boundary is required.

## 6. Double-rewrite blocking

Legacy rewrite is at RAGOrchestrator.answer line 184 and has no resolved flag. Add a request field such as query_as_resolved:

~~~text
original_query: What about last year?
standalone_query: Apple FY2023 Revenue
query_as_resolved: true
legacy rewrite: MUST BYPASS
~~~

Put the authoritative gate in QueryRequest/RAGOrchestrator or LegacyRewritePolicy, not in an implicit empty-history convention. Shadow resolver output may be recorded but must not alter V1.

## 7. Clarification contract

QueryResponse at models/schemas.py:139-183 requires answer: str and has no status/clarification. Recommend additive fields:

~~~text
status: ANSWER | CLARIFICATION_REQUIRED | OUT_OF_SCOPE | FAIL_CLOSED | ERROR
clarification:
  question: string
  reason_codes: string[]
  options: string[] (optional)
~~~

Normal answers preserve answer; control branches may use answer=null. SSE clarification is a final control/done event with no substantive token.

## 8. Structured Evidence IDs

Partial support:

- SourceInfo.chunk_id: models/schemas.py:55-62;
- retrieved_chunks has compact chunk identity;
- calculation operands expose evidence_chunk_id: models/schemas.py:64-73;
- internal EvidenceItem/validation structures carry IDs.

Missing:

- no top-level QueryResponse evidence_ids/citation_ids/calculation_ids;
- no unified AnswerResult provenance fields;
- _assistant_session_metadata stores sources/diagnostics only;
- record_assistant_turn accepts referenced_evidence_ids but /query and /query/stream never call it.

Only verified structured IDs may enter conversation state. Assistant text must never become a fact or calculator operand; absent IDs remain empty.

## 9. Streaming trust boundary

Current V1 runs answerability, deterministic calculation or generation, post-generation validation, and repair before main.py emits the complete answer at lines 1116-1121. Direct use of generate_stream would bypass post-generation validation. V2 must use buffered validation or control-event-first/final-answer-only streaming.

## 10. V1/V2 adapter seam

No FinancialQARuntime, LegacyFinancialRuntimeAdapter, runtime router or FINANCIAL_RUNTIME_MODE production entry was found.

Recommended:

~~~text
src/application/runtime_contract.py
  FinancialQueryRequest, FinancialQueryResult, RuntimeStatus/ReleaseStatus
src/application/runtime_adapters.py
  LegacyFinancialRuntimeAdapter -> RAGEngine.query/AnswerResult
src/application/runtime_router.py
  v1 | v2_shadow | v2_canary | v2
src/application/query_execution_service.py
  shared /query and /query/stream lifecycle
~~~

main.py should depend only on the application service and serializers. TrustedRAGRuntimeV2 currently expects SupervisorPlan plus trusted evidence packet, so it is not a drop-in HTTP replacement.

## 11. Component status

| Component | Master location | Status |
|---|---|---|
| RAGEngine/RAGOrchestrator | src/services, src/application | PRODUCTION_USED |
| SessionManager | src/services/session_manager.py | PRODUCTION_USED |
| QueryProcessor legacy rewrite | src/retrieval + LLMGateway | PRODUCTION_USED |
| Calculation/validation/repair | src/finance, src/validation | PRODUCTION_USED |
| V1 reranker | src/services/reranker.py (default heuristic) | PRODUCTION_USED |
| ConversationContextManager | src/conversation/service.py; eval runner only | COMPONENT_ONLY / EVALUATION_ONLY |
| ConversationStateStore | src/conversation/store.py; InMemory only | COMPONENT_ONLY; SQLite missing |
| Supervisor/Plan | rag_v2/supervisor, rag_v2/contracts | COMPONENT_ONLY / EVALUATION_ONLY |
| Bounded loop/Replan/Consistency | rag_v2/adaptive | COMPONENT_ONLY / EVALUATION_ONLY |
| Semantic Evidence Binder | rag_v2/evidence | COMPONENT_ONLY / EVALUATION_ONLY |
| R4 Slot-Aware Retrieval | src/pdf_retrieval_v4 | EXPERIMENT_ONLY / EVALUATION_ONLY |
| GeneratorRoutingPolicy/FinancialView | rag_v2/runtime, rag_v2/generation | COMPONENT_ONLY / EVALUATION_ONLY |
| SemanticClaimVerifier | rag_v2/runtime/semantic_claims.py | COMPONENT_ONLY / EVALUATION_ONLY |
| RuntimeValidatorChain | src/generation/runtime_validator_chain.py | EVALUATION_ONLY |
| Qwen3-Reranker-4B | src/pdf_retrieval_v4/qwen3_reranker*.py | EVALUATION_ONLY |
| TrustedRAGRuntimeV2 | rag_v2/runtime/runtime.py | EVALUATION_ONLY; no HTTP adapter |

137/140 comes from docs/showcase/multiturn-context-evaluation.md and scripts/evaluation/run_multiturn_context_eval.py, which directly instantiates ConversationContextManager and does not start FastAPI.

## 12. Expected first-stage files

I1-I2: add src/application/runtime_contract.py, runtime_adapters.py and runtime_router.py; map AnswerResult; make main.py depend on application service.

I3: add src/conversation/sqlite_store.py; extend SessionManager migration or add a state table in the same DB; persist schema version and (tenant_id, session_id); cascade clear/delete/expiry.

I4-I5: add src/application/query_execution_service.py; extend models/schemas.py and conversation contracts with status/clarification; keep release authority outside Conversation Layer.

I6-I8: keep main.py as serializer; extend services/streaming.py for buffered/control events; add evidence_ids/citation_ids/calculation_ids to domain/runtime result; commit verified provenance only.

I9-I10 tests: /query multi-turn, /query/stream parity, process restart, tenant/session isolation, clear/delete/expiry cascade, double-rewrite bypass, clarification, assistant-text-not-evidence, off/shadow no-impact. Keep 137/140 separate as component evaluation.

## Risks

1. Experimental worktree/process override must not be treated as master production evidence.
2. InMemoryConversationStore fails restart acceptance.
3. conversation_id-only state risks cross-tenant pollution; use (tenant_id, session_id).
4. Resolver output will be rewritten again unless bypass is explicit.
5. Blocking Bailian urllib can block the async event loop.
6. Separate message commits can interleave under concurrency.
7. Duplicated endpoint lifecycle code can drift.
8. Required answer: str makes clarification a compatibility change.
9. chunk_id is not automatically a V2 canonical evidence family.
10. Assistant text is history/UI only, never evidence.
11. Direct generate_stream would bypass validation.
12. MULTITURN_CONTEXT_ENABLED is documented but not read by master.
13. TrustedRAGRuntimeV2 is not a drop-in HTTP adapter.
14. FastAPI app version is 3.0.0 while root / returns 2.0.0; this is API metadata cleanup, not V2 evidence.
15. Context resolution adds latency; retain bounded timeout/fallback/fail-closed behavior.

## I0 gate

- master baseline audited: PASS
- /query call chain: PASS
- /query/stream call chain: PASS
- SessionManager schema/lifecycle: PASS
- V2 entrypoints/status: PASS
- ConversationContextManager in HTTP path: NO
- SQLite ConversationStateStore: NO
- unified FinancialQARuntime Port: NO
- V2 production switch: NO
- runtime behavior changed in I0: NO

I0 is ready to hand off to I1 after review. This audit does not authorize Multi-turn Production Integration Ready, MASTER_RUNTIME_V2_READY, or a production switch.
