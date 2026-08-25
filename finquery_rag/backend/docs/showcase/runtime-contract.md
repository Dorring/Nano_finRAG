# Unified Financial QA Runtime Contract

I1 introduces a stable port between the HTTP/conversation layer and future
financial runtime implementations.

~~~text
HTTP/API layer
      |
FinancialQARuntime
      |
+-----+----------------+
|                      |
V1 adapter             V2 trusted adapter
~~~

This commit defines the port only. The production /query and /query/stream
paths remain unchanged and still execute the existing V1 runtime.

## Request boundary

FinancialQueryRequest preserves the current production identity boundary:

- user_id and session_id identify the session.
- original_query is the immutable user wording.
- standalone_query is the query passed to the financial runtime and defaults
  to original_query.
- query_as_resolved is the explicit Conversation-to-financial-runtime rewrite
  gate. It remains false in off/shadow modes; active mode sets it true only
  after a successful contextual resolution.
- Conversation and request metadata are extension maps. No required
  tenant_id is introduced.

## Result boundary

FinancialQueryResult separates:

- RuntimeStatus: ANSWER, CLARIFICATION_REQUIRED, OUT_OF_SCOPE, FAIL_CLOSED,
  or ERROR.
- ReleaseStatus: RELEASED, NOT_RELEASED, or NOT_APPLICABLE.
- RuntimeVersion: V1 or V2.
- RuntimeRouterMode: ACTIVE, SHADOW, or CANARY.

A clarification is a structured ClarificationPayload; it is not an ordinary
answer string. Provenance fields (evidence_ids, citation_ids, and
calculation_ids) are optional empty lists in I1. They are not inferred by
parsing assistant text. Future adapters may populate them only from
structured runtime provenance.

The contract provides deterministic dictionary and JSON round trips for
request/result logging, shadow comparison, and later HTTP integration. It
does not define a second streaming runtime interface: the current
/query/stream can continue to package the validated V1 result as SSE.

## I2 legacy V1 adapter

I2 adds LegacyFinancialRuntimeAdapter as a thin implementation of
FinancialQARuntime:

~~~text
FinancialQARuntime
        ^
        |
LegacyFinancialRuntimeAdapter
        |
        v
existing RAGEngine instance
~~~

The adapter receives an already-created RAGEngine; it never constructs a
second retriever, gateway, orchestrator, validator, or calculator. Optional
V1 call arguments are read only from request_metadata:

- document_names
- n_results
- conversation_history
- memory_profile

The adapter converts the contract user_id string to the numeric user ID
required by the current V1 vector-store scope. It does not own session
loading, message persistence, or session deletion.

During I2, query_as_resolved=true failed fast with
UnsupportedResolvedQueryError. I6 replaces that temporary guard with an
explicit flag through RAGEngine into RAGOrchestrator: resolved requests bypass
the legacy rewrite, while a changed standalone_query without the flag still
fails fast. This keeps the bypass at the actual rewrite gate rather than
inferring it from conversation_history.

V1 result mapping is conservative:

- a validation status of passed maps to ANSWER and RELEASED;
- explicit answerability/calculation blocking or validation blocked/failed
  maps to FAIL_CLOSED and NOT_RELEASED;
- absent or not_applicable validation does not claim a trusted release;
- explicit out-of-scope and clarification states are not fabricated because
  the current V1 legacy result does not expose them structurally;
- sources are copied as citations, chunk_id and evidence_chunk_id are copied
  into evidence_ids, and absent citation/calculation IDs remain empty;
- engine exceptions become an explicit ERROR result without exception text.

I2 status: ADAPTER_IMPLEMENTED, NOT_PRODUCTION_ROUTED. The production
endpoints still invoke the legacy RAGEngine path directly.

## I3 /query production routing

I3 adds a narrow QueryExecutionService boundary and routes only the
non-streaming /query endpoint through the existing V1 adapter:

~~~text
/query
   |
QueryExecutionService
   |
FinancialQARuntime
   |
LegacyFinancialRuntimeAdapter
   |
the same cached RAGEngine
   |
existing V1 RAG pipeline
~~~

FINANCIAL_RUNTIME_ADAPTER_ENABLED defaults to enabled. Setting it to
false, 0, off, or no retains the original direct RAGEngine.query call as a
deterministic rollback/parity path. Both paths use the same engine instance
and the same V1 arguments. The endpoint constructs
original_query == standalone_query with query_as_resolved == false in off and
shadow modes. I6 adds an explicit on mode where ConversationResolution can
supply standalone_query and the active endpoint sends query_as_resolved=true;
the default remains off.

The adapter result is mapped back to the existing QueryResponse payload.
No public response fields, session lifecycle, validation behavior, or V1
rewrite behavior are changed. The adapter and execution service do not load
or persist sessions.

I3 status: QUERY_PRODUCTION_ROUTED for /query only. /query/stream remains
on the legacy direct V1 path in this milestone. Conversation and V2 are not
integrated.
