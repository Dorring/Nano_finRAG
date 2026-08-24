# I5 Conversation Shadow Integration

Status: IMPLEMENTED; endpoint-level verification is pending a complete backend
test environment.

## Runtime boundary

When MULTITURN_CONTEXT_MODE=shadow, POST /query runs two independent branches:

1. ConversationShadowService reads prior messages loaded by SessionManager,
   resolves context with the existing ContextualQueryResolver and persists only
   structured DialogueState through SQLiteConversationStateStore.
2. The existing FinancialQueryRequest is constructed with the original question
   as both original_query and standalone_query and query_as_resolved=false. The
   same LegacyFinancialRuntimeAdapter and V1 RAGEngine execute the official
   response.

Shadow standalone_query, clarification, ambiguity, topic-switch metadata and
errors are observation-only. They never enter the financial runtime or the
public QueryResponse. MULTITURN_CONTEXT_MODE=on is rejected until the active
rewrite-bypass gate is implemented. The deprecated boolean variable is only a
fallback when the mode variable is absent.

POST /query/stream remains on the I4 legacy direct V1 path in I5.

## State and trust boundaries

- SessionManager remains the raw user/assistant message source of truth.
- The Shadow service passes only the history loaded before the current user
  message is committed; the current turn is therefore supplied exactly once.
- SQLiteConversationStateStore persists semantic coordinates, compressed history
  and optional provenance IDs under (user_id, session_id).
- The structured state serializer removes recent raw turns. Assistant text is
  never promoted to a financial fact or calculator operand.
- After the V1 result exists, only structured IDs from FinancialQueryResult are
  eligible for the state provenance list.
- Session clear and clear-all now delete both raw messages and structured state;
  failures are explicit lifecycle errors.

## Resolver and context policy

The existing Qwen3.6-Flash client is reused. Its default model is
qwen3.6-flash and enable_thinking=false. Fast Path requests bypass the remote
resolver. Contextual requests use the existing relevance filter and bounded
L1/L2/L3 ContextBudgetManager; the full SessionManager history is not sent
directly.

## Verification

- Shadow lifecycle tests: 10 passed.
- I1-I4 conversation, SQLite and adapter regressions: 50 passed, 5 skipped.
- Static analysis: Ruff passed.
- Python compilation: passed.
- API-level tests: not sealed in this environment. The base interpreter lacks
  the full backend dependency set (the temporary dependency check still stops
  at the missing PyMuPDF module). No production environment was modified.
- Component context evaluation remains COMPONENT_CONTEXT_EVAL=137/140; it is
  not a production /query E2E score.

The code-level invariant is that Shadow cannot alter the official request or
response. Endpoint-level response-difference measurement is pending the
complete API test environment.
