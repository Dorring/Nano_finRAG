# I8 Query and Stream Conversation Semantics

I8 unifies the business lifecycle behind the two public query transports.

## Shared path

Both endpoints now call QueryLifecycleService.execute_user_turn():

HTTP request
  -> request normalization and auth
  -> QueryLifecycleService
       -> SessionManager raw history
       -> SQLite ConversationState
       -> off / shadow / on resolution
       -> clarification gate or V1 runtime
       -> final assistant/session/provenance commit
  -> transport serializer
       -> QueryResponse
       -> validated-final SSE

QueryLifecycleService has no FastAPI or SSE dependency. It returns a
UserTurnExecutionResult. /query maps that result to the existing QueryResponse;
/query/stream emits the same final result as one token event followed by one
done event.

## Transport and trust boundaries

The stream endpoint is deliberately not token-level generation. V1 retrieval,
generation, calculation, and validation complete before the SSE response is
released. No generate_stream path was added.

Conversation mode semantics are shared:

- off: no Conversation resolver; legacy V1 semantics.
- shadow: resolver and SQLite state observation run, but the official query
  remains the original query.
- on: a successful resolution supplies standalone_query with
  query_as_resolved=true; legacy query rewriting is bypassed. Ambiguity stops
  financial execution and is serialized as a control result.

The default remains MULTITURN_CONTEXT_MODE=off. Session raw messages remain
owned by SessionManager; structured state and provenance metadata remain owned
by SQLiteConversationStateStore. Assistant text is never parsed into evidence
or calculator operands.

Request idempotency and SQLite state-version/CAS behavior are shared by both
transports, so a request cannot duplicate raw turns, clarification state, or
provenance when retried through either endpoint.

## Release status

This seal means Conversation semantics are available consistently on /query
and /query/stream. The Financial Runtime is still V1 and the production
default remains off. Trusted Runtime V2 is not integrated.
