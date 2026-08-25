# I7 Conversation Lifecycle & Provenance Hardening

Status: 'CONVERSATION_LIFECYCLE_HARDENED'

Production financial runtime remains V1. The default
'MULTITURN_CONTEXT_MODE' remains 'off'. This phase changes only the
conversation state commit boundary; it does not change resolver algorithm,
financial retrieval, generation, validation, '/query/stream', or V2.

## Authority boundaries

~~~text
Raw dialogue messages
    = SessionManager

Structured semantic dialogue state
    = SQLiteConversationStateStore

Financial evidence, facts, calculations, and release status
    = FinancialQueryResult from the V1 runtime
~~~

Assistant text is never parsed into evidence, a numeric fact, or a calculator
operand. 'referenced_evidence_ids', 'last_assistant_provenance', and
'pending_clarification' are metadata/state coordinates only; they are not an
evidence cache and do not bypass later retrieval or binding.

## Logical turn lifecycle

~~~text
load prior raw turns
        |
resolve current turn
        |
tentative semantic update
        |
  +-----+------------------+
  |                        |
clarification          financial runtime
  |                        |
final control outcome  validated V1 result
  |                        |
persist raw user +     persist raw user +
assistant control      assistant answer
  |                        |
commit control state   commit structured provenance
~~~

'turn_count' means supported user-turn count. A clarification is one user turn;
the assistant clarification does not increment it. A normal successful logical
turn has a tentative semantic state write followed by one final outcome write,
so its SQLite 'state_version' advances twice. A duplicate request never advances
the state a second time.

## Pending clarification

Ambiguous active requests persist only:

- reason codes;
- semantic candidates;
- unresolved field names;
- source turn, entity, period, and topic coordinates.

Candidate values, assistant numeric text, and calculator operands are forbidden.
The next explicit choice can resolve the pending metric at the lifecycle
boundary without changing 'ContextualQueryResolver'. An explicit topic switch
continues through the existing resolver precedence and clears stale pending
clarification state.

Clarification is a user-visible control outcome:

~~~text
Runtime calls = 0
status = CLARIFICATION_REQUIRED
release_status = NOT_APPLICABLE
structured provenance = empty
~~~

The raw clarification assistant message is committed after the final control
response is known, so the next turn sees exactly what the user saw. Clearing or
deleting a session removes pending clarification and provenance with the
structured state row.

## Structured provenance

After V1 has produced its final result, '/query' persists:

~~~text
assistant_turn_id
evidence_ids
citation_ids
calculation_ids
release_status
outcome
~~~

These fields come only from the typed 'FinancialQueryResult'. Empty lists are
valid when the legacy V1 result has no structured identifier. No answer text
or source display string is parsed to manufacture provenance.

## Idempotency and conflict behavior

'/query' accepts the standard 'X-Request-ID' header. For a
'(user_id, session_id, request_id)' replay with the same original query:

- raw user/assistant messages are not duplicated;
- 'turn_count' is not incremented;
- structured provenance is not duplicated;
- the financial runtime is not silently replaced by a response cache.

Reusing a request ID with a different original query raises an explicit
conversation-state conflict. Final state writes use the SQLite expected
'state_version' CAS path, so an observed concurrent update cannot be silently
overwritten.

## Mode behavior

~~~text
off
    no Conversation state participation; I6 V1 behavior is preserved

shadow
    resolver and SQLite state run best-effort; official V1 input/response
    remains unchanged

on
    active resolution, clarification gate, V1 execution, and lifecycle commit
~~~

The default is still 'off'. '/query/stream' remains on the legacy V1 path and
was not modified in I7. V2, Supervisor, Binder, and R4 remain unintegrated.

## Verification

The I7 focused suite covers:

- normal answer, final assistant commit, and structured provenance;
- clarification with zero financial calls;
- clarification follow-up and pending-state clearing;
- topic switch cancellation;
- duplicate request IDs and conflicting request reuse;
- process-restart state continuity;
- session clear/delete cleanup;
- user/session isolation;
- SQLite CAS conflict behavior;
- assistant-text-not-evidence trust boundary;
- off/shadow/active I1-I6 regressions.

Component context evaluation remains named 'COMPONENT_CONTEXT_EVAL' (137/140);
it is not a full production E2E score.
