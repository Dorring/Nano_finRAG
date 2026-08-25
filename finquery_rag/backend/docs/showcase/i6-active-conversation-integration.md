# I6 Active Conversation Resolution Integration

Status: **ACTIVE_CONVERSATION_QUERY_INTEGRATED = PASS**

Base: c93d8e4e522a6c2a1c76c6ed488bf12ae216ec04

This milestone makes the existing Conversation Layer an available active
/query mode while keeping the default production mode and financial runtime
at V1. It does not integrate /query/stream or Trusted Runtime V2.

## Runtime modes

MULTITURN_CONTEXT_MODE is now validated as:

- off: no Conversation invocation; existing V1 behavior.
- shadow: I5 observation-only behavior; the original query reaches V1.
- on: active resolution; a successful contextual resolution supplies
  standalone_query and sets query_as_resolved=true.

The default remains off. The deprecated boolean compatibility setting still
maps true to shadow; it cannot silently activate active mode.

## Active path

~~~text
/query
  -> ConversationContextManager
  -> ConversationResolution
       -> clarification / out-of-scope: control response, no V1 call
       -> resolved: FinancialQueryRequest(original_query, standalone_query,
          query_as_resolved=true)
  -> QueryExecutionService
  -> LegacyFinancialRuntimeAdapter
  -> the same cached RAGEngine
  -> RAGOrchestrator rewrite gate
  -> V1 retrieval, calculation, generation and validation
~~~

original_query is preserved for audit and UX. Active success passes
standalone_query to V1 and removes uncontrolled raw conversation history from
the financial invocation. query_as_resolved=true is an explicit flag, not a
claim that retrieval or evidence binding has already succeeded.

## Rewrite safety

- unresolved V1 request: legacy rewrite call count is 1 when history is
  present;
- resolved request: legacy rewrite call count is 0;
- a changed standalone query with the flag false still fails fast in the
  adapter;
- the bypass is implemented at the actual RAGOrchestrator rewrite gate, not
  through a None/empty-history side effect.

## Clarification and failure policy

Active ambiguity returns a public status=CLARIFICATION_REQUIRED payload with
a structured clarification object and does not call the financial runtime.
The compatibility answer field carries the control question for existing
clients; no evidence, citation, calculation, or release provenance is
fabricated.

- context-dependent query with no prior state: clarification;
- context-dependent resolver/state failure: clarification;
- self-contained resolver failure: safe V1 execution without raw history;
- deterministic out-of-scope query: OUT_OF_SCOPE control response;
- clarification does not advance semantic dialogue state;
- assistant text is never converted into evidence or calculator operands.

## Verification

Focused I6/I1-I5 regression:

- 56 tests passed;
- full tests/conversation suite: 51 passed;
- targeted Ruff: passed;
- Python compilation: passed;
- no benchmark or model generation was run.

Covered cases include relative-period inheritance, explicit topic switch,
cross-turn calculation, ambiguous metric, stateless/no-state contextual
queries, resolver timeout isolation, rewrite call counts, raw-history
suppression, session isolation and existing off/shadow parity.

## Production boundaries

- Production financial runtime: V1.
- Default Conversation mode: off.
- Active /query mode: available by explicit configuration, not default.
- /query/stream: unchanged legacy direct V1 path.
- V2 Supervisor/Binder/R4/Validator runtime: not integrated.
- Component result 137/140 remains a component evaluation, not integrated
  /query accuracy.
