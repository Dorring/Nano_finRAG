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
- query_as_resolved is an explicit future rewrite gate. It is only data in
  I1; no legacy rewrite behavior is changed.
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
