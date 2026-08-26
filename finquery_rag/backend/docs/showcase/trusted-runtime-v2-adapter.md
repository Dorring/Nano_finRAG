# TV2-01 Trusted Financial Runtime V2 Adapter

- Base: 42943b01b4e943275b07981c55aa1d6894d1ff15
- Scope: V2 runtime boundary and coordinator shell only
- Production behavior changed: no
- Benchmark/model/GPU execution: none
- Production financial runtime: V1
- Production V2 routing: off

## Seal

TV2-01 TRUSTED_RUNTIME_V2_ADAPTER = PASS
V2 Adapter                = COMPONENT_READY
Full V2 Coordinator       = NOT WIRED
Production Financial RAG  = V1
Production V2 Routing     = OFF

TV2-01 makes a V2 implementation available as an injected
FinancialQARuntime component. It does not select that implementation in
QueryLifecycleService, and it does not claim that the V2 query-to-answer
pipeline exists.

## Why the coordinator is injected

TV2-00 established that rag_v2.runtime.runtime.TrustedRAGRuntimeV2.handle()
starts after a TrustedRAGQueryV2 already contains a Supervisor plan and a
verified evidence packet. It is a generation/validation/release component, not
the standalone-query execution root. The TV2-01 adapter therefore never calls
TrustedRAGRuntimeV2.handle() directly.

The boundary is:

FinancialQueryRequest
        |
        v
TrustedFinancialRuntimeV2
        |
        v
TrustedV2ExecutionCoordinator (Protocol)
        |
        v
V2ExecutionOutcome
        |
        v
FinancialQueryResult

TV2-02 through TV2-05 will build the real coordinator behind this protocol.
The injected fake coordinator in the tests is only a contract fixture; it is
not registered as a production fallback.

## Input contract

V2ExecutionRequest.standalone_query is the canonical V2 financial question.
original_query is retained for audit and conversation provenance. The
query_as_resolved bit is copied as conversation_resolved metadata only; it
does not control a V2 rewrite because V2 has no legacy conversational rewriter.

The V2 request does not forward raw conversation history, raw turns, messages,
or the legacy memory profile. Structured conversation_metadata remains
available for explicitly bounded metadata, while L1/L2/L3 context resolution
continues to belong to the Conversation layer.

## Output and release contract

V2ExecutionOutcome has three terminal states:

- READY_FOR_RELEASE requires release_status=RELEASED and a non-empty
  candidate answer.
- FAIL_CLOSED requires release_status=NOT_RELEASED.
- EXECUTION_ERROR requires release_status=NOT_RELEASED.

The adapter maps them to:

V2 outcome             FinancialQueryResult.status   release status
READY_FOR_RELEASE      ANSWER                        RELEASED
FAIL_CLOSED            FAIL_CLOSED                    NOT_RELEASED
EXECUTION_ERROR        ERROR                          NOT_RELEASED

A candidate answer on a FAIL_CLOSED outcome remains non-released. The
adapter never infers success from a non-empty answer.

Exceptions raised by a coordinator map to ERROR with
V2_COORDINATOR_EXCEPTION; they do not become a trust-policy refusal.
Malformed coordinator results map to ERROR with V2_OUTCOME_INVALID.

## Structured provenance

evidence_ids, citation_ids, and calculation_ids are copied only from the
structured V2ExecutionOutcome. The adapter never parses answer text,
citation text, or numbers to manufacture provenance. Empty lists are valid
until TV2-03/TV2-04 supply admitted evidence and deterministic calculation
objects.

The optional route, validator status, plan ID, evidence packet ID, and
calculation result ID are retained as runtime metadata. They do not grant
release authority.

## Current production graph

/query or /query/stream
  -> QueryLifecycleService
  -> QueryExecutionService
  -> LegacyFinancialRuntimeAdapter
  -> V1 RAGEngine

TV2-01 does not modify this graph. Conversation mode remains off by default,
and the same I8 lifecycle remains responsible for both transports.

## Future graph

QueryLifecycleService
        |
        v
FinancialQARuntime
        |
        +--> LegacyFinancialRuntimeAdapter -> V1
        |
        +--> TrustedFinancialRuntimeV2
                |
                v
        TrustedV2ExecutionCoordinator
                |
                +--> Supervisor / bounded runtime
                +--> R4 retrieval / Binder
                +--> Calculator / route preparation
                +--> VerifiedEvidencePacket
                +--> TrustedRAGRuntimeV2.handle()
                +--> validator release gate

Only the V2 adapter boundary is implemented in TV2-01. No Supervisor,
retrieval, Binder, Calculator, GeneratorRouting, or Validator production call
site is introduced by this change.

## Validation

TV2-01 uses CPU-safe fake-coordinator tests for released, fail-closed,
execution-error, malformed-result, query-selection, raw-context isolation,
provenance, serialization, and invalid status/release combinations. It does
not rerun sealed benchmarks and does not load models or use a GPU.
