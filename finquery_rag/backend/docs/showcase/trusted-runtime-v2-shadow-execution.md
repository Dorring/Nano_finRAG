# TV2-06 V1/V2 Shadow Execution

- Base: TV2-05 `12728190a4174aafc70fb3ce371d77e9e1533ce8`
- Status: `TV2-06 V1_V2_SHADOW_EXECUTION = PASS`
- Production financial runtime: V1
- Default runtime mode: `v1`
- V2 production authority: off
- V2 canary: not started

## Boundary

Both HTTP transports continue to enter the same I8 lifecycle:

~~~text
/query or /query/stream
        |
        v
QueryLifecycleService
        |
        v
FinancialRuntimeRouter
        |------------------------------|
        v                              v
LegacyFinancialRuntimeAdapter      TrustedFinancialRuntimeV2
V1 primary                         V2 observation branch
        |                              |
        v                              v
official result                    V2ShadowObservation
~~~

The router is the only runtime port visible to the lifecycle. It returns the
V1 `FinancialQueryResult` unchanged. The stream transport serializes that same
official result as the existing validated-final SSE; V2 never appears in the
wire payload.

Both branches receive the same `FinancialQueryRequest` object. V2 uses
`standalone_query` as its canonical query. The V2 adapter removes raw
conversation fields such as `conversation_history`, `raw_turns`, and
`memory_profile` before the coordinator boundary. Conversation state,
SessionManager messages, and official assistant provenance remain V1/lifecycle
responsibilities.

## Modes

~~~text
FINANCIAL_RUNTIME_MODE=v1
    V1 executes once; V2 is not invoked.

FINANCIAL_RUNTIME_MODE=shadow
    V1 executes as primary and the real TV2-05 V2 factory executes in a
    bounded observation branch. V1 remains the only returned result.

FINANCIAL_RUNTIME_MODE=v2
    rejected explicitly in TV2-06.
~~~

The safe default is `v1`. Shadow mode requires an explicit
`configure_trusted_v2_shadow_runtime_builder(...)` injection that returns a
real `TrustedFinancialRuntimeV2`. There is no fake default coordinator and no
V1 fallback hidden inside the V2 branch. This keeps an incomplete deployment
from claiming that it is observing V2.

`V2_SHADOW_TIMEOUT_MS` bounds the awaited V2 branch (default: 5000 ms).
Timeouts, exceptions, invalid V2 outcomes, and V2 fail-closed decisions are
recorded as shadow observations and cannot change the V1 result or HTTP status.

## Observation and comparison

Each completed shadow attempt produces a `V2ShadowObservation` containing:

- request and identity fields;
- original and standalone queries;
- V1/V2 status and release status;
- structured evidence, citation, and calculation IDs;
- V2 route, retrieval-round count, repair count, and latency;
- localized shadow error stage/code;
- a pure `ShadowComparator` result.

The comparator is diagnostic only. It distinguishes release-decision parity,
normalized answer parity, provenance availability/mismatch, and calculation
metadata. It never parses answer text into evidence or operands, triggers repair,
falls back between runtimes, or changes either result. V1-only release versus
V2-only release is classified as review-required rather than automatically
calling either runtime correct.

The default sink logs a bounded structured record. Evaluation runs also write:

~~~text
artifacts/evaluation/tv2-06-v1-v2-shadow/
    shadow-summary.json
    outcome-matrix.json
    case-results.jsonl
    v2-error-breakdown.json
    latency-summary.json
    runtime-config.json
    environment.json
    test-results.json
~~~

Shadow output has no path to SessionManager or
`SQLiteConversationStateStore`:

~~~text
V2 shadow -> Session writes                    0
V2 shadow -> ConversationState writes         0
V1 -> official assistant/provenance commit
~~~

## Verification

The focused TV2-06 suite passed:

- 12 router/real-factory shadow tests;
- 1 endpoint smoke covering `/query` and `/query/stream` in `v1` and
  `shadow`, with identical official JSON/SSE output;
- 90 existing TV2/runtime regression tests.

The endpoint smoke uses a test-only offline embedding constructor because the
legacy `vector_store` module currently requests HuggingFace metadata during
import when the model snapshot is incomplete. Production vector-store code
was not changed; the smoke still imports the real FastAPI app and executes both
endpoint handlers through the shared lifecycle and router.

Real TV2-05 factory smoke cases cover direct fact release, deterministic
calculation release, evidence fail-closed, and repair-once. No consumed
benchmark was rerun and no V2 algorithm was changed.

## Current status

~~~text
TV2-06 V1_V2_SHADOW_EXECUTION = PASS

V1 Official Runtime       = ACTIVE
V2 Shadow Runtime         = AVAILABLE
Default Runtime Mode      = v1
Full V2 Runtime           = REAL EXECUTABLE
V2 Production Authority   = OFF
V2 Canary                 = NOT STARTED
~~~

The next phase is TV2-07 integrated readiness evaluation. It must consume the
new disagreement artifacts and decide whether V2 is safe enough for canary; it
must not be inferred from shadow execution alone.
