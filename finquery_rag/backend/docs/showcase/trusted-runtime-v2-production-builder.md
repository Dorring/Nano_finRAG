# Trusted Runtime V2 production builder

## Purpose

The V2 runtime is now a complete query-to-release graph, but its components
are intentionally dependency-injected.  A deployment therefore needs one
canonical constructor that turns provisioned indexes, fact materialization,
providers, and the verified local specialist into a
`TrustedFinancialRuntimeV2`.

That constructor is:

```text
src.runtime.trusted_v2_production:build_trusted_v2_runtime_for_request
```

It is a real production builder, not a test coordinator and not a V1 hybrid.
It reuses the existing TV2 factories and creates request-scoped R4 and
capability adapters over process-scoped read-only resources.

## Deployment contract

The current application keeps the runtime seam explicit.  A V2 or shadow
deployment must set:

```ini
FINANCIAL_RUNTIME_MODE=v2
TRUSTED_V2_RUNTIME_BUILDER=src.runtime.trusted_v2_production:build_trusted_v2_runtime_for_request
```

The same builder can be registered through
`configure_trusted_v2_runtime_builder()` by an embedding application.  If the
builder is missing, the application fails closed with a configuration error;
it never silently falls back to V1.

The builder requires all of the following deployment assets:

1. `TRUSTED_V2_R4_INDEX_DIR`, containing the read-only
   `candidate-metadata.sqlite` and the four R4 lanes:
   `candidate_raw_bm25`, `candidate_structured_bm25`,
   `candidate_raw_dense`, and `candidate_structured_dense`.
2. `TRUSTED_V2_FACT_STORE_PATH`, a JSON/JSONL (optionally gzip-compressed)
   structured registry keyed by candidate ID.  Every materialized record must
   contain an evidence ID, citation ID, physical source identity, and
   `provenance_complete=true`.  Candidate text is never parsed into
   provenance.
3. A configured Supervisor provider (`bailian` or `api`) and a configured
   Semantic Binder provider (`bailian`).  Provider clients are constructed
   only from deployment configuration; deterministic test providers are not
   accepted by the production builder.
4. `TRUSTED_V2_SPECIALIST_CHECKPOINT`, loadable by the existing specialist
   loader.  The loader verifies the checkpoint identity/hash before serving
   requests.

The optional `V2_MAX_*` values configure the existing bounded runtime.  They
do not create a second budget policy.

## Construction and trust boundaries

```text
FinancialQueryRequest.standalone_query
        |
        v
request-scoped CandidateDirectRetriever + R4 policy
        |
        v
candidate evidence --(structured fact store)--> Semantic Binder
        |
        v
existing Trusted V2 Supervisor / bounded loop / Calculator /
Generator Routing / Validator / Repair-Once factory
        |
        v
TrustedFinancialRuntimeV2
```

The builder deliberately ignores the legacy engine argument and does not
read raw Conversation history.  Conversation resolution remains owned by
`QueryLifecycleService`; V2 receives the already resolved standalone query.
R4 returns candidates only.  Binder-admitted evidence is the only source of
V2 evidence provenance, and the final validator remains the release authority.

Resources are cached per configuration fingerprint, while the retrieval policy
and capability wrappers are request-scoped so document scope and execution
trace cannot leak across requests.  Cache clearing is available for reloads
and tests through `clear_trusted_v2_production_cache()`.

## Preflight and failure behavior

Before enabling traffic, operators can run the cheap preflight:

```python
from src.runtime import validate_trusted_v2_production_configuration

validate_trusted_v2_production_configuration()
```

It verifies the R4 layout, fact-store provenance contract, provider settings,
and Specialist checkpoint identity without making model calls.  The first
request (or an embedding application's startup hook) then loads the actual
provider clients and Specialist model.  Any missing or invalid dependency is a
`TrustedV2ProductionConfigurationError`; there is no deterministic fallback,
V1 retriever reuse, or silent V1 downgrade.

The repository clone does not include deployment-sized R4 indexes, the
structured financial-fact registry, provider secrets, or the Specialist
checkpoint.  Therefore a fresh clone is expected to fail the V2 preflight
until those assets are provisioned.  This is an explicit deployment
requirement, not a claim that the clone is an all-in-one V2 distribution.

## Runtime modes

The existing router remains the single selection point:

```text
v1     -> LegacyFinancialRuntimeAdapter (explicit rollback)
shadow -> V1 official result + V2 observation
v2     -> TrustedFinancialRuntimeV2 official result
```

The builder does not change Conversation semantics, `/query` versus
`/query/stream` transport behavior, or the V1/V2 shadow comparator.  It only
provides the missing real V2 construction seam.  A readiness benchmark is not
implied by the existence of this builder; production correctness and quality
evaluation remain separate concerns.
