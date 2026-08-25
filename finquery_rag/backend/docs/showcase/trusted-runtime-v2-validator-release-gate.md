# Trusted Runtime V2 - Validator and Release Gate

TV2-05 is the first V2 phase that can produce a released financial answer in
the component graph. Production API routing is unchanged: both API transports
still use the V1 runtime and V2 routing is off.

## Canonical execution boundary

~~~text
SupervisorPlan
    |
R4 candidate evidence
    |
Semantic Binder admission
    |
Calculator / Generator Routing
    |
CandidateExecutionResult
    |
TrustedReleaseValidationCapability
    +-- RuntimeGenerationValidatorV1
    +-- SemanticClaimVerifierV1 (non-simple-calculation routes)
    |
PASS -> READY_FOR_RELEASE / RELEASED
FAIL -> Repair Once when structurally eligible
        +-- full revalidation PASS -> RELEASED
        +-- fail -> FAIL_CLOSED / NOT_RELEASED
~~~

The validation packet is constructed only from Binder-admitted evidence. It does
not read raw retrieval candidates, conversation history, or assistant turns.
Evidence IDs, citation IDs, and calculation IDs remain structured metadata;
answer text is never parsed to create provenance.

TrustedRAGRuntimeV2.handle is intentionally not called from this boundary,
because that method owns generation as well as validation. Reusing it after the
TV2-04 candidate would create a non-repair double-generation path.

## Release authority

Only the final validation PASS branch in BoundedTrustedV2Coordinator creates
READY_FOR_RELEASE with RELEASED. Supervisor, retrieval, Binder, Calculator,
deterministic Renderer, and Financial Specialist return intermediate data only.

The canonical V2 validators are:

- RuntimeGenerationValidatorV1 for envelope, citations, numeric, period,
  unit/currency/scale, and calculation fidelity.
- SemanticClaimVerifierV1 for claim grounding against the admitted packet on
  fact and specialist routes.

The deterministic C1 ratio renderer uses a percent display. The V2 adapter
records this as an explicit structured ratio-to-percent display alias; it does
not weaken arbitrary unit checks.

## Two bounded loops

~~~text
Evidence recovery:
R4 -> Binder -> reason code -> targeted retrieval
bounded by tool/replan/no-progress budgets

Candidate repair:
Generator -> Validator -> Repair Once -> Validator
bounded by repair_count <= 1
~~~

Repair can only change candidate expression. It cannot add Binder-excluded
evidence or citations, alter the Supervisor plan, or change the C1 calculation
result. A repaired candidate is always fully revalidated.

## Factory and production status

build_trusted_v2_runtime requires explicit R4, Binder, Calculator, Generator,
and release-validator ports. Missing components fail fast; there is no V1
fallback and the factory is not registered with QueryLifecycleService.

~~~text
Full V2 Query -> Release Root = COMPONENT_READY
Production Financial Runtime = V1
Production V2 Routing        = OFF
/query and /query/stream     = unchanged
~~~

TV2-06 will add V1/V2 shadow execution and comparison. It is the first phase
allowed to call this complete factory from a production-adjacent path.
