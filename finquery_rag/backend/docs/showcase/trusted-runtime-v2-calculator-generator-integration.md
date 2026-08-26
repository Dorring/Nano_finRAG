# TV2-04 Deterministic Calculator and Generator Routing

- Base: TV2-03 9356ed0c1f83f34394c9b08552f41ca53a3f9d29
- Scope: evidence-ready candidate preparation only
- Status: TV2-04 CALCULATOR_GENERATOR_REAL_WIRED = PASS
- Production V2 routing: OFF
- Production financial runtime: V1
- Conversation/API/QueryLifecycleService: unchanged

## Execution boundary

TV2-04 starts after the real R4 retrieval and Semantic Evidence Binder
have admitted evidence. It replaces only the downstream fake calculation and
generation ports:

~~~text
SupervisorPlan.RequiredSlots
          |
          v
Bounded Runtime -> R4 -> Candidate Evidence
          |
          v
Semantic Evidence Binder
          |
          +--> DIRECT_FACT
          |      -> GeneratorRoutingPolicy
          |      -> DeterministicFactRenderer
          |
          +--> CALCULATION
          |      -> DeterministicCalculationCapability
          |      -> existing nine-operation registry/executor
          |      -> existing calculation renderer
          |
          +--> MULTI_EVIDENCE / qualitative
                 -> GeneratorRoutingPolicy
                 -> LocalSpecialistGenerationAdapter
          |
          v
       candidate result
          |
          v
 FINAL_VALIDATION_NOT_WIRED
 FAIL_CLOSED / NOT_RELEASED
~~~

TrustedRAGRuntimeV2.handle() is deliberately not called here. Its actual
contract includes the later generation/validation/release stage; using it as a
query entry point would bypass the staged V2 boundaries.

## Reused canonical components

| Capability | TV2-04 implementation | Authority |
| --- | --- | --- |
| Calculation | DeterministicCalculationCapability | CALCULATION_REGISTRY + execute_plan |
| Calculation rendering | existing render_calculation_result | CalculationResult / C1 contract |
| Direct fact rendering | thin structured DeterministicFactRenderer bridge | Binder-admitted packet fields only |
| Route selection | existing GeneratorRoutingPolicy | GeneratorRouteDecision |
| Specialist | LocalSpecialistGenerationAdapter | existing LocalSpecialistGenerator contract |

The direct-fact bridge is intentionally small because the existing canonical
renderer is calculation-specific. It does not retrieve, validate, or parse
answer text. The calculation path always retains deterministic arithmetic
authority; the Specialist cannot become a numeric fallback.

## Calculation contract

All nine existing CalculationOperation values remain the source of truth:

difference, growth_rate, percentage_share, sum, average, gross_margin,
net_margin, debt_ratio, and scale_conversion.

Operands are built only from Binder-admitted slot bindings. Retrieval
candidates that were not admitted are not visible to the calculator. The
stable calculation provenance identifier is a deterministic C1 identifier
derived from the operation, formula version, result, and bound operand IDs.
A missing operand, zero denominator, unsupported scale, or primitive refusal
returns a structured non-release result; it never falls back to the
Specialist.

For scale conversion, source_scale, target_scale, precision, and related
fields may be supplied through the existing structured V2 request metadata
(request_metadata.calculation or its explicit top-level keys). No scale is
inferred from an answer string.

## Candidate and provenance boundary

Every generated candidate is marked:

~~~text
candidate_status = CANDIDATE_READY_FOR_VALIDATION
validation_pending = true
status = FAIL_CLOSED
release_status = NOT_RELEASED
reason_codes includes FINAL_VALIDATION_NOT_WIRED
~~~

evidence_ids come from Binder-admitted evidence, not the raw R4 Top-K pool.
citation_ids are limited to citations on those admitted packets. Structured
calculation IDs come only from the deterministic calculator. Unknown citation
IDs returned by a Specialist are recorded as diagnostic metadata and are not
promoted to authoritative provenance. No answer text, citation text, or number
is parsed to create evidence, citation, or calculation provenance.

## Verification

The canonical backend environment is Python 3.12.2 with the locked NumPy,
OpenAI-compatible, Jose, and StrEnum dependencies. The existing Specialist
checkpoint hash matches its declared SHA256; a CPU load smoke and a
one-token generation smoke both pass. The TV2-04 unit path uses an injected
deterministic Specialist provider, so the focused suite remains reproducible.

Focused verification covers:

- 7 TV2-04 calculator/generator tests, including all nine registry operations;
- 27 TV2-02/TV2-03 coordinator, R4, and Binder regression tests;
- real R4 and real Semantic Binder objects in the integration fixtures;
- missing operand, zero denominator, candidate provenance, route selection,
  and candidate-never-released invariants.

No consumed benchmark, GPU benchmark, production router, Conversation code,
QueryLifecycleService, /query, or /query/stream path is changed by TV2-04.
