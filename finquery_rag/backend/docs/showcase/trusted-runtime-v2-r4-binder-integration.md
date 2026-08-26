# TV2-03 Real R4 Retrieval and Semantic Evidence Binder

- Base: TV2-02 3750263dd51cb9ab547014d85cc7a810743cb658
- Scope: replace only the TV2-02 retrieval and evidence-evaluation fakes
- Production behavior changed: no
- Production financial runtime: V1
- Production V2 routing: OFF
- Full backend environment verification: PASS

## Seal

TV2-03 R4_BINDER_REAL_WIRED = PASS

Supervisor + Plan       = WIRED
Bounded Runtime         = WIRED
R4 Retrieval            = REAL-WIRED
Semantic Binder         = REAL-WIRED
Evidence Recovery Loop  = REAL-WIRED
Calculator              = NOT REAL-WIRED
Generator               = NOT REAL-WIRED
Validator Release       = NOT REAL-WIRED
Production Runtime      = V1
V2 Routing              = OFF

## Canonical environment

The backend declares requires-python >=3.12. The verified worktree environment
uses Python 3.12.2 and imports NumPy, OpenAI, python-jose, onnxruntime,
transformers, StrEnum, the existing R4 modules, and the existing Semantic
Binder modules. No TV2-03 relevant test is skipped for an incomplete backend
environment.

The test fixtures are CPU-safe and deterministic. They do not call Bailian,
download a reranker, use a GPU, or rerun the consumed retrieval benchmarks.

## Actual execution graph

    FinancialQueryRequest.standalone_query
              |
              v
    BoundedTrustedV2Coordinator
              |
              +--> existing SupervisorService -> existing SupervisorPlan
              |
              +--> existing BoundedAdaptiveRAGV1
                      |
                      +--> R4RetrievalCapability
                      |       |
                      |       +--> existing CandidateDirectRetriever
                      |               (4-lane R4 Candidate RRF policy)
                      |
                      +--> SemanticEvidenceEvaluationCapability
                              |
                              +--> existing SemanticBinderService
                                      |
                                      +--> EvidenceBinding /
                                          BindingValidationResult
                                      |
                                      +--> bound slots or reason code

The TV2-03 adapters are thin boundaries. The bounded controller does not know
about BM25, dense lanes, alias expansion, structured metadata, or reranking
internals. The R4 policy remains responsible for those details. TV2-00 identified
Qwen reranking as outside the current CandidateDirectRetriever policy root, so
this gate makes no claim that a separate reranker path is production-wired.

## Candidate versus bound evidence

R4 output is candidate evidence. It is stored in the adaptive state's
evidence_packets and never becomes trusted merely because it was retrieved.

The Binder receives the SupervisorPlan's RequiredSlot objects plus the candidate
facts. Only Binder-admitted fact IDs are exposed as V2ExecutionOutcome.evidence_ids;
raw Top-K candidate IDs are never promoted to provenance. Bound citation IDs are
copied from the same admitted facts. No answer-text parsing is used.

The trace keeps the two populations separate:

    candidate_ids_per_round -> R4 candidate pool
    bound_evidence_ids      -> Binder-admitted evidence provenance

A single candidate may be returned by multiple R4 lanes. Candidate-aligned RRF
performs stable deduplication, while the Binder's slot-binding relation remains
explicit.

## Recovery and failure mapping

The existing bounded replanner remains the policy owner:

    MISSING_SLOT     -> targeted retrieval / bounded recovery
    WRONG_PERIOD     -> period-targeted retrieval
    MISSING_OPERAND  -> operand-targeted retrieval
    CONFLICT         -> controlled fail-closed

The Binder is the slot-satisfaction gate. A missing or wrong-period slot cannot
reach Calculator or Generator. Evidence conflict cannot be delegated to a
generation model.

R4 retrieval miss is a normal policy outcome and can end in
FAIL_CLOSED/NOT_RELEASED. Malformed candidate or Binder schema is a capability
error and maps to EXECUTION_ERROR/NOT_RELEASED.

Even when all slots are bound, TV2-03 stops at:

    EVIDENCE_READY
    FAIL_CLOSED / NOT_RELEASED
    reason_codes += DOWNSTREAM_EXECUTION_NOT_WIRED

Calculator, Generator, TrustedRAGRuntimeV2, and the final Validator Release Gate
remain deliberately unwired until TV2-04/05.

## Verification

The focused TV2-03 suite instantiates the real CandidateDirectRetriever and the
real SemanticBinderService around deterministic fixture dependencies. It covers:

- one-shot bound evidence and citation provenance;
- wrong-period recovery;
- missing-operand recovery with zero Calculator calls;
- candidate versus bound provenance;
- structured metadata lane / slot-crowding recovery;
- stable candidate deduplication;
- malformed R4 candidates;
- malformed Binder schema;
- unresolved Binder conflict;
- canonical environment imports.

Relevant TV2-01/02 runtime tests and focused R4/Binder contract tests also pass.
The consumed 120-case/NF-V2 benchmark is not rerun.

## Production boundary

    /query
    /query/stream
        -> QueryLifecycleService
        -> LegacyFinancialRuntimeAdapter
        -> V1

TV2-03 adds no V2 shadow call, no runtime router, no QueryLifecycleService
change, no Conversation change, and no production dependency-graph change.
