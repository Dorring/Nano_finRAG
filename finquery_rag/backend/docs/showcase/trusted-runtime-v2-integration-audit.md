# TV2-00 Trusted Financial Runtime V2 Production Integration Audit

- Base: `869cd08968f897071d38a010a07a7bd879f1b779`
- Scope: code-level production wiring and contract audit only
- Runtime behavior changed: no
- Conversation behavior changed: no
- Benchmark/model/GPU execution: none
- Production financial runtime: V1
- Default `MULTITURN_CONTEXT_MODE`: `off`

## Decision

`TV2-00_PRODUCTION_INTEGRATION_AUDIT = PASS`.

V2 components exist, but there is no production execution root that accepts a standalone query and performs Supervisor, bounded retrieval/replan, R4 retrieval, binding, calculation, verified packet construction, generation, validation, and release.

```
V2_PRODUCTION_INTEGRATION = NOT_READY
NEXT_GATE = TV2-01 TrustedFinancialRuntimeV2 Adapter
```

The closest root is `rag_v2.runtime.runtime.TrustedRAGRuntimeV2.handle()`. It is a post-evidence generation/release coordinator. It expects a prebuilt `SupervisorPlan` and a prebuilt trusted evidence packet; it does not call Supervisor, retrieval, R4, Binder, Calculator, or the bounded orchestration loop.

## 1. Current production graph

Both endpoints use the sealed I8 shared lifecycle:

```
/query or /query/stream
  -> QueryLifecycleService.execute_user_turn()
  -> QueryExecutionService.execute()
  -> injected FinancialQARuntime
  -> default LegacyFinancialRuntimeAdapter(existing RAGEngine)
  -> RAGEngine.query()
  -> RAGOrchestrator.answer()
     -> current RetrievalPipeline / BM25 / vector retrieval
     -> V1 CalculationPipeline
     -> V1 GroundedValidationPipeline + repair-once
  -> QueryResponse mapper or validated-final SSE serializer
```

`FINANCIAL_RUNTIME_ADAPTER_ENABLED` is an I3 migration flag (default true), not a V1/V2 router. The explicit false path invokes the same V1 engine directly. `/query/stream` executes the full validated result first and then emits SSE; it does not call token-level generation.

This confirms the I8 seam: replacing the injected `FinancialQARuntime` implementation can cover both endpoints without changing Conversation, SessionManager, Clarification, idempotency/CAS, or SSE business semantics.

## 2. Actual V2 root

```
TrustedRAGQueryV2
  (question + prebuilt SupervisorPlan + prebuilt trusted_evidence_packet)
    -> TrustedRAGRuntimeV2.handle()
       -> parse plan
       -> TrustedEvidenceGateV1.validate()
       -> GeneratorRoutingPolicyV1
       -> FinancialGenerationViewRendererV1
       -> TrustedGenerationStateMachineV1
          -> ProviderRegistryV1
          -> RuntimeGenerationValidatorV1
          -> optional SemanticClaimVerifierV1
          -> one primary attempt + one bounded fallback
    -> TrustedRAGResponseV2
```

The V2 generation state machine is the release authority for this component path: only `released=True` becomes `RELEASED`; all other paths become `ABSTAINED`. This is not proof of a full production runtime.

The separate `rag_v2/orchestration/state.py` state machine enforces transitions and budgets, but has no provider/retriever/binder/calculator/generator wiring. `rag_v2/adaptive/adaptive_state_machine.py` is an injected-tools loop used by tests/evaluation only.

## 3. Evaluation-only assembly

Evaluation scripts manually construct plans, select/rank frozen candidates, build evidence packets and calculation payloads, and then call `TrustedRAGRuntimeV2.handle()`. The assembly is not reachable from a normal `FinancialQueryRequest`, and must not be treated as production wiring.

Evidence checked:

- `artifacts/evaluation/nf-v2-08-r0-e2e-runtime/trusted-rag-runtime-contract.json`: input `TrustedRAGQueryV2`, output `TrustedRAGResponseV2`, retrieval calls 0.
- `artifacts/evaluation/nf-v2-08-r0-e2e-runtime/decision.json`: model/retrieval calls 0; future seam, not HTTP integration.
- `artifacts/evaluation/pdf-retrieval-v4-gate-08-r8-r3-1/acceptance.json`: `production_switch_allowed=false`.
- `artifacts/evaluation/pdf-retrieval-v4-gate-09-r5-2-r0/acceptance.json`: post-seal diagnostic; `production_switch_allowed=false`.

## 4. Component status matrix

| Component | Exists | Status | Production V2 call site | Gap |
|---|---:|---|---:|---|
| SupervisorService / SupervisorPlan | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | Request-to-plan adapter and plan-to-tool coordinator |
| StateMachine / BoundedAdaptiveRAGV1 | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | Production tool registry, replan driver, final result |
| R4 combined retrieval | yes, many `src/pdf_retrieval_v4` modules | COMPONENT_READY; EVALUATION_ONLY usage | no | No unified production R4 root; `src` does not import it |
| Alias / metadata / slot-aware retrieval | yes | EVALUATION_ONLY | no | Must be selected by V2 coordinator |
| Qwen reranker runtime | yes | EVALUATION_ONLY | no | No production call or V2 adapter |
| Semantic Binder / slotwise binders | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | Candidate-to-`VerifiedEvidencePacket` bridge |
| TrustedEvidenceGateV1 | yes | COMPONENT_READY; V2 component-only | no | Packet shape gate, not retrieval/binding |
| V1 CalculationPipeline | yes | PRODUCTION_USED by V1 | yes | Bridge to V2 `CalculationResultPacket`/C1 |
| V2 CalculationResultPacket | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | No mapping from `src.domain.calculation.CalculationResult` |
| GeneratorRoutingPolicyV1 | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | Provider IDs are configured; services are not wired |
| V1 LLM/deterministic renderer | yes | PRODUCTION_USED by V1 | yes | Not the V2 provider registry |
| LocalSpecialistGenerator | yes | COMPONENT_READY; EVALUATION_ONLY/EXPERIMENT_ONLY | no | Checkpoint/provider lifecycle absent |
| V2 validator/release state machine | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | Packet supply and API adapter absent |
| SemanticClaimVerifierV1 | yes | COMPONENT_READY; V2 component/eval only | no | No production V2 coordinator |
| V1 GroundedValidationPipeline | yes | PRODUCTION_USED by V1 | yes | Separate V1 release path |
| TrustedRAGRuntimeV2 | yes | COMPONENT_READY; EVALUATION_ONLY usage | no | Post-evidence root, not standalone-query root |
| FinancialQueryResult | yes | CONTRACT_READY | consumed by I8 | V2 adapter must populate it structurally |

No V2 component is production-wired. The production calculator and validator rows are the existing V1 path only.

## 5. Answers to the 12 audit questions

### 1. Unified execution root

No. `TrustedRAGRuntimeV2.handle()` starts after plan, retrieval, binding, and calculation preparation.

### 2. Minimum composition

```
FinancialQueryRequest.standalone_query
 -> SupervisorService.plan()
 -> bounded coordinator/state machine
 -> R4 retrieval/materialization
 -> SemanticBinderService and deterministic binding gate
 -> bounded repair or fail-closed
 -> V1 calculator when needed
 -> CalculationResultPacket/C1 bridge
 -> VerifiedEvidencePacket
 -> TrustedRAGRuntimeV2.handle()
 -> FinancialQueryResult
```

### 3. Supervisor entry

`rag_v2.supervisor.service.SupervisorService.plan(question)`. It makes one provider call, validates the plan, never retries, and never executes tools.

### 4. Bounded loop

Budget and no-progress logic exists, but no production tool registry connects it to R4, materialization, binding, or calculation.

### 5. R4 call

No. No production `src` call site imports `src.pdf_retrieval_v4`, and `TrustedRAGRuntimeV2` has no retrieval dependency.

### 6. Binder gate

The component contracts are real: `StateMachine.can_generate` requires a complete BOUND binding; `Action.BIND` rejects missing/ambiguous bindings; `TrustedEvidenceGateV1` requires VERIFIED evidence, fact IDs, citation IDs, and physical provenance. No production caller connects required slots to candidates, Binder, and packet construction.

### 7. Calculator contract

V1 returns `src.domain.calculation.CalculationResult` (status, operation, value, formula/version, operands, evidence chunk IDs). V2 expects `rag_v2.contracts.calculation.CalculationResultPacket` (status, operation/value, period/unit/scale/currency, supporting evidence IDs). No bridge exists.

### 8. Generator routing

`GeneratorRoutingPolicyV1` maps `DIRECT_FACT`, `CALCULATION`, and `MULTI_EVIDENCE` to provider IDs. It does not instantiate renderer, calculator, or Local Specialist. A TV2 adapter must make these route dependencies explicit. Local Specialist must not own retrieval, arithmetic, or release authority.

### 9. Release authority

V1 uses `RAGOrchestrator` plus `GroundedValidationPipeline` and repair-once. V2 uses `TrustedGenerationStateMachineV1` after packet gate; the generator cannot release. The missing V2 part is packet supply/coordinator wiring, not a post-generation gate.

### 10. Evidence/citation/calculation IDs

V2 packets require structured `fact_id`, `citation_id`, physical provenance, and calculation supporting IDs. `TrustedRAGResponseV2` exposes `citation_ids` only; it lacks top-level `evidence_ids` and `calculation_ids`. `FinancialGenerationViewV1` creates packet-local `E1...` and `C1` labels. The adapter must retain the verified packet/trace and map canonical IDs; it must never parse answer text.

### 11. FinancialQueryResult mapping

- released generation -> `ANSWER / RELEASED / runtime_version=V2`;
- missing evidence, incomplete binding, calculation-not-ready, validation failure, or budget exhaustion -> `FAIL_CLOSED / NOT_RELEASED`;
- proven provider/coordination infrastructure failure -> `ERROR / NOT_RELEASED`;
- clarification remains owned by Conversation before runtime;
- reasons and latency come from terminal reason, validator codes, trace, and coordinator timings.

V2 currently uses `ABSTAINED` for policy refusal and some execution failures, so TV2-01 must normalize `TerminalReason`; no answer-text inference is allowed.

### 12. Backend replacement and both endpoints

Yes, structurally. Both endpoints call `QueryLifecycleService.execute_user_turn()`, then `QueryExecutionService` calls an injected `FinancialQARuntime`. Selecting a V2 implementation at that point gives:

```
/query        -> shared lifecycle -> V2 adapter
/query/stream -> shared lifecycle -> V2 adapter -> SSE serializer
```

The V2 adapter must not bypass the shared lifecycle or create a second endpoint path.

## 6. Contract mapping

### Input

- V2 question: `FinancialQueryRequest.standalone_query`.
- `original_query`: audit/session UX only.
- `request_metadata`: document scope and request identifiers.
- `query_as_resolved`: V1 rewrite compatibility metadata; V2 must not re-run Conversation resolution.
- Raw conversation history: not forwarded as uncontrolled V2 context.

### Output

The V2 adapter must provide `runtime_version=V2`, explicit `ANSWER/FAIL_CLOSED/ERROR`, independent release status, structured evidence/citation/calculation IDs, reason codes, and stage latency metadata.

## 7. Non-negotiable gaps

1. No V2 `FinancialQARuntime` implementation.
2. No standalone-query-to-Supervisor production entry.
3. No production R4 provider/slot retrieval root.
4. No candidate materialization-to-Binder bridge.
5. No Binder-to-`VerifiedEvidencePacket` coordinator.
6. No V1 `CalculationResult` to V2 packet bridge.
7. No unified V2 route-to-renderer/calculator/specialist registry.
8. V2 response lacks complete top-level provenance.
9. No terminal-reason-to-`FinancialQueryResult` mapping.
10. V2 bounded orchestration and generation state machines are not one loop.
11. No V2 shadow/canary runtime selector.
12. No V2 production API/safety seal.

## 8. Next sequence

```
TV2-00  audit                                  PASS
TV2-01  TrustedFinancialRuntimeV2 adapter      next
TV2-02  Supervisor + bounded coordinator
TV2-03  R4 + Binder + verified packet
TV2-04  calculator/C1 + generator registry
TV2-05  validator release/fail-closed mapping
TV2-06  V2 shadow through shared lifecycle
TV2-07  canary and integrated evaluation
TV2-08  production decision and seal
```

TV2-00 does not switch production and does not modify Conversation.

## Final status

```
TV2-00_PRODUCTION_INTEGRATION_AUDIT = PASS
V2_PRODUCTION_INTEGRATION            = NOT_READY
PRODUCTION_FINANCIAL_RUNTIME         = V1
DEFAULT_MULTITURN_CONTEXT_MODE       = off
BENCHMARK_RERUN                      = false
RETRIEVAL_MODIFIED                   = false
NEXT_GATE                            = TV2-01
```
