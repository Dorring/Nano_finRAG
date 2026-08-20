# NF-V2-16 Runtime Audit

Base: `74910f27d9675a6537914581f9ff74ddd7d57f51`

This audit is recorded before implementation changes on the isolated
`exp/nf-v2-16-bounded-adaptive-rag` worktree.

## Existing entrypoints

- Production legacy entrypoint: `finquery_rag/backend/src/application/rag_orchestrator.py` (`RAGOrchestrator.answer`).
- NF-V2 control-plane entrypoint: `finquery_rag/backend/rag_v2/runtime/runtime.py` (`TrustedRAGRuntimeV2`) and `rag_v2/generation/state_machine.py`.
- Supervisor contract/provider: `finquery_rag/backend/rag_v2/supervisor/service.py`, `rag_v2/contracts/plan.py`.
- Retrieval implementation: `finquery_rag/backend/src/retrieval/retrieval_pipeline.py` and `finquery_rag/backend/src/pdf_retrieval_v4/`.
- Query rewrite/profile: `src/retrieval/query_processor.py`, `src/retrieval/query_profile.py`; no observation-driven rewrite loop exists.
- Missing-slot/binding: `rag_v2/evidence/` binders and `rag_v2/contracts/evidence.py`; no bounded adaptive state evaluator exists.
- Deterministic calculation: `src/finance/calculation_pipeline.py`, `rag_v2/contracts/calculation.py`, `src/finance/calculation_executor.py`.
- Trusted evidence packet: `rag_v2/contracts/evidence.py::VerifiedEvidencePacket`.
- Generator/Financial view: `rag_v2/generation/financial_view_v1.py`, `rag_v2/generation/state_machine.py`.
- Post-generation safety: `rag_v2/runtime/semantic_claims.py::SemanticClaimVerifierV1`, `rag_v2/generation/validator.py::RuntimeGenerationValidatorV1`.

## State and stop behavior before NF-V2-16

The existing `rag_v2/orchestration/state.py` state machine covers a linear
RECEIVED→PLANNED→RETRIEVED/MATERIALIZED→BOUND/CALCULATED→GENERATED→VALIDATED
path with repair budgets. It does not represent observation/progress
signatures, temporal scope, version succession, or a REPLAN state. The new
NF-V2-16 state machine is additive and will not replace this trusted path.

## Evidence / temporal limitations observed

`VerifiedEvidencePacket` carries metric, period, value, currency, scale and
unit on `BoundFact`, but no fiscal quarter, period start/end, report/filing
date, amendment or supersedes relation. Existing document metadata has
ingestion/creation timestamps; those are not financial effective time.
Temporal consistency is therefore implemented as a separate explicit
contract in this experiment and defaults unknown fields to `UNKNOWN` rather
than inferring from ingestion time or text similarity.

## Implementation map

The experiment adds `rag_v2/adaptive/` with serializable state/budget,
deterministic evidence evaluation and temporal consistency, bounded
replanning/progress detection, and an explicit state-machine driver. It is
covered by synthetic sealed component fixtures; production V1 wiring is not
changed and the frozen 72-question benchmark is not used for tuning.
