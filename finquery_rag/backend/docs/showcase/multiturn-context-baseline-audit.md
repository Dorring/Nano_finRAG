# Phase M0 — Multi-turn Context Extension Baseline Codebase Audit

**Project**: `nano_finance / finquery_rag`  
**Base Commit**: `8cadc78264f51b3a7ac5d81cf001209cdfbb2b89` (`master` / `nf-v2-interview-final`)  
**Branch**: `feat/nf-v2-24-multiturn-context`  
**Status**: **GATE M0 AUDIT PASS**

---

## 1. Baseline System Inventory

| Audit Item | Codebase Location | Current Mechanism / Status | Integration Role for Conversation Layer |
| :--- | :--- | :--- | :--- |
| **1. User Query Entry Point** | `src/main.py`<br>`@app.post("/query")`<br>`@app.post("/query/stream")` | Receives `QueryRequest` with `query`, `session_id`, `doc_names`, `mode`. | The Conversation Context Layer will be invoked at the top of the query handler when `session_id` has history. |
| **2. Supervisor Interface** | `rag_v2/runtime/runtime.py`<br>`TrustedRAGRuntimeV2.handle(...)` | Takes `TrustedRAGQueryV2(query_id, text, metadata_scope)`. | Receives the reconstructed `standalone_query` from Layer 1. |
| **3. Plan & Route Schema** | `src/generation/generator_routing_policy.py`<br>`rag_v2/runtime/routing.py` | Routes: `STRUCTURED_SINGLE`, `CALCULATION`, `QUALITATIVE`, `MULTI_EVIDENCE`, `TEMPORAL_COMPARISON`. | Remains completely unchanged; executes bounded capability dispatch. |
| **4. Metadata & Scope Parsing** | `src/retrieval/metadata_scope.py`<br>`MetadataFilterPlannerV1` | Parses `entity`, `period`, `form_type`, `fiscal_year`, `quarter`. | Layer 1 extracts semantic dialogue state without mutating downstream parsing. |
| **5. Provider Abstraction** | `rag_v2/generation/providers.py`<br>`ProviderRegistryV1` | Abstract provider interface with registration mechanism. | Bailian `Qwen3.6-Flash` will be integrated via standard OpenAI-compatible client wrapper. |
| **6. Session / Conversation ID** | `src/main.py`<br>`src/models.py` | `session_id` string passed in `QueryRequest`. | Mapped 1-to-1 to `conversation_id` in `ConversationStateStore`. |
| **7. History Message Storage** | `src/models.py` (`Session`, `Message`) | Fast API session database tracks historical messages. | Serves as backend persistence source for multi-turn history. |
| **8. Runtime Request Metadata** | `rag_v2/runtime/contracts.py`<br>`TrustedRAGQueryV2` | Contains `query_id`, `text`, `metadata_scope`. | Passes `standalone_query` with preserved metadata envelope. |
| **9. Token Counter Capability** | Python environment | `tiktoken` (cl100k_base) and `transformers.AutoTokenizer` available. | Used by `ContextBudgetManager` for sub-millisecond token counting. |
| **10. Test Suites & E2E Suites** | `tests/`, `scripts/runtime/` | `pytest tests/`, `scripts/runtime/run_nf_v2_23_retrieval_final_mile.py` | Target for contract, adversarial, budget, and parity test runs. |

---

## 2. Architecture Comparison

### Current Architecture (Single-Turn Dedicated)
```text
User Turn (Raw Query)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Financial RAG Supervisor / Runtime Entrance                 │
│ (rag_v2/runtime/runtime.py :: TrustedRAGRuntimeV2.handle)   │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Trusted Execution Pipeline                                  │
│  - R4 Combined Retrieval                                    │
│  - Semantic Evidence Binder                                 │
│  - Deterministic Calculator / 2.08B Financial Specialist    │
│  - Mandatory Validator Chain (0-Release Authority)          │
│  - Fail-Closed Gate                                         │
└─────────────────────────────────────────────────────────────┘
```

### Target Architecture (Multi-Turn Context Extended)
```text
User Turn (Raw Query + session_id)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Conversation Context Layer (Bailian Qwen3.6-Flash) │
│  ├── Context Relevance Filter (Query + State + Turns)       │
│  ├── Hierarchical Context Budget (L1 Raw / L2 State / L3)   │
│  ├── Reference / Ellipsis / Relative Period Resolution      │
│  ├── Ambiguity Detection (Clarification Required Gate)      │
│  └── Dialogue State Management (Semantic State Only)        │
└─────────────────────────────────────────────────────────────┘
         │
         ├──────────────────────────┐
         │ [If Clarification]       │ [If Standalone Query]
         ▼                          ▼
 [Clarification Response]   ┌─────────────────────────────────────────────────────────────┐
                            │ Layer 2: Existing Financial RAG Supervisor                  │
                            │ (TrustedRAGRuntimeV2.handle)                                │
                            └─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            ┌─────────────────────────────────────────────────────────────┐
                            │ Layer 3: Existing Trusted Execution Pipeline                │
                            │  - R4 Combined Retrieval                                    │
                            │  - Semantic Evidence Binder                                 │
                            │  - Deterministic Calculator / 2.08B Financial Specialist    │
                            │  - Mandatory Validator Chain (0-Release Authority)          │
                            │  - Fail-Closed Gate                                         │
                            └─────────────────────────────────────────────────────────────┘
```

---

## 3. Gate M0 Invariant Check

- [x] **No modification to Financial Specialist weights**: `model_000156.pt` (SHA256: `3bda9f03...`) remains completely frozen.
- [x] **No modification to Retriever data structures**: Index schema, BM25 tables, and structured sidecar formats remain untouched.
- [x] **No modification to Evidence Contract**: `FinancialGenerationViewV1` remains the authoritative view contract.
- [x] **No modification to Calculator / C1**: Arithmetic validation and deterministic calculations are preserved.
- [x] **No modification to Validator Release Authority**: The validator chain retains 100% authoritative veto and release control; Conversation Layer has 0 release authority.
- [x] **Feature Flag Isolation**: `MULTITURN_CONTEXT_ENABLED=false` preserves the exact legacy execution path.

**Gate M0 Status: PASSED — AUTHORIZED TO PROCEED TO PHASE M1.**
