# NanoFinance Conversation Context Layer: Architecture & Design

## 1. Executive Summary & Design Rationale

The **NanoFinance Conversation Context Layer** introduces an independent, modular multi-turn context interpretation layer in front of the existing NanoFinance Trusted Financial RAG runtime.

### Core Architectural Principle
> **Model Context Capacity $\neq$ Conversation Memory Strategy**
> 
> Although Alibaba Cloud Bailian `Qwen3.6-Flash` natively supports up to ~991K input tokens (1M physical context window), physical capacity is not a conversation memory strategy. Unbounded history concatenation leads to attention dilution, stale topic contamination, linear cost/latency growth, and hallucinated fact propagation.
> 
> Therefore, NanoFinance maintains a **3-Tier Hierarchical Context Architecture (L1/L2/L3)** and **Application-Level Context Budgeting** as core first-class capabilities.

---

## 2. Three-Layer Separation of Concerns

```
Layer 1 — Conversation Interaction (Conversation Context Layer)
---------------------------------------------------------------
• Intent Understanding & Query Reconstruction
• Dynamic Relevance Filtering (Query + State + Recent Turns)
• 3-Tier Hierarchical Context (L1 Raw / L2 State / L3 Compressed)
• Ambiguity Detection & Explicit Clarification Gate
• Fast-Path Self-Contained Query Bypass

                                   │ [Reconstructed Standalone Financial Query]
                                   ▼

Layer 2 — Agentic Orchestration (Financial RAG Supervisor)
---------------------------------------------------------------
• SupervisorPlan Request Interpretation
• Capability & Tool Routing (Deterministic Renderer / Calculator / Specialist)
• Bounded Runtime State Machine Control

                                   │ [Validated Tool Invocation]
                                   ▼

Layer 3 — Trusted Financial Execution (Trusted RAG Runtime)
---------------------------------------------------------------
• R4 Combined Retrieval (Structured Sidecar + Slot-Aware Ranking)
• Semantic Evidence Binder & RequiredSlot Enforcement
• Deterministic Calculator & C1 Verification
• Step-156 Local Financial Specialist Generator (2.08B)
• Mandatory Runtime Validator Chain (0-Release Authority)
• Deterministic Fail-Closed Gate
```

---

## 3. Core Component Design

### 3.1 Structured Data Contracts (`src/conversation/contracts.py`)
- **`DialogueTurn`**: Records turn ID, timestamp, raw query, standalone query, topic, and `referenced_evidence_ids` (provenance metadata only).
- **`DialogueState`**: Stores semantic state (`active_entity`, `active_metric`, `active_period`, `comparison_target`, `compressed_history`). **Zero authoritative facts or numbers stored**.
- **`ConversationResolution`**: Strongly typed output containing `standalone_query`, resolved coordinates, `inherited_fields`, `explicit_fields`, `topic_switch`, `ambiguity_detected`, `clarification_required`, `clarification_options`, and `reason_codes`.

### 3.2 Information Priority Hierarchy
$$\text{Current Explicit Query} > \text{Explicitly Referenced Turn} > \text{Structured Dialogue State} > \text{Relevant History} > \text{Compressed History}$$
- **`EXPLICIT_QUERY_OVERRIDE`**: Historical state is used strictly to fill missing slots (ellipses/pronouns). Explicit user fields in the current turn are never overwritten.

### 3.3 Strict Trust Boundary (`CONVERSATION_CONTEXT_NOT_EVIDENCE`)
- Historical assistant responses are never converted into `VerifiedEvidence` or fed directly into `Calculator` operands.
- All numbers and facts must be retrieved and validated afresh through the `Semantic Evidence Binder`.

### 3.4 Dynamic Relevance Filter (`src/conversation/relevance_filter.py`)
- Evaluates turns based on:
  $$\text{Score} = \text{state\_entity\_match} + \text{state\_metric\_match} + \text{topic\_match} + \text{explicit\_ref} + \text{recency} - \text{topic\_switch\_penalty} - \text{noise\_penalty}$$

### 3.5 Hierarchical Context Budget Manager (`src/conversation/context_budget.py`)
- **L1 (Recent Raw Turns)**: Window of last $N$ turns (default 4).
- **L2 (Structured Dialogue State)**: Active entity, metric, period.
- **L3 (Compressed History)**: Semantic topic summaries for dialogues $>8$ turns.
- **Trimming Priority**:
  1. *Never Drop*: Current Query, Structured State (L2), Explicitly Referenced Turns.
  2. *Retain Next*: Recent Relevant Turns (L1).
  3. *Trimming Candidates*: Compressed History (L3) $\to$ Unreferenced Old Turns.
- Sub-microsecond regex token estimation ensures zero external network dependency.

### 3.6 Bailian Qwen3.6-Flash Client & Fast Path (`src/conversation/bailian_client.py` & `resolver.py`)
- Configured via environment variables (`BAILIAN_API_KEY`, `BAILIAN_BASE_URL`, `BAILIAN_CONTEXT_MODEL`).
- `enable_thinking=False` (thinking mode disabled).
- Bounded retries (max 2) with exponential backoff + jitter.
- Output capped at `CONTEXT_RESOLVER_MAX_OUTPUT_TOKENS=512`.
- Fast Path bypasses external LLM calls for self-contained queries (0 latency overhead).

---

## 4. Runtime Configuration

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MULTITURN_CONTEXT_ENABLED` | `false` | Master feature flag. When false, runs legacy single-turn path. |
| `BAILIAN_CONTEXT_MODEL` | `qwen3.6-flash` | Alibaba Cloud Bailian model identifier. |
| `BAILIAN_CONTEXT_THINKING` | `false` | Disabled by default. |
| `CONTEXT_RECENT_TURNS` | `4` | L1 recent raw turns window. |
| `CONTEXT_SUMMARY_TRIGGER_TURNS` | `8` | Turn threshold to trigger L3 summarization. |
| `CONTEXT_TARGET_TOKENS` | `4096` | Application-level context budget target. |
| `CONTEXT_MAX_TOKENS` | `8192` | Application-level context hard ceiling. |
| `CONTEXT_SUMMARY_MAX_TOKENS` | `768` | Maximum token length for L3 summary. |
| `CONTEXT_RESOLVER_MAX_OUTPUT_TOKENS` | `512` | Cap on resolver output tokens. |
| `CONTEXT_MAX_RETRIES` | `2` | Bounded retries on 429/timeout. |
