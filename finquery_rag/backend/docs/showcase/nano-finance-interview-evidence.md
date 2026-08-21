# NanoFinance 2.08B: Financial Domain Specialist & Grounded RAG Generation System
## Authoritative Evidence Seal & Interview Snapshot

---

### 1. Executive Summary & Core Results

| Dimension | Baseline / Reference | NanoFinance / Final System | Delta / Metric | Scope / Evaluation Protocol |
| :--- | :--- | :--- | :--- | :--- |
| **Financial Domain Capability** | Qwen3.5-2B: **7.86%** | NanoFinance SFT: **19.78%** | **+11.92 pp** (2.5x) | Macro Financial Multi-Task Benchmark |
| **Grounded Specialist Fresh Holdout** | Baseline Gen: ~12% | Step-156 Specialist: **499 / 500 (99.8%)** | **99.8% Strict Correct** | ORCL Company-Held-Out Verified-Evidence Generation |
| **Generator Training Effect (Matched)** | Old SFT: **8 / 68 (11.76%)** | Step-156: **52 / 68 (76.47%)** | **+64.71 pp** (6.5x) | Matched 68-Packet Binder-Ready Regression (44 rescued, 0 regressed) |
| **End-to-End RAG Regression** | Baseline R0: **89 / 105 (84.76%)** | R4 Combined: **105 / 105 (100.0%)** | **+15.24 pp** | 120-Sample Consumed Benchmark (105 answerable + 15 unanswerable) |
| **Unanswerable Safe Refusal** | Non-gated: Hallucination | Runtime Guard: **15 / 15 (100.0%)** | **100% Safe Refusal** | Pre-generation fail-closed on unanswerable questions (TR7) |
| **Released Answer Precision** | Raw LLM: Untrusted | Validator Chain: **105 / 105 (100.0%)** | **100.0% Precision** | Zero wrong numeric, unit, period, C1, or phantom citations |
| **Safety Invariants** | False Execution risk | Runtime Invariants: **0.0%** | **0 Errors** | False Binding = 0, False Execution = 0, Unsafe Release = 0 |

---

### 2. Standardized Interview & Resume Claim Formulations

#### A. Financial Model Capability Claim
> **Claim**: "在 2.08B 参数规模下，自研 NanoFinance 经过 ~25B tokens 金融 CPT 与 40K 高质量指令微调，在金融多任务宏观评测中较同规模通用模型 Qwen3.5-2B 绝对提升 **+11.92pp**（**19.78% vs 7.86%**，2.5倍性能）。"
- **Scope**: 同一评测框架下的金融多任务宏观能力评测。

#### B. Grounded Specialist Holdout Claim
> **Claim**: "针对金融 RAG 中模型随意篡改数字和单位的痛点，构建 20K 严谨接地蒸馏数据集（V3），训练出的 Local Financial Specialist Generator 在完全未见过的 **ORCL 500 题盲测集（Company-Held-Out）** 上达到 **99.80% Strict Correct（499/500）**，释放答案正确率 **100.0%**，实质性错误释放数为 0。"
- **Scope**: 限定于已检索证据约束生成评测（Verified-Evidence Generation Holdout），不代表全流程无约束端到端召回。

#### C. Generator Training Effect (Matched Regression) Claim
> **Claim**: "在 68 个严格对齐的 Binder-Ready 消费回归样本上，Step-156 Specialist 将通过率从原模型的 **11.76%（8/68）** 提升至 **76.47%（52/68）**，绝对增益 **+64.71pp**，净拯救 44 个复杂金融问答场景且零回归。"

#### D. End-to-End RAG System Claim
> **Claim**: "在 120 题严格端到端财务问答回归中，可答问题正确率由 **84.76%（89/105）** 提升至 **100.0%（105/105）**；对 15 题无答案问题实现 **100.0% 安全拒答（15/15）**；释放答案正确率 **100.0%（105/105）**；False Binding 与 False Execution 率均为 **0.0%**。"
- **Status**: `CONSUMED_END_TO_END_REGRESSION`（经历过缺陷驱动的最终一公里检索优化）。

---

### 3. Architecture Snapshot

```
Financial Query
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 1. Retrieval & Financial Query Normalization           │
│    - Deterministic Financial Alias Expansion           │
│    - Structured Document Metadata Search (Sidecars)    │
│    - Slot-Aware Per-Slot Ranking & Dedup Merge (Multi) │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 2. Semantic Evidence Binder & Plan Routing             │
│    - Strict Evidence Binding (Pass / Fail-Closed)      │
│    - GeneratorRoutingPolicy:                           │
│      • STRUCTURED_SINGLE → Deterministic Renderer      │
│      • CALCULATION       → Deterministic Calculator/C1 │
│      • QUALITATIVE/MULTI/TEMPORAL → Local Specialist   │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 3. 2.08B Local Financial Specialist Generator          │
│    - Input: FinancialGenerationViewV1 (Contract Sealed)│
│    - Checkpoint: Step-156 (SHA256 Sealed)              │
│    - Role: LOCAL_FINANCIAL_SPECIALIST_GENERATOR        │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│ 4. Mandatory Runtime Validator Chain & Release Gate    │
│    - SemanticClaimVerifier                             │
│    - NumericValidator / UnitCurrencyScaleValidator     │
│    - PeriodValidator / CitationValidator / C1Validator │
│    - RepetitionDetector / CoTLeakageDetector           │
│    - AbstentionEvaluatorV2                             │
│    * MODEL HAS ZERO RELEASE AUTHORITY                  │
└────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────┬─────────────────────────┐
       ▼                         ▼                         ▼
 [RELEASED ANSWER]      [PRE-GEN FAIL-CLOSED]     [VALIDATOR FAIL-CLOSED]
 (Strict Verified)      (No Trusted Evidence)     (Safety Policy Violation)
```

---

### 4. Model Training & Distillation Pipeline

1. **Student Base**: `d24_sft_v2_best275` (2.08B parameters).
2. **Pre-training & Alignment**:
   - Financial CPT: ~25B tokens high-quality financial filings & disclosures.
   - Financial SFT: ~40K domain instructions.
3. **Grounded Distillation Mixture (V3)**:
   - Grounded V3 Data: 16,000 high-precision teacher-distilled samples.
   - Financial SFT Replay: 4,000 general domain capability replay samples.
   - Mixture Ratio: **80% Grounded / 20% SFT Replay**.
4. **Training Hyperparameters**:
   - Optimizer: AdamW, LR: `5e-6` with cosine decay, 1 Epoch response-only loss masking.
   - Sequence Length: 2048, Device: Logical `cuda:0` (NVIDIA RTX A6000).
5. **Lexicographic Selection**:
   - Selection Set: 500-sample NFLX Dev.
   - Safety -> Financial Capability Retention ($\ge 18.0\%$) -> Strict Correct.
   - Finalist: **Step 156** (`model_000156.pt`, SHA256: `3bda9f03...`).
6. **Fresh Holdout Verification**:
   - **ORCL 500-sample Holdout**: 499 / 500 (99.80% Strict Correct), 0 Unsafe Releases, Generalization Gap = 0.00pp.

---

### 5. Retrieval Final-Mile Recovery (NF-V2-23)

- **Problem Diagnosis**:
  1. *10 First-Stage Misses*: Legacy table retriever only queried statement rows; document sidecar/filing header metadata was not searched.
  2. *6 Multi-Evidence Failures*: Single global ranking allowed the primary table to crowd out cross-statement / note disclosure chunks.
- **Generic Solutions**:
  1. *Structured Document Metadata Search*: Indexed and retrieved filing header sidecars.
  2. *Slot-Aware Retrieval*: Decomposed multi queries into distinct slot phrases, retrieving top candidates per slot independently.
  3. *Deterministic Financial Alias Normalization*: Canonicalized financial terms.
- **Results**:
  - First-stage recovery: **10 / 10 (100.0%)**.
  - Multi-evidence recovery: **6 / 6 (100.0%)**.
  - Matched 105 Answerable regression: **16 Rescued, 89 Unchanged Correct, 0 Regressed**.
  - Retrieval Latency: P50 = **15.6ms**, P95 = **31.2ms** (Delta: +1.3ms mean).
  - External LLM rewrite calls: **0 ($0 cost)**.

---

### 6. Resource Profile & Serving Latency

- **Hardware**: NVIDIA RTX A6000 (48GB VRAM).
- **VRAM Footprint**:
  - Model Load: **5.46 GB**
  - Peak Concurrent Generation: **13.59 GB**
  - Steady-State: **5.46 GB**
- **Latency Breakdown**:
  - Retrieval P50 / P95: **15.6 ms / 31.2 ms**
  - Model Generation P50 / P95: **1894.7 ms / 6849.7 ms**
  - Validator Pipeline mean: **1.25 ms**
  - End-to-End System P50 / P95: **1911.4 ms / 6883.3 ms**
- **Concurrency**: Bounded 1, 2, 4 concurrent requests tested; 100% success rate, 0 OOM, queue backpressure stable.

---

### 7. Known Limitations & Honest Disclaimers

1. **Consumed Regression vs Blind Holdout**: The 120-sample E2E benchmark is a consumed regression suite used for defect diagnosis and closure; it is not a fresh blind evaluation.
2. **Generator vs System Scope**: The ORCL 99.8% metric evaluates Verified-Evidence Generation under supplied gold evidence, not the full retrieval-to-generation pipeline.
3. **Macro Retention**: The 36.26% capability retention metric for the Step-156 Specialist is an internal capability checkpoint evaluation.
4. **Production Routing**: Production remains `V1` (`production_switch: false`); the new runtime is verified in shadow mode and certified `READY_FOR_LIMITED_CANARY`.

---

### 8. Project Artifact Index

- Model Checkpoint: `/home/mxf/.cache/nanochat/chatsft_checkpoints/d24_grounded_specialist_v3_lr5e6/model_000156.pt`
- Checkpoint SHA256: `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`
- Generation View Contract: `FinancialGenerationViewV1` (SHA: `943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4`)
- Integration Runner: `finquery_rag/backend/scripts/runtime/run_nf_v2_21_runtime_integration.py`
- Shadow Verification: `finquery_rag/backend/scripts/runtime/run_nf_v2_22_shadow_verification.py`
- Retrieval Recovery: `finquery_rag/backend/scripts/runtime/run_nf_v2_23_retrieval_final_mile.py`
