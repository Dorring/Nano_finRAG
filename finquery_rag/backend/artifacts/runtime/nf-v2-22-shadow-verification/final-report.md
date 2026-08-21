# NF-V2-22 Shadow Production Verification & Metric Reconciliation Report

## 1. Executive Summary
- Decision: **SHADOW_VERIFICATION_SUCCESS_WITH_METRIC_CORRECTION**
- Primary Remaining Bottleneck: **RETRIEVAL**
- Promotion Readiness: **READY_FOR_LIMITED_CANARY**
- Production Status: **V1 (Production switch: false)**
- Checkpoint: `model_000156.pt` (SHA: `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`)

## 2. 120-Sample Exact Accounting Reconciliation
| Outcome Category | Count | Percentage | Description |
| :--- | :--- | :--- | :--- |
| **ANSWERABLE_RELEASED_CORRECT** | 89 | 74.17% | Answerable queries with complete retrieval, correct generation & validation pass |
| **ANSWERABLE_FAIL_RETRIEVAL** | 16 | 13.33% | 10 first-stage misses + 6 multi-evidence incomplete retrieval misses |
| **UNANSWERABLE_CORRECTLY_REFUSED** | 15 | 12.50% | Intentional pre-generation fail-closed on unanswerable queries (TR7) |
| **UNSAFE_RELEASE (ANY)** | 0 | 0.00% | Zero unsafe releases across all 120 samples |
| **Total Universe** | **120** | **100.0%** | **Unaccounted: 0** |

## 3. 31 Fail-Closed Reconciliation
- **15** = `UNANSWERABLE_CORRECTLY_REFUSED` (Expected safe refusal behavior)
- **10** = `FIRST_STAGE_RETRIEVAL_MISS` (Zero trusted evidence chunks found)
- **6** = `MULTI_EVIDENCE_INCOMPLETE` (1 of required chunks missing from retrieval top-k)
- **Total Fail-Closed**: **15 + 10 + 6 = 31 (100% Reconciled, 0 Unaccounted)**

## 4. Generator vs Retrieval Bottleneck Resolution
- **Generator Bottleneck**: **RESOLVED** (Conditional Strict Correct on Binder-ready: 93.68%, C1 preservation: 100%, Matched 68-packet comparison: 52/68 vs 8/68 = +64.71pp).
- **Retrieval Bottleneck**: **REMAINS AS PRIMARY BOTTLENECK** (16 answerable failures are all upstream evidence misses).

## 5. Route Breakdown (120 Benchmark)
- **QUANTITATIVE_TABLE_ROW**: 45/55 answerable correct (81.8%), 15/15 unanswerable refused (100.0%)
- **MULTI_EVIDENCE**: 29/35 answerable correct (82.9%), 6 partial retrieval misses fail-closed
- **CALCULATION**: 15/15 answerable correct (100.0% C1 preservation)

## 6. Safety, Latency & Concurrency Invariants
- Unsafe Releases: **0** | False Binding / Execution: **0.0%**
- Generation Latency: P50 = **1894.7ms**, P95 = **6849.7ms**, Validator = **1.25ms**
- Concurrency (1, 2, 4): **100% Stable**, 0 OOM, 0 Timeout, VRAM Peak = **13.59 GB**
