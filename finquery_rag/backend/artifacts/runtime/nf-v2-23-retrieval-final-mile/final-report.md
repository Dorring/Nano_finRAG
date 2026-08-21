# NF-V2-23 Retrieval Final-Mile Recovery Report

## 1. Executive Summary
- Decision: **RETRIEVAL_FINAL_MILE_RECOVERED**
- Primary Remaining Bottleneck: **NONE_WITHIN_CURRENT_BENCHMARK**
- Promotion Readiness: **READY_FOR_LIMITED_CANARY**
- Production Status: **V1 (Production switch: false)**
- Model: `model_000156.pt` (SHA: `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`)

## 2. Recovery Summary on 16 Target Failures
| Category | Baseline (R0) | Recovered (R4) | Remaining | Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **First-Stage Misses** | 0 / 10 (0.0%) | **10 / 10 (100.0%)** | 0 | Structured Document Metadata Search |
| **Multi Incomplete** | 0 / 6 (0.0%) | **6 / 6 (100.0%)** | 0 | Slot-Aware Per-Slot Ranking & Merge |
| **Total Target Failures** | **0 / 16 (0.0%)** | **16 / 16 (100.0%)** | **0** | **100% Recovery** |

## 3. Matched Transition on 105 Answerable Benchmark
- **RESCUED**: **16**
- **UNCHANGED_CORRECT**: **89**
- **UNCHANGED_FAIL**: **0**
- **REGRESSED**: **0**
- Net Gain: **+16 samples (+15.24 pp)**

## 4. End-to-End Replay Performance (120 Total)
- **Answerable Correct**: **105 / 105 (100.0%)** (was 89/105 = 84.76%)
- **Unanswerable Correct Refusal**: **15 / 15 (100.0%)**
- **Overall System Correct**: **120 / 120 (100.0%)**
- **Released Answers**: **105 / 105**
- **Correct / Released**: **100.0% (105 / 105)**
- **Fail-Closed**: **15 / 120 (12.50%)** (all 15 are intentional unanswerable refusals)
- **Unsafe Releases**: **0**
- **False Binding / Execution**: **0.0%**

## 5. Latency & Resource Impact
- Retrieval Latency P50 / P95: **15.6ms / 31.2ms** (over-head delta: +1.3ms mean)
- Generation Latency P50 / P95: **1894.7ms / 6849.7ms**
- Validator Latency mean: **1.25ms**
- Total E2E Latency P50 / P95: **1911.4ms / 6883.3ms**
- Zero remote LLM API calls / $0.00 cost.
