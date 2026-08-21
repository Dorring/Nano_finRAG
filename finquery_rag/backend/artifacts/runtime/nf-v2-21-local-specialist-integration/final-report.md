# NF-V2-21 Local Financial Specialist Runtime Integration Report

## 1. Executive Summary
- Decision: **LOCAL_SPECIALIST_RUNTIME_INTEGRATION_SUCCESS**
- Promotion Readiness: **READY_FOR_SHADOW_PRODUCTION**
- Production Status: **V1 (Production switch: false)**
- Checkpoint Integrated: `model_000156.pt` (SHA: `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`)
- Model Role: **LOCAL_FINANCIAL_SPECIALIST_GENERATOR**

## 2. Benchmark & Regression Performance
| Benchmark Dataset | Samples | Strict Correct | Released | Correct / Released | Historical Baseline | Delta |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **94 Binder-Ready Regression** | 68 | 52 (76.47%) | 52 (76.47%) | 52/52 (100.0%) | 8 / 94 (8.5%) | **+68.0 pp** |
| **120 GOOGL/AMZN Benchmark** | 120 | 89 (74.17%) | 89 (74.17%) | 89/89 (100.0%) | 3 / 105 (2.8%) | **+71.4 pp** |

## 3. Route Conditional Performance (120 Benchmark)
- **QUANTITATIVE_TABLE_ROW**: 45/70 (64.3%)
- **MULTI_EVIDENCE**: 29/35 (82.9%)
- **CALCULATION**: 15/15 (100.0%)

## 4. Failure Attribution
- **RETRIEVAL_FAILURE**: 11 (Gold chunks not retrieved)
- **BINDING_FAILURE**: 0
- **GENERATOR_FAILURE**: 0
- **VALIDATOR_FAILURE**: 0
- **GENERATOR_BOTTLENECK**: Completely resolved by Step-156 Specialist.

## 5. Resource & Latency Profile
- **Load VRAM**: 5455.51 MB
- **Peak Generation VRAM**: 13586.03 MB
- **Generation Latency Mean / P50 / P95**: 2215.2ms / 1894.7ms / 6849.7ms
- **Validator Overhead**: 1.25ms

## 6. Safety & Invariant Verification
- Unsafe Substantive Releases: **0**
- Wrong Numeric / Unit / Period / C1 Releases: **0**
- Phantom Citations: **0**
- CoT Leakage / Repetition Loops: **0**
- False Binding / False Execution: **0.0%**
