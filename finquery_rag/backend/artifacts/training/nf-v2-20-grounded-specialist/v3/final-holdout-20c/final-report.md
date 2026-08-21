# NF-V2-20C Final Fresh Company-Held-Out Evaluation Report

## 1. Executive Summary
- Decision: **SPECIALIST_V3_FRESH_HOLDOUT_SUCCESS**
- Runtime Readiness: **LOCAL_SPECIALIST_READY_FOR_RUNTIME_INTEGRATION**
- Checkpoint Evaluated: `model_000156.pt` (Step 156, SHA: `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`)
- Final Holdout Issuer: **ORCL (Oracle Corporation)**, 500 samples
- Holdout Status Transition: `FRESH_FINAL_HOLDOUT` -> `CONSUMED_FINAL_HOLDOUT`
- ORCL Strict Correct: **499 / 500 (99.80%)**
- ORCL Released: **499 / 500 (99.80%)**
- ORCL Correct / Released: **499 / 499 (100.00%)**
- Hard Safety Failures / Unsafe Releases: **0**

## 2. Abstention Evaluator V2 & Pre-Holdout Rescore
- Deterministic Regression Suite: **8 / 8 PASS**
- Corrected NFLX Dev Abstention Strict: **25 / 25 (100.0%)** (Old: 21 / 25)
- Corrected NFLX Dev Overall Strict Correct: **499 / 500 (99.8%)**

## 3. ORCL Route Performance
| Route Name | Samples | Strict Correct | Semantic Supported | Released | Correct / Released | Citation Valid |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| QUALITATIVE_GROUNDED_QA | 125 | 125 (100.0%) | 125 (100.0%) | 125 (100.0%) | 125 (100.0%) | 125 (100.0%) |
| MULTI_EVIDENCE_SYNTHESIS | 175 | 175 (100.0%) | 175 (100.0%) | 175 (100.0%) | 175 (100.0%) | 175 (100.0%) |
| TEMPORAL_VERSION_SYNTHESIS | 75 | 75 (100.0%) | 75 (100.0%) | 75 (100.0%) | 75 (100.0%) | 75 (100.0%) |
| CITATION_FORMAT_HARD_CASE | 50 | 50 (100.0%) | 50 (100.0%) | 50 (100.0%) | 50 (100.0%) | 50 (100.0%) |
| INSUFFICIENT_EVIDENCE_ABSTENTION | 25 | 24 (96.0%) | 25 (100.0%) | 24 (96.0%) | 24 (100.0%) | 25 (100.0%) |
| VERIFIED_C1_CONSUMPTION | 50 | 50 (100.0%) | 50 (100.0%) | 50 (100.0%) | 50 (100.0%) | 50 (100.0%) |

## 4. Multi-Evidence Cardinality Generalization
- 2 Evidence: **89/89 (100.0%)**
- 3 Evidence: **63/63 (100.0%)**
- 4 Evidence: **23/23 (100.0%)**

## 5. NFLX Dev vs ORCL Holdout Generalization Comparison
- Strict Correct: **NFLX 99.80% vs ORCL 99.80% (Gap: +0.00 pp)**
- Semantic Supported: **NFLX 99.80% vs ORCL 100.00%**
- Released: **NFLX 99.80% vs ORCL 99.80%**

## 6. Safety & Invariant Verification
- Unsafe Releases: **0**
- Wrong Numeric / Period / Unit / C1 Releases: **0**
- False Abstention Rate: **0.0%**
- Generation Template Collapse: **False** (Uniqueness: 97.8%)

## 7. Next Step Recommendation
Production remains `V1` (`production_switch: false`).
Proceed to `NF-V2-21` for full RAG runtime pipeline integration and production deployment.
