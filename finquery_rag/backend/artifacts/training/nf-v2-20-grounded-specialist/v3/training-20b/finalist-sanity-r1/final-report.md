# NF-V2-20B-R1 Specialist Finalist Sanity Audit Report

## 1. Executive Summary
- Decision: **SPECIALIST_FINALIST_READY_FOR_FRESH_HOLDOUT**
- Base Commit: `deea3b98ad8c990b3930e540741405074670eb31`
- Selected Finalist: `model_000156.pt` (Step 156)
- Checkpoint SHA256: `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`
- ORCL Final Holdout Status: **UNTOUCHED / READY FOR NF-V2-20C**

## 2. Abstention Route Audit (25 Samples)
- Strict Correct: **21 / 25 (84.0%)**
- Semantic Safe Refusals: **25 / 25 (100.0%)**
- Unsafe Substantive Answers on Insufficient Evidence: **0 / 25 (0.0%)**
- Released: **25 / 25 (100.0%)**
- Four Strict Failures: All classified as **SAFE_ABSTENTION_WORDING_VARIANT** (safe refusal phrasing differing from exact keyword search).
- Safety Claim Verification: **`unsafe_release = 0` IS FULLY VALID**.

## 3. Qualitative Single Failure (125 Samples)
- Strict Correct: **124 / 125 (99.2%)**
- Single Failure Classification: **SAFE_FAIL_CLOSED**
- Accounts for the single non-released response in total Dev (Released: 499 / 500).

## 4. Total Count Reconciliation (500 Samples)
- Total Dev Samples: **500**
- Released: **499 / 500 (99.8%)**
- Strict Correct: **495 / 500 (99.0%)**
- Non-strict Released Answers: **4 (All safe abstention wording variants)**
- Actual Unsupported / Unsafe Releases: **0**

## 5. Financial Macro Recomputation & Retention
- Reported Macro: **36.26%**
- Recomputed Macro: **36.26%** (Status: **PASS**)
- Delta vs SFT Baseline (19.78%): **+16.48 pp**
- Benchmark Status: **CONSUMED_CAPABILITY_REGRESSION**

## 6. Structural & Invariant Verifications
- C1 Calculation Accuracy: **50 / 50 (100.0%)**
- Template Collapse: **False** (High answer diversity, 0 repetition loops)
- Checkpoint Selection: **Valid** (Step 156 strictly best on NFLX Dev)
- Artifact Count: **23 Files (Bookkeeping confirmation: 22 content artifacts + 1 sha256)**

## 7. Gate Authorization
**SPECIALIST_FINALIST_READY_FOR_FRESH_HOLDOUT: TRUE**
ORCL Final Holdout evaluation is fully authorized for NF-V2-20C.
