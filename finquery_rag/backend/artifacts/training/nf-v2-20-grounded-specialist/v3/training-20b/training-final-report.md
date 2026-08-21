# NF-V2-20B Local Financial Specialist Generator Training Pilot - Final Report

## Executive Summary
- Decision: **SPECIALIST_V3_TRAINING_SUCCESS**
- Starting Checkpoint: `d24_sft_v2_best275 / model_000275.pt` (SHA `f6b93771b7d35d33f0dab5449b8568bc8135c176da442a74d12c0cc9941ca604`)
- Selected Checkpoint: `model_000156.pt` (Step 156, SHA `3bda9f032d7bfb29a3bdf7e0eeeee930a57a05e899e11e67e108483ca920894a`)
- Training Configuration: `LR = 5e-06`, `1 Epoch`, `20,000 samples (80% V3 + 20% Replay)`, `Response-Only Loss = True`
- NFLX Grounded Dev Strict Correct Gain: **27.8% -> 99.0% (+71.2 pp)**
- Financial Macro Capability Retention: **36.26% (Retention Floor: 18.0%, Gate: PASS)**

## Checkpoint Evolution on NFLX Grounded Dev (500 samples)
| Step | Strict Correct | Semantic Supported | Released | Correct / Released | Citation Valid |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Step 0 (Baseline) | 27.8% | 58.2% | 58.2% | 47.77% | 94.2% |
| Step 156 | 99.0% | 99.8% | 99.8% | 99.2% | 100.0% |
| Step 312 | 97.6% | 100.0% | 100.0% | 97.6% | 100.0% |
| Step 468 | 96.4% | 99.8% | 99.8% | 96.59% | 100.0% |
| Step 625 | 98.2% | 99.8% | 99.8% | 98.4% | 100.0% |

## Financial Macro Retention Benchmark (200 samples)
- Pre-training Financial SFT: **19.78%**
- Selected Grounded Specialist V3: **36.26%**
- Delta vs Baseline: **16.48 pp** (Retention Gate: **PASS**)
- Historical Grounding Alignment Step 7: **18.36%**

## Safety Gate & Integrity Results
- Unsafe Release: **0**
- Numeric / Period / Unit Corruption: **0**
- Citation Loops / CoT Leakage: **0**
- Repetition Rate: **0.0% (< 1.0%)**
- ORCL Final Holdout Status: **UNTOUCHED (0 access during 20B)**
