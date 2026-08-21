# NF-V2-17B3 One-Shot Fresh-Blind Trusted Agentic RAG Execution

## Sealed scope

Questions: 120; answerable: 105; unanswerable: 15; companies: GOOGL, AMZN.
One valid execution followed one archived INFRA_INVALID attempt; valid questions were not rerun after scoring.
Runtime output SHA: 9e02df6701268e83cd9dafdcc36d95736167c19cca8d244a134a69149538dd83
Trace SHA: 3b3b9ee227a49631dcf4f1c820172fd63cf465cbb85514d67ce9d06d2d5f6ebd
Gold/reference were loaded only after output and trace sealing.

## Retrieval

R@1: 2/120 (answerable 2/105)
R@3: 2/120 (answerable 2/105)
R@5: 2/120 (answerable 2/105)
R@10: 2/120 (answerable 2/105)
Multi Any@5: 0/20
Multi All@5: 0/20
Retrieval-complete answerable: 2/105; misses: 103
Frozen retrieval top-k for release was 5; R@10 is diagnostic from the same hard-scoped candidate trace.

## Agent and temporal

Expected replans: 15; attempted: 69; repairable recovered: 0/0
Mean/P50/P95 tool calls: 2.125/3.0/3
Mean/P50/P95 replan rounds: 1.133/2.0/2
Budget violations/infinite loops: 0/0
Scope correct: 119/120; annual/quarter labeled: 98/120
Created_at misuse: 0

## Conflict and calculation

Conflict cases: 7; unresolved conflict leakage: 0; fail-closed unresolved: 7
Calculation operand-ready/executed/canonical: 0/0/0 of 15
False execution/binding: 0/0

## Generation and final runtime

Grounded: 12/105; semantic unsupported: 40/105
Numeric/period/unit fidelity: 80/99/105 of 105
Citation valid/complete: 105/99 of 105
Answerable correct: 6/105; released: 12/105; released correct: 6
Release coverage: 0.114286; correct/released: 0.5; incorrect releases: 6
No-answer correct refusal: 15/15; fail-closed answerable: 93
Unsafe release/false binding/false execution: 0/0/0
Authorization leakage/budget violation/infinite loops: 0/0/0

## Latency and calls

All latency mean/P50/P95/max: 256.558/49.5/737.68/1507.984 ms
Generation-path P95: 798.07 ms; adaptive-path P95: 70.659 ms
Supervisor/general calls: 0; Financial generator calls: 52; semantic verifier model calls: 0

## Decision

Historical NF-V2-15 regression: safe retained 3/3; unsafe blocked 1/1.
Decision: FRESH_BLIND_RUNTIME_PARTIAL.
Production: V1; production switch: false; post-evaluation tuning: false.
