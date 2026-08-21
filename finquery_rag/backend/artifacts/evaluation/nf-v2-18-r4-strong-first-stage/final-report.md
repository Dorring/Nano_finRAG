# NF-V2-18A-R4 Strong First-Stage Retrieval

Base: `f34934b85b70ced100457f0b6c455bf8fed67572`
Development set: `CONSUMED_DEVELOPMENT_REGRESSION` (120 questions; not fresh-blind after R4 tuning)
Production: `V1`; production switch: `false`

## Embedding and index

- Model: `Qwen/Qwen3-Embedding-4B`; resolved revision: `5cf2132abc99cad020ac570b19d031efec650f2b`
- Snapshot manifest SHA: `54c2a3b9ef650ea026acab844c56917c23126f0426f1d4909154adbf485887dc`; bytes: `8059521975`; files: `14`
- Contract: `last_token_pool + L2 normalize`; query instruction: `Given a financial question, retrieve financial evidence that directly supports it.`; document instruction: none
- Dynamic GPU: physical `0`, logical `cuda:0`, tier `1`, free before `48536 MiB`
- General objects: `11991`; AtomicFact objects: `1924`; persisted vectors: `14190`; dimension: `2560`
- Scope: `{'atomic_objects': 1924, 'general_objects': 11991, 'reason': 'consumed development questions are GOOGL/AMZN; no production default changed', 'requested_ciks': ['1018724', '1652044']}`. This is a bounded GOOGL/AMZN development index; production/default indices were not replaced.

## A4 exact candidate headroom and union

| depth | A4 | Qwen dense | A4 ∪ Qwen |
|---:|---:|---:|---:|
| 20 | 77/105 | 18/105 | 77/105 |
| 50 | 88/105 | 23/105 | 89/105 |
| 100 | 92/105 | 33/105 | 93/105 |
| 200 | 95/105 | 48/105 | 95/105 |

A4 family-headroom artifact is preserved in `current-exact-candidate-headroom.json` (for example @100 answerable family metrics: `92`). A4 remains the hard-scope candidate provider. The union values measure exact canonical evidence inclusion before ranking; family recall is not substituted for fine evidence.

## General ablations (answerable denominator 105)

| stage | R@1 | R@3 | R@5 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|
| G0 A4 | 33/105 | 55/105 | 62/105 | 68/105 | 77/105 |
| G1 Qwen dense only | 7/105 | 9/105 | 13/105 | 18/105 | 21/105 |
| G2 A4 ∪ Qwen dense | 33/105 | 55/105 | 62/105 | 68/105 | 77/105 |
| G3 union + optional reranker | 33/105 | 55/105 | 62/105 | 68/105 | 77/105 |
| G4 route-specific | 33/105 | 55/105 | 62/105 | 68/105 | 77/105 |

G3 reranker status: `{'available': False, 'pairs': 5250, 'reason': 'disabled_for_r4_development; R3 historical effect retained'}`; frozen R3 global effect was rescued `8`, damaged `9`, net `-1`, so it was not selected globally.
- Frozen R3 selected reference (not overwritten): exact R@5 `63/120`, R@10 `71/120`; R4 G0 is the A4 first-stage baseline `62/120`, `68/120`.

## Route-specific multi and calculation

Multi ablations:
- `M0`: Any@5 17/20; All@5 3/20; All@10 6/20; All@20 8/20
- `M1`: Any@5 17/20; All@5 3/20; All@10 6/20; All@20 8/20
- `M2`: Any@5 17/20; All@5 3/20; All@10 6/20; All@20 8/20
- `M3`: Any@5 17/20; All@5 3/20; All@10 6/20; All@20 8/20
- Slot provider coverage is recorded for A4, Qwen dense, and union at depths 5/10/20 in `multi-slot-candidate-recall.json`; runtime slots are derived by the existing planner, never from Gold.

Calculation ablations (operand-complete counts at @5/@10/@20):
- `C0`: 3/15, 5/15, 6/15
- `C1`: 0/15, 0/15, 0/15
- `C2`: 0/15, 0/15, 0/15
- `C3`: 3/15, 5/15, 6/15
- `C4`: 3/15, 5/15, 6/15
- `C5`: 3/15, 5/15, 6/15
- R3 reference baseline: `{'denominator': 15, 'operand_complete_at_10': 5, 'operand_complete_at_20': 6, 'operand_complete_at_5': 5}`; AtomicFact → canonical TABLE_ROW mapping and period gating remain explicit.

## Selected configuration

- `A4 + Qwen3-Embedding dense union`
- Multi: `runtime-slot A4/Qwen dense union`
- Calculation: `runtime-operand A4 + AtomicFact lexical/dense union; iXBRL separate`
- Selected exact answerable R@1/R@3/R@5/R@10/R@20: `33/105`, `55/105`, `62/105`, `68/105`, `77/105`
- A4 no-loss invariant: `0` candidates lost due enrichment

## Safety and latency

- Safety counters: `{'a4_candidates_lost_due_enrichment': 0, 'authorization_leakage': 0, 'created_at_misuse': 0, 'document_type_violation': 0, 'entity_violation': 0, 'false_binding': 0, 'fiscal_violation': 0, 'silent_scope_relaxation': 0, 'version_violation': 0, 'wrong_period_binding': 0}`
- Embedding/index build: `{'batch_size': 32, 'device': 'cuda:0', 'dimension': 2560, 'dtype': 'bfloat16_model_float32_index', 'elapsed_seconds': 657.6471912378911, 'max_length': 512, 'peak_reserved_mib': 12562, 'reused': True, 'rows': 14190, 'rows_per_second': 21.576918732504772, 'truncated_rows': 11456}`
- A4 replay seconds: `182.17332205898128`; per-stage latency: `{'bm25_ms': 458.0013536149636, 'coarse_ms': 1417.2903248865623, 'dense_ms': 134.17461537756026, 'expand_ms': 823.9586208675368}`
- No generator calls, validator changes, calculator arithmetic changes, Gold edits, or production index writes were made.

## Decision

- Primary ceiling: **RANKING_CEILING**
- Decision: **RETRIEVAL_RECOVERY_FAILED**
- Recommendation: **CONTINUE_TARGETED_RETRIEVAL**
- Candidate target counts at @50/@100/@200: `{'answerable_at_100': 93, 'answerable_at_200': 95, 'answerable_at_50': 89}`

The embedding model improves first-stage candidate inclusion only modestly in this bounded development scope and does not reach the requested final operating point. A later sprint should target the remaining ranking/representation bottleneck; do not open full runtime based on this result alone.
