# NF-V2-18A-R3 A4-Preserving Hierarchical Qwen Retrieval

Base: 29103493ff6dfd07c470444424a11dd0d639dd32

This is a development-only run on the consumed 120-question regression set. No generator, validator, calculator arithmetic, Gold, B3 artifact, production default, or production switch was changed.

## Frozen Qwen runtime

- Model: Qwen/Qwen3-Reranker-4B
- Revision: 22e683669bc0f0bd69640a1354a6d0aebcfeede5
- Snapshot SHA: 32b52d29edef232618c27a728cde56d369ba862c81d49c6527528dfd82db39c1
- Runtime config SHA: d7c1841432198b60f2266e1d86c107354b11af8c03be5562fdc9f859b173f01a
- Historical scorer/instruction: reused
- dtype/batch: bfloat16 / 4
- Retrieval Python: /mnt/disk/mxf/anaconda3/bin/python
- Qwen Python: /mnt/disk/mxf/anaconda3/envs/QhChat/bin/python
- Dynamic GPU: physical 3, NVIDIA RTX A6000, UUID GPU-bea0ffce-0725-385d-8d4e-d2ff4f00e6e8, tier 1, 48536 MiB free before launch, logical device cuda:0
- Qwen pairs: 3168 (single 1766, multi 732, calculation 670)
- Qwen elapsed: 629.397 s; throughput 5.033 pairs/s; peak reserved 44858 MiB

## A4/S5 complementarity

S5 is diagnostic only and is not selected as the family retriever.

| depth | A4 only | S5 only | both | neither |
|---|---:|---:|---:|---:|
| @5 | 16 | 5 | 46 | 53 |
| @10 | 16 | 4 | 52 | 48 |

Oracle union: exact R@5 = 67/120, exact R@10 = 72/120; multi All@5 = 9/20 at both depths; calculation operand-complete = 4/15 at @5 and 9/15 at @10.

## A4 family headroom

Counts are family hits; denominator is shown explicitly.

| A4 depth | all/120 | answerable/105 | multi/20 | calculation/15 |
|---|---:|---:|---:|---:|
| @5 | 84 | 84 | 20 | 12 |
| @10 | 86 | 86 | 20 | 13 |
| @20 | 87 | 87 | 20 | 14 |
| @30 | 91 | 91 | 20 | 14 |
| @50 | 92 | 92 | 20 | 15 |
| @100 | 94 | 94 | 20 | 15 |

The A4 coarse replay remained the authoritative candidate-recall substrate.

## Ablations

Exact cells are numerator/denominator. All rows use /120; answerable rows use /105.

| stage | all R@1/3/5/10/20 | answerable R@1/3/5/10/20 | Any@5 | All@5 | All@10 | All@20 | operand @5/@10/@20 |
|---|---|---|---:|---:|---:|---:|---|
| U0 A4 | 33/55/62/68/77 | 33/55/62/68/77 | 17/20 | 3/20 | 6/20 | 8/20 | 3/15, 5/15, 6/15 |
| U1 semantic only | 5/20/30/52/74 | 5/20/30/52/74 | 8/20 | 0/20 | 0/20 | 7/20 | 1/15, 2/15, 4/15 |
| U2 A4 + Qwen | 43/56/63/71/75 | 43/56/63/71/75 | 19/20 | 8/20 | 8/20 | 8/20 | 3/15, 3/15, 6/15 |
| U3 semantic + Qwen | 42/56/63/71/76 | 42/56/63/71/76 | 19/20 | 8/20 | 8/20 | 8/20 | 3/15, 3/15, 4/15 |
| U4 + slot multi | 42/55/63/71/76 | 42/55/63/71/76 | 19/20 | 10/20 | 12/20 | 12/20 | 3/15, 3/15, 4/15 |
| U5 + operand fan-out | 42/55/63/71/76 | 42/55/63/71/76 | 19/20 | 10/20 | 12/20 | 12/20 | 5/15, 5/15, 6/15 |
| U6 + targeted iXBRL | 42/55/63/71/76 | 42/55/63/71/76 | 19/20 | 10/20 | 12/20 | 12/20 | 5/15, 5/15, 6/15 |

U5 is the aggregate selected stage. U6 is retained only as a targeted numeric route because it did not change the aggregate scores.

## Selected routes

- Single evidence: U3 (A4 + local semantic expansion + Qwen)
- Multi evidence: U4 (slot-aware fan-out)
- Calculation: U6 (U5 operand fan-out plus targeted iXBRL provider)
- Qualitative: U3 (A4 text plus Qwen)
- A4 candidates lost during enrichment: 0

Family-to-exact conversion for the unchanged A4 coarse path is 62/84 at @5, 68/86 at @10, and 77/87 at @20. With an A4 family depth of 20, the selected fine top-5 returned exact evidence for 63/87 family-hit questions; this is a diagnostic conversion, not a replacement for exact Recall.

## Fine-ranking and semantic effects

- U0 to U1: rescued 1, damaged 33, unchanged 86, net -32.
- U1 to U3: rescued 35, damaged 2, unchanged 83, net +33.
- U3 to U4: rescued 0, damaged 0, unchanged 120.
- U4 to U5: rescued 0, damaged 0, unchanged 120.
- U5 to U6: rescued 0, damaged 0, unchanged 120.
- Qwen rank effect at top-5: rescued 8, damaged 9, unchanged 103, net -1. By route: single 5/7/73, multi 0/1/19, calculation 3/1/11 (rescued/damaged/unchanged). Gold-content analysis: TABLE_ROW 8 rescued and 9 damaged; the remaining 15 scored cases were unresolved/UNKNOWN; no separate Gold TEXT rescue was observed in this rank-effect artifact.

## Known 22-case regression

| class | total | semantic-only U1 | Qwen U3 | combined U4/U5 | still missed |
|---|---:|---:|---:|---:|---:|
| MULTI_LEVEL_HEADER | 14 | 0 | 7 | 7 | 7 |
| PERIOD_COLUMN_MISMATCH | 5 | 1 | 2 | 2 | 3 |
| CHILD_RANK_FAILURE | 3 | 0 | 1 | 1 | 2 |
| total | 22 | 1 | 10 | 10 | 12 |

No damaged or wrong-period binding was admitted in this fixed regression summary; global wrong-period binding remains 0.

## Latency

Per-question stage distributions (milliseconds) are preserved in latency.json:

| stage | mean | P50 | P95 | max |
|---|---:|---:|---:|---:|
| BM25 | 781.01 | 391.87 | 1882.38 | 13906.10 |
| A4 coarse | 1794.23 | 1489.26 | 4135.41 | 15415.11 |
| dense component | 158.41 | 87.23 | 523.02 | 812.62 |
| semantic expansion | 853.41 | 787.22 | 1658.28 | 3965.43 |

Semantic graph build was 298.993 s. Qwen was measured as a batched child process (629.397 s for 3168 pairs), so it is reported separately rather than fabricated as a per-question latency.

## Safety invariants

All remained zero: A4 candidates lost, authorization leakage, entity violation, fiscal violation, document-type violation, version violation, silent relaxation, wrong-period binding, false binding, and created_at misuse.

## Decision

Decision: RETRIEVAL_PARTIALLY_RECOVERED

The selected path materially improves multi-evidence completion (All@10 6/20 to 12/20), multi Any@5 (17/20 to 19/20), and calculation operand completeness (5/15 remains at @10; @5 improves 3/15 to 5/15). It does not reach the requested answerable exact targets (R@5 79/105, R@10 90/105) or calculation target (11/15), so the next recommendation is CONTINUE_TARGETED_RETRIEVAL rather than opening the full-runtime gate.

Production remains V1; production switch is false. Next gate, if a later development sprint reaches the operating point, is NF-V2-18B_FULL_RUNTIME_RECOVERY.
