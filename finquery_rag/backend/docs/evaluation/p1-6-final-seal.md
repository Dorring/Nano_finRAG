# Final seal — trusted financial RAG, measured

Everything below is recomputed from case-level artifacts (`report_system_metrics.py`),
not read from a summary. Numbers are on the frozen benchmark
`tv2-canonical-v1-post-p1.6-0h` unless a row says otherwise.

## Retrieval — 75 scorable cases

| | R@5 | R@10 | R@20 |
|---|---:|---:|---:|
| shipped hybrid RRF | 50.7% | 58.0% | 65.3% |
| slot-local pool, no rerank | 77.333% | 87.333% | 93.333% |
| **structured rerank** | **85.333%** | **93.333%** | **95.333%** |

The reranker orders candidates by `(entity, metric, period)` agreement with the
plan, resolved through the system's own vocabularies — `canonical_entity_id`,
`canonical_metric_id`, a normalized period key — with token overlap as the
fallback where the ontology cannot name a label. No gold, no case ids, no branch
on question text. The 8-point gain over the unreranked pool came from comparing
*identities* rather than surfaces: `The Coca-Cola Company` vs `Coca-Cola`
matched nothing as strings, and `FY2025` vs `2025` threw the same period away.

## Trusted E2E — 120 cases, Gate ON, pinned plans

| | E1 | E2 | **E3** |
|---|---:|---:|---:|
| gate blocked (of 75 comparable answerable) | 66 | 57 | **48** |
| answerable release rate | 5.3% | — | **17.9%** |
| released correct | — | 11/11 | **17/17** |
| **incorrect release** | **0** | **0** | **0** |
| correct refusal (no-answer) | 25/25 | 25/25 | **25/25** |
| citation precision / recall vs gold | 0% / 0% | 60.9% / 73.7% | **79.4% / 90.0%** |

The gate was the bottleneck, not retrieval: production slot retrieval already
reached 94.7% gold-in-pool while the gate refused 66 of 75 answerable questions
before any search. Auditing those blocks and extending the ontology with 21
source-grounded canonical metrics (13 → 34) plus footnote-marker normalisation
took release from 5.3% to 17.9% **without relaxing a single fail-closed policy**.

## Structured reranker — benchmark-positive, production gain not demonstrated, DEFAULT DISABLED

This is the line that must not be dropped.

```
                      A shipped    B +rerank
  gate blocked              48          48
  gold in pool              25          25
  gold bound                17          17
  slots complete            21          21
  released                  16          16
  incorrect release          0           0
```

Six runs (3 per arm) — `16,16,16` vs `16,16,16` on released, `0` incorrect
release throughout. One earlier B run read 15; across four B runs the values are
15,16,16,16, so the honest statement is **no demonstrated production gain**,
not "neutral" and not "regression".

**Why, measured.** The rank-transfer audit over the traced cases:

```
GOLD_NOT_IN_POOL             87    the gold was never retrieved
UNCHANGED                    38
PROMOTED_ALREADY_VISIBLE     12
PROMOTED_INTO_WINDOW          0    <- the bucket that would move E2E
DEMOTED_OUT_OF_WINDOW         0
```

Promotions are real — rank 8→5, 7→2, 4→3 — but the pool is already 40 deep and
cut at 40, so **every promotion kept the gold inside the window the Binder was
already reading.** `PROMOTED_INTO_WINDOW = 0` is the whole explanation: the
offline metric measures rank quality over the whole list, the Binder reads a
window, and the reranker was improving ranks inside that window.

## Store, safety, reliability

```
Store                27,454 facts · 100% provenance · 80,563/80,563 source cells
Safety gates         wrong_scope / wrong_metric / value_mismatch / dangerous_authority = 0
Reliability          15 injected faults · 0 incorrect releases
Tests                966 passed, 1 skipped
Fixture store        frozen: benchmark-version.json pins every file hash
```

## The two results are independent

`85.333%` is a retrieval-benchmark number. `17.9%` is a trusted-E2E release
rate. **The A/B above shows the former does not cause the latter**, and the
rank-transfer audit says why. Any write-up that implies otherwise is false.

## Stopping here

Development stops. What is left is turning each number into an interview answer:
the experiment that produced it, the code path behind it, why it is credible,
and — for the reranker — why it is *not* claimed.
