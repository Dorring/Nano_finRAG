# P1.8 coverage sprint, step 1 — where the last 25 stop

D1 sealed at 52/77 released, all 52 correct. The remaining 25 answerable cases
each get exactly one `FIRST_FAILURE_STAGE`, earliest stage wins, and the audit is
finished only when `UNATTRIBUTED = 0`.

Measured on the sealed v9 run (`c3-full-v9`), V2 gold `a3d17211`, patch OFF.

```
python scripts/evaluation/audit_p1_8_first_failure.py \
    --predictions <v9 run>/replay-predictions.jsonl \
    --gold <V2>/gold-evidence-v1.jsonl --out first-failure.json
```

## Two views, because the gate is overridden in this track

The replay overrides a gate refusal so the rest of the chain can be measured, so
a gate-refused case has **two** answers and they are not the same number:

```
                        pipeline order    gate rule dropped
SEMANTIC_ALIGNMENT            10                 —
BINDING                       10                17
SLOT_COMPLETENESS              3                 4
ADAPTER_CONTRACT               2                 2
VALIDATION                     —                 2
                             ---               ---
                              25                25
UNATTRIBUTED                   0                 0
```

Both are true. The first is "where it stops in production". The second is "what
stops it once the gate is out of the way" — and since the gate is already
overridden in this track, those are the stages the ten refused cases actually
reached. A plan that reads only the first spends itself on the gate; one that
reads only the second never sees it.

## `BINDING` is the block, on either reading

17 of 25 with the gate dropped, 10 of 25 in pipeline order — the largest single
stage in both. The shapes, from the Binder's own final status:

```
binder ambiguous (EVIDENCE_CONFLICT)    msft-016  pfe-030  sum-007  crossdiff-004*
binder missing   (MISSING_*)            tsla-033  sum-001  compare-007  crossdiff-003  crossdiff-005  rank-005
binding did not reach evidence          compare-001  sum-007
```

`*` gate-refused as well. The ambiguous group is the same defect P1.8-B
identified and D1 did not touch: a coordinate holding several values, so the
Binder correctly refuses to choose. `msft-016` is the documented
`cost_of_revenue` case and `pfe-030` its sibling — both golds are correct and
both wait on the runtime resolving `cost_of_revenue`, which is why `p1-8-b9`
kept them out of the benchmark migration.

The `MISSING_*` group splits further and should not be worked as one bucket:
`BUDGET_EXHAUSTED` (crossdiff-005, rank-005) is a budget that ran out while
slots remained, not a coordinate that could not be resolved.

## The other three stages

`SLOT_COMPLETENESS` (3–4) is binding that returned something and still left an
operand missing — `CALCULATION_INVALID, INSUFFICIENT_OPERANDS` for
`compare-002/009`, and one slot short on `compare-010`.

`ADAPTER_CONTRACT` (2) is `CAPABILITY_EXCEPTION` on `compare-003` and
`rank-001`, the shape `p1-8-b9` already patched for company-level metrics. Both
are relation questions the adapter could not express.

`VALIDATION` (2, gate-dropped only) is `nvda-025` and `tsla-035`, where the
binding succeeded and the validator refused on period fidelity and unit support.

## What this cannot say

**`RETRIEVAL` is empty, and that is an artefact, not a finding.**
`reached.retrieval` records that the stage ran, not that the gold reached the
pool, and the sealed replay does not carry `gold_in_pool`. A case whose gold
never reached the pool is therefore attributed one stage late, to `BINDING`.
Separating the two needs a per-case pool-membership measurement the sealed run
does not have — the one missing measurement before this ladder can be trusted at
the retrieval boundary.

Cross-checking against P1.8-B's independent finding: all 22 gate-blocked gold
facts *are* in the store, so the expectation is that little hides at
`RETRIEVAL`. That is an expectation, not a measurement.

## Not started

No coverage change has been proposed, made or measured. This is the map.
