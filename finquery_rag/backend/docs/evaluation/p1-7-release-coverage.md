# The release-coverage round — what the gate was refusing, and what it was not

Goal for this round: raise trusted-E2E **release coverage** while holding
`incorrect release = 0`. Measured on 120 cases with pinned plans, Gate ON,
`document_names=[]`, Store / Binder / Validator / Finalizer unchanged.

```
                    baseline    this round
release coverage     20/95        46/95
                      21.1%        48.4%
released correct     20/20        44/46
incorrect release       0            0
correct refusal      25/25        25/25
citation P / R     84.4 / 96.4   83.3 / 96.8
```

Both columns were measured on the same host, same day, same code except the
gate's vocabulary, with the specialist on GPU. 48.4% is the pinned-plan
release rate; the live-planner rate is a different number and is not claimed
here.

## First: the previous numbers were measured with a broken generator

The deployment env pins `CUDA_VISIBLE_DEVICES=2`, which is the GPU holding
another user's 14 GiB job. Every earlier run in this line therefore fell back to
CPU, and on CPU the local specialist dies:

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: c10::BFloat16 key.dtype: float and value.dtype: float
```

Seven of the twenty `s3-compare` cases were failing with `GENERATION_EXCEPTION`
for exactly this reason, and they release once a free GPU is used. The published
`17.9%` was measured under that degradation. It is not wrong as a *baseline* --
both columns here are re-measured -- but it is not what the system does.

`/tmp/base-bench` reproduces it: the four changed source files restored from
`f212676`, hashes verified against the originals
(`cc2648db` / `0b2dce53` / `7c120859`).

## What the gate was actually refusing

E3 sorted the 48 blocked cases into vocabulary classes. That was the right
question for *extending the ontology* and the wrong one for *blocking*. The
gate's job is not "is this a financial concept" -- it is "can we tell what
single quantity the question asks for". For a table-scoped row those are
different questions, and the second one the **source** can answer:

```
iPad                       @ (Apple, FY2025)        2 facts, 1 value  -> admit
Deferred                   @ (Apple, FY2025)        3 values          -> refuse
Services                   @ (Apple, FY2025)        4 values          -> refuse
Colette M. Kress           @ (NVIDIA, FY2025)       5 values          -> refuse
U.S. GSEs and gov agencies @ (JPMorganChase, FY25)  4 values          -> refuse
Microsoft Cost of revenue  @ (Microsoft, FY2025)    4 values          -> refuse
```

`28,023` is Apple's iPad revenue. `Deferred` is deferred revenue, deferred tax
and deferred compensation in one balance sheet, and the row label does not say
which. The admission rule is determinacy at the coordinate -- the same invariant
`src/finance/operand_ambiguity.py` already enforces on a bound operand, asked one
layer earlier where answering is still possible. There is no list of labels and
nothing keyed by case or question text.

## The four changes

1. **`literal:` identity keeps footnote markers** (`semantic_alignment.py`).
   `_FOOTNOTE_SUFFIX` strips them for the *concept* lookup -- `Cost of revenues
   (1)` is cost of revenue -- and that is right. Applying the same strip to the
   *literal* identity merged rows the store keeps distinct: Pfizer files both
   `Acquired in-process research and development expenses` and the `(g)` row
   stating different numbers, and the slot for one bound the other. That was the
   only wrong answer in the bypass arm.

2. **Plan metrics resolve to `metric_identity`, not `canonical_metric_id`.** An
   unnameable plan metric is now a literal identity instead of an automatic
   mismatch, and a literal is admissible exactly when the question carries it.

3. **`SourceLabelGrounding`** (`src/finance/source_label_grounding.py`) is handed
   to the coordinator by the production builder and widens what the gate can
   *name*. The mention test stays in the gate: a caller that supplies a label the
   question never mentions authorizes nothing.

4. **A term that appears only inside the name of the thing asked for is part of
   that name.** `Weighted-average shares-diluted` is a metric, and reading its
   `average` as a request to average something made the gate contradict a plan
   that had read the question correctly. The same rule stops `revenue` inside
   `International transaction revenue` counting as a second metric. Adding
   `percentage change` as a longer alias of `growth_rate` is the same fix from
   the other side.

Gate blocks 48 -> 22. Nothing fail-closed was relaxed: `incorrect release` is 0
in every arm and every run, and the 25 abstention cases are refused by the same
policy as before.

## Three measurement corrections

These changed numbers without changing behaviour, and each is a defect in the
measuring apparatus rather than in the system.

- **`score_nf_v3_final.py`** compared a growth rate's two conventions as
  written. The system renders `-202.04%` where the gold stores `-2.0204`; both
  are the same quantity, and the case bound exactly the gold facts. Rescaling
  cannot rescue a wrong answer -- a role inversion's reciprocal (`-23.1429`
  against a gold of `-0.0432`) still fails at every scale.
- **`report_system_metrics.py`** counted `released` only among comparable cases,
  so the headline read 16/95 where the rows said 17.
- **`analyse_release_gap.py`** bucketed unreleased `s3-compare` cases as
  `GOLD_NOT_IN_POOL`. Their gold carries a relation, not a fact id, so the gold
  set is empty and `gold_in_pool` is 0 by construction. Eleven of fourteen
  "retrieval misses" were this.

## Where the remaining 49 answerable cases go

```
GATE_BLOCKED_UNKNOWN          21   the coordinate does not identify one value
NONCOMPARABLE_UNRELEASED      13   s3-compare relation cases
GOLD_NOT_BOUND                10   gold in pool, not bound
VALIDATION_FAILED              2
GATE_BLOCKED_OTHER             1
SLOTS_INCOMPLETE               1
GOLD_NOT_IN_POOL               1
```

**The largest block and the largest binding loss are the same defect.** Seven of
the ten `GOLD_NOT_BOUND` cases have the Binder returning `AMBIGUOUS` at a
coordinate whose facts disagree -- Microsoft's `Cost of revenue` FY2025 is four
numbers, Visa's `U.S. Treasury securities` is three. The gate refuses the
question and the Binder refuses the binding, both correctly, for one reason: a
stored fact's coordinate is `(entity, metric, period)` and for many rows that
does not identify one value.

Recovering those needs the missing dimensions restored -- statement, table,
column, row hierarchy (P1.6-A). It is not a gate policy and not a retrieval
policy, and loosening either would admit guesses rather than answers.

The `s3-compare`/`rank` cases are a separate subsystem: cross-entity plans whose
slots each name a different filer, where the Binder returns `MISSING` for most
slots. Entity-bearing slot queries were tested as the fix and **do not work**
(arm C on GPU: 5 of 20 compare cases against 7 for the shipped query, 46.3%
overall against 48.4%).

## Not claimed

- **60% was not reached.** 48.4% is the measured value.
- The two structural blocks above are diagnosed, not fixed.
- `pctshare-003` releases a wrong answer: the planner maps `What percentage of
  A was B` with A as numerator, and it is B/A. It is the only wrong release and
  it is a planner construction error, reported rather than papered over.
- Citation precision is 83.3%, below the 90% asked for.
