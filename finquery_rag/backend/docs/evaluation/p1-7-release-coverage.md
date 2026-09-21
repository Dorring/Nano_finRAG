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

## The ceiling, measured rather than argued

Before concluding that the remaining blocks need representation work, the
obvious objection has to be closed: maybe the admission rule is simply too
strict. So the gate was removed entirely and the run repeated on GPU.

```
gate fully bypassed      50/95 = 52.6%
of the 22 my rule blocks, bypass releases   3
the other 19 fail as:    EVIDENCE_CONFLICT 16
                         QUERY_EVIDENCE_SEMANTIC_MISMATCH 2
                         MISSING_OPERAND  1
```

**No gate policy reaches 60%.** The maximum any admission rule can achieve on
this benchmark is 52.6%, because the sixteen cases that remain are refused
downstream by the Binder at the same coordinates -- with the gate silent, the
Binder says `AMBIGUOUS` for the same reason the gate did. Loosening the gate
further buys three cases and hands the rest to a later refusal.

That also disposes of the "the values must be arbitrary, so the rule is too
strict" reading. Five of the eighteen conflicting coordinates do hold a
total-and-components structure -- Microsoft's `Cost of revenue` FY2025 is
87,831 = 22,422 + 40,171 + 25,238, confirmed by the prior-year column where
74,114 = 19,611 + 29,611 + 24,892. A total-detecting rule would recover about
five cases, taking the ceiling to roughly 54%, and it would be choosing on the
system's behalf which of several reported numbers the question meant.

Thirteen of the eighteen have no such structure. `Deferred` is deferred
revenue, deferred tax and deferred compensation; `Income tax effect` is five
unrelated numbers; `Intersegment` is seven.

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

### The one path that would reach 60%

The `s3-compare`/`rank`/`crossdiff` cases are a *separate* subsystem and are not
limited by the representation defect: their metrics are `Net income`,
`Total assets`, `Interest expense` -- unambiguous labels. They fail in
cross-entity binding, and the metrics are not the problem.

```
CAPABILITY_EXCEPTION        4   binder_returned_invalid_schema
MISSING_SLOT                3   (with NO_PROGRESS / BUDGET_EXHAUSTED)
SCV_RELATION_AMBIGUOUS      2
MISSING_OPERAND             2
QUERY_EVIDENCE_SEMANTIC_MISMATCH 1
EVIDENCE_CONFLICT           1
```

Recovering all thirteen takes release to 59/95 = 62.1%. Two of the obstacles are
deliberate design decisions rather than bugs, and neither is mine to reverse
silently:

- **`binder_returned_invalid_schema` is not retried.** `TransportRetryPolicy`
  freezes one semantic response and one transport retry, and `retryable_failures`
  lists only transport and HTTP errors. Re-rolling a *malformed* response is not
  the same act as re-rolling one the model disagreed with -- malformed output
  carries no semantics to re-roll -- but the freeze is explicit and says R0E.
- **Entity-bearing slot queries do not help.** Tested on GPU as the natural fix
  for a cross-entity pool: 5 of 20 compare cases released against 7 for the
  shipped entity-less query, and 46.3% overall against 48.4%. The hypothesis is
  refuted, not untested.

### The pool defect is real, and fixing it does not help

`s3-compare-002` asks which of Apple and Visa had the higher FY2025 net income.
Both slots issue `Net income FY2025`, and the twenty-candidate packet holds
seven Pfizer rows, four Tesla rows, one Apple row and **no Visa row**. The lanes
hold **747** candidates for that query, sixty of them Visa, best rank 9. So the
filer is reachable and pool construction discards it.

`CandidateDirectRetriever` now takes an `entity_key_lookup` that narrows each
slot to its own filer, before the pool is cut. The filter cannot lose a valid
binding -- `_entity_matches_slot` already rejects a fact from another company,
so a wrong-entity candidate was never bindable -- and the packet does gain the
missing filer. Measured:

```
                      shipped   entity-isolated
released (of 95)          46            47
compare cases           7/20          7/20
slots complete         47/75         45/75
EVIDENCE_CONFLICT          1             4
```

One case is inside this binder's run-to-run spread, so the honest reading is
**no demonstrated gain**: the packet was missing the filer, and the Binder still
could not choose once it had it -- `EVIDENCE_CONFLICT` rose, which is the
Binder seeing *more* competing facts and refusing them.

The lookup is therefore **off on every production call path**, exactly as the
structured reranker is, and for the same reason. The seam and the store API stay
so the next attempt starts from a measured position rather than a hypothesis.

## The arithmetic, so the decision is not a matter of opinion

Every lever that does not touch a safety invariant has now been tried and
measured:

```
source-grounded admission        +30 cases   (shipped, this round)
ontology aliases                  +1   2 refusals in 49 blocked cases
entity isolation at retrieval     +1   inside the Binder's spread, off
consolidated-total preference     +5   5 of 18 conflicting coordinates hold one
plan-entity inference             +2   refused by derive_slot_identities' design
                                  ---
ceiling from policy alone      <= 52.6%   measured with the gate fully removed
target                            60.0%
```

The gap cannot be closed by policy. What is left is (A) the Binder's conflict
and retry semantics, which the entity-isolation experiment showed is where the
refusals actually happen -- the packet had the filer and the Binder still would
not choose -- or (B) the table, column and row-hierarchy dimensions in the fact
representation, which is where the twenty-one gate blocks and seven of the ten
binding losses come from. Both change something this project has deliberately
frozen, and neither is a decision to take on the way to a metric.

## Not claimed

- **60% was not reached.** 48.4% is the measured value, and 52.6% is the
  measured ceiling for any gate policy.
- The two structural blocks above are diagnosed, not fixed.
- `pctshare-003` releases a wrong answer: the planner maps `What percentage of
  A was B` with A as numerator, and it is B/A. It is the only wrong release and
  it is a planner construction error, reported rather than papered over.
- Citation precision is 83.3% on strict gold-fact ids. Six of the twelve misses
  are the same row reached through a different table fragment -- the store holds
  each row twice, and `iPad` FY2025 is 28,023 under two fragment ids -- so
  precision by fact identity is 91.7%. Both numbers are reported; the strict one
  is the one in the table.

## The last two levers, measured and closed

Both remaining ideas were tried rather than argued, and both are small.

**Ontology aliases.** The second firewall -- `align_bound_evidence_to_query` --
refuses a binding whose fact's metric identity differs from its slot's, and that
refusal is invisible in the outcome trace (`semantic_check` stays in the Binder's
own round trace). Wrapping the function and re-running all 49 blocked cases
gives **two** refusals in the whole set:

```
fact_metric_not_matching_slot ... coca_cola_net_income_fy2025   (s3-compare-001)
fact_metric_not_matching_slot ... pfe-030 cost_of_sales
```

The second is incidental: `pfe-030` is a coordinate-conflict case whose Binder
already said `AMBIGUOUS`. So one case in the whole benchmark turns on a missing
alias -- Coca-Cola files the line as `Consolidated Net Income` where the slot
says `Net income`.

It is not added. `net_income` already carries `归母净利润`, which is net income
*attributable to the parent*; `Consolidated Net Income` is the total including
non-controlling interests, and the two are different numbers in any filing that
has one. The collision is pre-existing, and resolving it needs a decision about
which quantity `net_income` denotes -- not another alias, and certainly not one
whose measured effect is a single case.

**Entity isolation at retrieval.** Covered above: +1, inside the Binder's spread,
off by default.

**Two binding failures look like a plan defect and are not one.** `tsla-031` and
`tsla-033` have a slot whose `entity` is null, so nothing constrains the fact to
Tesla and the Binder returns `MISSING`. Deriving the entity from the question is
refused by design rather than overlooked: `derive_slot_identities` is one-way
from the slot's own mention, because "a plan cannot assert an identity its own
mention contradicts" and a mention the vocabulary cannot name means *constrained
and unnamed*, never *unconstrained*. Inventing one from the question is exactly
the inference that function exists to refuse, and it is worth two cases.

**Period misattribution is not the cause.** The one conflict shape that would
have been a correctable extraction defect -- a coordinate holding last year's
number under this year's label -- does not occur. Coca-Cola's `Operating income`
is clean: `FY2025 13,426`, `FY2024 12,536`, `FY2023 11,868`, one value each. The
conflicts are rows and columns, which is why they need the representation.

**And the record does not contain what would settle them.** Each fact's `content`
is its own row, so the row's cells survive:

```
Microsoft, Cost of revenue, FY2025
  value=87,831   content=| Cost of revenue | | 87,831 | 74,114 | 65,863 |
  value=22,422   content=| Cost of revenue | | 22,422 | 19,611 | 17,202 |
```

-- and the *label is identical on every one of them*. The header that tells them
apart, `Productivity and Business Processes`, sits in the table fragment and is
flattened onto all four rows equally. So the store records which cells a row
holds and not which row it is, which is exactly the missing dimension, and no
rule over the existing fields recovers it. `Colette M. Kress` is the same shape
one level down: its five "competing values" are four columns of a single row
plus the year column, read as if they were five quantities.
