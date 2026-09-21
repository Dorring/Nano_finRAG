# P1.8-B2 — the coverage target against the representation gap

This answers the question P1.8-B1 left open: **how much of the residual gate
headroom is real?** Measured, not estimated. No code changed; no backend
stopped.

## Headline

> **The 60% target is not reachable by any retrieval optimisation on the current
> store.** Every lever this phase has examined is bounded, and the bounds do not
> add up to 57/95. The binding constraint is a representation gap in the fact
> store, not a retrieval defect — and the fix for it is the store migration the
> Benchmark Authority Contract deliberately deferred.

---

## 1. All 22 gate refusals are the same failure, and the gate is right

Every one of the 22 is `QUERY_PLAN_SEMANTIC_MISMATCH` with
`unknown_query_fields = ['metric']`, terminal state `PLAN`, `pool_depth = 0`.
They never reach retrieval.

The metric strings are:

```
Deferred · Services · Hedge accounting fair value adjustments · Cash reserves
at non-U.S. central banks · Common stockholders' equity(f) · U.S. GSEs and
government agencies · Intersegment · Other contracts · Colette M. Kress ·
Ajay K. Puri · Direct Customer A value · Total automotive cost of revenues ·
State value · Total nominal payments volume(4) · Income tax effect ·
Beginning balance at January 1 · Commercial(3) · International-based companies ·
Total noninvestment-grade · Direct Customer B · Other letters of credit(d) ·
Foreign currency contracts
```

These are **free-text row labels lifted out of filing tables**, several still
carrying their footnote markers (`(f)`, `(3)`, `(4)`). The gate's
`STRICT_DIRECT_FACT` policy refuses to bind a direct fact to a metric it cannot
interpret, and that refusal is correct behaviour, not a defect.

## 2. Extending the gate is not the lever either

The natural fix is to add these to the `MetricDefinition` ontology. Measured
against what would then have to hold — the bound value must *also* resolve to
exactly one undimensioned iXBRL fact, or the operand guard refuses downstream:

```
CO_LEVEL_UNIQUE      4      both the gate extension AND the company-level rule work
PARTIAL              1
NOT_CO_LEVEL        17      the gold value is not a company-level fact at all
```

```
GATE_ON 46/95  +  4  =  50/95 = 52.6%        need 57
```

And those 4 are weak. Two are quantity-shaped and genuinely defensible:

```
aapl-001   Deferred 604           1 concept
v-039      Income tax effect 4    2 concepts -- tied, i.e. NOT unique
```

Two are artefacts the contract's §1.1 already excludes as cell-values:

```
tsla-034   State value 5          matches 4 unrelated concepts -- a column
                                  header read as a quantity
msft-020   Other contracts 21     matches us-gaap:EffectiveIncomeTaxRate
                                  ReconciliationAtFederalStatutoryIncomeTaxRate
                                  -- a coincidence of arithmetic, not an authority
```

So the defensible yield from extending the gate ontology is **one case**, and it
costs a change to the gate's vocabulary. That does not justify the change.

## 3. Why the arithmetic cannot work

```
                                cases   running total   coverage
Gate-ON baseline                   --        46/95        48.4%
+ all 22 gate refusals released    22        68/95        71.6%   <- unreachable
+ what a gate extension clears      4        50/95        52.6%
+ canonical candidates (B1)         3        53/95        55.8%
                                ----        -----
target                            ---        57/95        60.0%
```

The 22 gate cases are disjoint from the 9 `EVIDENCE_CONFLICT` cases, so the two
levers do compose — and they compose to **53/95 = 55.8%, four cases short of the
target**, on the most generous reading of both.

## 4. The constraint, stated once

Look at what the 17 `NOT_CO_LEVEL` cases actually ask for:

```
"the figure for Services in FY2025"                       a product line
"the figure for U.S. GSEs and government agencies"        a securities category
"Coca-Cola's Intersegment"                                a segment elimination
"NVIDIA's Direct Customer A value"                        a customer concentration
"JPMorganChase's Commercial(3)"                           a portfolio segment
"the Beginning balance at January 1, change"              a rollforward line
```

Every one is a **scoped disclosure question** — class S in the Benchmark
Authority Contract — and class S resolves only against a fact whose context
carries the named scope.

The legacy store carries no scope, no dimension, and no table identity: its
fields are enumerated in `p1-8-b0-independent-verification.md` §5, and there is
no axis, no caption, and `table_id` is `null` on all 20,394 records. Meanwhile
the filings do carry the scopes — the iXBRL layer holds **125 axes and 1,083
members**.

So the system is refusing correctly, for a reason it cannot express:

> **The value is in the store. The scope that would make it the right value is
> not. No retrieval change can supply what extraction discarded.**

## 5. What this means for the phase

The Benchmark Authority Contract already named this and required it be settled
first. §5 of that contract:

> The store does not record this labelling. logical-table authority is therefore
> NOT ESTABLISHED IN THE LEGACY STORE, and a row's membership of a table is not,
> by itself, evidence that the row is about the filer.

The measurement above is the consequence of that, quantified. Three independent
levers have now been bounded:

```
levers measured this phase         yield      reaches 57?
canonical-candidate retrieval     +3 (2)      no
gate ontology extension           +4 (1)      no
both together                     +7          no -- 53/95
```

**The 60% target requires the store to carry statement/table scope.** That is a
single, well-scoped data project — the fields are already identified
(`statement_or_table_scope`, `row_role` from P1.7's audit) and the source
material is confirmed present (125 axes, 1,083 members). It is a store rebuild,
and a store rebuild invalidates the gold's `v2fact:` ids and the pinned
`benchmark-version.json` hash, so it is a benchmark version bump taken
deliberately.

## 6. What I did not do

```
did not modify        any file under src/, or the store, benchmark, gate,
                      binder, validator, finalizer, retriever or sidecar
did not stop          the backend or any GPU process
did not extend        the gate ontology, or apply the B1 seam
did not treat         60% or 80% as an adjudication criterion
```

Analysis ran read-only against `/tmp/g-*/` and the two frozen stores.

## 7. Recommendation

The honest next step is not a retrieval change. It is a decision about whether
to scope the store rebuild that would give these 17 cases an authority — and
before that, a decision about whether the benchmark's gold for them is even
right, since a scoped question currently has no defined answer to be scored
against.

Two smaller items can proceed independently and are worth their cost:

1. **The 4 arithmetic `CO_LEVEL_UNIQUE` cases within reach** — `msft-016` and
   `pfe-030` are defensible standard-taxonomy company-level totals. Small, real,
   and safe.
2. **The `s3` net-income concept split** (`ProfitLoss` vs `NetIncomeLoss`) is
   already handled by `CONCEPT_ALIGNMENT` ordering; verify it holds after any
   store change.

Neither is worth a freeze lift on its own. Both belong inside the rebuild scope.
