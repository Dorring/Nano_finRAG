# P1.8-B5 — source adjudication of the ambiguous coordinates

The deliverable P1.8-B0 defined the contract for. Each conflicting coordinate
read against the competing rows the store actually holds, classified into the
contract's classes. **No files modified; correction of any gold is proposed, not
applied.**

Every row below is a measurement from the frozen legacy store. The question text
and the gold are from the frozen benchmark.

---

## The finding that overrides the coverage question

> **Most of these cases are not store defects and would not become answerable
> under a perfect store. The questions are underdetermined, and in several the
> gold is not an eligible reading of the question.**

Three distinct causes are present, and only one of them is the store:

```
1. STORE COLLAPSE     several rows share one label; the store lost which
                      column and which row.  Fixable by extraction.
2. QUESTION UNDER-    the question names a row label that several real
   DETERMINATION      quantities satisfy and names no column.  NOT fixable by
                      any store or retriever -- the eligible answer is undefined.
3. CELL-VALUE         the store emitted a fact per table cell, so labels,
   ARTEFACT           years and percentages became quantities.  Fixable at
                      emission (contract section 1.1).
```

---

## Class S — scoped, and the scope resolves it (4 cases)

The question names a scope and the store holds exactly one value for it.

```
tv2f01-s1-msft-016   "Microsoft's Cost of revenue"
   87,831  "Cost of revenue | 22,422 | 19,611 | 17,202"   <- the total row
   22,422 / 40,171 / 25,238                                <- segment rows
   eligible: 87,831.  Confirmed twice: it is the standalone cost-of-revenue row,
   and 22,422 + 40,171 + 25,238 = 87,831 exactly.
   VERDICT: gold correct, authority = the company-level total. RESOLVABLE.

tv2f01-s1-pfe-030   "Pfizer's Cost of sales FY2024"
   $17,851  "Costs and expenses follow:"                   <- the line
   14,997 / (1,360) / (5,656)                              <- components
   VERDICT: gold consistent with a company-level total. RESOLVABLE.

tv2f01-s1-tsla-033   "Tesla's Revenues FY2025"
   $82,056  "Revenues | $ 82,056 | $ 87,604 | $ 90,738"
   $12,771  "Revenues | $ 12,771 | $ 10,086 | $ 6,035"
   Two genuine revenue rows (total, and a component). The company-level one
   is 82,056. VERDICT: resolvable by class C rule, not class S.

tv2f01-s1-tsla-034   "Tesla's State value"
   5 and 21, both "State | 5 | 45 | 57" / "State | 21 | (49) | (653)"
   "State" is a tax-jurisdiction line, not a quantity the question can name.
   VERDICT: class A -- no scope can be named. NOT RESOLVABLE.
```

**Three of these four are resolvable**, all by the class-C company-level rule
rather than by scope. Combined with the canonical seam this is the same small
headroom measured in B1–B3; it is recorded here per-case rather than counted
twice.

## Class A — the question does not determine its answer (11 cases)

The decisive group. In each, the question names a row label and the store holds
several **genuinely different** quantities under it, with no column named and no
scope to appeal to.

```
tv2f01-s1-aapl-001   "what was the reported Deferred?"
   (1,804) / (139) / 604   all "Deferred | ...".  Three different deferred
   balances (tax assets/liabilities by class).  The question names none.
   NO ELIGIBLE READING.  -> abstention side

tv2f01-s1-aapl-003   "the figure for Services in FY2025"
   82,314 revenue / 75.4% margin / 109,158 / 26,844
   Four real figures, one label, no column named.
   NOTE: the benchmark's current gold is 75.4% -- a MARGIN -- for a question
   asking for "the figure for Services".  The natural reading is the revenue,
   82,314.  This is the sum-007 failure mode again.
   -> abstention side, or re-scope the question to name the column.

tv2f01-s1-ko-011   "Net foreign currency translation adjustments"
   2,868 / (32) / $(12,673)   three different adjustments (OCI, cumulative
   translation, tax effect).  No column named.  -> abstention side

tv2f01-s1-nvda-022   "how much did NVIDIA report for Colette M. Kress"
   2025 / 47,890 / 6,830,072 / 95,780 / 13,660,144
   A compensation table. "How much" has no single column: salary, stock
   awards and total compensation are all real.  The store also emitted the
   year 2025 as a value.  -> abstention side

tv2f01-s1-nvda-023   "the figure for Ajay K. Puri"   -- identical shape.
   -> abstention side

tv2f01-s1-jpm-008    "U.S. GSEs and government agencies"
   $92,112 / $1,075 / $2,215 / $90,972 and 89,073 / 57 / 9,200 / 79,930
   Two different tables, each with several columns (fair value, amortized
   cost, ...).  -> abstention side

tv2f01-s1-tsla-032   "Total automotive cost of revenues"
   57,165 / (5,708) / (9)%
   The gold is (5,708); 57,165 is Tesla's automotive cost of revenues in the
   income statement.  The gold does not match the natural reading.
   -> adjudicate; likely a second sum-007.

tv2f01-s2-diff-002   "Beginning balance at January 1, change FY2024->FY2025"
   seven values at FY2024, seven at FY2025 -- an allowance-for-credit-losses
   rollforward with one column per portfolio segment.  "The" beginning balance
   does not exist.  -> abstention side

tv2f01-s2-diff-004   "Visa's Commercial(3) change"
   1,042 / 613 / 1,655 (FY2024), 1,084 / 1,739 (FY2025)
   Rows labelled "Consumer debit(2)" -- the metric and the row text do not even
   agree.  -> abstention side (and a benchmark metric-label defect)

tv2f01-s2-growth-001 "growth rate in International-based companies"
   325 / 248 / 56 / 455 in one row; a different table adds 379 / 381 / 691 /
   10 / 2 / 703.  The gold 0.3105 = 325/248 - 1, i.e. two adjacent columns of
   the first row.  -> abstention side unless the question names the column
```

## Class B — cell-value artefacts (contract §1.1) (6 cases)

Not competing values at all: the store emitted a fact per table cell, so labels
and percentages became quantities. These are emission defects.

```
tv2f01-s1-aapl-005  two rows whose content is blank; values (294) and (358)
tv2f01-s1-v-036     2,101 / 15 / 2,116 -- 2,101 is the row's own total column
tv2f01-s1-v-037     $6,788 / $13,894 -- two regions' payment volumes
tv2f01-s1-v-039     5 / 4 / 36 / (13) / 101 under one "Income tax effect" label
tv2f01-s1-ko-015    "Intersegment" bound to a row reading
                    "Total net operating revenues | 11,513 | ..." -- the
                    metric and the row do not agree; 16 records for one row
tv2f01-s1-msft-020  (1) and 21 under "Other contracts"; 21 is a
                    statutory-tax-rate line
```

## Class C — resolvable company-level (2 remaining)

```
tv2f01-s2-sum-007  Coca-Cola operating income -- resolved in B0: the source
                   determines 13,762 + 9,992 = 23,754; gold 25,962 is the
                   investee aggregate.  Needs a gold correction (benchmark).
tv2f01-s2-sum-009  foreign currency contracts; one of two operands resolves
                   company-level, the other does not.
```

---

## The adjudication, stated as an outcome

```
resolvable, authority defined today                          5
  msft-016, pfe-030, tsla-033, sum-007 (with gold correction), sum-009 (partial)
class A -- no eligible reading without a question change    11
     aapl-001, aapl-003, ko-011, nvda-022, nvda-023, jpm-008,
     tsla-032, diff-002, diff-004, growth-001, tsla-034
class B -- emission artefacts                                6
     aapl-005, v-036, v-037, v-039, ko-015, msft-020
```

**Eleven cases would not become answerable under any store the project could
build.** They are questions whose text does not determine an answer — and for
`aapl-003` and `tsla-032` the current gold appears to point at the wrong
quantity, which is the sum-007 failure mode rather than a coverage problem.

That is the answer to the question P1.8 was convened to ask:

> When one filing reports several real values for the same entity / metric /
> period, what source-level evidence decides which is eligible?
>
> **The question text. When it names neither a column nor a scope, nothing
> does — and the case belongs on the abstention side, not in the coverage
> numerator.**

## Consequence for the coverage target

Coverage cannot reach 60% by resolving these, because 11 of 24 are not
resolvable by resolution — they are resolvable only by rewriting the questions
or by removing them from the answerable denominator. Both are benchmark
decisions, and both are `不修改` in this phase.

```
46/95 released today
24 of the 49 unreleased are in this ambiguous set
11 of those 24 have no eligible reading
```

Any coverage number that counts them as recoverable is counting cases whose
answer is undefined.

---

## The mechanism, confirmed by running the gate's own extractor

Every one of the 22 gate-blocked questions was run through
`extract_query_semantic_frame` and `canonical_metric_id` — the gate's real code,
not a model of it:

```
case                     plan metric                          query_metric_ids  canonical_metric_id(plan)
aapl-001                 'Deferred'                           []                None
aapl-003                 'Services'                           []                None
aapl-005                 'Hedge accounting fair value adj…'   []                None
ko-015                   'Intersegment'                       []                None
nvda-022                 'Colette M. Kress'                   []                None
tsla-032                 'Total automotive cost of revenues'  ['revenue']       None
...                      (all 22)                             [] (bar tsla-032)  None
```

**In all 22, the plan's metric resolves to `None` — it is not an ontology
metric — and the query text yields no metric id.** The gate's rule is
`unknown_query_fields = ("metric",) if not query_metric_ids`, and under
`STRICT_DIRECT_FACT` that makes the plan inadmissible.

The gate is not misclassifying anything. **These questions do not name a
metric.** `Deferred`, `Services`, `Intersegment`, `Colette M. Kress`,
`Commercial(3)`, `State value` are row labels, a person's name, and a table
cross-reference. A direct-fact question that names no metric has no defined
answer, and refusing it is the correct behaviour.

`tsla-032` is the one instructive variant: the query text *does* yield a metric
id — `revenue`, matched out of "revenues" — which then disagrees with the plan's
metric and produces `MISMATCH` instead. Same refusal, different branch.

### What this settles

There is no gate defect to fix and no vocabulary entry to add. Extending the
ontology with `Deferred` would be **worse**, not better: it would admit a
question whose answer is undefined and let the system bind one of three real
deferred balances and release it. That is precisely the incorrect-release
failure the phase's invariants exist to prevent.

**Coverage cannot reach 60% while these 22 remain in the answerable
denominator**, and they remain because the benchmark classifies them as
answerable direct-fact questions. Changing that is a Benchmark change.

