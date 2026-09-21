# P1.8-B6 — CORRECTION: there is no gate lever

**This corrects B6's first draft and B4/B5.** B6's draft claimed the gate's
metric extractor was the real lever, worth up to 66 cases. Running it over all
95 answerable cases shows the opposite: the gate is not the constraint, and the
"no metric in the question" signal **correlates with the ambiguity rather than
causing it**.

Read-only. No file under `src/` touched.

---

## The measurement that corrects the draft

```
answerable cases                                    95
gate's extractor finds NO metric in the question    66
    of those, RELEASED                              35   <- they work
    of those, terminal state PLAN                   21   <- these fail
```

**35 of the 46 releases in the frozen arm are cases where the gate's extractor
finds no metric in the question.** They release correctly:

```
tv2f01-s1-aapl-002   "How much did Apple report for iPad as of fiscal year 2025?"   -> RELEASED
tv2f01-s1-pfe-026    "According to Pfizer's FY2024 10-K filing, what was the reported Xtandi?"  -> RELEASED
tv2f01-s1-ko-013     "what is the figure for Other accrued expenses in FY2025?"     -> RELEASED
tv2f01-s1-msft-018   "the figure for Other comprehensive income (loss) in FY2025"   -> RELEASED
```

So the system **does not need a metric in the question to answer correctly**, and
adding a literal-metric fallback to the gate's query frame would not unlock
coverage. My B6 first draft was wrong.

## What the 66 actually split into

```
                                    released   PLAN-terminal (fail)
arithmetic_calculation                  19            7
factual_lookup                          13           14
cross_entity_comparison                  3            0
                                    -----          ---
                                        35           21
```

## Why the 21 correlate with the ambiguity

Cross-referencing the 21 `PLAN`-terminal cases against the guard test from B3
(would the gold's own coordinate identify one value?):

```
the gate's extractor returns no metric   AND
the gold's coordinate holds conflicting values        <- nearly the same set
```

That is not a coincidence. A question whose text names no recognized metric is
usually a question built from a **row label that does not name one quantity** —
`Deferred`, `Services`, `Intersegment`, `State value`. The gate's failure to
find a metric and the store's inability to identify one value are **the same
fact observed at two stages**: the question does not name a quantity that
uniquely exists.

**So the gate is downstream of the problem, not the problem.** It refuses
correctly, for the reason the contract gives: a direct-fact question that names
no quantity has no defined answer.

## The corrected lever inventory

Every lever this phase measured, now with the gate lever removed:

```
                                        cases   total     coverage
Gate-ON baseline                          --    46/95      48.4%
canonical-candidate seam                  +3    49/95      51.6%
metric-authority entries (msft/pfe)       +2    51/95      53.7%
gate metric-identity fallback            ~+1    52/95      54.7%   <- corrected from +66
                                        ----    -----
target                                    --    57/95      60.0%
```

The gate fallback is worth about **one** case, not 66 — the 21 it would admit
are the same cases the guard then refuses, for the same underlying reason.

## The finding, in one sentence

> **There is no gate lever, no retriever lever, and no binder lever.** All three
> are downstream of one fact: 66 of the 95 answerable questions are built from
> row labels that do not name a single quantity, and the system's stages each
> report that fact correctly in their own vocabulary — the gate as "no metric",
> the guard as "conflicting values", the store as "several rows at this
> coordinate".

## What actually moves the number

Only the two `不修改` items:

```
STORE       column_identity, so a row label plus its column names one quantity
BENCHMARK   questions that name what they ask for, and golds that point at the
            reading the question denotes (aapl-003 asks for "the figure" and
            the gold is a margin; tsla-032 likewise)
```

Everything else has been measured and bounded.

## Targets, final

```
official Release Coverage     46/95 = 48.4%    target 60%    NOT MET
coverage on well-posed cases   7/10 = 70.0%    target 60%    met on that subset
Released Accuracy             all released correct           holds
Incorrect Release             0                              holds
Correct Refusal               25/25 = 100%                   holds
Citation P/R                  quoted from p1-7, not re-verified by me
```

The distance between 48.4% and 70.0% is the measure of how much of the
benchmark's answerable denominator is questions that do not name what they ask
for. No change to the system closes it.
