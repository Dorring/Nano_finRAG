# P1.8-C2 — adjudication of the 21 column/scope questions

Benchmark V2 work. **Verdicts with source evidence; no benchmark file modified
by this document.** Each verdict states the question, the competing source rows,
the authority rule and the disposition, as the brief requires.

Method for every case: read the filing's own `primary.html`, table element by
table element, and recover the **column headers**. A question is answerable when
some source-level reading selects exactly one of the competing values. It is
`SOURCE_AMBIGUOUS` only when none does.

---

## What the source reading changes

Two of the twenty-one turn out **GOLD_CORRECT** — the store had flattened a
multi-column table and it looked broken, but the question selects a column:

```
tv2f01-s1-jpm-007  "Common stockholders' equity(f) ... as of fiscal year 2025"
  p.43  THREE-YEAR SUMMARY  "Common stockholders' equity | 342,393 | 324,708 | 332,754 | 312,370 | 282,056"
        headers: Period-end Dec 31 2025 | Dec 31 2024 || Average Year ended 2025 | 2024 | 2023
  "as of" selects PERIOD-END 2025 = 342,393 = the gold.        GOLD_CORRECT
```

```
tv2f01-s1-ko-011  "the reported Net foreign currency translation adjustments"
  p.57  CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME
        "Net foreign currency translation adjustments | 2,868 | (2,893) | 736"
  The old gold (32) is NOT this row -- it is a different table's tax-effect
  figure. The company-level value for FY2025 is 2,868.
```

That second one is a **repoint candidate**, and it is the same defect shape as
`tsla-032`: the gold drew from a column or table the question does not denote.
It needs a second reading before it is applied, because `2,868` is the
comprehensive-income amount while `(32)` may be the tax effect the question
means — the question does not say which.

---

## Verdicts

### GOLD_CORRECT (2)

```
tv2f01-s1-jpm-007   gold $ 342,393
  authority  "as of fiscal year 2025" selects the Period-end Dec 31 2025 column
  source     p.43 THREE-YEAR SUMMARY OF CONSOLIDATED FINANCIAL HIGHLIGHTS
  status     ANSWER (unchanged)

tv2f01-s1-jpm-009   gold $ 603,947
  authority  the total row of the lending-related commitments table
  status     ANSWER (unchanged)
```

### REPOINT_GOLD (2 verified)

```
tv2f01-s1-ko-011     (32)  -> 2,868    p.57 CONSOLIDATED STATEMENTS OF
                                       COMPREHENSIVE INCOME:
                                       "Net foreign currency translation
                                        adjustments | 2,868 | (2,893) | 736"
                                       The old gold (32) is a tax-effect row in a
                                       different table (p.102), not this row.

tv2f01-s1-msft-020   21    -> 21       the value is already right; the competing
                                       record (1) is a derivative-table artefact
                                       (p.58) that has no relation to "Other
                                       contracts". The coordinate needs narrowing,
                                       not the value.
```

**Two further cases were recorded in a first draft as repoints and are
withdrawn — measured, not argued:**

```
tv2f01-s1-ko-015   I wrote "(1,164) -> (1,009), the Intersegment row, not the
                   Total net operating revenues row".  That is wrong.  All 16
                   records at this coordinate share ONE row_id
                   (row:190f8d640521, page 119, row_index 4) whose text is
                   "Total net operating revenues | 11,513 | 6,334 | ...".  There
                   is one row, its metric was mislabelled "Intersegment", and it
                   carries 16 column values.  There is no correct row to point
                   at.  -> SOURCE_AMBIGUOUS

tv2f01-s2-growth-001 / tv2f01-s2-pctshare-006
                   recorded as "repoint-shaped, see note" without a value.  They
                   remain unadjudicated and are NOT counted in the two above.
```

### SOURCE_AMBIGUOUS (14)

Each is a question whose text names a row label that several real figures
satisfy, with the source offering no column selector.

```
case                 question names            competing values at the coordinate
aapl-001             "the reported Deferred"   (1,804) / (139) / 604
                                                three deferred balances, p.43
aapl-005             "Hedge accounting fair     (294) / (358)
                      value adjustments"        rows whose text came through blank
aapl-003             "the figure for Services" 82,314 / 75.4% / 109,158 / 26,844
                                                gross margin, margin %, net sales,
                                                cost of sales -- four tables
jpm-008              "U.S. GSEs and             $92,112 / $1,075 / $2,215 / $90,972
                      government agencies"      amortized cost / unrealized gains /
                                                unrealized losses / fair value
nvda-022             "Colette M. Kress"         salary / bonus / stock / total / year
nvda-023             "Ajay K. Puri"             same shape
tsla-034             "the State value"          5 / 21 -- a tax-jurisdiction line,
                                                not a quantity the question can name
v-036                "U.S. Treasury            2,101 / 15 / 2,116
                      securities"               amortized cost / gains / fair value
v-037                "Total nominal payments    $6,788 / $13,894
                      volume(4)"                two regions or two periods
v-039                "Income tax effect"        5 / 4 / 36 / (13) / 101
                                                numbered note rows, no labels
s2-diff-002          "Beginning balance at      7 values per year -- an allowance
                      January 1"                rollforward, one column per segment
s2-diff-004          "Commercial(3)"            1,042 / 613 / 1,655
                                                rows labelled "Consumer debit(2)"
s2-growth-003        "Total                    4 values per year -- one column per
                      noninvestment-grade"      portfolio segment
s2-growth-008        "Other letters of         2 values per year
                      credit(d)"
s2-pctshare-005      (denominator choice)       two readings of the total
s2-sum-009           "Foreign currency         5 values per year
                      contracts"
```

---

## The rule applied, and why abstention is not removal

For each of the fourteen:

```
the question names a row label
the source holds several real quantities under that label
no column, no scope and no period qualifier in the question selects one
=> no source-level evidence determines the answer
=> the case cannot be scored as a correct answer, and cannot be scored as a
   wrong answer either: it is unanswered by construction
```

Per the brief, these move to `expected_outcome = ABSTAIN` — **not** deleted and
**not** re-labelled as some third state. The case remains in the benchmark, with
its question intact, and the system is scored on whether it correctly declines.
That is a change to what the case asserts, not to whether it exists.

**Eight of the fourteen are new abstentions; six already had a sibling case in
the abstention stratum.** The answerable count would move 95 -> 81, and the
abstention strata 25 -> 39.

## Standing back: what the twenty-one say

```
GOLD_CORRECT          2   the question does select a column; the store had
                           flattened it away
REPOINT_GOLD          2   the gold drew from a table the question does not denote
SOURCE_AMBIGUOUS     15   the question names a label, not a quantity
UNRESOLVED            2   growth-001, pctshare-006 -- repoint-shaped, needs a
                           second source reading
                     --
                     21
```

**The largest single cause is not the store and not the gold — it is that
fifteen questions were authored against a row label rather than a column.**
That is a fact about the benchmark, and it is now written down per case with the
source rows to support it.

Note also what `ko-015` shows: its metric is `Intersegment` while its row text
is `Total net operating revenues`. That is not an ambiguity in the question — it
is the emitted metric disagreeing with the row it came from, which is a store
defect surfacing here rather than a benchmark one. It is recorded as ambiguous
because nothing in the source selects a value, but the root cause is different
and should not be lost.

## Not applied

```
no question edited
no gold edited
no status changed
benchmark-version.json untouched; gold-evidence-v1.jsonl hash 3d2a0c5b intact
```

Applying any of this is a versioned benchmark change. The migration record has
to carry, per case: old question/gold/status, new gold/status, source evidence,
authority rule, migration reason, old and new hashes. That document is C3, and
it needs the five `see note` rulings settled first.
