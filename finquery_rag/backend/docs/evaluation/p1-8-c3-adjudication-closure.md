# P1.8-C3 — adjudication closure and migration proposal

**UNRESOLVED = 0.** Benchmark files still untouched; `gold-evidence-v1.jsonl`
hash `3d2a0c5b` intact. This is the migration **proposal** the brief asks for —
reviewable, not applied.

Verdicts for the five remaining cases, each with its source evidence and
authority rule, plus two `STORE_SEMANTIC_DEFECT` flags per the brief's third
principle: *if the case comes from a Store semantic-attribution error rather
than the gold pointing at a wrong cell, do not paper over it with a repoint.*

---

## The five remaining verdicts

### `tv2f01-s1-ko-011` → **SOURCE_AMBIGUOUS**

```
Q        "According to The Coca-Cola Company's FY2025 10-K filing, what was
          the reported Net foreign currency translation adjustments?"
old gold (32)
```

Three rows at this coordinate, in three tables:

```
p.57  CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME
      "Net foreign currency translation adjustments | 2,868 | (2,893) | 736"
      -> the consolidated comprehensive-income amount
p.102 "Net foreign currency translation adjustments | (32) | 25 | 1 | (2)"
      -> four columns; the tax-effect or allocation reading   <- old gold
p.108 "Net foreign currency translation adjustments | $ | (12,673) $ | (15,610)"
      -> a cumulative-translation or rollforward balance
```

**Asked for a disambiguator in the question and there is none.** The question
names an entity, a period and a row label. It does not say *in comprehensive
income*, *before tax*, *tax effect*, or *cumulative*. Each of the three is a
legitimate answer to the question as written.

`2,868` is the most natural reading — it is the consolidated statement's own
line — but "most natural" is not "determined by the question", and choosing it
would be this audit writing the question. `SOURCE_AMBIGUOUS`, as you indicated.

### `tv2f01-s1-msft-020` → **STORE_SEMANTIC_DEFECT**

```
Q        "What was Microsoft's Other contracts in FY2025?"
gold     21
```

```
p.60  "| Other contracts | | 21 | | (15) | | (41) |"    <- the gold; correct
p.58  "Gross amounts of derivatives ... (1)"           <- competing record
```

The gold is right. The competing record is a **derivative-table cell** whose
metric was attributed `Other contracts` although the row has nothing to do with
it. The value `21` is also `(15)`'s neighbour, i.e. a change column of the same
financial-instruments table.

**Not a repoint.** The gold does not point at a wrong cell; the store attributed
a row to the wrong metric. Recorded `STORE_SEMANTIC_DEFECT` and the gold is kept.

### `tv2f01-s1-ko-015` → **SOURCE_AMBIGUOUS + STORE_SEMANTIC_DEFECT**

```
Q        "What was The Coca-Cola Company's Intersegment in FY2025?"
old gold (1,164)
```

Measured: **all 16 records at this coordinate share one `row_id`**
(`row:190f8d640521d1…`, page 119, row_index 4) whose text is

```
Total net operating revenues | 11,513 | 6,334 | 19,586 | 5,638 | 5,735 |
                               48,806 | 144 | (1,009) | 47,941
```

One row, nine columns, sixteen emitted cells. The metric is `Intersegment`
while the row is `Total net operating revenues`.

Both flags apply. The store attributed the row to the wrong metric (**semantic
defect**), and the question names `Intersegment` — which in this table is the
Eliminations column — without naming a column (**ambiguous**). There is no
correct row to repoint to; the row itself is the thing that is mislabelled.

### `tv2f01-s2-pctshare-005` → **SOURCE_AMBIGUOUS**

```
Q        "What percentage of Apple's Total net sales was Cost of sales in FY2025?"
old gold -0.5309   operands {part: (220,960), total: $ 416,161}
```

The operands are right and the magnitudes are consistent — 220,960 is Apple's
cost of sales and 416,161 its total net sales, both from the consolidated
statement of operations. **The reading is what is not determined.** Apple states
cost of sales as a positive expense in the statement and as a parenthesis in the
segment table; the question asks "what percentage … was Cost of sales", which
does not say whether the answer carries the expense's sign. `-0.5309` and
`0.5309` are both defensible.

Against the brief's principle this is the question lacking a measure, not the
gold pointing at a wrong cell. `SOURCE_AMBIGUOUS`.

### `tv2f01-s2-pctshare-006` → **SOURCE_AMBIGUOUS**

```
Q        "For Apple in FY2025, what is Total non-current portion of term debt
          as a share of iPhone?"
old gold 0.3737    operands {part: $ 78,328, total: $ 209,586}
```

The denominator is unambiguous — `iPhone | $ 209,586` in the products-and-services
table, and the same figure in the consolidated statements. The numerator's
reading is not. `$ 78,328` is the **total non-current portion of term debt**, and
the question asks for it "as a share of iPhone"; the answer therefore depends on
whether the share is of *iPhone* or of *total net sales*, and the coordinate
holds `$ 85,750` as well — a second non-current debt figure.

`SOURCE_AMBIGUOUS`.

---

## Closure summary — all 25 adjudicated, `UNRESOLVED = 0`

```
                        GOLD_      REPOINT_   SOURCE_     STORE_SEMANTIC_
                        CORRECT    GOLD       AMBIGUOUS   DEFECT
four gold cases            0         3           1            0
twenty-one column cases    2         1          15            2   (overlapping)
                          --        --          --           --
                           2         4          16            2
```

`ko-015` carries both flags; it is counted once in each column.

## Migration diff — proposed, NOT applied

### Repoints (3)

```
case                 old gold     new gold     reason
sum-007              25962.0000   23754.0000   gold drew from the equity-method
                                               investee table; question names
                                               Coca-Cola with no scope
tsla-032             (5,708)      57165        gold drew from the table's
                                               year-over-year CHANGE column
tsla-033             82,056       94,827       gold drew from the automotive
                                               segment table; question names no
                                               scope
```

Each carries, in `p1-8-c1` / `p1-8-c2`, its source table, caption, column
headers and the authority rule that selects the new value.

### Status changes (18) — `expected_outcome` ANSWER -> ABSTAIN

```
from the four gold cases        aapl-003
from the twenty-one             aapl-001  aapl-005  jpm-008  nvda-022  nvda-023
                                tsla-034  v-036     v-037    v-039
                                ko-011    ko-015
                                diff-002  diff-004  growth-003  growth-008
                                pctshare-005  pctshare-006
                                (17)
                               ----
                                18
```

### Flagged, not migrated (2)

```
msft-020  STORE_SEMANTIC_DEFECT    gold kept unchanged; the coordinate is
                                   polluted by a wrongly attributed record
ko-015    STORE_SEMANTIC_DEFECT    gold moves to abstention AND the store
                                   attribution is recorded as the root cause
```

### Denominator impact

```
                     before   after
answerable              95       77
abstention              25       43
                      ----     ----
total                  120      120
```

No case is deleted. No case changes stratum. The 18 that move were authored
against a row label the source does not resolve to one quantity, and they are
now scored on whether the system declines them.

### Hashes

```
old  gold-evidence-v1.jsonl    3d2a0c5b7839656ce1414923ab90d03b844bc5c5fc17d82dc5e68224f466eb05
old  canonical-eval-v1.jsonl   227f0341d94ab8d4b9e7ee033feaa9b40e86ec32c3137a74d2931657e9281ece
new  both files rewritten; benchmark-version.json gains a p1.8-c migration entry
     with the pre/post hashes in the same shape as migrations[0] (p1.6-0h)
```

The new hashes can only be computed after the files are written, which is why
they are not in this document. The migration entry will record them.

## Two things I want on the record before anything is applied

**`msft-016` and `pfe-030` are not in this migration.** They were in `p1-8-b9`'s
runtime patch, and they should stay there: their questions *do* name a metric,
their golds are correct, and only the runtime's inability to resolve
`cost_of_revenue` blocks them. Migrating them would be using the benchmark to
compensate for a runtime gap — the thing the brief's fourth requirement forbids.

**`ko-015`'s store defect is not fixed by moving it to abstention.** It will
score as a correct refusal, which is better than a wrong answer, but the
underlying attribution error remains and will affect other cases. Moving it is a
benchmark action; fixing it is not, and this migration should not be read as
having addressed it.
