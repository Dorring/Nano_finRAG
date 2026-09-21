# P1.6-A3-W4-A5 — is the period the store would lose actually the cell's own?

Diagnosis only.

W4-A2 measured 8,369 cells the legacy stores and the new producer withholds, and W4-A3
showed that for the 474 oracle-PRIMARY ones the legacy period is **not** the cell's period.
That made "the store would lose these" the wrong description of them. This asks the same
question of the other 7,895, which is what decides whether W4-B's deficit is a cost or
mostly a correction.

```
python diagnose_loss_period_truth.py --delta <artifact> --store <store-v2.jsonl> --out <dir>
```

## The lens

It generalises off the equity case rather than reusing its anchors, because the other tables
are not equity statements. What every fact in Store V2 does carry is its own column header,
so the question becomes: **does the column this fact sits in name a period, and is the
fact's period that one?**

```
                                       count    share
the column names one period, and it is the fact's    1649    20.5%   <- plausibly right
the column names one period, and it is not              1     0.0%
the column names several periods                    1521    18.9%   <- a choice, not a statement
the column names no period                          4883    60.6%   <- from outside the column
```

**The period of 79.5% of the deficit is not something the cell's own column states.**

## By table role

```
                      total   declared   declared %
OPEN                   5660        964        17.0
WEAK_NON_PRIMARY       1006        396        39.4
DANGEROUS_NEGATIVE      916        289        31.6
PRIMARY                 472          0         0.0
```

**The primary figure is zero, and it is an independent confirmation of W4-A3.** That
diagnosis read equity anchors out of the filings and concluded the legacy names the opening
balance of each section. This one reads only the column header stored beside the fact and
asks whether it names that period. Two different methods, two different inputs, the same
answer: none of the 472 is a period its column declares.

## What the lens can and cannot say

**20.5% is an upper bound**, not a measurement of correctness, and the first version of this
showed why. It counted *years*, and `June 29, 2025 to August 2, 2025: / August 3, 2025 to
August 30, 2025:` names one year and several periods -- so a weekly-reporting column scored
as declaring a single period and the fact's. Counting month-day expressions as periods in
their own right moved 27 cells and nothing else, which is the useful part: the headline
moved by 1.6% when the obvious false positive was closed, so it is not an artefact of that
bug.

It still cannot separate a period column from prose that happens to contain one date, so a
column header reading `Indenture, dated as of October 28, 2021` is counted as declaring a
period when it is describing a document. The over-count is in the 1,649, not in the 6,378.

## The other half: where the periods V2 would *add* come from

The loss direction is measured above. The gain direction is not in the store -- the legacy
never admitted those cells -- so it is checked from the binding the producer recorded:
every source cell it read has to be in the column the fact sits in.

```
rule_gain: every source cell is in the fact's own column      6110
rule_gain: row-scoped declaration (INLINE_PERIOD_DATA_ROW)     276
offenders                                                         0
```

**All 6,386 cells V2 would newly admit carry a period sourced from their own column** (or
from a row-scoped declaration in that row). This is not a coincidence of the sample: the
column rules read `raw[col]`, that column's own header cells, and nothing else. It is the
same property that makes the producer refuse the equity statements, and it is why the two
directions are not symmetric.

## What this does to W4-B

> **Overclaim corrected.** An earlier version of this section called the 6,405
> "corrections" and said the store "is holding them wrongly". That does not follow from the
> measurement and is not claimed here. `the column does not name this period` is **not**
> `this period is wrong`: a legitimate period may come from a row, a row-group, or the
> table's own context, none of which the column-scoped producer can see. What is measured
> is that **V2's column-local producer does not recognise them** — nothing more. The equity
> 474 are the exception, and only because they were proved wrong by an independent source
> geometry, not by this lens.

The deficit is not what it looked like in W4-A2.

```
would add       6,386   period sourced from the cell's own column       100%
would remove    8,369   period the cell's own column names            <= 19.7%
```

The 1,649 are the load-bearing number, and they are the *only* candidates that satisfy

```
legacy period  ~=  the period the cell's own column explicitly names
```

so they are the closest thing to a genuine V2 capability regression. They are audited
separately in W4-A6. The other 6,405 are **not** called corrections: they are facts whose
period this lens cannot confirm and cannot refute, which is a different state and one that
must not be settled by deleting them.

Against a store of 26,977, the arithmetic cost of switching is at most 6% of the store; the
addition is 6,386 facts that are period-correct by construction and whose periods the
legacy path cannot supply at all (all 14,802 rule-gain cells carry no period on the legacy
axis -- W4-A, "why there is no rule-only escape"). But the gate W4-B is decided on is not
the arithmetic.

The one thing that has not moved is `producer_gain = 0`, and that is **not** a blocker in
its own right. It says V2 found no cell the legacy had not touched; it does not say V2
added no capability. V2 supplies a usable binding for cells the legacy had **no** usable
period for, which is where the 6,386 come from. The number that matters is not
`producer_gain` but:

```
newly admissible source-grounded facts   6386
unexplained valid losses                 ?      <- W4-A6
```
