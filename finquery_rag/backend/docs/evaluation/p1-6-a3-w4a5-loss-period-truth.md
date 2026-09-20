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

## What this does to W4-B

The deficit is not what it looked like in W4-A2. Of 8,369 cells:

- **6,405 (76.5%)** carry a period their own column does not name — the equity defect in
  shapes that do not need equity anchors to see, plus notes, exhibit indexes and tables of
  contents. Withholding these withholds facts the store is holding wrongly.
- **at most 1,649 (19.7%)** sit in a column that names exactly their period. That is the
  only part that is genuinely coverage, and it is an upper bound.

Against a store of 26,977, the honest cost of switching is **at most 6% of the store**, and
the remaining 94% of the deficit is correction rather than loss. That is a very different
proposition from the one W4-A2 recorded, and it is the number W4-B should be decided on.

The other half of the entry condition is unchanged and matters more: `producer_gain` is
still **0**. The new producer has never yet bound a cell the legacy axis missed, so a switch
today is still strictly narrower regardless of how right the withheld facts are.
