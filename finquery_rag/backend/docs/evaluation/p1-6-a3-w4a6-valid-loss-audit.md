# P1.6-A3-W4-A6 — auditing the 1,649 losses that look column-local

Diagnosis only, and **narrow on purpose: 1,649 cells, not 8,369.**

```
python audit_valid_losses.py --delta <artifact> --store <store-v2.jsonl> --out <dir>
```

## Why only these

W4-A5 measured that 79.5% of the deficit carries a period the cell's own column does not
name. The correction that follows from it is a boundary this step exists to respect:

```
the column does not name this period   !=   this period is wrong
```

A legitimate period can come from a row, a row-group, or the table's own context, none of
which the column-scoped producer can see. The 6,405 are therefore neither confirmed nor
refuted and are not touched. What is audited is the 1,649 that satisfy

```
legacy period  ~=  the period the cell's own column explicitly names
```

because those are the only ones that can be a genuine V2 capability regression, and they are
the number W4-B has to be decided on. "At most 6% of the store" is a budget; this turns it
into a fact.

## The verdicts

```
A  V2_PRODUCER_GAP                    876  (53.1%)
B  LEGACY_FALSE_POSITIVE               28  ( 1.7%)
C  VALID_BUT_OUT_OF_SCOPE_GEOMETRY    171  (10.4%)
D  SOURCE_AMBIGUOUS                   574  (34.8%)
```

**`V2_PRODUCER_GAP` is not zero, and that is the answer to the question this step asked.**

## A — 876 cells, and one cause

Every column is asked why the producer has no binding, using the producer's own predicates.
Two buckets were built to catch a producer defect and both came back **empty**:
`FULL_DATE_ACCEPTED` (a date rule 1 accepts, yet unbound) and `BARE_YEAR` (rule 3 should
have bound it). So the producer's rules fire whenever their predicates are met, and the
question became why the predicates are not met.

```
  340  A  'Fair value at'                    an as-of date naming the column's point
  282  A  'For the Year Ended'               a period header in words
  144  A  'Change in unrealized gains/...'   the same as-of date in a longer clause
   48  A  'Central case assumptions at'      an as-of date
   30  A  'Fair value purchase price allocation as of'
   24  A  'Contractual rate in effect at'
    8  A  'As of or for the year ended'
```

All of these are one defect in one predicate. `_is_period_header_cell` refuses a cell when
what is left of it after the date is longer than twelve characters, and:

```
For the Year Ended September 30, 2025          prefix is 19 characters -> refused
As of or for the year ended December 31, 2025  prefix is 27 characters -> refused
Fair value at Jan. 1, 2025                     prefix is 14 characters -> refused
```

**A twelve-character limit is a prose heuristic, and `For the Year Ended` is not prose.**
The fix is not a bigger number — `Indenture, dated as of October 28, 2021, between ...` is
a document description and must stay refused — it is that a cell whose prefix is a *period
phrase* and whose suffix is empty is a period header however long the phrase is.

## B, C, D

```
B  28   signature blocks (`Chairman of the Board of Directors ... February 20, 2026`), a
        policy description, and a share-vesting clause (`... Performance Share Units
        vesting on March 25, 2026`).  These are the refusal working.
C  171  NO_HEADER_CELL 127 -- the column has no cell in any header row the producer reads,
        so its period lives in a row or in the table's own context; and MONTH_DAY_NO_YEAR
        44 -- `Year Ended December 31,` with the year outside the column.  Both are
        geometries the column model cannot express.  Registered as capability debt, NOT as
        corrections.
D  574  NO_DATE_IN_COLUMN 518 -- the column names a scope or a basis (`Managed basis`,
        `Recorded Basis`, `Cash and Cash Equivalents`, `Adjusted Cost Basis`, `Actual`,
        `12 Months or Greater`) and the legacy period came from elsewhere in the table --
        plus two families whose date sits inside a parenthetical aside or a note title.
        **Not deleted, and not callable a false positive**: the lens cannot prove these
        wrong, only that the column does not state the period.
```

## The rule that was tried first, and why it was thrown away

The families were first split by a regex: a cell is a period header if the prefix ends in a
date-introducing word and the suffix is empty. It put **950 of 960 in class A** — the
signature of a rule that has stopped discriminating. `Fair value at December 31, 2025` and
`... Performance Share Units vesting on March 25, 2026` have the same shape — clause,
preposition, date — and only one of them names a period.

So the split is a hand-written adjudication table, family by family, each with the reason it
was reached, in the same shape as `table_authority_adjudications.json`. Any family that
matches nothing returns **D** rather than being quietly dropped into a neighbour.

## What this means for W4-B

The frozen entry condition was:

```
UNEXPLAINED_VALID_LOSS = 0
V2_PRODUCER_GAP        = 0
```

```
V2_PRODUCER_GAP = 876   ->  not met
```

So W4-B does not switch yet, and it is no longer a judgement about a six-percent budget: it
is one predicate, one fix, and a re-measurement. `UNEXPLAINED_VALID_LOSS` is 574 and every
one of those is registered as ambiguous rather than absorbed.
