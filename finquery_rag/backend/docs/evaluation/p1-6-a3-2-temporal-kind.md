# P1.6-A3-2 — why a column with a period still classifies as `bucket`

Diagnosis only. **No temporal rule change, no period change.**

A3-1a2 found NVIDIA's and Tesla's cash flow statements bind a period on 15 of 18 columns
and still emit nothing. This settles why before any header repair lands, because that loss
sits at the *shared downstream gate* of every period repair: fixing Visa's, Pfizer's and
Coca-Cola's headers and then finding the same zero at the same gate is the outcome this
pass exists to prevent.

## The cascade, and which step fires

```
nvda_fy2025#8766  CASH_FLOW  (fails)     {'bucket': 18}              step 2 bucket  on 'rating'
tsla_fy2025#10407 CASH_FLOW  (fails)     {'comparison': 3,
                                          'bucket': 15}              step 1 comparison on 'increase'
                                                                     step 2 bucket  on 'rating'
aapl_fy2025#6383  CASH_FLOW  (control)   {'comparison': 3,
                                          'duration': 15}            step 6 duration on 'Years ended'
msft_fy2025#17151 CASH_FLOW  (control)   {'bucket': 1,
                                          'duration': 12}            step 6 duration on 'Year Ended'
aapl_fy2025#5498  BALANCE_SHEET (control) {'category': 3,
                                           'duration': 9}            step 7 normalized_period
```

`_classify_column_temporal` is a priority cascade over `header_text + " " + cell_text`. It
checks comparison, bucket, segment and category **before** point and duration, so a period
that is correctly bound never reaches the rule that would read it.

## The two collisions

```
nvda#8766 col 3   '2025 / Cash flows from operating activities: Y…'
                                                       ^^^^^^
                                          _BUCKET_RE's `rating`, written for
                                          credit-rating buckets, matching inside
                                          "ope·rating"

tsla#10407 col 3  '2025 / Cash Flows from Operating Activit  Year…'   same

tsla#10407 col 0  'Cash Flows from Operating Activities Ca…'
                                          _COMPARISON_RE's `increase` — no; see below
```

`_BUCKET_RE` carries `|range|rating|grade|tier` for maturity and credit-rating columns. The
word **`Operating`** contains `rating`. Every column of a cash flow statement whose header
path carries `Cash flows from operating activities` is therefore classified `bucket` — not
because anything about it is a bucket, but because a substring of an ordinary statement
heading matched a pattern written for a different disclosure entirely.

Tesla's comparison matches are the same species from the other pattern: `_COMPARISON_RE`
carries `\bincrease\b|\bdecrease\b`, which match the ordinary cash-flow row wording
`Increase (decrease) in …`.

## Why the controls survive

Apple's cash flow statement reaches step 6 and binds `duration`. It does not match
`rating`, because **its header path never absorbed the row-label column's text**:

```
nvda#8766  col 3 header_path  '2025 / Cash flows from operating activities:'   <- polluted
aapl#6383  col 3 header_path  'Years ended September 27, 2025 …'               <- clean
```

So the discriminator is not the statement or the filer. It is whether the first column's
text leaked into the header path of the data columns — the same over-segmentation that
A3-1a2 saw in the grid. **A pollution defect upstream is what lets a substring collision
downstream fire.**

## What this changes about A3's shape

**NVIDIA's and Tesla's zero-record is not a period defect and A3-1 will not touch it.** They
are removed from the period work and carried as their own root cause, as A3-1a2 proposed —
and now with a named mechanism rather than an observation.

Two more patterns in the same cascade have the same shape and were not triggered here but
carry the same risk: `_SEGMENT_RE` matches `foreign`, `domestic` and `international` (cash
flow statements have `Effect of foreign exchange rates` rows), and `_CATEGORY_RE` matches
`other`, which appears in nearly every financial table. Both are checked before point and
duration. **This is a class of defect, not two instances of one.**

The repair is not obviously "reorder the cascade", which would be a guess about which kind
wins. The two things the evidence supports are that **a bound period should outrank a
substring match on prose**, and that **the pattern set needs to distinguish a column that
*is* a bucket from one whose heading merely contains the letters**. Which of those A3-3
adopts is a decision, not a finding.

## Status

Nothing changed. Evidence at
`artifacts/evaluation/p1-6-a3-2-temporal/temporal-kind.json`, which carries every column's
header text, cell text and firing step for all five tables.
