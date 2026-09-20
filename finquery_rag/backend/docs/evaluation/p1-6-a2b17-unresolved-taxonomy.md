# P1.6-A2B-17 — unresolved slot taxonomy

Classification only: no resolver change, no classifier change, no fixture change. The point
is a priority matrix, so the next step is chosen from a distribution rather than from
whichever case was looked at last.

## The distribution

```
C_NON_PRIMARY_SOURCE_ONLY           28
B_STATEMENT_CLASSIFICATION_ERROR     8
F_TRUE_SOURCE_ABSENCE                3
A_METRIC_ALIAS                       2
G_SOURCE_AUTHORITY_UNRESOLVED        2
                                   ---
                                    43
```

(The 43 counts every slot this pass did not find a primary-statement candidate for. It is
larger than the resolver's 34 refusals because the two use different criteria — noted below
rather than reconciled here, since reconciling them is a change and this step is not.)

## The two buckets are one root cause

`B` and `C` were separated because a table wrongly labelled a primary statement and a figure
that only exists in the notes want opposite fixes. **On this corpus they are not two
problems.** Both are the statement classifier failing:

```
B   eight tables labelled INCOME_STATEMENT / BALANCE_SHEET that do not name
    themselves as a statement -- a hedging note described as BALANCE_SHEET is the
    clearest
C   28 slots whose row exists but sits in a table classified UNKNOWN
```

**36 of 43 unresolved slots have the statement classification as their cause.** And the
`C` cases are not obscure rows: `Operating income`, `Net income`, `Total assets`,
`Total liabilities`, `Research and development`, `Interest expense` — line items that belong
on a primary statement and are being found in tables the classifier never labelled.

The supporting measurement is the same one from A2B-15: 10,046 of JPMorganChase's 11,713
records are `UNKNOWN`, and Apple's filing has **no `INCOME_STATEMENT` table at all**
(`UNKNOWN 59, CASH_FLOW 2, NOTES 1`).

So `C` is not "the answer is only in the notes". It is "the classifier did not recognise the
statement", and its fix is the same as `B`'s: **the statement classifier, not the resolver
and not the notes.**

## The small buckets

```
F  TRUE_SOURCE_ABSENCE            3   compare-005/008 Apple Diluted EPS,
                                      crossdiff-004 Visa Long-term debt
A  METRIC_ALIAS                   2   a row carries the metric's words under
                                      another label
G  SOURCE_AUTHORITY_UNRESOLVED    2   crossdiff-004 Coca-Cola (a hedging note
                                      labelled BALANCE_SHEET) and rank-004
                                      Microsoft (BUSINESS and NOTES)
```

`F` is worth a second look rather than acceptance: `Diluted earnings per share` and
`Long-term debt` are statements every filing makes, so "the source does not state it" more
likely means the row exists under a name this pass did not recognise. That would move them
to `A`, and it is the kind of claim that should be checked against the filing before it is
believed.

`G` for `crossdiff-004` is the same hedging note that `B` describes, seen through a
different branch — the two are the same defect and the taxonomy says so twice.

## What the matrix says to do next

```
A  2    metric canonicalization          small, and needs a real alias table
B  8    statement classifier             \
C 28    statement classifier             / 36 of 43, one root cause
D  0    period normalization             none
E  0    parser extraction                none
F  3    source reality                   probably A in disguise; verify first
G  2    keep fail-closed                 no new authority signal yet
```

**The next step is the statement classifier, not the resolver.** Fixing it addresses 36 of
43, and it is the layer where `crossdiff-004`'s hedging note and Apple's unrecognised income
statement are the same defect.

Two cautions carried forward:

- The classifier change must be checked across all eight filings, not fitted to
  JPMorganChase — a rule fitted to one example is what produced the column-name rule that
  failed 40 of 47 slots.
- It must not be a loosening that admits notes. `B` exists because the classifier was
  already admitting a note; a change that admits more would trade `C` for more `B`.

## Not established

The 43-versus-34 difference is unexplained here. This pass asks a slightly different question
from the resolver — "is there a primary-statement candidate" rather than "do the company-level
candidates agree" — so some slots the resolver resolved are counted here and vice versa. It
does not affect the distribution's shape, but it should be reconciled before either number is
quoted as the coverage figure.

## Status

No behaviour changed. No fixture, denominator, legacy store, view, index or production
behaviour touched. Evidence at `artifacts/evaluation/p1-6-a2b17-taxonomy/`.
