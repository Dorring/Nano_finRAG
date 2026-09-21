# P1.6-A2B-19B — the two-headed shadow classifier

**Shadow only.** The resolver keeps its current logic and production behaviour does not
change. This emits, per table, what a replacement would say and what it rests on.

## The denominator, corrected before anything was classified

`table_eligibility` is now its own layer, because "is this a data table" and "may it speak
for the company" are two questions and folding the first into the second is what put 592
pieces of page furniture into every metric:

```
906  DATA_TABLE        a table of figures
592  LAYOUT_SCAFFOLD   one- and two-row page layout: page numbers, cover-page
                       checkboxes, empty spacers, with no tagged fact and no row
                       of figures
--- 
1498
```

Nothing downstream counts the scaffolds. A two-row table with no tagged fact that *does*
carry a row of numbers stays a data table — the test is `facts == 0 and rows <= 2 and not
has_figures`, and a page footer reading `Pfizer Inc. 2024 Form 10-K 55` is not figures.

## The hard oracle, and what it does not include

```
42   PRIMARY             read one by one, all 42 genuine
229  DANGEROUS_NEGATIVE  the table's own caption names a disaggregation
---
271  the safety gate
```

The dangerous set is **229, not the 281** quoted by caption alone: 52 of those are page
furniture whose nearest line happened to contain a danger word. The remaining 635 data
tables are **weak** — 152 the rules call non-primary and 483 they never decided. Those are
reported as `extended` and never as the gate. The rules' negative verdicts are the filing
declaring the scope of its own facts, which is sound, but nobody has checked them one by
one, so they cannot decide whether a classifier passes.

## The classifier

Two heads, because A2B-19A proved one field cannot carry both. A derivatives note is
entirely about balance-sheet line items and is still not the balance sheet.

```
statement_type   INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW / EQUITY /
                 COMPREHENSIVE_INCOME / SEGMENT / DEBT / DERIVATIVES / ... / UNKNOWN
table_role       PRIMARY_FINANCIAL_STATEMENT / NON_PRIMARY / UNKNOWN
```

**Authority precedence**, which the Pfizer supplement forced:

```
the table's own explicit identity  (its title, or the subject it names)
  > the nearest external heading
    > body semantic inference
```

`Supplemental Cash Flow Information` sits two tables below the cash flow statement and
inherits its title from the page above. A title outside a table cannot outrank a caption
inside it — and that has to hold for **both** heads, or the table is called a cash flow
statement while being denied the authority of one.

**Evidence aggregation, not keyword matching.** Each signal is an item with a polarity and
a weight, the items are listed with every table, and the decision is a threshold plus
blocking conditions:

```
+6  OWN_STATEMENT_TITLE         the table's own first rows name a statement
+5  EXTERNAL_STATEMENT_TITLE    a statement title stands within three lines above
+2  COMPANY_LEVEL_FACTS         at least half the tagged facts are undimensioned
+1  OWN_UNITS_CAPTION           the table opens with a units caption
+1  STATEMENT_FAMILY_CONCEPTS   it reports NetIncomeLoss / Assets / ...

-5  OWN_SUBJECT_CAPTION         the table's own first row names a subject   BLOCKING
-6  ALL_FACTS_DIMENSIONED       every tagged fact is a disaggregation      BLOCKING
-3  EXTERNAL_DANGER_HEADING     the nearest line names a subject
-2  EXTERNAL_NOTE_HEADING       the nearest line is a note heading
-2  MOSTLY_DIMENSIONED          under 15% of facts are company-level
-2  SEGMENTATION_AXIS           broken out by segment/geography/product/customer
                                while not mostly company-level
```

`PRIMARY` requires the score to clear 5 **and** a positive identity **and** at least one
company-level fact — so every predicted primary can be pointed at the words in the filing
that say so. Blocking evidence cannot be outvoted at any score. Between the two thresholds
the answer is `UNKNOWN`, which stays a real output: 78 of the 906 data tables.

### Two corrections forced by measurement

Both were found by running the first version and reading what it got wrong.

**An axis in a table is not evidence against it.** The first model penalised any
`DISAGGREGATING_AXIS`, which cost Visa's balance sheet (`PledgedStatusAxis`), Visa's and
Tesla's income statements (`ProductOrServiceAxis` — Apple tags its Products and Services
revenue lines with it and they are income-statement rows) and every statement of changes
in equity (`StatementEquityComponentsAxis`). Presence of an axis means nothing; the fact
counts already measure how much of the table is company-level, and the axis now only
sharpens that to "broken out by part of the business". The dimensionality penalty is also
waived for the equity family outright, because a statement of changes in equity is tagged
by equity component and none of these eight companies is an exception — penalising it threw
away eight statements for a property of the statement itself.

**A note heading anywhere in the window is not a signal.** Reading every note heading
punished Visa's statement of comprehensive income and Pfizer's income statement, each of
which still had a note heading from the previous page inside the six lines the evidence
pack carries. It now counts only when it is the *nearest* line.

## The gate

```
dangerous tables promoted to PRIMARY : 0      PASS
hard-set primary precision           : 1.0    PASS
recall improves on 5/42              : 42/42  PASS
every filing independently           : PASS
```

```
                                  old classifier        shadow classifier
aapl_fy2025          (Apple)        0 / 5                  5 / 5
jpm_fy2025           (JPMorgan)     0 / 5                  5 / 5
ko_fy2025            (Coca-Cola)    0 / 5                  5 / 5
msft_fy2025          (Microsoft)    5 / 5  (11 wrong)      5 / 5  (0 wrong)
nvda_fy2025          (NVIDIA)       0 / 5                  5 / 5
pfe_fy2024           (Pfizer)       0 / 5                  5 / 5
tsla_fy2025          (Tesla)        0 / 5                  5 / 5
v_fy2025             (Visa)         0 / 7                  7 / 7
                                  ----------            ----------
                                   5 / 42                42 / 42
                                  precision 0.312       precision 1.000
```

**Seven filings go from 0.0 to 1.0 recall, each on its own, with a different HTML authoring
style.** Microsoft is the one that was already at 5; what improves there is precision,
16 promotions down to 5, and the eleven notes it used to promote are now
`NON_PRIMARY` with a subject type (`DEBT`, `DERIVATIVES`, `QUARTERLY_DATA`, `SEGMENT`,
`LEASES`, `EPS`).

## What this does not establish

**The 42/42 is not independent confirmation of anything.** The oracle's positive rule and
this classifier's `EXTERNAL_STATEMENT_TITLE` are the same signal, so agreement was the
expected outcome, not a finding. What the oracle contributed was reading all 42 and
confirming the signal is right on this corpus; the classifier then spends it. The
independent results are narrower: the safety property holds under two separately-derived
decision paths, `statement_type` is produced by a head that did not exist, and `UNKNOWN`
is a reachable outcome. On a ninth filing — or a filer that titles its statements
differently from all eight here — neither has been tested.

**Nothing was promoted outside the hard oracle either**, which sounds stronger than it is:
of the 906 data tables the classifier predicts exactly the 42 and no others, so
`EXT_FP` is 0 and no `OPEN` table was promoted. That is the conservative direction, and it
is the same conservatism that leaves 78 tables `UNKNOWN`.

**The `statement_type` head is not measured.** No gold exists for it, so the distribution
below is a description, not an accuracy:

```
UNKNOWN 373   DEBT 89   INCOME_TAXES 64   DERIVATIVES 56   FAIR_VALUE 53
SEGMENT 52    LEASES 43  PENSION 27   SHARE_BASED_COMPENSATION 22   GOODWILL 16
QUARTERLY_DATA 13   ALLOWANCE 11   RECONCILIATION 11   NON_GAAP 11
BALANCE_SHEET 10   EQUITY 10   EPS 10   SUPPLEMENTARY 10   CASH_FLOW 9
INCOME_STATEMENT 8   COMPREHENSIVE_INCOME 8
```

Three tables outside the 42 carry a statement family, and one of them is worth keeping:

```
jpm_fy2025#132518   BALANCE_SHEET   'Parent Company-only financial statements'
jpm_fy2025#132857   CASH_FLOW       'Parent Company-only financial statements'
jpm_fy2025#63771    BALANCE_SHEET   inherits the balance sheet's title from the page above
```

The first two are JPMorganChase's Schedule I. They **are** balance sheet and cash flow
statements — of the parent alone, not consolidated. `statement_type` says what they are
about and `table_role = NON_PRIMARY` says they may not speak for the company, which is
exactly the split this step exists to make. The third is a balance-sheet continuation
whose family is defensible and whose role is inert.

## Status

Resolver and production behaviour unchanged. Evidence at
`artifacts/evaluation/p1-6-a2b19b-shadow/` — `shadow-classification.json`, per table with
its evidence, its score and its status; `shadow-vs-hard-oracle.json`, the gate above.

Next, once this is accepted: **A2B-20 — carry `table_role` into Store V2 and re-run the
47-slot resolver.**
