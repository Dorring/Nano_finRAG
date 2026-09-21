# P1.6-A holdout — Amazon and Alphabet, unseen filers

The first test of the table-authority rules on filings that contributed nothing to them.
**No rule was changed before running**: no title alias, no weight, no blocking condition,
no threshold. The scripts that ran are the sealed ones, plus one configuration edit that
adds the two documents and a `--holdout` switch; the sealed oracle reproduces
byte-identically after that edit (`dccbe5d0221ccb67`), which is how the edit is shown to be
behaviour-preserving.

```
                             corpus  data  scaffolds | oracle          | predicted
                                                    | primary danger  | P   N   U
amzn_fy2025   Amazon FY2025      92    65        27 |     5      23   | 5  58   2
googl_fy2025  Alphabet FY2025   185    81       104 |     5      35   | 5  70   6
                                                    |                 | ---
                                                                       10 128   8
```

## The gate

```
wrong PRIMARY authority      0        PASS
main consolidated statements identified, both filers   PASS
```

**Precision 1.0, recall 1.0**, but read the next section before believing either.

## Every predicted PRIMARY, read

Both filers yield their complete statement set, in consecutive document order, and every
title was verified verbatim in the filing's own HTML:

```
amzn_fy2025#7070   CASH_FLOW           CONSOLIDATED STATEMENTS OF CASH FLOWS
amzn_fy2025#7689   INCOME_STATEMENT    CONSOLIDATED STATEMENTS OF OPERATIONS
amzn_fy2025#8180   COMPREHENSIVE_INCOME CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME
amzn_fy2025#8441   BALANCE_SHEET       CONSOLIDATED BALANCE SHEETS
amzn_fy2025#8861   EQUITY              CONSOLIDATED STATEMENTS OF STOCKHOLDERS' EQUITY

googl_fy2025#8833  BALANCE_SHEET       CONSOLIDATED BALANCE SHEETS
googl_fy2025#9344  INCOME_STATEMENT    CONSOLIDATED STATEMENTS OF INCOME
googl_fy2025#9820  COMPREHENSIVE_INCOME CONSOLIDATED STATEMENTS OF COMPREHENSIVE INCOME
googl_fy2025#10105 EQUITY              CONSOLIDATED STATEMENTS OF STOCKHOLDERS' EQUITY
googl_fy2025#10868 CASH_FLOW           CONSOLIDATED STATEMENTS OF CASH FLOWS
```

Amazon's run 35-39 and Alphabet's 78,80,82,84,86 — consecutive statement blocks, which is
what the spine of a 10-K looks like. **All ten are genuine; wrong PRIMARY authority is 0.**

Two signals that were fitted to the eight carried over without adjustment. The
typographic apostrophe in `STOCKHOLDERS' EQUITY` — which is why the alias list carries both
`'` and `’` — is what Alphabet and Amazon use. And the equity statement's dimensionality
exemption held on both: both equity statements are tagged
`us-gaap:StatementEquityComponentsAxis` with a low company-level share (Amazon 13 of 52,
Alphabet 25 of 74), which under the unexempted rule would have cost two more statements.

## The case worth keeping

```
googl_fy2025#24129   '1. Consolidated Financial Statements'
    OWN_STATEMENT_TITLE  +6   the table's own first rows read 'Consolidated Balance Sheets'
    -> UNKNOWN / INSUFFICIENT_EVIDENCE
```

Alphabet's Note 1 is an **index of the statements** — its rows are literally
`Consolidated Balance Sheets`, `Consolidated Statements of Income`, and so on. A rule that
promoted on a matching title would have taken it. Two independent guards stand in the way,
and the one that fired is the one that generalises: the table carries **0 tagged facts**, so
it fails candidacy (`facts >= 1 and undim >= 1`) before the score is consulted at all.

The other guard — `INDEX_LINE`, which catches `Index to Consolidated Financial Statements` —
did **not** fire here, because Alphabet's nearest line is a note heading rather than an
index heading. The guard that saved this is the one that was not written for this shape.

## Safety

**All 58 oracle-known dangerous tables are `NON_PRIMARY`. None was promoted.** The eight
`UNKNOWN` tables are an SEC cover-page checkbox, and seven notes (`Other income (expense),
net`, `Accrued expenses and other current liabilities`, `Components of OI&E`, goodwill
activity, non-marketable securities carrying values and gains). No primary statement is
hiding among them, so the `UNKNOWN` bucket is not concealing a false negative.

## What this does not establish

**The authoring platform is the same.** All eight design filings and both holdouts are
Workiva. This tests generalisation across **filers**, not across authoring styles, and a
filing from a different platform has not been run. The corpus contains a
Donnelley-produced Microsoft filing but it is a proxy statement, not an annual report, so
it has no primary statements to test against.

**Precision 1.0 and recall 1.0 on two filings is two filings.** Ten statements is not a
sample from which a rate can be estimated; it is ten statements that came out right. The
honest statement is that no counterexample was found, not that none exists.

**Neither holdout is in the benchmark.** There is no 47-slot result here, no store, no
resolver run — this is the classifier's output read against the filings, which is what the
holdout is for.

## Verdict

The authority contract held on two unseen filers with no adjustment: every promoted table
is the statement of record, no dangerous table was promoted, and the one table that should
have worried us was refused. The residual risk is authoring-style generalisation, which
this did not test.

Evidence at `artifacts/evaluation/p1-6-a2b21-holdout/` — `table-authority-oracle.json`,
`shadow-classification.json`.
