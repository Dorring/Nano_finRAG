# P1.6-A2B-18 — statement authority diagnosis

Diagnosis only. No classifier change, no resolver change, no fixture change.

## The measurement

```
                        tables   currently primary   FN (own text says primary)
aapl_fy2025                 62                   0                            10
jpm_fy2025                 679                   0                            ?
ko_fy2025                  117                   0                            10
msft_fy2025                 85                  16                             8
nvda_fy2025                 68                   0                             7
pfe_fy2024                 322                   0                            27
tsla_fy2025                 79                   0                             5
v_fy2025                    86                   0                            13
                        ------              ------                        ------
                          1498                  16                           120
```

**Seven of the eight filings have no primary-statement label at all.** Every one of Apple's
62 tables, Coca-Cola's 117, Pfizer's 322, Tesla's 79 and Visa's 86 is `UNKNOWN`. 120 tables
whose own text reads as a primary statement are labelled something else.

Only Microsoft has any labels, and that is the whole of the 16.

## Why, precisely

`section_type` reaches a table by one of two routes, and **neither reads the table**:

```python
# make_blocks -- a running value advanced by headings
if k == "HEADING":
    inferred = section(tx + ...)
    if inferred != "UNKNOWN": cur = inferred
"section_type": cur

# parse_table -- the caption, the preceding text, and the column headers
inferred = section(caption + " " + prior + " " + " ".join(headers))
sec = inferred if inferred != "UNKNOWN" else sec
```

The table's **body** — the rows that say `Net sales … Cost of sales … Income before provision
for income taxes` — is never consulted for this.

For Apple that is fatal in a second way: its document produced **62 blocks, all of them
TABLE and none of them a paragraph or heading**, so `cur` never advances past its initial
`UNKNOWN` and every table is labelled from a running value that never changed.

## The signal that is being missed

`section()` applied to the table's own text gives the right answer, and the parser already
holds that text:

```
Apple, table order 4731
  'Years ended September 27, 2025 ... Net sales: Products $ 307,003 ... Cost of sales ...'
  section(own text) -> INCOME_STATEMENT      (current label: UNKNOWN)
```

That is a **source-derived** signal — the words in the filing — not a keyword guessed at from
a heading that may not exist.

## The measurement I do not trust

The false-positive count came out at 16, all Microsoft, and **it is wrong**. The oracle
required phrases like `statements of income`; Microsoft titles its statements `INCOME
STATEMENTS`, so the oracle scored four genuine statements as false positives:

```
INCOME_STATEMENT  '(In millions, except per share amounts) Year Ended June 30, 2025 2024 2023 Revenue: Product …'
BALANCE_SHEET     '(In millions) June 30, 2025 2024 Assets Current assets: Cash and cash equivalents $ 30,242 …'
CASH_FLOW         '(In millions) Year Ended June 30, 2025 2024 2023 Operations Net income $ 101,832 …'
```

Two of the sixteen are genuine — a quarterly MD&A table and a debt-maturity note, both
labelled `CASH_FLOW` — but **the number as reported is not usable and should not be quoted**.
The false-positive side needs its own oracle before it can be measured, and that is part of
what A2B-19 has to build.

So: **the false-negative side is measured and large (120); the false-positive side is
attested by two examples and unquantified.**

## What the diagnosis says to build

The taxonomy and this diagnosis agree on the shape of the fix. `statement_type` describes
what a table is *about*; what the resolver needs is whether the table has the standing to
speak for the company. Those are two properties and one field was carrying both:

```
statement_type    INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW / NOTES / MDA / …
                  what the table is about

table_role        PRIMARY_FINANCIAL_STATEMENT / SUPPLEMENTARY / NOTE_TABLE /
                  SEGMENT_TABLE / MD&A_TABLE / UNKNOWN
                  whether it may speak for the company
```

And the resolver's test becomes `table_role == PRIMARY_FINANCIAL_STATEMENT and
statement_type in (expected family)` rather than `statement_type` alone — which is the
contract A2B-15 showed to be missing when `INCOME_STATEMENT` alone could not separate
JPMorganChase's consolidated statement from its per-share summary.

## The exit gate this passes, and the one it does not

```
1. an authority inventory for all eight filings                         yes
2. known primary statements explained as false negatives                yes -- 120, and
                                                                        the mechanism is
                                                                        the running heading
3. known note misclassifications explained as false positives           partly -- two
                                                                        examples, oracle
                                                                        too narrow to count
4. source-derived classifier signals proposed                          yes -- the table's
                                                                        own text
5. no dependence on case id or company-specific special case            yes
```

Item 3 is the one not met, and it is the one that matters most: the classifier's existing
false positives are what the primary-statement rule was built on, and a new classifier that
fixes recall without measuring them would trade `C` for more `B`. **A2B-19 should build the
false-positive oracle first.**

## Status

Nothing changed. Evidence at `artifacts/evaluation/p1-6-a2b18-diagnosis/`.
