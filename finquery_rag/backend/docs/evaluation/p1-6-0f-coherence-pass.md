# P1.6-0F — the cross-entity stratum is not repairable by slot

Every entity of all 20 cross-entity cases was located in its filing, and the evidence
frozen in `cross-entity-coherence.json` (47 of 47 values found). Read as a whole, the
stratum's defect is **not** eleven mis-scoped operands. Only **2 of the 20 cases** hold a
coherent quantity for every entity.

That changes the plan: the eleven-slot migration in `p1-6-0e-metric-replacements.md` is
not sufficient, and should not be executed.

## Verdicts

| verdict | meaning | cases |
|---|---|---|
| **COHERENT** | every entity's value is the asked-for quantity at company level | **2** |
| **SCOPE** | one quantity, but some entity carries a component | 3 |
| **SIGN** | one quantity, but a value's sign is wrong | 1 |
| **INCOHERENT** | the metric matches a *different concept* for different entities | 11 |
| **NEEDS CHECK** | evidence gathered, adjudication not yet made | 3 |

## The two that work

```
compare-008  Diluted earnings per share
  Apple  $7.46  p32  "Basic $ 7.49 … Diluted $ 7.46"        consolidated income statement
  JPM    20.02  p2   "Net income per share: Basic $ 20.05 … Diluted 20.02"

rank-004  Interest expense
  JPM       97,898  p197  "Interest expense 97,898 101,350 81,321"
  Microsoft (2,385) p29   "Interest expense (2,385) (2,935)"
  Coca-Cola  1,654  p51   "Interest expense was $1,654 million in 2025"
  Visa       (589)  p63   "Interest expense … $ (589) $ (641)"
```

Note `rank-004` mixes signs legitimately — all four are interest expense, and Visa and
Microsoft present it as an outflow.

## SCOPE and SIGN — one quantity, one bad operand

```
compare-009  Net income       Apple 112,010 ✓ | JPM 4,520 = Corporate segment column
compare-010  Other current liabilities
             Apple  44,452  p43  "Other current liabilities 44,452 … Total other current liabilities $ 66,387"
             Microsoft 5,424 p71 (lease note) — balance sheet is 25,020
crossdiff-004 Long-term debt  Visa 19,602 ✓ balance sheet | Coca-Cola 850 = held-for-sale package
rank-002     Research and development
             Apple "(34,550)" p27 — the line reads "$ 34,550", positive; parentheses are a sign defect
             Microsoft 32,488 ✓   Tesla $6,411 ✓
```

`rank-002` matters most: with the sign corrected the true ranking is
`Apple > Microsoft > Tesla`, so `expected_ranking = ["Microsoft","Tesla","Apple"]` is
itself derived from the defect, and the runtime's released answer was **wrong**, not
lucky.

## INCOHERENT — the metric matches different concepts

Eleven cases. In each, the two entities' values are not the same kind of number, so the
question has no single subject and no value substitution can repair it.

```
compare-001  "United States"
  Tesla 47,627  p129  heading "revenues by geographic area"        -> US REVENUE
  Coca-Cola 5,678 p103 "Income before income taxes consisted of"   -> US PRETAX INCOME

compare-003  "Gross profit"
  Tesla 13,292 p57  "Gross profit total automotive & services and other segment"
  Coca-Cola 42,178 p87 heading "financial information for our equity method investees
                              in the aggregate"                    -> the INVESTEES' gross profit,
                                                                     not Coca-Cola's

compare-004  "Foreign exchange contracts"
  Apple 62,647    p42 "The notional amounts of the Company's outstanding derivative instruments"
  Microsoft (809) p58 "Fair Values of Derivative Instruments"      -> notional vs fair value

compare-005  "Current"
  Apple 11,487 p43 "Federal: Current $ 11,487"                     -> a jurisdiction component
  JPM 840      p147 context is a loan-portfolio discussion

crossdiff-001  "Additions"
  NVIDIA 1,203 p165 "product warranty liabilities … Additions 1,203"
  Tesla 1,083  p74  "Changes in operating assets and liabilities: Accounts receivable (261) (1,083)"
                    -> a receivable movement, not an addition

crossdiff-003  "Total"
  Apple 9,683   p43 "Total 9,683 …" under "Federal:"              -> a tax sub-total
  Tesla 13,279  p68 "Accrued liabilities and other 13,279 10,723" -> accrued liabilities

crossdiff-005  "Land"
  NVIDIA 511 p161 "Property and Equipment: Land $ 511 $ 218"
  Coca-Cola 265 p57 context is a treasury-stock repurchase discussion

rank-001  "Prepaid expenses and other current assets"
  Tesla  130  p106 "Total finance lease expense $ 130"            -> lease expense
  Coca-Cola 147 p67 equity-attributable-to-noncontrolling block
  Visa   1,208 p88 restricted cash reconciliation
                    -> NONE of the three is the asked-for line

rank-005  "Interest rate contracts"
  Apple 12,875 p42 notional amount
  Microsoft 0  p26 "Productivity and Business Processes / Revenue"
  Coca-Cola 16 p3  a table-of-contents page number
```

## GARBAGE — the values are not measurements

Three cases where the gold value is a document artifact rather than a number from a
table. These are not scope errors at any level.

```
compare-006  metric "2025"
  JPMorganChase "2025" p1  "Annual Report 2025"
  Tesla         "2025" p1  an XBRL header fragment
                    -> the metric is a year and the values are page-1 strings

compare-007  "Discount rate"
  JPMorganChase "5.49 %" p255 "the weighted-average discount rate … as of December 31,
                               2025 and 2024, was 5.39% and 5.49%"
                    -> 5.49% is the 2024 rate, not 2025
  Pfizer "$5" p2   a table-of-contents page number

crossdiff-002  "Other"
  Apple 410     p44 "Impact of the State Aid Decision (486) 10,246 — Other 410 279 (188)"
                    -> a tax-reconciliation line, at least a real number
  Microsoft 72  p73 context is a column of totals; 72 is not established as this metric
```

## NEEDS CHECK

Evidence is captured in the artifact but not yet adjudicated:

```
compare-002  "General and administrative"   Apple (8,077) in a segment-column table;
                                            Visa 1,926 in a personnel breakdown
rank-003     "Comprehensive income"         JPM $65,214 ✓ and Tesla 4,886 ✓ on their
                                            statements; Microsoft $104,075 and
                                            Visa $20,614 sit in dotted-leader contexts
```

## What this means for the plan

The eleven-slot migration assumed the other operands were sound. They are not: the
stratum's gold was built by matching a metric *string*, and for most metrics that string
matches several different concepts across filings. Repairing eleven values would leave
thirteen cases still incoherent and three still garbage, with a fixture hash that claims
otherwise.

Recommended:

1. **Do not execute `p1-6-0e`.** It is superseded.
2. **Re-derive the stratum, do not patch it.** Which cases are salvageable is a
   judgment: `compare-008` and `rank-004` need nothing; `compare-009`, `compare-010`,
   `crossdiff-004`, `rank-002` need one operand corrected (with `rank-002`'s
   `expected_ranking` re-derived, which changes its answer); the rest need a metric that
   names a concept, or retirement.
3. **Retire rather than repair the GARBAGE cases.** `compare-006`, `compare-007` and
   `crossdiff-002` are not measuring anything — their gold is a document artifact.
4. **The denominator cannot be 20.** It was going to stay at 20 on the reasoning that the
   eleven contested operands were *resolved* rather than excluded. That reasoning is
   void: resolution is not available for most of the stratum.

No fixture has been changed. The remaining work is a re-derivation with a per-case
decision, not an edit.
