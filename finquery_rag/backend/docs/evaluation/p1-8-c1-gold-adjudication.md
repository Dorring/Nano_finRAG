# P1.8-C1 — adjudication of the four gold-attribution cases

Benchmark V2 work. **Verdicts only; no benchmark file is modified by this
document.** Each verdict carries its source evidence and authority rule, as
required. The migration itself is C3.

Source of record for every row below: the filing's own `primary.html`, table
elements read directly (not the store, not the iXBRL layer).

---

## 1. `tv2f01-s2-sum-007` → **REPOINT_GOLD**

```
question    "What is the sum of The Coca-Cola Company's Operating income
             across FY2024 and FY2025?"
entity      The Coca-Cola Company     document ko_fy2025
old gold    25962.0000   operands {$ 12,536, 13,426}
```

**Source.**

```
CONSOLIDATED STATEMENTS OF INCOME   (preceded by "THE COCA-COLA COMPANY AND SUBSIDIARIES")
    Operating Income | 13,762 | 9,992 | 11,311        FY2025 FY2024 FY2023

"A summary of financial information for our equity method investees in the aggregate"
    Operating income | $ | 13,426 $ | 12,536 $ | 11,868
```

**Authority rule.** Benchmark Authority Contract §2 — a question's coordinate
denotes the company-level fact unless the question names a scope. The question
names none. Coca-Cola's company-level `us-gaap:OperatingIncomeLoss` is 13,762
(FY2025) and 9,992 (FY2024), both `dimension_count == 0`; the investee row
carries `EquityMethodInvestmentNonconsolidatedInvesteeAxis`.

**Verdict.**

```
new gold    23754.0000   operands {9,992 ; 13,762}
new fact_ids   the company-level facts for us-gaap:OperatingIncomeLoss FY2024/FY2025
status      ANSWER (unchanged)
```

The question names one entity, one metric and two periods, all of which
determine one value each. It is answerable; the gold was wrong.

---

## 2. `tv2f01-s1-aapl-003` → **SOURCE_AMBIGUOUS**

```
question    "In the Apple annual report, what is the figure for Services in FY2025?"
entity      Apple     document aapl_fy2025
old gold    75.4%
```

**Source.** The coordinate (Apple, Services, FY2025) is four rows in four
different tables, each with its own caption:

```
p.23  Gross Margin                     "Services | 82,314 | 71,050 | 60,345"
                                        -> Services GROSS MARGIN ($m)
p.24  Gross margin percentage          "Services | 75.4 | % | 73.9 | % | 70.8 | %"
                                        -> Services GROSS MARGIN PERCENTAGE   <- old gold
p.22  CONSOLIDATED STATEMENTS OF OPERATIONS  "Services | 109,158 | 96,169 | 85,200"
                                        -> Services NET SALES ($m)
p.22  (same statement, Cost of sales)  "Services | 26,844 | 25,119 | 24,855"
                                        -> Services COST OF SALES ($m)
```

Corroborating: p.22 "Products and Services Performance" and p.35
"disaggregated net sales" both state Services net sales = 109,158.

**Authority rule.** Contract §3 class A — where the question names neither a
column nor a scope and several real quantities satisfy it, no source-level
evidence selects between them.

**Verdict.**

```
new gold    none
new status  ABSTENTION (expected_outcome ABSTAIN)
```

**Why this is not a repoint.** The natural reading of "the figure for Services"
is net sales (109,158), and the old gold — a gross margin *percentage* — is
plainly not a "figure for Services" in the sense the question means. But the
question does not say "net sales". It says "the figure", and four figures exist.
Choosing 109,158 would be this audit writing the question rather than reading
it. Per the brief, abstention is permitted **only** where question + source
structure cannot uniquely determine the answer — and that is exactly this case.

---

## 3. `tv2f01-s1-tsla-032` → **REPOINT_GOLD**

```
question    "How much did Tesla report for Total automotive cost of revenues
             as of fiscal year 2025?"
entity      Tesla     document tsla_fy2025
old gold    (5,708)
```

**Source.**

```
p.56 "Cost of Revenues and Gross Margin"
   Year Ended December 31, | 2025 | 2024 | 2023 | 2025 vs 2024 ($) | 2025 vs 2024 (%)
   Total automotive cost of revenues | 57,165 | 62,873 | 66,389 | (5,708) | (9) %

p.49 Consolidated Statements of Operations
   Total automotive cost of revenues | 57,165 | 62,873 | 66,389
```

**The defect, named.** `(5,708)` is not a value of this row — it is the row's
**year-over-year change**, read from a change column of the same table. `(9) %`
is the percentage change beside it. The store emitted the comparison columns as
if they were values.

**Authority rule.** The row "Total automotive cost of revenues" states exactly
one value for FY2025 in both tables that carry it: 57,165. The question names
the row exactly.

**Verdict.**

```
new gold    57165
new fact_ids   the FY2025 "Total automotive cost of revenues" fact
new status  ANSWER (unchanged)
```

---

## 4. `tv2f01-s1-tsla-033` → **REPOINT_GOLD**

```
question    "In the Tesla annual report, what is the figure for Revenues in FY2025?"
entity      Tesla     document tsla_fy2025
old gold    $ 82,056
```

**Source.**

```
p.49 Consolidated Statements of Operations        Total revenues | 94,827 | 97,690 | 96,773
p.54 "The following table disaggregates our revenue by major source"
                                                  Total revenues | $ 94,827 | ...
p.59 "The following table presents revenues, cost of revenues and gross profit
      by reportable segment"
      Automotive segment   Revenues | $ 82,056 | $ 87,604 | $ 90,738
p.33 "Revenues" (MD&A)  Total revenues | $ 94,827 | ...
```

`82,056` is Tesla's **automotive segment** revenue. `94,827` is company total
revenue, stated in three tables — including the consolidated statement of
operations.

**Authority rule.** Contract §2. The question names no scope, so the
company-level fact governs. Measured against the iXBRL layer: 94,827 appears at
`dimension_count == 0` under two concepts (`us-gaap:Revenues` and
`us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax`); **82,056 appears
at no company-level concept.**

**Verdict.**

```
new gold    94827
new fact_ids   the FY2025 Total revenues fact
new status  ANSWER (unchanged)
```

---

## Summary

```
case         verdict              old gold    new gold   status
sum-007      REPOINT_GOLD         25,962      23,754     ANSWER
aapl-003     SOURCE_AMBIGUOUS     75.4%       --         ABSTENTION
tsla-032     REPOINT_GOLD         (5,708)     57,165     ANSWER
tsla-033     REPOINT_GOLD         82,056      94,827     ANSWER
```

Three repoints and one abstention. Two distinct root causes appear:

```
WRONG TABLE         the gold drew from a different table than the question
                    denotes                                                    sum-007, tsla-033
COLUMN MISREAD      the gold drew from a CHANGE column of the correct table   tsla-032
UNDER-DETERMINED    the question names a row label four real figures satisfy  aapl-003
```

`COLUMN MISREAD` is the same defect class as the 8,737 physical keys over the
logical count: the store reads every printed number as a value. It is worth
checking for in the remaining 21.
