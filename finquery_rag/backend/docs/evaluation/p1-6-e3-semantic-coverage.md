# E3-1 — the remaining 57 gate blocks, audited

After E2 added 12 canonical metrics, 57 comparable answerable cases are still
refused at the semantic gate. Every one is audited here into a recovery class,
and the point of the exercise is the split, not the release rate:

```
SAFE_RECOVERABLE   = A + B     the system should be able to answer these
CORRECTLY_BLOCKED  = C + D     it should not, and the refusal is right
```

The bypass ceiling (43/75) is not the target. Chasing it would mean admitting
labels whose only argument is that they make a case pass.

## Two findings that are not about vocabulary at all

**1. Three cases are blocked by period handling, not by metric coverage.** They
are the only cases where *every* slot now resolves and the gate still refuses:

```
tv2f01-s1-pfe-028   Weighted-average shares—diluted        MISMATCH   unknown: []
tv2f01-s2-growth-002  Finished goods x2 (FY2024 & FY2025)  MISMATCH   unknown: []
tv2f01-s2-growth-006  Basic weighted average shares x2     AMBIGUOUS  unknown: []
```

All three are two-period questions — a growth rate and a fiscal-year-scoped
lookup — and the gate reads them as mismatched or ambiguous with **no unknown
field at all**. That is not a metric-vocabulary gap and no ontology entry fixes
it. It is reported separately as `PERIOD_SCOPED_BLOCK` rather than forced into
A–D, because filing it under a metric class would misattribute the cause and
send the next person to the wrong layer.

**2. Seven labels are the right concept wearing table furniture.** `Interest
expense(a)`, `Interest^{(f)}`, `Cost of revenues (1)`, `Other letters of
credit(d)`, `Common stockholders' equity(f)`, `Commercial(3)`, `Total nominal
payments volume(4)`. The parenthetical is a footnote marker the extractor
welded onto the row label. Stripping it before the ontology lookup is
deterministic, source-grounded and table-independent — and `Cost of revenues (1)`
then resolves against `cost_of_revenue`, which is **already in the registry**.
This is the `ALIGNMENT_NORMALIZATION_GAP` the E2 spec defined, and it is a
separate fix from extending vocabulary.

## A — GLOBAL_CANONICAL_CONCEPT (12 labels, to add)

Each survives the E2 test: it names one quantity, it would be recognised in any
filing that reports it, and the mapping would still be wanted with a different
benchmark.

| label | store | canonical target |
|---|---:|---|
| Interest expense(a) · Interest^{(f)} | 4 · 4 | `interest_expense` |
| Stock-based compensation expense | 19 | `stock_based_compensation` |
| Share-based compensation | 4 | same |
| Other comprehensive income (loss) | 8 | `other_comprehensive_income` |
| Total Comprehensive Income | 4 | same |
| Foreign currency translation adjustment | 4 | `foreign_currency_translation_adjustment` |
| Net foreign currency translation adjustments | 13 | same |
| Return on equity | 4 | `return_on_equity` |
| Risk-free interest rate | 4 | `risk_free_interest_rate` |
| Statutory federal income tax rate | 4 | `statutory_federal_income_tax_rate` |
| Total non-current portion of term debt | 3 | `noncurrent_term_debt` |

## B — SCOPED_CONCEPT (7 labels)

Legitimate reported facts whose meaning comes from the entity or the table. These
need `scoped_concept(entity, metric_family, member)`, not a global entry.

| label | why scoped |
|---|---|
| iPad · iPhone | members of Apple's product-revenue breakdown |
| Xtandi | a Pfizer product line |
| International transaction revenue | Visa's own revenue definition |
| Medicare rebates | a pharma-specific accrual |
| Allowance for loan losses to total retained loans | a bank-specific ratio |
| Change in fair value of derivative instruments | a cash-flow-statement line |

## C — SOURCE_AMBIGUOUS (23 labels)

Each names a position, a bucket or a family rather than one quantity, and the
source cannot settle which. `Deferred` (21 rows) is deferred revenue, tax or
compensation; only the table says which.

```
Deferred · Deferred taxes · Deferred income taxes · Services · Intersegment
Additions · Other contracts · Other contracts sold · Other accrued expenses
Total Fees
Total noninvestment-grade · International-based companies · Direct Customer B
U.S. GSEs and government agencies · Foreign currency contracts · Fair value hedges
Income tax effect · Total automotive cost of revenues
Hedge accounting fair value adjustments
Cash reserves at non-U.S. central banks and held for other general purposes
Balance in AOCI at beginning of year · Beginning balance at January 1
Revenues - excluding Comirnaty and Paxlovid
```

## D — NON_CONCEPT / CORRECT_FAIL_CLOSED (12 labels)

Names, roles, and labels with no rows behind them.

```
Colette M. Kress · figure for Ajay K. Puri · Direct Customer A value
President and CEO · State value · Services (1) · Commercial(3)
Cost of revenues (1)* · Common stockholders' equity(f) · Other letters of credit(d)
Total nominal payments volume(4) · Acquired in-process research and development expenses(…)
```

\* `Cost of revenues (1)` moves to A once the footnote marker is stripped — the
concept is already in the registry. The rest keep their markers as the reason.

## Totals

```
A  GLOBAL_CANONICAL_CONCEPT   12 labels
B  SCOPED_CONCEPT              7 labels
C  SOURCE_AMBIGUOUS           23 labels
D  NON_CONCEPT                12 labels
                             ---
                              54 labels on 57 cases   UNCLASSIFIED = 0

PERIOD_SCOPED_BLOCK            3 cases  (not a metric gap; reported apart)
```

**SAFE_RECOVERABLE (A + B) = 19 labels.** The A set is addable now. The B set
needs the scoped layer, which is the more interesting engineering work and the
better answer than flattening `iPad` into a global metric.

## Stopping condition

This round stops when `GLOBAL recoverable gaps = 0`. After A is added and the
footnote normalisation lands, every remaining block should have a `C` or `D`
reason, with the 3 period-scoped cases held apart as the one open question.
