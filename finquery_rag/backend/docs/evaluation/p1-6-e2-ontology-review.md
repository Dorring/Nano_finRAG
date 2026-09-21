# E2-2B final — the 22 candidates, reviewed

Scope: the labels E2-2B's mechanical rules left as `ADD_ALIAS` candidates. They
are reviewed here by hand, one at a time, against a single test:

> **Would I still want this mapping in a production ontology tomorrow, with a
> different benchmark?**

The question "does adding this release a case" was not asked and is not recorded
anywhere below. Adding a mapping because it makes a case pass is test-set fitting
with extra steps, and it would be invisible in the result.

**One structural fact decides most of the classification.** The ontology is not a
list of aliases onto 13 concepts — a new label that names a *new kind of quantity*
needs a new canonical metric, not an alias onto an existing one. `Leasehold
improvements` mapped onto `assets` would be a false equivalence, which is the
failure `NO_COLLISION` exists to prevent. So a class-A entry is a new canonical
metric, and the ontology grows from 13 to 25.

## A — GLOBAL_CANONICAL_ALIAS (12): add in E2-3

Each names one quantity, table-independent, and would be recognised in any filing
that reports it.

| label | source evidence | canonical target |
|---|---|---|
| Accumulated depreciation | Apple, Microsoft; `(76,014)` | `accumulated_depreciation` |
| Finished goods | Pfizer, Coca-Cola; `1,375` | `finished_goods` |
| Other long-term liabilities | Microsoft; `43,000` | `other_long_term_liabilities` |
| Selling and marketing | Apple; `(19,524)` | `selling_and_marketing` |
| Basic earnings per share | Apple, JPMorganChase; `$ 7.49` | `basic_eps` |
| Basic weighted average shares | NVIDIA; `24,555` | `basic_weighted_average_shares` |
| Weighted-average shares—diluted | Pfizer; `5,700` | `diluted_weighted_average_shares` |
| Tangible book value per share | JPMorganChase; `107.56` | `tangible_book_value_per_share` |
| Total lending-related commitments | JPMorganChase; `$ 24,358` | `lending_related_commitments` |
| U.S. Treasury securities | Apple, Visa; `16,074` | `us_treasury_securities` |
| Leasehold improvements | Apple, Microsoft, Tesla; `15,091` | `leasehold_improvements` |
| Land | Microsoft, NVIDIA, Pfizer; `265` | `land` |

## B — SCOPED_METRIC_CONCEPT (4): legal facts, no global mapping

Each is real reported data whose meaning comes from the entity or the table it
sits in. `iPad` is a member of Apple's product-revenue breakdown; it is not a
quantity that maps to a canonical metric, and `iPad -> revenue` would be a false
equivalence for any other filer.

| label | source evidence | why not global |
|---|---|---|
| iPad | Apple; `28,023` | a product member of a revenue breakdown |
| iPhone | Apple; `$ 209,586` | same |
| Xtandi | Pfizer; `$ 2,039` | same |
| International transaction revenue | Visa; `14,166` | Visa's own revenue definition |

Registered as a future **scoped ontology** capability — `scoped_concept(entity,
metric_family, member)` — not enabled this round.

## C — KEEP_BLOCKED (6): no semantic contract exists to admit them

| label | source evidence | why blocked |
|---|---|---|
| Revenues - excluding Comirnaty and Paxlovid | Pfizer; `$ 51,331` | bespoke non-GAAP measure, defined by this filer |
| Total noninvestment-grade | JPMorganChase; `46,670` | a credit-quality bucket of a portfolio table |
| International-based companies | Coca-Cola; `325` | a grouping row |
| Direct Customer B | NVIDIA; `11 %` | an anonymised counterparty; the label means nothing outside its table |
| Hedge accounting fair value adjustments | Apple; `(294)` | an AOCI component; ambiguous without the statement |
| Cash reserves at non-U.S. central banks and held for other general purposes | JPMorganChase; `9.6` | a specific regulatory disclosure line |

These are not extraction errors and not bad data. The current global ontology
simply has no semantic contract that could carry them, and admitting them to
raise release coverage is the trade this work exists to refuse.

## Totals

```
GLOBAL_CANONICAL_ALIAS   12
SCOPED_METRIC_CONCEPT     4
KEEP_BLOCKED              6
                         --
                         22      UNREVIEWED = 0
```

`KEEP_BLOCKED_AMBIGUOUS` (29) and `KEEP_BLOCKED_NON_CONCEPT` (16) from E2-2B are
not re-reviewed here; nothing in them changes.
