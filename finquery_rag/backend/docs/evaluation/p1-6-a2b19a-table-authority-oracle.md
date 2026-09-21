# P1.6-A2B-19A — the Primary/Non-primary table authority oracle

No classifier change, no resolver change, no fixture change. This builds the oracle the
classifier will be measured against, which had to come first: the 120 false negatives in
A2B-18 were counted with the same rules under criticism, and the false-positive side —
the side that matters, because promoting a note to company-level authority is worse than
missing a statement — could not be counted at all.

## The question, and why it is not `statement_type`

```
statement_type   INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW / NOTES / ...
                 what the table is about
table_role       PRIMARY_FINANCIAL_STATEMENT / NON_PRIMARY / UNRESOLVED
                 whether it may speak for the company
```

The resolver needs the second. `statement_type` has been carrying both.

## What the oracle reads that the classifier cannot

`section_type` reaches a table by a running heading (`make_blocks`) or by
`section(caption + prior + headers)` (`parse_table`). `make_blocks` emits blocks only for
`table` / `h1`-`h6` / `p` / `li` / `dt` / `dd`. Apple, Coca-Cola, Visa, NVIDIA, Pfizer and
Tesla all render their statement titles as

```html
<div><span>CONSOLIDATED STATEMENTS OF OPERATIONS</span></div>
```

which is emitted by none of those, so the title produces no block and the table's own body
is never consulted either. **Microsoft is the eighth filing and the only one that works**,
and only because it alone sets `id="income_statements"` on its title `<p>` — which
`make_blocks` does read. The classifier's entire 16-label success is one filing's
idiosyncratic id convention.

The oracle reads instead:

- **the standalone lines above the table** — a standalone line being an element whose whole
  text is its own text node, which is what a title is and a sentence is not. This finds
  `CONSOLIDATED STATEMENTS OF CASH FLOWS` above the cash flow statement and does not find
  the same words inside an MD&A sentence;
- **the iXBRL context each fact points at.** An undimensioned context states a
  company-level amount; a dimensioned one states a disaggregation. This is the filing
  declaring its own scope and is the one signal here that is not a reading of prose. Apple
  tags 969 facts against 182 contexts, 168 of them dimensioned;
- **the table's own body** — its first row, its row labels, its column headers.

## The decisions, and only these

Three rules, each the source stating the answer rather than this script opining. Everything
else is left `UNRESOLVED` and is not guessed at, because a third guess-derived bucket would
spend the oracle's only asset.

```
TITLE_STATEMENT_OF_RECORD     a statement title stands within three lines above the table
                              and the table carries company-level facts        -> PRIMARY
DIMENSIONED_CONTEXT_ONLY      every tagged fact sits in a dimensioned context -> NON_PRIMARY
OWN_CAPTION_DISAGGREGATION    the table's own first row names a subject rather
                              than a units caption                            -> NON_PRIMARY
LAYOUT_SCAFFOLDING            no tagged fact, at most two rows, no row of figures
                                                                              -> NON_PRIMARY
```

`TITLE_STATEMENT_OF_RECORD` had to tolerate two things the filings do. The units caption
usually sits between the title and the table (Apple, Coca-Cola, NVIDIA: title, then
`(In millions)`, then the table), which is why the distance is three and not one; and
Workiva splits titles across adjacent spans, so Microsoft's income statement is two lines
reading `INC` and `OME STATEMENTS` and `BALANCE` / `SHEETS` is two more. Contiguous runs of
up to three lines are therefore tried joined both with and without a space — which is how
Microsoft is found by its *text* and not only by its `id`.

`OWN_CAPTION_DISAGGREGATION` exists because of one measured failure. Pfizer's
`Supplemental Cash Flow Information` sits two tables below the cash flow statement and
inherits its title from the page above:

```
pfe_fy2024#25973  row_labels[0] = '(MILLIONS)'                        -> the statement
pfe_fy2024#26989  row_labels[0] = 'Supplemental Cash Flow Information' -> the note
```

What separates them is that the statement opens with a units caption and the note opens by
naming its own subject. A title outside a table cannot outrank a caption inside it. The
rule as first written also swallowed six real statements — it read
`(In millions, except per share amounts)` and `(MILLIONS, EXCEPT PER SHARE DATA)` as
subject captions, because both contain `per share` — so units captions are excluded, and
the corrected rule removes the one false positive and none of the 42.

## The result

```
                        tables   PRIMARY   NON_PRIMARY   UNRESOLVED
aapl_fy2025                 62         5            16           41
jpm_fy2025                 679         5           461          213
ko_fy2025                  117         5            30           82
msft_fy2025                 85         5            10           70
nvda_fy2025                 68         5            16           47
pfe_fy2024                 322         5           237           80
tsla_fy2025                 79         5            15           59
v_fy2025                    86         7            23           56
                        ------      ----         -----         ----
                          1498        42           808          648
```

Every filing yields its own statements and nothing else. Apple 23-27, JPMorganChase
329-338, Coca-Cola 24-28, Microsoft 24-28, NVIDIA 22-26, Pfizer 115-123, Tesla 22-26, Visa
25-31 — income, comprehensive income, balance sheet, equity and cash flow, in filing order,
with Visa's equity statement correctly spanning three continued tables. **All 42 were read
and all 42 are genuine.**

**The A2B-18 denominator was wrong and this corrects it.** 592 of the 1498 "tables" are
page-layout scaffolding — Workiva lays a page out as a stack of one- and two-row tables,
carrying page numbers, cover-page checkboxes and empty spacers. A2B-18's 1498 over-counts
by 40%; the tables of figures number 906.

## The safety set: what must never be promoted

281 tables carry a caption naming a disaggregation, a note subject or a summary. **None of
them is promoted by the oracle to PRIMARY** — the property held mechanically across all
eight filings at every stage of this build.

```
LEASE_PENSION_OR_TAX           66
DEBT_OR_MATURITY               44
SEGMENT_OR_GEOGRAPHIC          40
DERIVATIVES_OR_HEDGING         38
FAIR_VALUE                     36
SUPPLEMENTARY_RECONCILIATION   14
EPS_OR_PER_SHARE               12
ALLOWANCE_OR_CREDIT             6
MDA_OR_NONGAAP_SUMMARY          4
QUARTERLY_OR_SELECTED           4
OTHER                          17
                             ----
                              281
```

Segment breakdowns, debt maturity schedules, derivatives and hedging notes, quarterly
selected data, MD&A and non-GAAP summary tables, EPS detail and supplementary
reconciliations are all present and all excluded.

## The current classifier, measured against it

```
TP 5   FP 11   FN 37   TN 804        precision 0.3125   recall 0.119
```

The classifier calls 16 tables primary. **Five of them are.**

**Seven of the eight filings score recall 0.0** — every primary statement in Apple,
JPMorganChase, Coca-Cola, NVIDIA, Pfizer, Tesla and Visa is `UNKNOWN`. The 37 false
negatives are those statements.

Microsoft scores recall 1.0 and precision 0.312. Of its 16 promotions:

```
 5  true            the five statements, via id="income_statements" and siblings
 4  provably wrong  all facts dimensioned -- derivative notional amounts, derivative
                    gains and losses, OCI derivative movements, changes in ...
 7  adjudicated     read by hand, all notes:
                      #11784  unearned revenue recognition schedule
                      #11982  contractual obligations / debt maturity schedule
                      #34702  derivative instruments
                      #36735  fair value of derivatives, by hierarchy level
                      #50208  supplemental cash flow information related to leases
                      #50757  supplemental balance sheet information related to leases
                      #51968  maturities of lease liabilities
```

The four provable and the seven adjudicated are the same defect seen from two sides. The
adjudications are in `scripts/evaluation/table_authority_adjudications.json`, with the
source evidence for each, so they can be disagreed with; the scorer folds them in only when
passed `--adjudications`.

**The 116 danger-captioned tables the classifier leaves alone are the reason its false
positive count is 11 and not 116.** It is not that the classifier refuses danger — it is
that it refuses everything, so on seven filings it accidentally refuses the statements too.

## Not established

**648 tables are `UNRESOLVED`** — 43% of the corpus. 347 carry no tagged fact at all and
301 carry company-level facts with no statement title standing above them. A table the
classifier promotes from that set is counted separately and never as a success or a
failure; after the adjudications above, no promoted table remains in it.

The oracle's own error rate is measured only on its positive side — 42 read, 42 correct —
and not on the 808 it calls `NON_PRIMARY`. The one error found during the build (Pfizer's
supplement) was found by reading, not by a rule, which is the reason the digest exists.

## Status

Nothing changed in the classifier or the resolver. Evidence at
`artifacts/evaluation/p1-6-a2b19a-oracle/` — `table-authority-oracle.json`, the full
per-table evidence; `adjudication-digest.md`, the same evidence as readable text;
`baseline-vs-oracle.json`, the measurement above.
