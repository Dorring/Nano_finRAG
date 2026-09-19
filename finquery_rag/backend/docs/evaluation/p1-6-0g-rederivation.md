# P1.6-0G — re-derivation of the cross-entity stratum

Chosen over patching (option A). `p1-6-0e` is superseded. **Audited and corrected**:
this revision incorporates the independent review at
`artifacts/evaluation/p1-6-0g-rederivation/p1-6-0g-audit-review.md`, with the reviewer's
load-bearing claims re-verified against the filings before acceptance (see
"Review reconciliation" at the end — two of its citations do not match the source).

## The design rule

A case survives only if **one concept is disclosed as a single company-level line by
every entity in it**. `rank-004` is the model: all four companies put `Interest expense`
on their own income statement, so the question has one subject.

This replaces the old construction, which matched a metric *string* against the store.
That string matched several concepts per filing — `Total` lives in every note, `United
States` is both a revenue geography and a tax jurisdiction — and it is why only 2 of the
20 cases held one quantity for all entities.

**The rule is not decorative; it removed a case.** `crossdiff-005` was assigned
`Total liabilities` here, and Coca-Cola's balance sheet has no such line — it itemises
`Total Current Liabilities 21,281`, `Long-term debt 42,119`, `Other noncurrent
liabilities 4,735`, `Deferred income tax liabilities 2,406` and then goes straight to
`Shareowners' Equity`. A `Total liabilities` question would force the evaluator to sum
four lines, which is the defect the rule exists to prevent. The case moves to
`Total assets`, where both filers publish one line.

## Status

`C` = read and confirmed against the filing in this work (or re-verified here).
Every value in the stratum is now `C`; nothing remains machine-proposed.

## The stratum (20 cases, none retired)

### Comparison (10)

| case | entities | concept | values (USD m) | gold verdict |
|---|---|---|---|---|
| compare-001 | Coca-Cola, Tesla | Net income | KO `13,137` · TSLA `3,855` | Coca-Cola > Tesla |
| compare-002 | Apple, Visa | Net income | AAPL `112,010` · V `20,058` | Apple > Visa |
| compare-003 | Tesla, Coca-Cola | **Total assets** | TSLA `137,806` · KO `104,816` | Tesla > Coca-Cola |
| compare-004 | Apple, Microsoft | Net income | AAPL `112,010` · MSFT `101,832` | Apple > Microsoft |
| compare-005 | JPMorganChase, Apple | Diluted earnings per share | JPM `20.02` · AAPL `7.46` | JPMorganChase > Apple |
| compare-006 | JPMorganChase, Tesla | Comprehensive income | JPM `65,214` · TSLA `4,886` | JPMorganChase > Tesla |
| compare-007 | JPMorganChase, Pfizer | Net income | JPM `57,048` · PFE `8,062` | JPMorganChase > Pfizer |
| compare-008 | JPMorganChase, Apple | Diluted earnings per share | JPM `20.02` · AAPL `7.46` | JPMorganChase > Apple |
| compare-009 | Apple, JPMorganChase | Net income | AAPL `112,010` · JPM `57,048` | Apple > JPMorganChase |
| compare-010 | Apple, Microsoft | **Operating income** | AAPL `133,050` · MSFT `128,528` | Apple > Microsoft |

### Difference (5)

| case | entities | concept | values (USD m) | gold verdict |
|---|---|---|---|---|
| crossdiff-001 | NVIDIA, Tesla | Net income | NVDA `72,880` · TSLA `3,855` | `69,025` |
| crossdiff-002 | Apple, Microsoft | Net income | AAPL `112,010` · MSFT `101,832` | `10,178` |
| crossdiff-003 | Apple, Tesla | Total liabilities | AAPL `285,508` · TSLA `54,941` | `230,567` |
| crossdiff-004 | Coca-Cola, Visa | Long-term debt | KO `42,119` · V `19,602` | `22,517` |
| crossdiff-005 | NVIDIA, Coca-Cola | **Total assets** | NVDA `111,601` · KO `104,816` | `6,785` |

### Ranking (5)

| case | entities | concept | values (USD m) | gold ranking |
|---|---|---|---|---|
| rank-001 | Visa, Coca-Cola, Tesla | Net income | `20,058` · `13,137` · `3,855` | Visa, Coca-Cola, Tesla |
| rank-002 | Apple, Microsoft, Tesla | Research and development | `34,550` · `32,488` · `6,411` | Apple, Microsoft, Tesla |
| rank-003 | JPMorganChase, Microsoft, Visa, Tesla | Comprehensive income | `65,214` · `104,075` · `20,614` · `4,886` | Microsoft, JPMorganChase, Visa, Tesla |
| rank-004 | JPMorganChase, Coca-Cola, Visa, Microsoft | Interest expense | `97,898` · `1,654` · `(589)` · `(2,385)` | JPMorganChase, Coca-Cola, Visa, Microsoft |
| rank-005 | Apple, Microsoft, Coca-Cola | **Operating income** | `133,050` · `128,528` · `13,762` | Apple, Microsoft, The Coca-Cola Company |

## Source evidence for the values added in this revision

```
KO net income          13,137  ko   p63  "Consolidated Net Income 13,137 10,649 10,703"
KO operating income    13,762  ko   p63  "Operating Income 13,762 9,992 11,311"
KO total assets       104,816  ko   p65  "Total Assets 104,816" (consolidated balance sheet)
KO no total-liabilities line ko p65      section ends "Deferred income tax liabilities 2,406"
                                          then "Shareowners' Equity"
TSLA net income         3,855  tsla p70  "Net income 3,855 7,153 14,974"
TSLA operating income   4,355  tsla p70  "Income from operations 4,355 7,076 8,891"
TSLA total assets     137,806  tsla p68  "Total assets 137,806 122,070"
AAPL operating income 133,050  aapl p32  "Operating income 133,050 123,216 114,301"
MSFT net income       101,832  msft p38  "Net income $ 101,832 $ 88,136 $ 72,361"
MSFT total assets     619,003  msft p40  "Total assets $ 619,003 $ 512,163"
MSFT total liab.      275,524  msft p40  "Total liabilities 275,524 243,686"
NVDA operating income  81,453  nvda p144 "Operating income 81,453 32,972 4,224"
NVDA total assets     111,601  nvda p146 "Total assets $ 111,601"
```

## Two properties the re-derivation had to fix, and one it could not

**Concept clustering.** The first draft put nine cases on `Net income`, with exact
duplicates: `compare-001` and `compare-003` were both (Coca-Cola, Tesla) on net income,
and `compare-004`, `compare-010` and `crossdiff-002` were all (Apple, Microsoft) on it.
Two cases asking the identical question of the identical pair is not two measurements.
`compare-003` moves to `Total assets` (which also flips the expected direction — Tesla
larger, where its net income is smaller) and `compare-010` to `Operating income`.

**`rank-005` is rescued, not retired.** The first draft retired it because the store
holds no shared concept. That was a *store* limitation, not a disclosure one: Apple,
Microsoft and Coca-Cola each publish `Operating income` as a single line. The
re-derivation's candidate finder searches the store, so an XBRL label normalisation
gap read as "no such concept exists". Recorded because the same inference — "the store
has nothing, therefore nothing exists" — would retire other cases wrongly.

**`compare-007` spans two fiscal years and cannot be silently aligned.** JPMorganChase
is FY2025 (`pfe` is the only FY2024 filing in the corpus). Comparing `57,048` against
`8,062` is a legitimate cross-entity comparison only if the question says so; the case's
question text and any case metadata must state **JPMorgan Chase FY2025 vs Pfizer
FY2024**, or an automated runner will align the periods and score a false result either
way.

## Review reconciliation

Accepted after re-verification: the Coca-Cola `Total liabilities` defect; the
`rank-005` rescue via `Operating income`; the deduplication of `compare-001/003` and
`compare-004/010/crossdiff-002`; the `compare-007` fiscal-year trap; and every value in
the tables above.

Two citations in the review do not match the source and are recorded rather than
copied:

- It lists Coca-Cola's liability lines as including `Other Liabilities 10,741`. The
  filing reads `Other noncurrent liabilities 4,735` followed by `Deferred income tax
  liabilities 2,406`. The conclusion — no `Total liabilities` line — is unaffected and
  independently confirmed.
- It cites Microsoft's `Operating income 128,528` to page 38. The value is on **page
  26**; Microsoft's PDF splits labels and values into separate blocks so the income
  statement page does not carry the number in one block. The value is right, the page
  is not.

## Files

```
docs/evaluation/
  p1-6-0g-rederivation.md              this document
  p1-6-0f-coherence-pass.md            why the old stratum is not patchable
  p1-6-0f-partner-coherence-audit.md   the partner check that started this
scripts/evaluation/
  derive_cross_entity_metrics.py       candidate finder (store-side)
  read_company_line.py                 source reader (proposal only)
  audit_cross_entity_coherence.py      per-entity evidence workbook
artifacts/evaluation/p1-6-0g-rederivation/
  cross-entity-metric-candidates.json  candidates per case, with store statuses
  p1-6-0g-audit-review.md              the independent review
```

## Not yet done

**No fixture has been changed.** The stratum above is a specification. Writing it into
`benchmarks/` — gold values, `fact_ids`, plan `required_slots[].metric`, question text,
and the manifest hash — is the migration, and it is a single batch.
