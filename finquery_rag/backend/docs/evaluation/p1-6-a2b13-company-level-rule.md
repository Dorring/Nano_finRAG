# P1.6-A2B-13 — the company-level rule was mis-specified, and eight filings showed it

Rolling Store V2 out to all eight filings made the stratum answerable for the first time,
and the answer is mostly refusal:

```
slot statuses   RESOLVED_COMPANY_LEVEL 4 | NO_COMPANY_LEVEL_FACT 40 | NO_FACT 3
case states     PARTIALLY_RESOLVED 4 | UNRESOLVED 16
```

Forty of forty-seven slots refuse. The rule that identifies a company-level figure is wrong,
and the data says how.

## What a company column actually looks like

For the refusing slots, the candidates' column headers are:

```
'Year Ended December 31 / 2025'                   a period
'Year Ended / Jan 26, 2025'                       a period
'Statements of income and comprehensive income …' a table title
'(In millions) / Year Ended June 30 / 2025'       a unit and a period
'September 27, 2025 / ASSETS: / Current assets …' a balance-sheet section
'Financial performance of JPMorganChase / …'      a table title
```

**Not one of them names a company column**, and that is not a coincidence. In a consolidated
income statement the columns *are* the periods — 2025, 2024, 2023 — because the table as a
whole is the company. There is no `Total` column to find, and looking for one was the wrong
question.

## What the rule should be

The distinguishing property is not the column's name but **which statement the row is in**:

```
primary statement (income / balance sheet / cash flow)
    the period column is the company figure

breakdown (segment, jurisdiction, note)
    the columns name scopes, and a company figure exists only where one is headed Total
```

That is also why the two slots that did resolve, resolved: `JPMorganChase Net income` and
`The Coca-Cola Company Net income` sit in segment tables that *do* carry a `Total` column. The
rule worked exactly where a breakdown table handily labelled its total, and failed everywhere
else — which is a fair description of a rule fitted to one example.

## What this needs

The store does not currently carry the statement type. The parser computes a per-table
`section_type` (`INCOME_STATEMENT`, `BALANCE_SHEET`, `CASH_FLOW`, `NOTES`, `MDA`, …), and it
is not among the fields `to_v2` writes — so the one signal that would separate a primary
statement from a breakdown is the one that did not travel.

So the next step is small and well-defined: **carry `section_type` from the parsed table onto
the store record, and key company-level on it rather than on the column name.** The eight
filings make it checkable rather than a guess — 47 slots with known expected values is
enough to tell whether the new rule is right.

Two things to keep in view while doing it:

- `section_type` is `UNKNOWN` for most tables in a filing — 645 of JPMorganChase's 679. That
  is expected, since most tables are notes, but it means `UNKNOWN` must be treated as "not a
  primary statement" and never as "probably one".
- Some rows are genuinely company-level *and* in a note — a note restating the consolidated
  figure. The rule will refuse those, which is the right default until there is a way to
  tell.

## What the rollout did establish

- All eight filings build: **26,977 records**, every one carrying `column_header`, `row_id`
  and `cell_id`, and every filing reproducible byte for byte.
- The five audited JPMorganChase facts still hold.
- The entity name now comes from the corpus manifest with **no fallback** — the ticker
  fallback is what once produced a store whose every record said `JPM` while the benchmark
  says `JPMorganChase`, silently, so that every slot matched nothing.

## Status

No fixture, denominator, legacy store, view, index or production behaviour changed. Store V2
is a new file beside them and nothing reads it in production.
