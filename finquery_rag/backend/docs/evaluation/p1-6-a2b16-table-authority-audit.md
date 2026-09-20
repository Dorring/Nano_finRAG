# P1.6-A2B-16 — Primary Table Authority: two different problems, not one

The audit asked whether the SEC source carries a signal that says which of several
`INCOME_STATEMENT` tables is the consolidated one. The answer is yes for one of the three
slots and **there is no such candidate at all** for another, so they are not one problem.

## `crossdiff-004` — Coca-Cola / Long-term debt: no candidate is company-level

All six candidates sit in **one table**, and the table is a hedging note:

```
table_b36435626e07212385e0b69e   statement_type = BALANCE_SHEET

'Cumulative Amount of Fair Value Hedging Adjustments / Carrying Values of Hedged Items /
 Included in the Carrying Values of Hedged Items / Remaining for Which Hedge Accounting
 Has Been Discontinued'
```

Its columns are hedging scopes, not periods and not segments, and the values `-705`, `97`
and `11,648` are hedging adjustments rather than long-term debt.

**The consolidated balance sheet is not among the candidates.** So this slot's correct
outcome is `NO_COMPANY_LEVEL_FACT`, not `AMBIGUOUS`, and what is wrong is the
`statement_type` classification: a note was labelled `BALANCE_SHEET`, which is why the
primary-statement rule treated its rows as company-level.

That is a classification defect with a clear shape — a table whose columns name hedging
categories, filed in the notes — and it is fixable at the classification rather than by a
tie-break. It also means the fix will *reduce* resolution, which is the right direction
here.

## `compare-007` / `compare-009` — JPMorganChase / Net income: a real signal exists

The candidates span more than one table, and the tables are not equivalent:

```
table_8cae026a59f757ed3cd4be7c
  'Year ended December 31, (in millions, except per share amounts) 2025 2024 2023
   Basic earnings per share Net in…'
  -> a per-share summary

table_53f41623440829a3b830e2fb
  'Statements of income and comprehensive income Year ended December 31, (in millions)
   2025 2024 2023 Income Divi…'
  -> the consolidated statement of income, named by its own text
```

**The statement names itself.** `statement_type = INCOME_STATEMENT` is not enough to tell
these apart, but the table's own heading is, and that is a fact about the filing rather than
a rule invented here.

So a source-derived authority plausibly exists for this shape. Two cautions before adopting
it:

- it must be checked against the other filings rather than fitted to JPMorganChase, because
  a rule fitted to one example is exactly what produced the column-name rule that failed 40
  of 47 slots;
- it must **narrow** rather than widen: if a table does not name itself as the statement of
  record, the slot stays AMBIGUOUS and fails closed.

## What this changes

The three ambiguities are two defects and neither is a tie-break:

```
crossdiff-004      a note misclassified as BALANCE_SHEET; no company-level candidate exists
compare-007/009    several tables, one of which names itself as the consolidated statement
```

Neither should be resolved by taking the first table, the most common value, or the earliest
in filing order. The first would pick a hedging adjustment; the second and third would
resolve without evidence.

And the priority stands: a resolver that returns **nothing** for `crossdiff-004` is correct,
even though it lowers coverage.

## Status

No resolver change in this step. No fixture, denominator, legacy store, view, index or
production behaviour changed. The audit's evidence is at
`artifacts/evaluation/p1-6-a2b16-authority/`.

Next, in the order agreed: finish the authority question for `compare-007/009` across the
other filings — and separately treat the note-misclassified-as-statement defect, since it
will also be behind some of the 29 `NO_COMPANY_LEVEL_FACT`.
