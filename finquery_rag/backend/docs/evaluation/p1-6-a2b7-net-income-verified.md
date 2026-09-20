# P1.6-A2B-7 — the 11,993 verified against the filing

The check that a count moving cannot make: are the facts right? Read back out of the
source, on the row the benchmark cannot separate.

## The disputed pair, from the filing

`compare-009` is the case whose two candidates the legacy store filed under one coordinate.
Read from JPMorganChase's segment table now:

```
table_f6415249a02006fe5c6a0468

   4,520  | As of December 31 / Corporate / 2025
  10,601  | As of December 31 / Corporate / 2024
   2,821  | As of December 31 / Corporate / 2023
  57,048  | As of December 31 / Total     / 2025
  58,471  | As of December 31 / Total     / 2024
  49,552  | As of December 31 / 2023
```

And the row as the source states it, from the same table's text:

```
Net income   4,520   10,601   2,821   —   —   —   57,048   58,471   49,552
```

**Value for value, column for column.** `4,520` is the Corporate column's 2025 figure;
`57,048` is Total's. The period binds to the column that carries it, three years each, and
the two are separated by the filing's own header rather than by anything inferred.

That is the whole thing P1.6-A exists to establish, and it now holds at the level of an
individual fact.

## The populated row

`Net income` across all of JPMorganChase's tables:

```
atomic facts with leaf_metric ~ "Net income"   199
values   57,048 (17) | 58,471 (17) | 49,552 (15) | 17,603 (6) | 21,232 (6)
         24,846 (6) | 20,272 (6) | 4,520 (6) | 10,601 (6) | 2,821 (6)
```

The three highest-count values are the consolidated statement's three years, and each binds
to the right period — `57,048` to `2025-12-31`, `58,471` to `2024-12-31`, `49,552` to
`2023-12-31`. All three match what was read out of the filing by hand earlier.

The rest are segments and the managed-basis view, and the column header says which:
`Consumer & Community Bankin…` (18,245, matching page 97 as read earlier),
`Financial performance of JPMorganChase`, `Managed basis` against `Reported`.

## What is verified, and what is not

**Verified:** the values on the rows that matter are the filing's, the periods bind to the
right year, and the dimension that separated nothing before now separates the two cells
`compare-009` turns on. The row-index fix is not merely a count improvement.

**Not verified:** all 11,993. This checked the `Net income` population (199 facts) and the
disputed table. The rest has not been read back, and saying "the 11,993 are correct" would
be the same overreach as saying "7,322 is the right number".

Also noted and not chased: some column headers carry the year and some do not
(`As of December 31 / 2023` against `... / Corporate / 2023`), and one row produced a
`(2)` change-column value with `FYNone`. Neither affects the values above, and both are
worth a pass before the pipeline is trusted wholesale.

## Status

No code changed in this step. Two source changes from the row-index fix stand. Legacy store
frozen and untouched, no fixture or denominator changed.
