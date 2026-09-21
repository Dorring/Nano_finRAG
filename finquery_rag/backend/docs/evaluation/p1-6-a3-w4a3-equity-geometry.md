# P1.6-A3-W4-A3 — the stockholders' equity statement, and what its periods actually are

Diagnosis only. No fix, no producer change.

W4-A2 left 474 oracle-PRIMARY cells the legacy stores and the new producer refuses, across
seven tables — six of them statements of stockholders' equity. That concentration was the
reason to look, and looking changed what the number means.

```
python diagnose_equity_geometry.py --delta <artifact> --out artifacts/evaluation/p1-6-a3-w4a3-equity
```

## 1. The geometry is row-major, everywhere, without exception

Every one of the six statements carries its periods as **row labels**, and its columns as
equity *components*:

```
tsla 9172   27x57   Balance as of December 31, 2022 / 2023 / 2024 / 2025   at rows 3, 10, 19, 26
nvda 7899   29x36   Balances as of Jan 30, 2022 / Jan 29, 2023 / Jan 28, 2024 / Jan 26, 2025
                                                                           at rows 4, 12, 20, 28
v    12525  19x48   Balance as of September 30, 2023 / 2024               at rows 4, 18
v    11902  18x54   Balance as of September 30, 2024 / 2025               at rows 4, 17
v    13135  18x48   Balance as of September 30, 2022 / 2023               at rows 4, 17
jpm  63944  33x21   `Balance at January 1` / `December 31` -- no year in the label
```

A column here is `Retained Earnings` or `Accumulated Other Comprehensive Income (Loss)`.
**A column has no period at all.** The column binder is being asked a question this table
does not answer, and its refusal is the correct response to it.

## 2. The legacy selector treats the balance rows as header rows

```
tsla 9172   header_idx [0, 1, 2, 3, 10, 19, 26]        3, 10, 19, 26 are balance rows
nvda 7899   header_idx [0, 1, 2, 3, 4, 12, 20, 28]     4, 12, 20, 28 are balance rows
v    12525  header_idx [0, 1, 2, 3, 4, 16, 18]         4, 18 are balance rows
```

Every date-bearing row is classified as a header. So the legacy's per-cell date extraction
reads `header_path + cell_text` and finds **several** balance dates — the whole table's
worth — and takes `dates[0]`.

## 3. What the legacy actually assigns — and it is not a missing period

All 474 carry a `normalized_period`. The first version of this diagnosis printed only
`period_start`/`period_end` and concluded "legacy has no period at all" for 436 of them.
That was wrong, and checking the field the store actually carries is what showed it.

```
                                                   count   legacy period names
matches the OPENING balance above the cell           281   the section's opening date
matches neither -- it is the table's earliest date   191   2022-12-31 / 2022-01-30 / ...
no period at all                                       2
                                                     ---
                                                     474
```

```
nvda 7899  r5 'Net income'   legacy 2022-01-30   between 'Balances as of Jan 30, 2022'
                                                        and 'Balances as of Jan 29, 2023'
v    12525 r5 'Net income'   legacy 2023-09-30   between 'Balance as of September 30, 2023'
                                                        and 'Balance as of September 30, 2024'
tsla 9172  r20+              legacy 2022-12-31   between the 2024 and 2025 balances
```

**The activity rows sit between two balances and report the period those balances bracket.
The legacy names the earlier one — the instant the period *started*, or for 191 cells the
table's first date no matter where the row is.**

Whether the correct answer is specifically the *closing* balance date is an accounting
question and is not settled here. What is settled does not need it: **the assigned date is
the opening instant of the period the row reports activity during, so it is not that
period.** A fact cannot carry the moment its own period began as the period it belongs to.

## What this changes

The 474 were carried in W4-A and W4-A2 as **coverage the store would lose if V2 switched**.
They are not that. They are primary-statement facts the store is holding **under a period
that is not theirs**, and V2's refusal withholds them rather than storing them wrongly.

That does not make V2 correct — the right outcome is these cells stored under their real
period, which is neither what the legacy does nor what the producer does today. It makes
the direction of the change different from the one the count suggested, and it means the
entry condition for W4-B is not simply "producer coverage >= legacy coverage": a producer
that *matched* the legacy here would be reproducing a defect.

## What a real fix would be, and why it is not this phase

The period of a cell in a row-major statement is a **section** — the span between one
balance row and the next. That is not any of the five methods, and it is not
`COLUMN`, `ROW` or `CELL_GROUP`:

```
new scope     SECTION   the binding covers the rows between two declarations
new method    SECTION_BRACKET   a period bounded by the balance that opens it and the
                                balance that closes it
```

It is a contract change, and it interacts with the legacy header selector that currently
calls the bounding rows headers. It is not a widening, and doing it as a side effect of W4
would be exactly the kind of unowned scope change this line of work keeps refusing.

## Registered

- The `matches OPENING balance` / `matches neither` split is measured, not adjudicated cell
  by cell; the mechanism behind the second bucket (`dates[0]` over a header path holding
  every balance in the table) is inferred from the code and matches all 191, but has not
  been stepped through.
- Whether these wrong-period facts reach the 47-slot resolver is **not** checked here. The
  seal's gates read zero, which says the slot queries do not land on them; it does not say
  they are absent from the store. They are: 474 of them.
- `v_fy2025#13696`'s 2 cells have an empty header column and no period. Unrelated to the
  rest, and correctly unbound.
