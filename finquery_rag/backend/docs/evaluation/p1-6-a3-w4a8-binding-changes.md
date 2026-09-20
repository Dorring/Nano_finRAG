# P1.6-A3-W4-A8 — what the period-header repair actually changed

Diagnosis only. The predicate is not touched.

W4-A7 replaced the period-header predicate and reported 27 columns unbound and 645 columns
changed, as counts. This says what those numbers mean, against the gate that replaced
strict monotonicity:

```
CORRECT_SOURCE_GROUNDED_BINDING_LOST = 0
NEW_UNSUPPORTED_BINDING              = 0
UNEXPLAINED_BINDING_CHANGE           = 0
V2_PRODUCER_GAP                      = 0
```

Strict monotonicity was the right gate when the change was a pure regex widening, where
every new binding was additional and no old one had a reason to move. It is the wrong gate
now: A7 both removes false positives and corrects wrong periods, so requiring every old
binding to survive requires V2 to reproduce the legacy defect.

```
python audit_binding_changes.py --out artifacts/evaluation/p1-6-a3-w4a8-changes
```

## Result

```
gained: supported                            519
gained: SOURCE CELLS OUTSIDE THE COLUMN        0
unbound: NOT_A_PERIOD_DECLARATION              9
unbound: REHOMED_TO_ROW_BINDING               18
unbound: ROW_PRODUCER_GAP                      0
changed: SOURCE_GROUNDED_REFINEMENT          645
changed: NEW_BINDING_REGRESSION                0
```

**All four gates are zero.**

## The 27, resolved

### 9 — `NOT_A_PERIOD_DECLARATION`

```
Date: October 31, 2025      Dated: February 27, 2025      Date: January 28, 2026
```

Signature dates. The old predicate bound them as column periods; the repair refuses them,
which is the same family W4-A6 filed as `LEGACY_FALSE_POSITIVE` and the same judgement the
new predicate's negative oracle already encodes. **No recovery wanted.**

The first classifier called these `ROW_PRODUCER_GAP` — it counted any row holding a number
as fact-bearing, and a signature block holds one — which is how a heuristic turns a
signature date into a finding. The row test now mirrors `bind_table`'s own row rule: at
least two value-like cells *and* a cell stating exactly one year.

### 18 — `REHOMED_TO_ROW_BINDING`

```
(MILLIONS, EXCEPT PER SHARE DATA) / Balance, January 1, 2022 / Balance, December 31, 2022 / ...
```

Stockholders' equity and rollforward statements. These columns were bound because the legacy
header selector treats the `Balance, ...` **data rows** as header rows — the W4-A3 geometry —
so a row label was being read as a column period. The repair refuses that, and the period
does not disappear: it moves to the row bindings, which are correct.

```
pfe ord=24395   row bindings  {4: 2022-01-01, 15: 2022-12-31, 26: 2023-12-31, 37: 2024-12-31}
pfe ord=31263   row bindings  {3: 2023-01-01,  6: 2023-12-31,  9: 2024-12-31}
pfe ord=34766   row bindings  {4: 2022-01-01,  7: 2022-12-31,  9: 2023-12-31, 11: 2024-12-31}
```

Every `Balance, ...` row is bound. **Authority moved from a wrong `COLUMN` scope to the
correct `ROW` scope** — a migration, not a loss. The activity rows between them (`Net
income`, `Provision`) stay unbound, and that is the `SECTION` geometry W4-A3 registered as
capability debt.

## The 645, resolved

All of them one transition, and it is the repair working:

```
645   2025 (YEAR) -> 2025-12-31 (DAY) via ADJACENT_YEAR_JOIN
      2024 (YEAR) -> 2024-12-31 (DAY)
      2025 (YEAR) -> 2025-09-30 (DAY)
      ...
```

**"More precise" is not "better"**, so the geometry was read before the label was applied.
JPMorganChase ord=25581:

```
r1  c0-2:'As of or for the year ended December 31,'   c6-8:''   c12-14:''   c18-20:''
r2  c6-8:'2025'                                        c12-14:'2024'  c18-20:'2023'
```

The table declares `As of or for the year ended December 31,` across cols 0-2, and each
column supplies its own year. The month-day cell is 27 characters, so the old predicate
refused it, `pending` was never set, and every column stayed YEAR-only. **The source states
both halves and the column's own cells carry them** — the complete date, not a finer guess.

The same shape, column-local, in Visa: `For the Years Ended September 30, / 2025 /
(in millions, ...)` with all three source cells at the same column.

The provenance check is the one W4-A5 used: every source cell the binding read must be in
the fact's own column, unless the method is a row-scoped declaration. `519 gained:
supported`, `0 gained: SOURCE CELLS OUTSIDE THE COLUMN`.

## State

```
CORRECT_SOURCE_GROUNDED_BINDING_LOST = 0   met
NEW_UNSUPPORTED_BINDING              = 0   met
UNEXPLAINED_BINDING_CHANGE           = 0   met
V2_PRODUCER_GAP                      = 0   met
```

W4-B's entry condition is met on all four. What W4-B still **does not** claim: the 541
`SOURCE_AMBIGUOUS` cells stay ambiguous, the 171 `VALID_BUT_OUT_OF_SCOPE_GEOMETRY` cells
stay unsupported, and the 37 `LEGACY_FALSE_POSITIVE` cells stay deleted. None of those is a
correction, and the Store V2 delta accounting has to keep them apart.
