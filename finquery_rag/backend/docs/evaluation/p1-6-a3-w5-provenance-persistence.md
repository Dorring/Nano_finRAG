# P1.6-A3-W5 — provenance persistence, and the 1,375 facts that had a kind but no evidence

W5 persists **why** a period identity is believed. W4-B2 saved *what* it is
(`normalized_period`, `status`, `granularity`); this saves the method that produced it, the
scope it applies to, and the cells it was read from.

It is green on all four checks, and — the harder half — it moved nothing:

```
store        26311 records            unchanged from W4-B2
47-slot      37 / 5 / 5               unchanged, all five safety gates 0
removals     691 / 4188 / 2173        exact, classes still kept apart
additions    3368 PARTIAL/YEAR, 3018 RESOLVED/DAY   exact
period       2593 moved               period_end moved on 0
```

## What was added, and the one name that had to be split

```
period_binding_method          ADJACENT_YEAR_JOIN      13241
                               DIRECT_HEADER            6026
                               YEAR_ONLY_PERIOD         5008
                               INLINE_PERIOD_DATA_ROW   1375
                               HEADER_ROW_SELECTION_EXTEND 661
period_target_scope            COLUMN 19928 | CELL_GROUP 5008 | ROW 1375
period_source_cells            one per source cell, 1..22 per fact, none empty
period_conflict_candidates     always empty on a stored fact, by construction
temporal_kind                  point 17514 | UNKNOWN 5008 | duration 3789
temporal_kind_method
temporal_kind_source_cells
temporal_kind_matched_text
legacy_temporal_kind
```

`temporal_kind` needed a decision rather than a passthrough. `AtomicFact.temporal_kind` is
the **legacy axis** kind, and `semantic_equivalence` groups canonical facts on it, so it
cannot be renamed. The A3 kind is a different value that can legitimately disagree with it,
and it is the one whose method and source cells are recorded here. So the fact carries it as
`binding_temporal_kind` and the store maps it onto its own `temporal_kind` — which the store
never had before, so the bare name was free there — keeping the axis kind beside it as
`legacy_temporal_kind` rather than dropping it. One name for two meanings is how the
disagreement would get lost.

## Acceptance

```
1. provenance integrity
   facts admitted on a binding                          26311
   carrying method, scope, cells, granularity, period   26311
   conflict candidates on a stored fact                     0

2. resolvability
   source cells checked                                 77883
   that fail to resolve to a real filing/table/row/column   0

3. round-trip   producer -> AtomicFact -> record -> reloaded
   ok  ko_fy2025    YEAR_ONLY_PERIOD            CELL_GROUP   2 source cells
   ok  pfe_fy2024   INLINE_PERIOD_DATA_ROW      ROW          1 source cell
   ok  v_fy2025     HEADER_ROW_SELECTION_EXTEND COLUMN       7 source cells
   ok  jpm_fy2025   ADJACENT_YEAR_JOIN          COLUMN       2 source cells

4. the store is otherwise untouched
   records 26311   previous 26311
   fields that differ: the nine W5 additions, and nothing else
```

Resolvability is checked by **re-parsing each filing and confirming the table exists and the
cell is inside its grid**, not by testing the fields are non-empty — a tuple of
plausible-looking numbers would satisfy the weaker check. The four round-trip cases are the
ones W3 and W4 spent their time on, and they are selected by the evidence they must carry
rather than by a cell coordinate, so a re-parse that moves a row cannot turn the check into a
no-op.

`period_conflict_candidates` is empty on every stored fact **because a conflict is withheld
before the fact is built**, not because the field is unused. The structure is persisted so
that a disagreement would stay legible as a disagreement; today the candidates live on the
admission decision, which is where a fact that does not exist should keep them.

## What the acceptance caught: 1,375 facts asserting a kind with no evidence

```
temporal_kind != UNKNOWN                     21303
  with EMPTY temporal_kind_source_cells       1375   <- all of them INLINE_PERIOD_DATA_ROW
```

The row-binding path built its `TemporalKindEvidence` without `source_cells`
(`period_binding_shadow.py`, the `INLINE_PERIOD_DATA_ROW` branch). The binding itself named
the cell it read; the kind attached to it named nothing. So a fact could say `point` — a
claim about the *shape* of its time — with no way to check it, which is exactly the state W5
exists to end.

The fix attaches the same `SourceCell` the binding already carried. It is provenance only:
nothing consults `temporal.source_cells` to decide anything — `decide_emission_admission`
takes the kind as a separate argument and never reads the evidence's cells — so the record
count, the 47 slots and every admission are unchanged, which the rebuild confirmed at 26,311.

The pre-fix store is kept at `p1-6-a3-w5-store-prefix/` rather than overwritten, so the
failure this phase was built to detect stays on disk as an artifact rather than as a
sentence in a report.

## A defect found and deliberately not fixed

`trusted_v2_canonical_fact_store.py` has **two `normalized_period` keys in one dict literal**
— the W4-B2 line and the legacy fallback below it. The second wins, so the first is dead
code, and the surviving value is `normalized_period or period_end or period`: a sensible
chain, reached by accident.

They differ only when the payload's `normalized_period` is empty while `period_end` or
`period` is set. Removing either one is therefore a **behaviour change in a corner case**,
and W5's whole standard is that it changes nothing. It is registered here with the condition
that would distinguish the two, so a later phase can decide it with the facts in hand rather
than as a side effect of a persistence migration.

## Registered debt, unchanged

* **SECTION scope / SECTION_BRACKET** for row-major equity statements (`VALID_BUT_OUT_OF_SCOPE_GEOMETRY`).
* **The resolver's period matching** does not consume `YEAR(2025)` or granularity. 3,368 facts wait on it.
* **Row-label numeric extraction** (`Balance at January 1` → `1`).

## Verdict

```
W5  full provenance persistence   DONE
```

W6 is the final enablement: funnel, Store, 47-slot and seal — the point at which this round
of P1.6-A3 closes.
