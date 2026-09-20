# P1.6-A2B-2 — the A→B hop holds; C is blocked by a parser duplication

The propagation the plan asked for, measured on JPMorganChase.

## A → B: the column is carried

```
A  parsed cells 123,120   cell anchors 13,682
B  atomic facts 1,543     with column_header 1,543   (100%)     with anchors 896 (58%)
```

**Every atomic fact carries `column_header`.** That is the dimension whose absence made
`(JPMorganChase, Net income, FY2025)` hold `4,520` and `57,048` at once, and it now travels
from the parsed cell, through `emit_atomic_facts`, onto the fact.

Carried as an explicit field rather than inside `source_traceback` — a missing dict key is
silent, a missing field is a schema error.

Two smaller numbers worth stating plainly rather than burying:

- **896 of 1,543** facts carry `ixbrl_anchors`, not all. Anchors are recorded on cells whose
  `<td>` contains a tag; cells whose value is presented without an inline tag have none
  even though the cell is real. That is the honest coverage of that field, and it is
  separate from the 100% on `column_header`.
- **1,543 facts** is far below the store's ~7,322 for this document. The re-parse is not
  reproducing the store's population, which nothing has yet explained.

## Why B could not be reached before

`build_semantic_corpus` reads `document["tables"]`, iterates it, and emits one atomic fact
per numeric cell. **The parser computed `tables` and then dropped them** — the persisted
`norm` carried `blocks` and `ixbrl_facts` only.

Measured consequence, not inferred:

```
building from the persisted corpus
  atomic_facts    0
  logical_tables  0
  store records   0
```

So the corpus on disk **cannot reproduce the benchmark store at all**. That closes the
question left open when the store rebuild was scoped: the parsed documents behind the
store are not merely absent, they are *different* — the version that built the store had
`tables`, and the version on disk does not.

## C: blocked, and by something pre-existing

```
ValueError: duplicate canonical candidate key
```

```
TABLE blocks 679   distinct table_ids 676   duplicated 3
```

**Three tables are emitted as two blocks each.** The same table produces the same cells
twice, so the same cell yields two identical atomic facts, and the store builder's
duplicate-key guard fires. The duplication is in `make_blocks`, not in anything added here,
and it would fire on any build that had tables — which is why the corpus on disk, which has
none, never hit it.

That is a real defect to fix before C can be measured, and it should be fixed at the source
rather than by relaxing the guard: a duplicate block means a table is being read twice.

## What is and is not established

**Established:** the column dimension survives from the parsed cell to the atomic fact,
completely. The hop that the whole plan depended on — A→B — works.

**Not established:** that it reaches the store (C), that it reaches a view or an index (D),
or that the re-parse reproduces the corpus. The fact-count gap between 1,543 and ~7,322 is
unexplained and is the next thing to look at, because a rebuild that produces a fifth of
the facts is not a rebuild.

## Status

Four files changed: the parser persists `tables`, `AtomicFact` gains `column_header`,
`row_label` and `ixbrl_anchors`, and the store builder carries them into its records. No
fixture, denominator, store, view, index or production behaviour changed. The existing
store is untouched and no document has been rewritten on disk.
