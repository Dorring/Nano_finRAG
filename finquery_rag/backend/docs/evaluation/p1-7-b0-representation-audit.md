# B0 — fact-representation feasibility audit (Go / No-Go)

Nothing was changed to produce this: no store, no gold, no `src/`. It is an
audit plus a ceiling calculation, and it is meant to be read as a decision.

## Verdict: **NO-GO** for the 60% coverage target

```
                              releases    coverage   vs 57 needed
today                            46         48.4%
+ recoverable from the record     +3.2      51.8%     SHORT
+ both recoverable classes        +7.7      56.5%     SHORT
                                    ---
needed for 60%                   +11       60.0%
```

Even counting **every** case the migration could plausibly touch, and converting
them at the direct-fact rate rather than a blended one, the ceiling is **56.5%**.
A fact-representation migration cannot reach 60% on this benchmark.

The migration also does not own most of what it appeared to. Of the 36 blocked
comparable answerable cases, **five need no new field at all** -- the relation
that identifies the answer is already arithmetic in the record -- so they are a
Binder rule, not a representation change. B's own contribution is at most the
seven conditional cases: **+4.5 releases, 53.2%.**

## Q1 — how many are really representation loss

36 blocked comparable answerable cases, each attributed. **UNCLASSIFIED = 0.**

```
BINDER_SEMANTIC                        14
RECOVERABLE_IF_SOURCE_SELECTS_ONE       7
SOURCE_AMBIGUOUS                        7
RECOVERABLE_FROM_THE_RECORD             5
NOT_ACTUALLY_BLOCKED_BY_B               3
```

The classifier does not judge. Every fact carries its own row in `content`, so
at a shared `(entity, metric, period)`:

```
all facts share one content  ->  the values are COLUMNS of one row
the content strings differ   ->  the values are DIFFERENT ROWS
```

- **SOURCE_AMBIGUOUS (7).** All competing values sit in one row, so they are
  columns. `column_identity` would let the system *describe* the row; it would
  not tell it which column the answer is, because the question asks for the row.
  `Colette M. Kress` is four numeric columns and a year column; the question is
  "how much did NVIDIA report for Colette M. Kress".
- **RECOVERABLE_FROM_THE_RECORD (5).** Different rows, and one is the arithmetic
  sum of the others -- Microsoft's FY2025 `Cost of revenue` is 87,831 =
  22,422 + 40,171 + 25,238, confirmed by the prior-year column. Nothing outside
  the store is needed to know which row is the total.
- **RECOVERABLE_IF_SOURCE_SELECTS_ONE (7).** Different rows with no arithmetic
  relation. Which row is meant is settled by the statement or table it sits in,
  and that is in the source rather than in the record. Counted as conditional,
  never as a gain.
- **BINDER_SEMANTIC (14).** Not a representation question at all: the Binder
  refuses at the coordinate, or the evidence firewall rejects the binding.
- **NOT_ACTUALLY_BLOCKED_BY_B (3).** The symbol count at the point of failure was
  zero; see the stage breakdown.

## Q2 — the minimal field set, if it were pursued

Derived from the seven conditional cases, not from an ontology. The smallest set
that explains them is **two fields**, and no more:

```
statement_or_table_scope   which statement or table the row sits in
row_role                   total | segment | member, where the source says so
```

The five arithmetic cases need **neither**. A `table_id`/`logical_table_id`
split is a *separate* concern -- see the identity finding below -- and
`column_identity` explains none of the coverage failures, because a column the
question does not name is not selectable.

## Q3 — can it be rebuilt deterministically

- The five arithmetic cases: **already determinable today**, from values the
  store holds. No rebuild.
- The seven conditional: **unknown, and this audit cannot settle it.** It needs
  the source HTML table structure to say whether a row's enclosing statement or
  table is recoverable. That check is cheap and should precede any migration.
- The 43% identity duplication (below): **deterministic**, by construction.

## Q4 — the ceiling, computed rather than assumed

Conversion is measured, not assumed, and it is measured **conditionally** -- a
blended rate would flatter this case:

```
                    gate-passed   released   conversion
DIRECT_FACT              25          16         0.640
CALCULATION              28          23         0.821
```

The recoverable cases are direct-fact, and direct-fact converts *worse* than
calculation, because identifying one fact is exactly where a coordinate that
does not identify one value bites. So the conditional rate is **0.640**, not the
blended 0.736:

```
                              releases    coverage   vs 57 needed
today                            46         48.4%
+ recoverable from the record     +3.2      51.8%     SHORT
+ both recoverable classes        +7.7      56.5%     SHORT
                                    ---
needed for 60%                   +11       60.0%
```

Even the optimistic column -- every conditional case recovered *and* converting
at the direct-fact rate -- reaches **56.5%**. The verdict is No-Go on the
generous bound, and no assumption left in it is favourable.

## What B would invalidate

| | classification |
|---|---|
| runtime fact store (20,394 records; 11,657 logical facts) | **must rebuild** |
| retrieval index built over candidate keys | **must rebuild** |
| gold `v2fact:` ids resolving into the store | **must regenerate** |
| `benchmark-version.json` gold hash (currently `3d2a0c5b`) | **must regenerate** |
| A3 provenance seal, resolver behaviour, alias maps | **must rerun** |
| questions, plan fixtures | **expected byte-identical** |
| retrieval recall rows | **expected semantically equivalent** |

Two of these matter more than the rest. The freeze pins **questions, gold and
plan fixtures** -- not the fact store -- so rebuilding the store does not by
itself break it. But the gold names `v2fact:` candidate keys, and new keys mean
the gold stops resolving, which means regenerating the gold, which *does* break
the pin. A "store migration" is therefore a benchmark-migration as well, and
should be scoped as one.

## The finding that is worth acting on separately

The store holds **20,394 records for 11,657 logical facts** -- 1,354 logical
facts carry several physical keys, and there are **8,737 physical keys above the
logical count, 43% of the store**. That is `physical fragment identity ≠ logical
fact identity`, exactly as suspected, and it is what caps citation precision:

```
citation precision, strict gold ids        60/72   83.3%
the same quantity via another fragment      6/72    8.3%
genuinely different quantity                6/72    8.3%
                                           ----
precision by logical fact identity         66/72   91.7%   >= 90%
```

So the citation-precision target is reachable, and it is reachable by
**deduplicating to a logical fact identity** -- a contained identity change, not
a representation migration. It also does not require an answer to "which row is
this", which is the question the migration cannot answer from the source.

## Recommendation

1. **Do not start the representation migration for coverage.** The ceiling is
   57.7% optimistic and 53.7% on its own contribution, against 60% needed.
2. **Before any migration, run the cheap Q3 check** against the HTML corpus: can
   a row's statement or table be recovered deterministically? If no, the seven
   conditional cases are zero and the migration is definitively dead.
3. **The five arithmetic cases are worth doing on their own**, as a Binder rule
   with a guard, because they need no new data. They are option (A) in scope,
   bounded to five cases, and should be judged on `incorrect release`, not on
   coverage.
4. **The logical-fact-identity change is worth doing on its own merits.** It is
   the only lever here that closes one of the stated targets, and it fails
   toward recall rather than correctness.
