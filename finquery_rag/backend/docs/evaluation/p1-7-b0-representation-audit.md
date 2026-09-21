# B0 — fact-representation feasibility audit (Go / No-Go)

Nothing was changed to produce this: no store, no gold, no `src/`. It is an
audit plus a ceiling calculation, and it is meant to be read as a decision.

## Verdict: **GO**, and it flipped on a base that moved

```
                              releases    coverage   vs 57 needed
baseline at audit time           46         48.4%
recoverable, on that base        +7.7      56.5%     SHORT  <- the first verdict
                                    ---
current                          50         52.6%
recoverable, on the current base +7.7      60.7%     MET
```

The first version of this audit was a No-Go. It was computed from a base of 46
releases, and this round then recovered four more -- a slot with no entity had
been read as *the source has no such row*, and a `BOUND` binding that named its
own gaps was being discarded whole. The recovery classes were unchanged at 12;
the base under them moved.

So the Go condition holds on the measurement that matters: **50 -> 57.7 releases,
60.7%, at the direct-fact conversion rate** (0.640, the rate the recoverable
cases actually convert at). At the blended rate it is 61.9%.

**The estimate is not comfortable.** 57.7 against a threshold of 57 is a margin
of 0.7 releases, and it rests on twelve cases all converting at the direct-fact
rate. Two things could take it under: a conversion rate below 0.60, or any of the
eight conditional cases turning out unrecoverable once the source is inspected.
Both are cheap to check and neither has been.

## Q1 — how many are really representation loss

33 blocked comparable answerable cases, each attributed. **UNCLASSIFIED = 0.**

```
SOURCE_AMBIGUOUS                        9
BINDER_SEMANTIC                         9
RECOVERABLE_IF_SOURCE_SELECTS_ONE       8
RECOVERABLE_FROM_THE_RECORD             4
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

Answered as far as the record can answer it, and it narrows the question to one
that only the source can settle.

Measured over the 21 blocked coordinates that hold more than one value:

```
SEVERAL_ROWS       12   genuinely different rows
ONE_ROW_COLUMNS     9   one row's cells, read as if they were quantities
```

The two are different problems and only one of them is a migration.

**The nine column cases are not recoverable at all.** Visa's FY2025 `U.S.
Treasury securities` is not three competing values: `2,101` and `15` and the
rest carry the *same* `row_id`, the same `row_index` and the same `row_bbox`.
They are cells of one row, and the store emitted a fact per cell. Adding
`column_identity` would let the system describe that row; it would not tell it
which column the answer is, because the question says "the reported U.S. Treasury
securities" and names no column. These are `SOURCE_AMBIGUOUS` and stay blocked.

**The twelve row cases are the whole Go margin**, and the record already
identifies the row: `row_id`, `row_index` and `row_bbox` are all present. What is
missing is not *which row* but *what the row means* -- the caption it sits under.
Coca-Cola has two `Operating income` rows on different pages:

```
Operating Income | 13,762 |  9,992 | 11,311
Operating income | 13,426 | 12,536 | 11,868
```

Both are real rows with real `row_id`s. Nothing in the record says the second is
the consolidated statement's and the first a segment table's, and that is what
decides the answer.

So the migration's Go verdict rests on exactly one unverified fact: **does the
source table's caption attach to the row in a form that survives?** If it does,
the twelve are recoverable and the ceiling is 60.7%. If it does not, they are
zero and the ceiling is 55.4%. This audit cannot see the source, and that check
is cheap.

Of the earlier list, what B0 called `RECOVERABLE_FROM_THE_RECORD` are the four
where the arithmetic settles it without any of this.

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
blended 0.736.

```
                              releases    coverage   vs 57 needed
current (after this round)       50         52.6%
+ recoverable from the record     +2.6      55.4%     SHORT
+ both recoverable classes        +7.7      60.7%     MET
                                    ---
needed for 60%                   +7        60.0%
```

The strict column -- only the four the record alone can settle -- still misses.
The full column clears, by 0.7 of a release.

## B1 — the source check, and it answers yes

B0's Go verdict rested on one unverified fact. It is now verified against the
filing the store was built from: `raw_sec_html/KO/SEC_21344_000162828026010047/
primary.html`, whose metadata gives `source_url` ending `ko-20251231.htm` and
`period_hits: December 31, 2025` -- Coca-Cola's FY2025 10-K.

`Operating income` occurs six times. Two of them are the rows the store holds,
and the source separates them structurally:

```
occurrence 3
  Operating income $ 13,426 $ 12,536 $ 11,868 Consolidated net income $ 9,355 $ 8,439 ...
  -> a two-row window of the consolidated statement of income

occurrence 4
  Operating income (loss) $ 4,298 $ 3,742 $ 5,070 $ 2,042 $ 426 $ 15,578 $ (1,816) $ -- $ 13,762
  -> a segment table, with EMEA / Latin America / ... / Total / Corporate /
     Eliminations / Consolidated as its column headers
```

The distinguishing information is not a caption that happens to be nearby, it is
the **table's own structure** -- the column set. A consolidated statement has
`2025 | 2024 | 2023`; the segment note has one column per reporting segment plus
Total, Corporate, Eliminations and a Consolidated total. Both are deterministic
and neither is a guess.

**That second sentence is wrong, and the sidecar build is what showed it.** The
table that actually holds Coca-Cola's segment `Operating income` is not the
reconciliation table. It is this:

```
0  Year Ended December 31 | 2025 | 2024 | 2023
1  Net Operating Revenues  | $ | 47,941 | $ | 47,061 | $ | 45,754
2  Cost of goods sold      | 18,397 | 18,324 | 18,520
...
6  Operating Income        | 13,762 | 9,992 | 11,311
```

Its column structure is *identical* to the consolidated statement's -- both are
`Year Ended December 31 | 2025 | 2024 | 2023`. The column set separates nothing
here. What separates them is the **section heading above the table** (the
segment's name), which is prose. The `EMEA | Latin America | ... | Total |
Corporate | Eliminations` row is a different table again, the note's
reconciliation.

So B1's answer stands -- the source does carry the discriminator -- but it is a
heading, not a structure, and recovering it means identifying the nearest
heading reliably. The sidecar's first version read 4,000 characters of prose and
matched keywords loosely, which is why it returned CONSOLIDATED_STATEMENT for
both; the correction is a nearest-heading reader, not a column reader.

So the eight `RECOVERABLE_IF_SOURCE_SELECTS_ONE` cases are recoverable in
principle, and the Go verdict stands on evidence rather than on an estimate:
**50 -> 57.7 releases, 60.7%.**

Two things this does *not* establish. It is one filing and one label, so it
shows the source carries the information rather than that all eight carry it in
the same shape. And it does not make the migration smaller: recovering a table's
column structure means re-extracting from the corpus, which is the data project
the invalidation table above describes.

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

1. **The Go condition holds, but on a 0.7-release margin.** Before unfreezing
   anything, close the two cheap uncertainties: (a) the Q3 source check -- can a
   row's enclosing statement or table be recovered deterministically from the
   HTML corpus? If not, the eight conditional cases are zero and the whole path
   is dead; (b) re-measure the conditional conversion on the recoverable cases
   specifically rather than borrowing the direct-fact rate.
2. **The four arithmetic cases need no unfreezing at all.** They are
   recoverable from values already in the record, so they are a Binder rule with
   a guard, and they should be judged on `incorrect release`, not coverage. They
   are worth +2.6 releases and 55.4% -- not the target, but not nothing.
3. **The logical-fact-identity change is done** and closed the citation
   precision target; it needed no migration either.
4. If (1) comes back positive, the migration is a benchmark migration as well as
   a store migration -- see the invalidation table above -- and should be scoped
   as one, not as a data refresh.
