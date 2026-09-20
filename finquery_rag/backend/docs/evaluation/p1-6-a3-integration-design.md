# P1.6-A3 — Unified Temporal / Period Integration Design

Design for review. **No code changed.** A3's diagnosis is complete: five period-recovery
shapes, one temporal-semantics defect, and an admission rule that is currently welded to
temporal kind. This document freezes the contract before any of it is wired, because the
next change is the first that touches the whole chain at once —

```
parser -> period binding -> temporal kind -> atomic emission -> Store V2 -> 47-slot resolver
```

— and a regression there should be bisectable, not a rewrite.

Every claim below is marked **evidenced** (a traced, measured finding, with the doc that
records it) or **designed** (a contract decision with no measurement behind it yet).

---

## 1. Authority: five layers, and none may substitute for another

```
HTML / grid geometry   owns the source structure        evidenced
PeriodBindingV2        owns WHEN                        evidenced
TemporalKind           owns temporal semantics          evidenced
EmissionAdmission      owns structural completeness     designed
table_role             owns company-level authority     evidenced (sealed)
```

The two separations this line of work has already had to pay for:

```
TemporalKind != EmissionAdmission
Admission    != authority
```

`TemporalKind` must not decide whether a fact is admitted, and admission must not decide
whether a fact may represent the company. The first is what A3-1d exposed — a fully
traceable `YEAR(2025)` dropped for the same reason as junk — and the second is what
A2B-19/20 already fixed and this work must not undo.

---

## 2. `PeriodBindingV2` — frozen schema

```
PeriodBindingV2
  normalized_period    str | None      2025, 2025-12-31
  granularity          DAY | YEAR | UNRESOLVED
  temporal_kind        point | duration | comparison | segment | bucket
                       | category | non_temporal | YEAR | UNRESOLVED
  source_cells[]       [(row, col)]    the actual HTML cells this came from
  target_scope         COLUMN | ROW | CELL_GROUP
  method               DIRECT_HEADER | ADJACENT_YEAR_JOIN
                       | HEADER_ROW_SELECTION_EXTEND
                       | INLINE_PERIOD_DATA_ROW | YEAR_ONLY_PERIOD
  status               RESOLVED | PARTIAL | UNRESOLVED
```

**`status` is not decoration.** Coca-Cola's equity statement is

```
normalized_period = 2025    granularity = YEAR
temporal_kind     = UNRESOLVED        status = PARTIAL
```

Period *identity* is resolved; the temporal *shape* was not stated by the source. Without
`status` that reads the same as a binding that failed, and the next reader drops it —
which is the defect A3-1d was written to prevent. `PARTIAL` means: usable where the
granularity is compatible, never as a finer claim.

---

## 3. Method precedence — evidence locality, not a cascade

```
1  INLINE_PERIOD_DATA_ROW        the row states its own period        ROW
2  DIRECT_HEADER                 a complete source header            COLUMN / CELL_GROUP
3  HEADER_ROW_SELECTION_EXTEND   recovers a real header row, then
                                 re-runs DIRECT_HEADER               COLUMN
4  ADJACENT_YEAR_JOIN            month-day + adjacent bare years     COLUMN
5  YEAR_ONLY_PERIOD              the source states a year only       CELL_GROUP / YEAR
```

This is **not** "later ones override earlier ones". The rule is **the more local and more
direct the source evidence, the higher it ranks** — a row stating its own period may not be
overridden by a column header above it. Pfizer's dated balance rows sit under a column
period and the row wins. **Evidenced** for the scope of each method (A3-1a/1b2/1c/1d);
**designed** as an ordering, since no table in the corpus exercises 1 against 2.

---

## 4. Conflict contract

```
one target cell, two sources:
    row-local period   2024-12-31
    column period      2025-12-31
        -> CONFLICT
        -> do not choose
        -> the fact enters with no certain period (status UNRESOLVED)
```

Precedence resolves *which method produced a binding*; it does **not** license silently
picking a winner when two bindings disagree about the same cell. Merge only when the
source evidence agrees.

**Designed, not observed.** No conflict of this shape was found in the corpus — which is a
statement about the corpus, not about the contract. It is written now because a silent
tie-break is exactly the class of defect this project keeps finding.

---

## 5. `TemporalKind` input domain — frozen

```
allowed
  raw source header cells for the logical column
  explicit axis / group metadata
  PeriodBindingV2

forbidden
  ordinary row labels
  cell prose
  the assembled header path
```

and `rating | range | grade | tier` are matched as **whole tokens**.

**Evidenced.** A3-3b: the collision fires on genuinely raw column evidence
(`['Operating Leases'] → bucket` on `rating`), so token-safety is contract, not defence in
depth; and raw scope removes the row-prose dependency for every column that emits.

**A3-3b also showed the boundary is imperfect and must be stated rather than assumed:**
NVIDIA's columns 0-2 keep `Cash flows from operating activities:` in a *raw* header cell,
because the label prose sits inside the grid's header row. **This is residual R1 below**,
not something A3 fixes.

This section should be enforced by a test, not by a comment: an architecture guard that
fails if a row label or `cell_text` reaches the classifier.

---

## 6. `EmissionAdmission` — decoupled

```
old   temporal_kind not in {point, duration, comparison}  ->  drop
new   the fact is admitted when it has sufficient, traceable PERIOD IDENTITY
```

```
DAY  + point/duration          admissible
YEAR + temporal_kind UNRESOLVED  admissible          <- A3-1d's case
no period identity             per fact type; not a blanket drop
```

A `YEAR(2025)` fact belongs in Store V2. What it may answer is the resolver's question, not
admission's. **Designed**, and it is the change that most needs to be visible in the diff:
it does not add recall by loosening a filter, it stops a filter from doing a job it never
owned.

---

## 7. Period compatibility — a pure function, not string equality

```
periods_are_compatible(source_period, requested_period)

DAY  2025-12-31   <->  DAY   2025-12-31          yes
YEAR 2025         <->  FY    2025                yes
YEAR 2025         <->  DAY   2025-12-31          NO -- a year does not pin a day
DAY  2025-12-31   <->  FY    2025                yes
UNRESOLVED        <->  anything                  no
```

It lives in one function, outside the resolver. **Designed.** Quarter and month granularity
are explicitly out of scope until a filing needs them.

---

## 8. Commit plan — unified architecture, bisectable history

```
A3-W1  PeriodBindingV2 contract + provenance fields, carried through unchanged
       no behaviour change; the schema exists and is populated where known
A3-W2  period reconstruction methods, dual output (old binding still used)
A3-W3  TemporalKind raw column-local evidence + token-safe patterns
A3-W4  EmissionAdmission decoupled from TemporalKind
A3-W5  provenance persisted into AtomicFact and Store V2
       normalized_period, period_granularity, period_target_scope,
       period_binding_method, period_source_cells,
       temporal_kind, temporal_kind_source_cells, temporal_kind_method
A3-W6  enable the new path for A3 evaluation
```

W2-W4 emit both the old and the new answer until W6, so each step can be scored against the
frozen baseline on its own.

---

## 9. Baseline and gates

```
sealed baseline (tag p1.6-a-table-authority-seal, commit 2e492e8c)
  47 slots   RESOLVED 37   NO_COMPANY_LEVEL_FACT 5   NO_FACT 5
  wrong_scope 0   wrong_metric 0   value_mismatch 0   dangerous_authority 0
```

```
hard gates, must all hold
  the 37 sealed resolutions       0 regression
  wrong_scope / wrong_metric / value_mismatch / dangerous_authority   0
  the 9 missing primary statements  no unexplained zero-record table
  period provenance integrity     100% -- every admitted fact can name
                                   source_cells, method, target_scope,
                                   granularity, and why its temporal_kind
                                   has the value it has
```

Capability (how far `37` moves) is **not** pre-specified. A change that trades one of the
hard gates for recall is a failure regardless of the number.

**Provenance integrity is new and is a gate, not a metric.** Without it A3 would replace
"the structure was dropped" with "the structure is opaque", which is the same loss one layer
down.

---

## 10. Exclusions — registered, not fixed here

```
metric identity / the 5 NO_FACT slots
label-column prose inside the grid's header row        residual R1
the one-row offset between adjacent columns            residual R2
cross-authoring-platform validation
production switch
retrieval / index rebuild
```

**R1 and R2 are the same over-segmentation, seen three times (A3-1d, A3-3b) and diagnosed
zero times.** They are recorded so that a later reader does not rediscover them as new, and
so that nobody repairs them inside this wiring where their effect would be indistinguishable
from the intended change.

---

## What this design does not claim

The precedence order and the conflict contract are **designed against a corpus that does not
exercise them**. The temporal fix is evidenced only on five tables. The methods are
evidenced on nine. Nothing here has been tested on a filing outside the eight, and the
authoring platform is Workiva in every case.

Those limits carry into A3-W6 unchanged; they are the same limits the seal already records.
