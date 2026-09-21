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
  granularity          DAY | YEAR | UNKNOWN
  temporal_kind        point | duration | comparison | segment | bucket
                       | category | non_temporal | YEAR | UNKNOWN
  source_cells[]       [(row, col)]    the actual HTML cells this came from
  target_scope         COLUMN | ROW | CELL_GROUP
  method               DIRECT_HEADER | ADJACENT_YEAR_JOIN
                       | HEADER_ROW_SELECTION_EXTEND
                       | INLINE_PERIOD_DATA_ROW | YEAR_ONLY_PERIOD
  status               PeriodBindingStatus
  conflict_candidates[]  the competing bindings, when status is CONFLICT
```

```
PeriodBindingStatus
  RESOLVED      identity and shape both stated by the source
  PARTIAL       identity resolved, shape not stated
  CONFLICT      two sources disagree about this cell
  UNRESOLVED    no source evidence found
```

**`status` is not decoration, and it is a separate enum from `TemporalKind` on purpose.**
Coca-Cola's equity statement is

```
status            = PARTIAL
normalized_period = 2025      granularity = YEAR
temporal_kind     = UNKNOWN
```

Period *identity* is resolved; the temporal *shape* was not stated. Without `status` that
reads the same as a binding that failed and the next reader drops it — the defect A3-1d was
written to prevent. `PARTIAL` means: usable where the granularity is compatible, never as a
finer claim.

`TemporalKind` therefore says **`UNKNOWN`, not `UNRESOLVED`.** A binding is `UNRESOLVED`
when nothing was found; a temporal kind is `UNKNOWN` when the source did not state a shape.
The words were one word apart and would have been read as the same condition.

---

## 3. Period evidence: two classes, not a precedence order

```
A. DIRECT DECLARATION
   INLINE_PERIOD_DATA_ROW      the target row itself states WHEN

B. INHERITED CONTEXT
   DIRECT_HEADER
   HEADER_ROW_SELECTION_EXTEND
   ADJACENT_YEAR_JOIN
   YEAR_ONLY_PERIOD            inherited from a header or group above
```

```
a legal DIRECT DECLARATION on the target shadows inherited context
    -- scope semantics, not a tie-break
two DIRECT declarations that disagree        -> CONFLICT
two inherited bindings that disagree         -> CONFLICT
```

**There is no total order over methods.** An earlier draft ranked them
`INLINE > DIRECT > EXTEND > ADJACENT > YEAR` and that was wrong twice over: it contradicted
the conflict rule (if ROW always wins, a row/column disagreement could never be a conflict),
and it would have turned `method` — which is provenance, a record of where a value came
from — into a business ranking. A method enum must never quietly become an authority order.

So Pfizer's `Balance, December 31, 2022` legitimately shadows the column context above it
because the row *declares* its period, not because its method outranks the column's. The
distinction is why that case is not a conflict while a genuine two-source disagreement is.

**Evidenced** for the scope of each method (A3-1a/1b2/1c/1d). **Designed** for the two-class
rule, since no table in the corpus pits a declaration against an inherited binding on the
same cell.

---

## 4. Conflict contract

```
one target cell, two sources of the same class:
    inherited  2025-12-31
    inherited  2024-12-31
        -> status = CONFLICT
        -> normalized_period carries no single certain value
        -> every competing source evidence is retained in conflict_candidates
```

**A conflict is a first-class binding outcome. It is not an admission decision.**
`PeriodBindingV2` says *I do not know which period is correct*; whether that is still worth
storing, and in what form, is `EmissionAdmission`'s to say and is decided in **A3-W4**.
Deciding it here would re-weld the two layers this design exists to separate — and it would
do so in the layer that has no standing to decide it.

Precedence resolves *which method produced a binding*; it does **not** license silently
picking a winner when two bindings disagree about the same cell. Merge only when the source
evidence agrees.

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

**What is frozen here is the minimum the evidence supports, and no more:**

```
RESOLVED period     satisfies period completeness
PARTIAL YEAR period must not be rejected merely because temporal_kind is UNKNOWN
```

That is enough to retire the welded rule, and it is all that is retired.

**Where `CONFLICT` and `UNRESOLVED` go — the main store, a side artifact, or a rejection
ledger — is decided in A3-W4 by implementation and measurement, not here.** Designing it
now would settle a question no evidence has been gathered for, and would leave `W4` with
nothing to measure.

**Designed**, and it is the change that most needs to be visible in the diff: it does not
add recall by loosening a filter, it stops a filter from doing a job it never owned.

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
       period_binding_method, period_source_cells, period_binding_status,
       temporal_kind, temporal_kind_source_cells, temporal_kind_method
A3-W6  enable the new path for A3 evaluation
```

W2-W4 emit both the old and the new answer until W6, so each step can be scored against the
frozen baseline on its own.

**W1's acceptance is hard and mechanical**, precisely because a contract commit that moves
behaviour means introduction and migration were mixed:

```
Store V2 records            26,977        unchanged
47 slots                   37 / 5 / 5     unchanged
sealed safety gates         0 0 0 0       unchanged
```

W1 may introduce types, enums, provenance containers and their serialisation round-trip. It
may not change period selection, emission count, store count or any slot result.

**W4 is not to be pulled forward.** The delta between the new binding and the old path, and
between the new temporal kind and the old, must be measured before anything changes what is
admitted — otherwise the admission change and the binding change are inseparable, and a
47-slot regression cannot be attributed to either.

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
  period provenance integrity     100%
```

**The provenance-integrity denominator, written down so it cannot drift:**

> every `AtomicFact` that obtained its period through a new `PeriodBindingV2` path with
> `status in {RESOLVED, PARTIAL}` **and** was admitted by `EmissionAdmission`

must carry all of

```
source_cells   method   target_scope   granularity   status
```

and, whenever `temporal_kind != UNKNOWN`, must also be able to trace

```
temporal_kind_source_cells / structural evidence
temporal_kind_method
```

**`CONFLICT` bindings are excluded from this denominator by construction** — they retain
all competing candidates rather than a single resolved provenance, so they cannot satisfy a
"one traceable provenance" test, and counting them would make the gate either fail
trivially or get quietly weakened until it passed.

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
