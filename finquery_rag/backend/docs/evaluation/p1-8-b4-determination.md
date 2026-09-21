# P1.8-B4 — DETERMINATION: the coverage target under this phase's constraints

This consolidates B1–B3. Read this one; the others are the working.

## The determination

> **Release Coverage ≥ 60% is not reachable in P1.8's current constraints.**
> Not by retrieval optimisation, and not by any change the phase permits.
> The binding constraint is `不修改` on Store and Gate, and what stands behind it
> is a data-resolution problem in the fact store.

This is a measurement, not a judgement, and the arithmetic is below.

---

## 1. The baseline, re-derived

Your brief quoted 50/95 = 52.6%. That figure is the **gate-bypass** arm. From
the P1.7 frozen per-case artifacts (`/tmp/g-*/` on 4090-qh, dated 2026-09-21):

```
BASE           /tmp/g-base/A_pinned_production-cases.jsonl    20/95   21.1%
GATE_ON        /tmp/g-new/A_pinned_production-cases.jsonl     46/95   48.4%   <- the config
GATE_BYPASS    /tmp/g-bypass/B_gate_bypass-cases.jsonl        50/95   52.6%
```

`p1-7-release-coverage.md` says the same in its own words: "48.4% is the
pinned-plan release rate." The distance to 60% is **11 cases**, not 7.

## 2. The 49 unreleased answerable cases

```
gate ontology does not know the metric (SEMANTIC_MISMATCH, PLAN)   22
the guard refuses the gold's own coordinate (conflicting values)   27
```

and split by whether the operand guard would admit the gold's coordinate:

```
                                    conflicting   passes the guard
arithmetic_calculation                   9               3
factual_lookup                          18               6
cross_entity_comparison                 13               0   (blocked earlier)
```

**9 cases pass the guard.** Every one of those 9 is gate-blocked, so the gate
must also admit its metric — which is a Gate change.

## 3. Every lever, bounded

```
                                              cases   total    coverage
Gate-ON baseline                               --     46/95     48.4%
+ gate ontology admits the 4 safe metrics      +4     50/95     52.6%
+ canonical-candidate seam                     +3     53/95     55.8%
+ ALL 9 guard-passing gate cases               +9     55/95     57.9%
                                              ----    -----
target                                         --     57/95     60.0%
```

**The most generous bound I can construct — every gate case whose operand the
guard would admit, plus the canonical seam, all landing — is 55/95 = 57.9%.**
It still misses 57. And each step of it is either forbidden this phase (the Gate
is explicitly `不修改`) or worth one or two cases.

## 4. Why the remainder cannot move

24 unreleased cases need the guard's ambiguity resolved. Resolving it means
choosing among several real values at one coordinate. Measured against the
store's own fields:

```
observations with NO separating store field at all          15 / 35
observations separable only by row_id / table_fragment_id   20 / 35
```

`row_id` and `table_fragment_id` do distinguish the rows — that is real. But
they distinguish *which row*, never *which is right for the question*. The store
records where a value was printed, not what it is about. Using them to pick a
winner is reading the store's own label back as evidence, which the Benchmark
Authority Contract §5 rules out as circular.

The three classes the store cannot express:

```
column_identity     (Apple, Services, FY2025) holds 82,314 revenue,
                    75.4% margin, 109,158 and 26,844 -- four rows, one label
row_identity        three rows labelled "Deferred" on page 43, one fragment
statement_scope     KO 13,762 (p63 consolidated) vs 13,426 (p87, investee-
                    dimensioned) sharing the row label "Operating income"
```

## 5. What this costs, stated as a decision

The target needs `column_identity` — an extraction-time property. Adding it is
a Store rebuild, which is `不修改` this phase. And because a Store rebuild
invalidates the gold's `v2fact:` ids and the pinned `benchmark-version.json`
hash, it is also a Benchmark change — the second `不修改` item.

**So there is no in-constraint path.** That is the determination.

I want to be plain that this is not "the task is hard": the three levers were
each implemented in measurement, and they compose to four cases short even read
generously. Continued retrieval work would produce changes worth one or two
cases per iteration and would not reach the number.

## 6. Where the targets actually stand

```
Release Coverage     46/95 = 48.4%    target >= 60% (57)     NOT MET, not reachable here
Released Accuracy    all released correct in the frozen arm  holds
Incorrect Release    0 in all three arms                     holds
Correct Refusal      25/25 = 100%                            holds
Citation P/R         92.4% / 98.5% by logical fact           quoted, not re-verified
```

The citation row is from `p1-7-release-coverage.md`. My attempt to recompute it
from the `g-new` artifact returned 0/46 because that artifact's `citation_ids`
are in a different id space from the gold's — the attempt was wrong, not the
figure. `report_system_metrics.py` is the tool that produced it; I did not
reproduce it here.

## 7. What would actually reach 60%, with its cost

```
1. STORE: add column_identity (binding), a meaningful row label
          (row_id already distinguishes rows), and table/statement scope.
   Cost: a Store rebuild. Invalidates gold v2fact: ids and the pinned
         benchmark-version.json hash -> a Benchmark version bump.
   Evidence it is feasible: the iXBRL layer already carries all three
         (125 axes, 1,083 members; the KO rows resolve exactly).

2. BENCHMARK: for the ~18 cases where the coordinate holds several real
   values, decide which is eligible -- the gold currently points at an
   arbitrary member of the ambiguity, and sum-007 shows it can point at the
   wrong one.
   Cost: a Benchmark change. REQUIRED before coverage is meaningful, because
         a benchmark whose gold is undefined for a case cannot score it.

3. GATE: admit the ~4 metrics whose operand the guard would already accept.
   Cost: a Gate vocabulary change. Worth +4, and safe, but alone it does not
         reach the target.
```

Only (1) is on the critical path for coverage. (2) is on the critical path for
the coverage number to *mean* anything.

## 8. What I did not do

```
did not modify   any file under src/, or the Store, Benchmark, Gate, Binder,
                 Validator, Finalizer, Retriever or Sidecar
did not stop     the backend (uptime 1d12h, port 18002) or any GPU process
did not apply    the reviewed seam, any gate vocabulary change, or any migration
did not treat    60% or 80% as an adjudication criterion
```

All measurements read-only, against the frozen per-case artifacts and the two
frozen stores. `git diff HEAD` is empty; six documents added under
`docs/evaluation/`.
