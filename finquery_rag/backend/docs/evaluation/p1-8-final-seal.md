# P1.8 — Final Seal

Coverage-C1 closed the identity measurement. The stop rule fired: **no single
failure bucket holds ≥6 deterministic-recoverable cases**, so the Coverage
Sprint ends here and the benchmark is sealed at its current measured capability.

**No runtime behaviour was changed. No Store, Retriever, Binder, Gate, Validator
or benchmark file was modified. The benchmark is Benchmark V2 with
`plan-fixtures-v9` `c20afaec`.**

---

## The sealed numbers

```
Benchmark                 tv2-canonical-v1, post-p1.8-d1-c3
gold                      a3d17211   ·  eval set 227f0341  ·  fixture v9 c20afaec

Answerable                        77
Abstention                        43
Released                          52
Release Coverage                  52/77 = 67.53%
Correct Released                  52/52
Released Accuracy                 100%
Incorrect Release                  0
Correct Refusal                   43/43 = 100%
```

Citation, over the 52 released cases:

```
                          strict ids            canonical identity
citation precision        68/99  = 68.7%         95/99  = 96.0%
citation recall           68/86  = 79.1%         82/86  = 95.3%
```

**The strict-id row is not a grounding measurement and should not be quoted as
one.** Gold for the cross-entity stratum names rebuilt-iXBRL keys while every
citation names a legacy candidate id, so the two sides are in different
namespaces and the strict comparison reports a miss for a correct citation.
Measured, split by the namespace the gold is written in:

```
iXBRL-keyed gold     missed 16 / 16      (100% -- every one)
legacy-keyed gold    missed  2 / 70      (97.1% recall)
```

The 16 are the whole of the strict-id gap. Resolving both sides to the canonical
financial identity -- entity, canonical metric, normalised period, value -- is
the resolution `load_alias_map` already performs one namespace down, and it
gives the 96.0% / 95.3% row. **The identity row is the citation number for
Benchmark V2.**

---

## C1 — identity measurement closure

C0's bridge used `(entity, period, value)` as *the* identity. Four released cases
whose gold it called absent showed that was an over-correction: it is a quantity
fingerprint, not an identity. C1 replaced it with a ladder, strongest rung wins,
and reports which rung each of the 133 gold/pool slot comparisons reached:

```
MATCH_EXACT                79    ids resolve to one candidate via load_alias_map
MATCH_PROVENANCE            0    shared row / table fragment / citation / doc+page
MATCH_CANONICAL            20    entity + canonical metric + period + value
AMBIGUOUS_MATCH             2    the winning rung holds for >1 candidate
POSSIBLE_QUANTITY_MATCH     4    same quantity, canonical metric does not agree
NO_MATCH                   28
```

Authoritative reachability is `EXACT | PROVENANCE | CANONICAL`. A
`POSSIBLE_QUANTITY_MATCH` alone is deliberately **not** reachability, which is
the whole reason for splitting the rung out.

**`MATCH_PROVENANCE = 0`, and that is a finding rather than a null.** The iXBRL
records carry `concept`, `context_ref` and `ixbrl_fact_id`; the legacy records
carry `row_id`, `table_fragment_id` and `page`. Neither vocabulary has a field
the other can name, so no gold/candidate pair ever shares a provenance anchor —
the rung is unrunnable on this store pair, not unhelpful. Where it would earn its
keep is *within* the legacy store, comparing a cited fact to a gold fact; the
gold of the released cases is EXACT-matched there already.

### The re-attribution

```
A. PRODUCTION (gate ON -- the real FIRST_FAILURE_STAGE)
     SEMANTIC_ALIGNMENT      10
     TRUE_RETRIEVAL_MISS     12
     BINDING                  3

B. GATE-BYPASS (diagnostic; the replay overrides refusals)
     TRUE_RETRIEVAL_MISS     15
     BINDING                  8
     VALIDATION               2

BINDING sub-bucket
     STRUCTURAL_DISAMBIGUATION  8

UNATTRIBUTED = 0  in both
```

### What the ladder changed, and what it did not

Case-level, the split is unchanged from C0 — 14 miss at retrieval, 11 do not.
Two cases moved inside it, and both moves matter:

* **`tsla-033` is no longer a retrieval miss.** Its gold matches **exactly**, so
  the C0 bridge was wrong about it; the failure is at binding.
* **`compare-001` becomes one.** Its `s1` reaches only `POSSIBLE_QUANTITY_MATCH`
  — the same quantity under a metric that does not agree, which is the
  Coca-Cola `Consolidated Net Income` / `Net income` collision P1.7 recorded.

### A limitation of the trace, measured rather than assumed

The trace models **one** retrieval pass. The Binder may run more: `compare-005`,
`compare-008` and `pctshare-002` all have `binder_round_count = 2`, all slots
bound, and released correctly, while their gold is absent from the pool the first
pass built. So `TRUE_RETRIEVAL_MISS` means "the gold did not reach the pool *on
the first pass*", and it is an **upper bound** on retrieval failure — exactly as
C0 predicted, now with the mechanism named.

---

## The stop rule

```
a single bucket with ≥6 DETERMINISTIC_RECOVERABLE cases is the only
thing that licenses a single-variable capability fix

DETERMINISTIC_RECOVERABLE      0
IMPLEMENTATION_BUG             1     tsla-035
CAPABILITY_GAP                22
SOURCE_AMBIGUOUS               1     nvda-025
POLICY_CORRECT_BLOCK           1     compare-001
UNCLASSIFIED                   0
```

**No bucket qualifies. The Sprint stops.** No A/B was run, no safety layer was
unfrozen, and nothing was fixed to reach a number.

`CAPABILITY_GAP = 22` is a floor, not a bucket: 14 cases whose gold never reached
the first-pass pool and 8 whose gold was in the pool at a coordinate holding
several values. Both are the same representation finding this phase reached three
times, and neither is a rule.

---

## Future work — registered, not started

**1. Row / column / logical-table representation gap.**
The largest block, and the one that appears at every layer. `compare-010` is the
clean statement of it: the pool holds Microsoft FY2025 Operating income at
`14,166`, `69,773` and `44,589` — three segments — and the gold `128,528` is
exactly their sum. The retrieved pool contains the row's **parts** and not the
row's **total**. Each record's `content` is its own row, so the cells survive;
what is missing is which row it is, and the header that would say so sits in the
table fragment and is flattened onto every row equally. No rule over the existing
fields recovers it.

**2. `tsla-035` unit emission.** The source row reads
`Risk-free interest rate | 3.95 % | 3.92 % | 3.90 %` and the emitted record's
`unit` is `null`, so the validator refuses with `SCV_UNIT_UNSUPPORTED`. The unit
is in the row; the emitter dropped it. One case, deterministic, and deliberately
**not** fixed in this phase.

**3. Source ambiguity.** `nvda-025` is stored with `value = "2024"`, the year
column of `| President and CEO | 2024 | 996,514 | … |`. The question asks for a
person and the gold is a year, so the gold does not denote what the question
names. `compare-001` is the same class from the other side: Coca-Cola files
`Consolidated Net Income` where the slot says `Net income`, and P1.7 declined the
alias because consolidated net income includes non-controlling interests and the
two are different numbers in any filing that has one.

**4. The identity bridge, one level further.** `MATCH_PROVENANCE` is unrunnable
across the two stores (see above) because the iXBRL and legacy schemas share no
provenance field. A cross-store provenance vocabulary would make the rung live
and would turn `TRUE_RETRIEVAL_MISS` from an upper bound into a measurement.

**5. Binder repair rounds.** The trace sees one retrieval pass; the Binder may
run more. Capturing the later rounds' pools would close the last gap between
"not in the pool" and "not answerable".

---

## What this phase fixed, and what it did not

Fixed, and sealed:

* `crossdiff-001` / `crossdiff-002` — the pinned fixture carried the operand
  order of the pre-P1.6-0H question, and the runtime executed it faithfully.
  Regenerated through the authoring contract into `plan-fixtures-v9`; two wrong
  releases became correct ones and `Released Accuracy` went 96.15% → 100%.
* A fixture integrity guard, at generation and as a standalone verifier, which
  fails on a `difference … between A and B` whose slots do not name the question's
  companies in the question's order.

Not fixed, and now registered above: the representation gap, the unit emission
bug, the source-ambiguous cases. **67.53% is the system's measured capability on
this benchmark, and it is reported as such rather than as a number to be moved.**
