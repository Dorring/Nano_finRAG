# P1.8 Coverage-C0 — the pool trace, two attributions, and the answer to "can 75% be reached?"

**Report only. No runtime behaviour was changed, no code was modified, and the
sealed baseline is intact.**

Baseline under test: Benchmark V2, fixtures **v9** `c20afaec`, answerable 77,
abstention 43, released **52/77 = 67.53%**, correct released **52/52**, incorrect
release **0**, correct refusal **43/43**.

**Verdict: no single bucket holds ≥6 deterministic-recoverable cases.
`DETERMINISTIC_RECOVERABLE = 0`. The C0 rule therefore points at a Final Seal at
67.53%, not at a capability fix.**

---

## 1. The pool-membership trace

`scripts/evaluation/trace_p1_8_pool_membership.py` runs the real retrieval policy
per case and records, per `RequiredSlot`: the gold candidate's identity, whether
it reached the final Binder-visible pool, its final pool rank, plus
`gold_pool_complete`, `gold_bound_per_slot`, `gold_binding_complete`,
`slot_complete`, the adapter/validation result and the release status. Retrieval,
materialisation, ordering and the cap are all non-model steps, so this is
deterministic and needs no GPU.

### The canonical identity bridge is not the one that was assumed

Gold for the cross-entity stratum names rebuilt-iXBRL keys; the retriever's pool
holds legacy keys. `load_alias_map` resolves ids *within* a record's namespaces —
it maps `ixbrl:X` to `ixbrl:X` — so comparing raw ids reports **every**
cross-entity gold as absent, and the runtime's own correct releases prove that is
false.

```
tv2f01-s3-crossdiff-003  ixbrl:7eda20a1…  ->  ixbrl:7eda20a1…   (unchanged)
runtime store            TRUSTED_V2_FACT_STORE_PATH = financial-facts.jsonl  (legacy)
pool candidate space     v2fact:…
```

The bridge used instead is the **asserted quantity**, `(entity, period, value)` —
`logical_fact_id`'s payload minus its `metric`. The metric is dropped
deliberately: the iXBRL row carries `canonical_concept: total_liabilities` where
the legacy row carries the printed label `Total liabilities`, and a bridge that
demanded they match would rebuild the same false negative. (The host runtime
does not carry `logical_fact_id` at all — it is in the Windows checkout, not in
the run host's copy.)

**Direction of the error, measured not argued.** The bridge can only *over*-report
reachability. Four **released** cases show gold absent from the pool and released
correctly anyway:

```
tv2f01-s2-pctshare-002  s2 absent      tv2f01-s3-compare-005  s1 absent
tv2f01-s2-pctshare-004  s2 absent      tv2f01-s3-compare-008  s1 absent
```

So the bridge has false negatives and `RETRIEVAL` below is an **upper bound**.
That is the one number in this document that a later phase should tighten.

### What the trace found

```
the 25 unreleased, by whether the gold reached the pool
  gold fully in pool       11     (failure is at or after BINDING)
  gold PARTLY in pool       6
  gold absent from pool     8
```

---

## 2. The two attributions

`scripts/evaluation/attribute_p1_8_coverage.py`. Both have `UNATTRIBUTED = 0`.

```
A. PRODUCTION (gate ON — the real FIRST_FAILURE_STAGE)
     SEMANTIC_ALIGNMENT    10
     RETRIEVAL             11
     BINDING                4

B. GATE-BYPASS (diagnostic only — the replay overrides refusals, so these are
   the stages the refused cases actually reached; NOT a production count)
     RETRIEVAL             14
     BINDING                9
     VALIDATION             2

BINDING sub-buckets (not merged)
     STRUCTURAL_DISAMBIGUATION   8
     OTHER                       1
```

**This corrects the previous first-failure pass**, which had no pool
measurement and put `BINDING` at 17 and `RETRIEVAL` at 0. With the gold's
presence in the pool actually measured, `RETRIEVAL` is the largest stage under
either reading and `BINDING` shrinks to the cases where the gold *was* in the
pool and the Binder still would not choose.

### The column/row identity defect, caught in the act

`compare-010` asks which of Apple and Microsoft had the higher FY2025 Operating
income. The pool holds, for Microsoft FY2025:

```
14,166    69,773    44,589          <- three segments
14,166 + 69,773 + 44,589 = 128,528  <- the gold
```

The retrieved pool contains the row's **parts** and not the row's **total**, and
Apple does not appear at all. That is the defect B0 and B1 named — a coordinate
that does not identify one value — reproducing exactly, with the arithmetic
closing.

---

## 3. The 17 question drifts

`scripts/evaluation/classify_p1_8_question_drift.py` parses **the canonical**
question for its metric, companies and period and compares those with the
fixture's `required_slots` as they now stand. The recorded sentence is never
read as evidence — it is the thing under test.

Order is treated as semantic only where the operation makes it so: `difference`
has a minuend and a subtrahend, so the mention order decides the sign and must
match; `comparison` asks which company is higher, a set question, and the order
the sentence happens to enumerate them in is not part of what is asked;
`ranking` enumerates the companies to rank and the answer is the ordering, which
the gold carries separately.

```
drifted rows: 15
  METADATA_ONLY_DRIFT    15
  SEMANTIC_SLOT_DRIFT     0
  UNCLASSIFIED            0
```

**No re-seal is needed.** Note the count is 15, not 17: v8 carried 17, and the
two that are gone are exactly the pair `crossdiff-001`/`crossdiff-002` whose
slots were wrong — the only two that were ever `SEMANTIC_SLOT_DRIFT`. Every
surviving drift is a fossil sentence over slots that already agree: `compare-001`
still reads `United States`, `compare-006` reads `a larger 2025`, `rank-005`
reads `Interest rate contracts`.

---

## 4. Recoverability

```
DETERMINISTIC_RECOVERABLE      0
IMPLEMENTATION_BUG             1     tsla-035
CAPABILITY_GAP                22
SOURCE_AMBIGUOUS               1     nvda-025
POLICY_CORRECT_BLOCK           1     compare-001
UNCLASSIFIED                   0
```

Three cases could not be labelled by rule and were decided on evidence, recorded
in `JUDGED` in the attribution script with the record that decides each:

* **`tsla-035`** — gold in the pool at rank 1 and bound; the validator refuses
  with `SCV_UNIT_UNSUPPORTED`. The source row reads
  `Risk-free interest rate | 3.95 % | 3.92 % | 3.90 %` while the record's `unit`
  is null. The emitter dropped a unit the source states. Recoverable by parsing
  a unit already in the row — but it is **one** case.
* **`nvda-025`** — the stored value is `2024`, which is the **year column** of
  `| President and CEO | 2024 | 996,514 | 26,676,415 | … |`, a compensation
  table. The question asks for a person; the gold is a year. The gold does not
  denote what the question names.
* **`compare-001`** — gold in the pool at ranks 6 and 1, Binder `BOUND`, refused
  by `QUERY_EVIDENCE_SEMANTIC_MISMATCH` because Coca-Cola files the line as
  `Consolidated Net Income` where the slot says `Net income`. P1.7 measured this
  and declined the alias: `net_income` already carries 归母净利润, and
  consolidated net income includes non-controlling interests, so they are
  different numbers in any filing that has one.

`CAPABILITY_GAP` at 22 is not a bucket but a floor: 14 cases whose gold never
reached the pool (a retrieval/representation problem) and 8 whose gold was in
the pool at a coordinate holding several values (the column/row identity
problem). Both are the same finding this phase has reached twice already, and
neither is a rule.

---

## 5. The decision

The C0 rule: a single bucket with `≥6` source-grounded, deterministic-recoverable
cases is the only thing that licenses a single-variable capability fix.

```
DETERMINISTIC_RECOVERABLE                    0
largest bucket with a rule that recovers it  tsla-035, n=1
target                                       58/77 = 75.3%   (+6 releases)
```

**No bucket qualifies.** Nothing here is a lever that returns six cases without
simultaneously touching several safety layers — the recovery paths are
representation (a pool that carries the row's total, not its parts) and
disambiguation (a coordinate that identifies one value), and both are the changes
P1.6/P1.7 froze rather than a fix this round can make.

**Recommendation: Final Seal at 52/77 = 67.53%,** with `incorrect release = 0`
and `correct refusal = 43/43` held.

### What would have to be true for 75% to be reachable

Six releases out of the fourteen `RETRIEVAL` cases, without a single one of them
turning into a wrong release. The one measured fact bearing on that is the
control group above: gold absent from the pool is *demonstrably survivable* —
four released cases did it. So the retrieval bucket is not hopeless; it is
unmeasured. Tightening the identity bridge so `RETRIEVAL` stops over-reporting is
the prerequisite, and it is a measurement, not a fix.
