# P1.8-D1-C3/C5 — the stale fixture, regenerated; and the guard that would have caught it

**Verdict: `ROOT_CAUSE = STALE_PINNED_PLAN_FIXTURE`.** The runtime was never
wrong. It executed the plan it was given, in the order the plan gave it, and the
plan was written for a question the benchmark no longer asks.

Nothing in the runtime was changed to produce this. The narrow cross-entity
binding patch stays **OFF**, the global structured-operand flag stays **OFF**,
and no question or gold was touched.

---

## 1. What was actually stale

`crossdiff-001` and `crossdiff-002` are pinned with the operand order of the
pre-P1.6-0H question. v8 says:

```
tv2f01-s3-crossdiff-001
  fixture question  "What is the difference in Additions between Tesla and NVIDIA in FY2025?"
  slots             s1 minuend=Tesla   s2 subtrahend=NVIDIA
```

while the benchmark's canonical question, the gold's `fact_ids` and the gold's
`expected_value` all say NVIDIA first:

```
canonical question  "What is the difference in Net income between NVIDIA and Tesla in FY2025?"
gold fact_ids       [ixbrl:aa028d0d… (NVIDIA, 72880), ixbrl:05cf9d28… (Tesla, 3855)]
gold expected_value 69025            = 72880 - 3855
```

The runtime computes `current - previous` (the answer text names the formula
`difference.v1`), reads `current` from `slot[0]`, and therefore returned
`Tesla - NVIDIA = -69,025`. Correct execution, wrong plan.

### How it got that way — from the migration's own source

`migrate_cross_entity_v8.py` is the P1.6-0H migration. Its per-case loop does
three things and stops:

```python
gold["values"] = dict(spec["values"])          # -> fact_ids rebuilt in spec order
...
for slot in plan["plan"]["required_slots"]:
    slot["metric"] = spec["metric"]            # <- metric only
...
entities = list(spec["values"])
question = _QUESTION_TEMPLATES[operation].format(..., a=entities[0], b=entities[1])
evaluation["question"] = question              # <- canonical question rewritten
```

It rewrites the **gold**, the **slot metric** and the **canonical question**. It
never re-derives the slot **entities** or their **order**, and it never refreshes
the fixture's own `question` field. So the entity order stayed as v7 had it —
the order of the old `Additions between Tesla and NVIDIA` wording — while
everything around it was moved forward.

That is the whole defect: a partial migration leaving one field behind.

### It is visible without running anything

The fixture's recorded question says `between Tesla and NVIDIA` and its slots
are `[Tesla, NVIDIA]` — **internally consistent**. The drift is only visible
against the canonical question. This matters for §3 below and is why the first
version of the guard passed the stale fixture.

---

## 2. C3 — regenerate through the contract, not by hand

`scripts/evaluation/build_p1_8_d1_fixture_v9.py` invokes the authoring contract
(`build_p1_2_plan_fixtures.author_plan` → `_multi_evidence_slots`, which takes
one slot per `gold.fact_ids` entry *in the gold's order*, reading each fact's
entity from the store) and writes what it emits. No slot is swapped by hand.

```
python scripts/evaluation/build_p1_8_d1_fixture_v9.py --apply
```

```
v8  7463b45a5bfc309d0a5ab29cdb15b3f070c13c8526ae5d1486ca67847dda362d
v9  c20afaec24bcab1360bbcbf9e880d9c8661c24551a1bcef2c59dc0248f3147da
rows moved: ['tv2f01-s3-crossdiff-001', 'tv2f01-s3-crossdiff-002']
```

| case | old slot entities | new slot entities | old question → new question |
| --- | --- | --- | --- |
| `crossdiff-001` | `[Tesla, NVIDIA]` | `[NVIDIA, Tesla]` | `…in Additions between Tesla and NVIDIA…` → `…in Net income between NVIDIA and Tesla…` |
| `crossdiff-002` | `[Microsoft, Apple]` | `[Apple, Microsoft]` | `…in Other between Microsoft and Apple…` → `…in Net income between Apple and Microsoft…` |

Checked independently of the generator: **exactly 2 of 120 lines differ**, ids
and line order identical.

`crossdiff-003/004/005` are carried as a **control** — the contract must still
reproduce their slot entities, and does. If it had not, the contract would be
what moved and none of the above would be trustworthy.

The full provenance — old/new fixture hashes, per-row hashes, canonical question
hashes, the generation command and code path, and the fact-store and gold
hashes — is in
`/disk/qh/nano-finrag/artifacts/evaluation/p1-8-d1-c3/p1-8-d1-c3-migration.json`.

### Result, with the narrow patch verified absent

```
== preflight: runtime source state ==
narrow cross-entity patch: ABSENT (flag OFF)
```

| stage | runs | `crossdiff-001` | `crossdiff-002` |
| --- | --- | --- | --- |
| `c3-probe-001` | 1 | `+69,025.00` RELEASED | — |
| `c3-probe-002` | 1 | — | `+10,178.00` RELEASED |
| `c3-crossdiff-r1..r3` | 3 | `+69,025.00` RELEASED ×3 | `+10,178.00` RELEASED ×3 |

Both acceptance values are met, and they are stable across repeats — the
direction here is fixed by the fixture's slot order, not by the binder, so the
run-to-run spread that the binder's LLM introduces does not reach it.

`crossdiff-003/004/005` remain `FAIL_CLOSED` in every run, unchanged from before.
They were never operand-order defects; they are blocked elsewhere and remain out
of D1's scope.

### Full Benchmark V2, flag OFF

120 questions, V2 gold `a3d17211`, eval set `227f0341`, fixture v9 `c20afaec`.

| | v8 (Arm A, patch OFF) | v9 (regenerated, patch OFF) |
| --- | --- | --- |
| released | 52 | 52 |
| released correct — canonical checker | 50 | **52** |
| released correct — judge | 50 | **52** |
| incorrect releases | 2 | **0** |
| released accuracy | 50/52 = 96.15% | **52/52 = 100%** |
| false release rate | 0.026 | **0.0** |
| answerable correctness | 50/77 | **52/77 = 67.53%** |
| correct refusal | 43/43 = 100% | **43/43 = 100%** |

The only rows that move are the two the fixture moved. `candidate_correctness`
goes 52/77 → 54/77; `trusted_release` is 52 with `ungrounded_release` 0.

The two scorers agree row for row — the judge's `false_release` was exactly
`{crossdiff-001, crossdiff-002}`, and the canonical checker's
`COMPARATOR_MISMATCH` was the same two, `-69,025` and `-10,178` against golds
`+69,025` and `+10,178`.

### Consequence

Because the corrected fixture with the patch OFF produces the right answers, the
runtime does **not** need to override an authoritative plan with the question's
surface wording. The narrow patch is not promoted to production, its flag stays
OFF, and `ROOT_CAUSE` is `STALE_PINNED_PLAN_FIXTURE` alone — not
`LIVE_RUNTIME_OPERAND_AUTHORITY`.

---

## 3. C5 — the fixture integrity guard

For a `difference in <metric> between <A> and <B>` question whose two mentions
uniquely resolve to two distinct companies, the fixture's `slot[0].entity` must
be A and `slot[1].entity` must be B. Otherwise fixture verification fails.

`scripts/evaluation/verify_fixture_integrity.py` implements it;
`fixture_integrity.assert_plan_integrity` is called from
`build_p1_2_plan_fixtures.build_fixtures`, so a fixture that cannot be verified
is not written in the first place.

### The one design point that matters

**The check is made against the canonical question, passed in separately — not
against the fixture's own `question` field.**

This is not a detail. The first version of this guard read the fixture's own
question, and it **passed the stale v8 fixture**: v8's question says `between
Tesla and NVIDIA` and v8's slots say `[Tesla, NVIDIA]`, so the fixture agreed
with itself. A stale fixture is internally consistent by construction; the drift
is against `canonical-eval-v1.jsonl`. A guard that cannot see the authority
cannot see the defect. The eval set is therefore a required argument.

### It catches the real defect

```
v8   rows 120  checkable 5  violations 2   exit 1
       FAIL tv2f01-s3-crossdiff-001   ['Tesla','NVIDIA']     == ['NVIDIA','Tesla']
       FAIL tv2f01-s3-crossdiff-002   ['Microsoft','Apple']  == ['Apple','Microsoft']
v9   rows 120  checkable 5  violations 0   exit 0
```

### A second finding the guard surfaced

17 of 120 v8 rows record a `question` that is not the canonical question —
`compare-001` records `Which company had a higher United States in FY2025…`,
`compare-006` records `…a larger 2025 in FY2025?`, `crossdiff-003` records
`difference in Total between…`. These are the same fossil class as
`crossdiff-001`'s `Additions`: question text left behind by an earlier rewrite.
v9 brings that to 15, by fixing the two rows it touches.

This is reported, not fatal, and **v9 does not change it** — those rows'
operand order already agrees with the contract, so they are not operand-order
defects, and widening D1 to cover them would make the change set unattributable.
It is recorded here as the next stratum-wide cleanup.

### Where the guard is *not*

Not in the runtime. The runtime is supposed to treat the plan as authoritative,
and a runtime that silently corrected a stale plan from the question's wording
would be the `LIVE_RUNTIME_OPERAND_AUTHORITY` behaviour this phase exists to rule
out. The guard belongs at generation and verification, which is where it is.

Wiring it into the replay harness's fixture load — so a stale fixture cannot be
*run* silently either — is recommended and deliberately left until after the
seal, so the sealed run is not confounded by a harness change.

---

## 4. Open, and deliberately not fixed here

* **`enrich_and_validate` cannot run over this benchmark.** It aborts on
  `compare-001` with `an answerable slot selects no fact`. The coordinate index
  folds a record's `metric` field, which is empty on every rebuilt-iXBRL row,
  while a slot names the metric in prose — so the index and the slot vocabulary
  disagree. Pre-existing, unrelated to D1, and it means the full
  `build_p1_2_plan_fixtures` build has not been runnable since that change; v8
  was produced by patching v7, not by a build.
* **15 rows carry a stale recorded question** (§3), as do 17 before this change.
* **`crossdiff-003/004/005` remain unreleased** for reasons outside D1.
* **`msft-016` / `pfe-030`** stay where `p1-8-b9` put them — a runtime gap, not a
  benchmark one.

---

## 5. D1 Seal

```
ROOT_CAUSE            STALE_PINNED_PLAN_FIXTURE
LIVE_RUNTIME_OPERAND_AUTHORITY   not implicated

benchmark V2          gold a3d17211 · eval set 227f0341 · fixture v8 7463b45a
fixture v9            c20afaec  (2 of 120 rows changed, both cross-entity)

released              52
released correct      52          (47 STRICT + 5 relational)
incorrect releases    0
released accuracy     52/52 = 100%
answerable correctness 52/77 = 67.53%
correct refusal       43/43 = 100%
false release rate    0.0

narrow cross-entity binding   OFF -- not promoted to production
global structured-operand flag OFF
questions / gold              unmodified
runtime sources               unmodified
```

The runtime is not what was wrong, so nothing in it was changed. The
`difference ... between A and B` shape is now answered correctly by the ordinary
arithmetic path, on the authoritative plan, with the sign the gold states.

**What the seal does not claim.** It does not say coverage improved: 52/77 was
already the *candidate* count, and two of those candidates were being released
with the wrong sign. This change converts two wrong releases into correct ones.
It does not touch `crossdiff-003/004/005` or the 23 `blocked_and_wrong` cases.

## 6. Promotion — applied

v9 is the benchmark's fixture. Written by
`scripts/evaluation/promote_p1_8_d1_fixture_v9.py`, which refuses to promote
unless the source bytes hash to the digest the C3 record states:

```
benchmarks/tv2_canonical_v1/plan-fixtures-v9.jsonl          c20afaec   (new)
benchmarks/tv2_canonical_v1/plan-fixtures-v9.manifest.json
benchmarks/tv2_canonical_v1/migrations/p1.8-d1-c3.json      the C3 record
benchmarks/tv2_canonical_v1/benchmark-version.json          -> tv2-canonical-v1-post-p1.8-d1-c3
artifacts/evaluation/p1-8-c-v2/plan-fixtures-v9.jsonl        c20afaec   (beside the run)

plan-fixtures-v8.jsonl                                      7463b45a   (kept, not overwritten)
```

`benchmark-version.json`'s `files` now names v9; v8 is identified by its hash
under `migrations[p1.8-d1-c3].pre`, which is how the `p1.6-0h` entry already
identifies v7. The C5 verifier passes over the promoted fixture in the benchmark
tree, against the benchmark's own canonical questions.

Both the Windows checkout and the 4090-qh run host carry the promoted tree. The
only difference between them is that the host checkout predates
`benchmark-version.json`, so this migration adds it there.
