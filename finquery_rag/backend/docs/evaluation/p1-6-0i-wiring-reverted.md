# P1.6-0I — the canonical wiring is reverted; it made the stratum worse

Step 3's second half was built, measured, and taken back out. The measurement is the
reason, and it is unambiguous.

## What was measured

Twenty cross-entity cases, same fixtures, same commit, the only difference being whether
`_build_fact_store` returned the canonical wrapper:

```
canonical ON    released_correct 0    false_release 0    blocked_and_wrong 20
canonical OFF   released_correct 5    false_release 2    blocked_and_wrong 13
```

The wrapper is not neutral and not slightly negative: cases that released stopped
releasing. `compare-004`, `compare-005` and `crossdiff-001` went from
`READY_FOR_RELEASE` to `FAIL_CLOSED / CALCULATION_INVALID / INSUFFICIENT_OPERANDS`,
**with the binder reporting `BOUND`** — so the operand guard refused after a successful
binding.

## Why

Instrumenting the guard:

```
GUARD s1 entity=JPMorganChase metric=None candidate='20.02' siblings=[] -> conflicting_values
```

`siblings=[]`: the lookup found no facts at the coordinate it was asked about, and the
no-siblings case is treated as conflicting, so the operand is refused.

The cause is a **store mismatch**, not a defect in the wrapper. The packet's candidates
come from R4 retrieval over the legacy `v2fact:` key space, and the operand guard asks
the fact store about *those* candidates' coordinates. Redirecting `facts_at_coordinate`
to a different store makes the guard compare a legacy candidate against canonical
siblings, find no agreement, and refuse — for every operand, in every case.

So the wrapper was answering a question the rest of the path was not asking. The A/B in
P1.6-A9 passed because it called `facts_at_coordinate` **directly**; it never exercised
the operand path, which is where the two stores meet. That is the gap the full run
found, and it is the argument for running the whole system rather than the unit that
changed.

## What was reverted, and what was kept

**Reverted:** `_build_fact_store` returns the legacy store. The canonical path is not in
production.

**Kept:**

- `CanonicalFactStore`, with its tests. It resolves the whole stratum correctly — that
  was measured, twice — it simply needs a retrieval path that speaks its key space.
- The `inspect_r4_fact_store_compatibility` change. It tested `isinstance(...)` where the
  contract is "every R4 candidate is materializable"; that check passed on the wrapped
  store and failed on its wrapper, which is backwards regardless.
- `CanonicalFactStore.candidate_keys` reporting both key spaces.
- The `load_coordinate_index` dedent fix, which was a genuine pre-existing bug.

## What the requirement now is

Precisely stated, because the last attempt failed by not stating it:

> **Retrieval must return canonical candidates.** The packet's candidate keys and the
> fact store's keys must be the same store. Until they are, replacing the fact store
> underneath a retrieval path that still speaks the legacy key space will refuse every
> deterministic operand.

That is a change to the retrieval/index path — how R4 candidates are produced and keyed —
rather than another wrapper.

## Consequence for the 120-case run

`p1-6-0i-full` was run **with the harmful wiring in place**. Its counts (45 / 0 / 45 / 0
across the three runs; 0 of 20 cross-entity) describe the broken configuration and should
not be read as the state of the system. It needs re-running on the reverted build before
any of its numbers are used.

The three non-cross-entity strata were scored under the same broken configuration, and
they moved little — arithmetic 25/35, factual_lookup 20/40 — which is itself consistent
with the mechanism: the operand path only bites where deterministic execution happens,
and the wrapper only redirected metrics that map to a canonical quantity.

## Status

Fixtures stay at v8. The store stays rebuilt and verified. The wiring is out. No
production behaviour changed by this commit beyond returning to what it was.
