# P1.6-A9 — retrieval resolves the stratum through the canonical store

Step 2. `CanonicalFactStore` wraps both stores: a slot's metric is mapped to a canonical
quantity and resolved through `concept_alignment`; anything unmapped or unanswerable goes
to the legacy store exactly as before.

```
iXBRL store available: yes (19,795 records)
retrieval through CanonicalFactStore, using the re-derived metric names:

  RESOLVED 47 / 47    AMBIGUOUS 0    EMPTY 0
  expected values 45 / 45 matched
  every resolution served via=ixbrl
```

This is the same call retrieval makes — `facts_at_coordinate(entity, metric, period)` —
so it exercises the wiring rather than the resolver underneath it.

## The fallback is what keeps step 2 and step 3 separable

The fixtures still name the old strings (`United States`, `Current`, `Total`), and those
are **deliberately absent** from the metric map. They name no single quantity; mapping
them to something would be the guess this work exists to stop. So:

- a metric that maps and resolves → the iXBRL fact
- a metric that does not map → the legacy store, unchanged
- a mapped metric with no iXBRL fact → the legacy store, unchanged

Without that, switching would break every case whose fixture has not been re-derived, and
step 2 could not be measured before step 3. A test pins each branch.

## An ambiguous quantity returns nothing rather than one value

If a canonical quantity still holds more than one value for an entity, `resolve_canonical`
returns **nothing** and the caller falls back. Returning the first value would look like a
successful resolution and be a guess — which is precisely the failure the earlier phases
kept finding, and the reason the operand guard exists. Pinned by a test.

## Two properties of the wrapper worth stating

**Ordering happens at query time.** `canonical_concept` alone collapses `ProfitLoss` and
`NetIncomeLoss` into `net_income` and cannot say which wins. Resolution goes through
`CONCEPT_ALIGNMENT` in order — that is what returns Tesla's 3,855 rather than its 3,794,
and a test asserts it.

**Periods match by year, not by label.** A balance-sheet fact labelled `ASOF2025-09-27`
and an income-statement fact labelled `FY2025` are the same fiscal year, and a query that
compared labels would miss one of them.

## What is switched, and what is not

**Switched:** nothing in production. `CanonicalFactStore` exists, is tested, and resolves
the whole stratum, but no runtime path constructs it — that wiring plus the fixture
migration is step 3.

**Not switched:** the facts themselves. The 20 cross-entity cases resolve against the
re-derived metric names; the fixtures still carry the old ones, so a production run today
would fall back throughout — which is the correct behaviour until step 3 lands.

## Status

No fixture changed, no production retrieval changed, existing store unchanged. The iXBRL
store and the wrapper are both additive.
