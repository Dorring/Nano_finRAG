# Decision brief — the two ways past 48.4%

Everything here is measured. Where a number comes from an experiment, the
experiment is named; where it comes from reasoning, it says so.

## Where the round landed

```
                    baseline        now        target
Release Coverage     20/95         46/95       57/95
                      21.1%         48.4%       60%
Released Accuracy    20/20         44/46        95%
INCORRECT RELEASE        0             0           0
Correct Refusal      25/25         25/25        95%
Citation Recall   27/28 96.4%   62/64 96.9%      90%
Citation Precision 27/32 84.4%   62/74 83.8%      90%
```

Both arms re-measured on GPU on the same day. The previously published `17.9%`
was measured with the specialist degraded to CPU (the deployment env pins
`CUDA_VISIBLE_DEVICES=2`, the GPU holding another user's job, and the bf16
specialist dies there), so the honest baseline is 21.1%.

## Why no policy change closes the gap

The gate was removed entirely and the run repeated: **50/95 = 52.6%**. Of the
22 cases the source-grounded rule still blocks, bypass releases 3; the other 19
are refused downstream by the Binder at the same coordinates. So 52.6% is the
ceiling for *any* admission rule, measured rather than argued.

Every invariant-safe lever was then tried and measured:

```
source-grounded admission   +30 cases    shipped this round
ontology aliases             +1          2 refusals across 49 blocked cases
entity isolation             +1          inside the Binder's spread, off
consolidated-total preference +5         5 of 18 conflicting coordinates
plan-entity inference        +2          refused by design
                            ----
need                        +11
```

## The one root cause underneath most of it

A fact's coordinate is `(entity, metric, period)` and for many rows that does not
identify one value. Each fact's `content` is its own row, so the cells survive:

```
Microsoft / Cost of revenue / FY2025
  value=87,831   content=| Cost of revenue | | 87,831 | 74,114 | 65,863 |
  value=22,422   content=| Cost of revenue | | 22,422 | 19,611 | 17,202 |
```

**The label is identical on all four.** The header that separates them,
`Productivity and Business Processes`, sits in the table fragment and is
flattened onto every row equally. The store records which cells a row holds and
not which row it is. No rule over the existing fields recovers that, which is
why 21 gate blocks and 7 binding losses cannot be closed by policy. It is also
why citation precision stalls at 83.8%: six of the twelve misses are the same
quantity reached through a different table fragment, because the store holds
each row twice.

## Option A — the Binder's conflict and retry semantics

**What it changes.** Two things, both in `src/runtime/trusted_v2_binder.py` and
the transport-retry policy:

1. A Binder that cannot decide among several facts at one coordinate refuses
   (`EVIDENCE_CONFLICT`). The entity-isolation experiment showed this is where
   the refusals actually happen: the packet was given the missing filer and the
   Binder still would not choose. Changing it means giving the Binder a rule for
   choosing.
2. `binder_returned_invalid_schema` is not retried — `TransportRetryPolicy`
   freezes one semantic response and one transport retry, and `retryable_failures`
   lists only transport and HTTP errors. Four compare cases fail here. Re-rolling
   a *malformed* response is arguably not re-rolling one the model disagreed
   with, since malformed output carries no semantics to re-roll.

**Expected gain.** Up to 13 compare cases, but they are not all this: 4 schema,
~4 coordinate conflict, 2 relation ambiguity. Realistically **+6 to +9**.
Reaching 57 needs the high end *plus* the +5 total rule.

**Risk.** This is the surface that makes `Incorrect Release = 0` true. A Binder
that picks among competing values can pick wrong, and the wrong pick is a
confident released answer, which is the one failure mode this project has spent
its whole history preventing. It would need a new guard, and the guard would
have to be designed and measured before the change is worth anything.

**My recommendation:** do not do this to reach a coverage number.

## Option B — restore the table, column and row-hierarchy dimensions

**What it changes.** The fact representation. Facts would carry which row they
are, not only which cells they hold — the segment header, the column, the row's
position in its table.

**Expected gain.** It addresses 21 gate blocks and 7 of the 10 binding losses,
plus the citation-precision misses — **+28 addresses, the largest single block
in the project.** It is also the only option that improves citation precision,
since deduplicating a row held twice makes the cited row the gold's row.

**Risk.** It does not touch a safety invariant; the failure mode is recall, not
correctness. The cost is that it is a data project, not a code change.

**Blocker.** A standing decision records the P1.6-0D PDFs as *read-only audit
material* — never rebuild the store or gold from them — and the store is
hash-pinned in `benchmark-version.json`. So this cannot start without an explicit
decision to reopen the representation, and it should not be done as a side
effect of a coverage target.

**My recommendation:** this is the one worth doing, on its own schedule, as its
own piece of work.

## What I will not do

Raise 48.4% to 60% by loosening fail-closed semantics on the way past. The
`Incorrect Release = 0` in the table above is the number this project's
credibility rests on, and a 60% that was bought by making it less true is worse
than a 48.4% that is true.

## What I need

One line: **(A)**, **(B)**, or **stop at 48.4%**.

- On **(A)** I will design the conflict rule and its guard first, and measure it
  against `incorrect release` before it goes near the shipped path.
- On **(B)** I will scope the representation change, including what re-extraction
  costs and what it invalidates, before touching the store.
- On **stop**, the round is already documented and committed, and the seal in
  `p1-7-release-coverage.md` stands as written.
