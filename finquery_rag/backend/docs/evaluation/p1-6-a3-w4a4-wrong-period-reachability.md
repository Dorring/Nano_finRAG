# P1.6-A3-W4-A4 — do the 474 wrong-period facts reach a query?

Diagnosis only.

W4-A3 established that 474 oracle-PRIMARY cells are stored under a period that is not
theirs. The open question was whether that is a latent defect or an active one, and it is
answered by asking the resolver rather than by reasoning about it.

```
python diagnose_wrong_period_reachability.py --store .../store-v2.jsonl \
    --equity .../equity-geometry.json --out artifacts/evaluation/p1-6-a3-w4a4-reach
```

**Answer: latent. None of the 47 slots' answers comes from them, and the facts a query can
reach at all are not metric-shaped.**

## Three findings, in the order they matter

### 1. The seal is not affected

```
slots whose answer is one of the 474 wrong-period facts: 0
```

Across all 47 slots under `table_role` authority, not one resolves to one of them.

### 2. Most of them the resolver cannot see at all

```
472  in Store V2
432  stored with period_end NULL -- unreachable by construction
 40  stored with a period_end -- a query can reach these
```

The resolver matches periods with `str(r.get("period_end") or "")[:4] == want_year`, and
**Store V2 carries no `normalized_period` field** — its period fields are
`period`/`period_start`/`period_end`. The legacy axis set the normalised string on these
cells while leaving `period_end` unset, so 432 of them are stored with a period the resolver
cannot read.

That is a property worth its own line: *a fact can be in the store carrying a period the
store cannot state.* It is why the wrong period does no damage here, and it is not a
safety mechanism — it is an accident of which field got populated, and it would stop
protecting anything the moment `period_end` were filled in from the other side.

### 3. The 40 it can reach are not metrics

Every one of the 40 has a value scraped out of the **row label**:

```
JPMorganChase  'Balance at January 1'                        value 1
JPMorganChase  'Balance at December 31'                      value 31
JPMorganChase  'Balance at January 1 and December 31'        value 131
Visa           'Anniversary release (2)'                     value 2
Visa           'Class B-1 common stock exchange offer'       value -1
Visa           'Cash dividends declared and paid, at a quarterly amount of $
                per class A common stock 0.59'               value 0.59
NVIDIA         'Cash dividends declared and paid ($ 0.016 per common share)'  0.016
```

`January 1` -> `1`, `(2)` -> `2`, `B-1` -> `-1`. Queried under the year they claim
(331 `NO_FACT`, 72 `NO_COMPANY_LEVEL_FACT`, 61 agree with another record at that period,
8 win outright) the eight that win do so because a query built from a row label like
`Anniversary release (2)` matches that row and nothing else. **No plausible query names
those rows.**

## The hypothesis that was wrong

The reasoning going in was that a year-only period match would put a wrong-period fact and
a correct one in the same pool and produce false `AMBIGUOUS_COMPANY_LEVEL` readings — a
refusal where the store holds the answer, which is the class of regression the seal's gates
exist to catch.

**Nothing landed in AMBIGUOUS. Not one.** The count is zero, and it is zero because the
wrong-period facts are not in the pools: 432 are unreadable and the rest do not match the
rows a slot asks about. The hypothesis was reasonable and it was wrong, which is the second
time in this phase that checking beat reasoning — the first being the lexicon-vs-structure
bucket in W4-A2.

## Registered as separate defects, not fixed here

- **`Store V2` has no `normalized_period`.** A fact's period identity survives only through
  `period_start`/`period_end`, and the legacy path leaves those unset while setting the
  normalised string. 432 of these 474 are in that state.
- **A row label containing digits becomes the fact's value.** `Balance at January 1` -> `1`,
  `Class B-1` -> `-1`, `June 29, 2025 to August 2, 2025:` -> `29202522025`. Independent of
  the period work, and it reaches the resolver.
