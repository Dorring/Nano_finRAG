# P1.6-A3-3b — column-local evidence taken before assembly

Diagnosis only. **No rule change.** A3-3 tested scoping with `column_headers[col]`, which is
`col_headers`' assembled string — carry-forward plus a row-level scope phrase plus whatever
the label column leaked in. Its failure proved only that a polluted string stays polluted.
This takes the evidence one layer earlier: **the raw header cells that geometrically cover
the logical column**, `grid[i][col]["raw_text"]`, with no assembly.

```
                     assembled_raw        assembled_safe       raw_raw              raw_safe
nvda CF  (failing)   bucket 18            duration 18          bucket 3,            duration 18
                                                                 duration 15
tsla CF  (failing)   bucket 18            duration 18          bucket 3,            duration 18
                                                                 duration 15
aapl CF  (control)   unknown 3,           unknown 3,           unknown 3,           unknown 3,
                     duration 15          duration 15          duration 15          duration 15
msft CF  (control)   duration 13          duration 13          duration 13          duration 13
```

## 1. Raw scope removes most of the pollution, and shows where the rest lives

```
column 1 raw cells [(3, 1)]  raw ['Cash flows from operating activities:']
```

Fifteen of NVIDIA's eighteen columns come back as `duration` under the raw scope. The three
that do not are **columns 0, 1 and 2 — and column 1's raw header cell genuinely reads
`Cash flows from operating activities:`.** The label column's prose is in the grid's *header
row* at that position, so the contamination is **in the grid expansion, upstream of the
assembly**, not created by it.

That is why A3-3's scoping looked useless: it scoped to a string assembled from an already
polluted grid. And it is why the scope alone cannot finish the job here — the three columns
it fails on are the label-adjacent ones, which emit nothing either way.

## 2. The token-safe regex is contract, not defence-in-depth

The decisive evidence is a **positive control**, not the failing tables:

```
kind=bucket   matched 'rating'   aapl_fy2025 col 3   raw header cell: ['Operating Leases']
```

`Operating Leases` is a raw header cell. The `rating` alternative matches inside
**ope·rating** on **unassembled column evidence**. So `raw + old regex` is not safe, and
hardening the pattern is not belt-and-braces — it is the second half of a two-part
requirement.

On NVIDIA and Tesla the remaining three collisions happen to sit on label columns, which is
why `raw_raw` and `raw_safe` agree on the 15 that matter. Had that been the only evidence,
the wrong conclusion — "the scope fixes it, the regex is optional" — was available and
wrong.

## 3. Genuine non-temporal columns survive

The positive controls are real header cells that declare a kind:

```
segment     'Americas'        aapl col 3    ['Americas']                          genuine
segment     'Europe'          aapl col 9    ['Europe']                            genuine
segment     'Greater China'   aapl col 15   ['Greater China']                     genuine
segment     'Japan'           aapl col 21   ['Japan']                             genuine
segment     'Rest of'         aapl col 27   ['Rest of Asia Pacific']              genuine
segment     'United States'   jpm col 0     ['December 31, 2025 …', 'United States (a)']
comparison  'Increase'        aapl col 6    ['Hypothetical Interest Rate Increase']  genuine
bucket      'rating'          aapl col 3    ['Operating Leases']                  FALSE
```

The hardening touches only `rating|range|grade|tier` as whole tokens. **It removes
`Operating Leases → bucket` and leaves `Americas`, `Europe`, `Japan`, `Greater China` and
`Hypothetical Interest Rate Increase` exactly as they are.** The change is not a blanket
softening of the cascade; it is the removal of a substring collision, and the controls show
the difference.

## What is now settled, and what is not

```
raw column-local evidence     removes the row-prose dependency for the columns that emit
token-safe patterns           required; a collision fires on genuinely raw column evidence
genuine segment/comparison    unaffected
```

**Not settled:** whether the label-adjacent columns' prose in the header row is a defect
worth its own repair. It affects three columns per table that emit nothing, so it is not on
A3's path — but it is the same over-segmentation A3-1d saw as a one-row offset in Coca-Cola,
and it now has three sightings without a diagnosis.

## Status

Nothing changed. Evidence at
`artifacts/evaluation/p1-6-a3-3b-rawscope/column-local-scope.json`.
