# P1.6-A3-W3 — TemporalKind shadow: closed

```
W3 DONE — 10 PASS + 1 KNOWN COVERAGE GAP (REAL_BUCKET_POSITIVE_UNOBSERVED)
```

The shadow producer is `scripts/evaluation/temporal_kind_shadow.py`, the acceptance verifier
is `scripts/evaluation/accept_temporal_kind.py`, and the two changes it makes — column-local
input domain, whole-token collateral words — were both measured in A3-3/A3-3b before they
were written.

## The gate

```
Visa mixed DAY/YEAR resolved                  PASS
NVDA/Tesla false bucket/comparison removed    PASS
Apple control preserved                       PASS
MSFT UNKNOWN columns source-adjudicated       PASS
true segment columns preserved                PASS
true comparison columns preserved             PASS
true bucket columns preserved                 UNTESTED
all new non-UNKNOWN kinds have provenance     PASS
legacy authoritative kind path unchanged      PASS
1498-table emission unchanged                 PASS
production still consumes legacy path         PASS
```

## The one that is UNTESTED, and why that is not a failure

The scan found **no genuine bucket column in these eight filings** — only the false
`Operating Leases`, which the token-safe pattern correctly stops calling a bucket. So
"buckets are preserved" is asserted by the *absence of a false positive*, not by a surviving
positive. A real `Rating / Tier / Grade / Range` column has not been exercised.

It is deliberately not closed by manufacturing a fixture. A corpus case invented to make the
table green would test the fixture, not the producer, and this line of work has spent its
whole budget refusing exactly that trade.

It does not block W4: it is not a known error and not a safety hole. It bounds what may be
**claimed**, not what may be built:

```
may be said
  bucket false positive `rating` inside `operating`  measured, eliminated
  segment / comparison true positives                measured, preserved

may not be said
  real bucket positive recall                        verified
```

## What W3 established

**The oracle is the source, not the legacy.** Microsoft's six columns come back `UNKNOWN`
and that is the correct answer: their raw header cells carry no duration phrase, no `as of`,
no date expression and no binding. The legacy called them `duration` — through the polluted
header path, not because the filing said so. Catching up to the legacy count would have been
the wrong goal, and the acceptance script says so in its first check.

**Provenance is per-kind, not per-table.** Every non-UNKNOWN kind names its trigger and the
header cells it came from, so a later reader asks *which HTML cell said `Americas`* instead
of being told a concatenated string happened to match. 816 non-temporal columns, 26 distinct
triggers, all with provenance.

**The legacy path did not move.** This is not `new kind == legacy kind` — NVDA and Tesla can
never satisfy that. It is that W3's code changed nothing the authoritative path outputs:
MSFT#17151 still classifies `{bucket: 62, duration: 744}`, the production modules still do
not import the shadow, and the emission is byte-identical across all 1498 tables.

## Entry condition for W4

`PeriodBindingV2` and `TemporalKind` are both now determinate, explicable and free of
unresolved anomalies — the condition W4 was waiting on. **No YEAR/point mixing is carried
forward.**
