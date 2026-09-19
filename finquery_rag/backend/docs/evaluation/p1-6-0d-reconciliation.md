# P1.6-0D reconciliation — what the verification settled, and the two it did not

The independent verification returned `INTENDED_AMENDMENT_ONLY` for the corpus and
`1 VALID_GOLD / 10 WRONG_SCOPE_GOLD / 0 SOURCE_STILL_AMBIGUOUS` for the eleven
contested operands. This records which of those I confirmed against the filings, and
where the "company-level value" it supplies is **not** the straightforward fix the
label implies.

## Confirmed

**VERIFY-01 — accepted, and the discrepancy was my oversight.** My count listed
`raw/SEC` but never `raw/` itself, so I missed `raw/version_candidates/` entirely.
It holds exactly the three amendments, and their `primary.html` SHA256 match the
verifier's report:

```
TSLA/SEC_1318605_000110465926053166  primary.html  35e8e3c985ad540e…
TSLA/SEC_1318605_000110465925042659  primary.html  c28e39f1152a8210…
KO/SEC_21344_000002134424000019      primary.html  d5f8c22f1f670fd2…
```

No material is missing. `raw/SEC` is 60 by design; the `/A` amendments are archived as
`VERSION_SIDECAR_CANDIDATE` and normalization consumes both roots, which is why
`normalized/SEC` holds 63.

**`rank-002` — `VALID_GOLD`, confirmed at source.** `6,411` appears on **page 70**
(the consolidated income statement, row `Research and development`) and again on
page 58 (MD&A, block titled `Research and Development Expense`). It is the company's
FY2025 R&D, so the operand the runtime bound (`1,332`) is genuinely ungrounded and
the coordinate guard refusing it is correct. **The guard's cost against the valid
denominator is therefore zero on this case.**

**`compare-001`, `compare-003`, `compare-005`, `compare-009`, `compare-010`,
`crossdiff-003`, `crossdiff-004` — accepted as `WRONG_SCOPE_GOLD`.** Each gold takes a
component of one quantity while the company figure is a different row or column of the
same quantity. `crossdiff-003`'s Apple `9,683` (Federal sub-total → `Provision for
income taxes $20,719`) and `compare-001`'s `$5,678` (`United States` → `Total
$15,998`) I had already read out of the filings myself and they agree.

## Not accepted as stated — two slots need different treatment

Both are cases where "`WRONG_SCOPE_GOLD`, company-level truth is X" would put a number
into fixture v6 that the filing does not support.

### `compare-004` Apple — the "company value" is a derived sum

```
aapl_fy2025_10k.pdf p42
  Derivative instruments designated as accounting hedges:
      Foreign exchange contracts   62,647
  Derivative instruments not designated as accounting hedges:
      Foreign exchange contracts  109,079
```

`171,726` is `62,647 + 109,079`. **The filing never states it.** The two figures are
two categories of the same metric, each a reported number; their sum is a construction.

This is not necessarily wrong — a question *can* legitimately ask for the combined
total — but it must be a deliberate fixture decision about what the question asks, not
a silently computed value presented as source truth. Before v6, `compare-004`'s
**question text** has to be checked: if it asks about "foreign exchange contracts"
without distinguishing designated from not-designated, the case may need the question
narrowed rather than the gold replaced with a sum.

The same shape applies to `compare-004` Microsoft, where `(853)` is `(809) + (44)`.

### `rank-005` — the gold values are different *kinds* of quantity

```
tsla… (question) rank Apple, The Coca-Cola Company, Microsoft by "Interest rate contracts"

Apple   gold  12,875   aapl p42: "The notional amounts of the Company's outstanding
                        derivative instruments … Interest rate contracts $ 12,875"
                        -> a NOTIONAL AMOUNT

Coca-Cola gold    16   ko p84: table "Gain (Loss) Recognized in OCI"
                        -> a GAIN recognised in other comprehensive income

Microsoft gold     0   (type not established)
```

`16` is not a mis-scoped `13,674`. They are different quantities — an OCI gain and a
notional principal — and no row or column choice makes one into the other. So the
verifier's proposed company value for Coca-Cola (`13,674`, the notional of fair-value
hedges) is the right *kind* of number to compare against Apple's `12,875`, but that is
a **fixture redesign**, not a scope correction: the gold for at least two of the three
entities is currently measuring something else.

Until that is settled, `rank-005` should not be counted as a fixable `WRONG_SCOPE_GOLD`.
It is a case whose gold is internally incoherent, and it needs either a re-posed
question over a single consistent quantity or retirement.

## What this changes

The `WRONG_SCOPE_GOLD` label is not uniform, and treating it as such would put two
unsupported values into fixture v6. Before the v5 → v6 migration, each of the eleven
slots needs its **fix** classified, not just its verdict:

| fix | slots |
|---|---|
| company total of the *same* quantity — substitute the value | compare-001, compare-003, compare-005, compare-009, compare-010, crossdiff-003 (×2), crossdiff-004 |
| gold is already correct; nothing to change | rank-002 |
| needs a decision about what the **question** asks before any value is chosen | compare-004 (×2) |
| gold is internally incoherent across entities; needs re-posing or retirement | rank-005 |

No fixture has been changed. This is recorded so the migration is one deliberate batch
rather than eleven independent edits.
