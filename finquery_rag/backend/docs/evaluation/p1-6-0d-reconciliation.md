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

## Not accepted as stated — three slots need different treatment

Three slots are not "wrong scope, substitute the company total". In each, the gold
values for the entities compared are **not the same kind of quantity**, and no row or
column choice makes them one. That is a fixture defect one level up from scope, and
the fix is to make the *metric* specific, not to find a better value.

### `compare-004` — a notional amount against a fair value

```
Q: "Compare Apple and Microsoft: which reported a larger Foreign exchange contracts
    in FY2025?"                       metric: "Foreign exchange contracts"

Apple      gold  62,647  aapl p42  "Derivative instruments designated as accounting
                                    hedges: Foreign exchange contracts 62,647"
                                    under  "The notional amounts of the Company's
                                    outstanding derivative instruments"
                                    -> a NOTIONAL AMOUNT

Microsoft  gold   (809)  msft p58  under  "Not Designated as Hedging Instruments"
                                           "Fair Values of Derivative Instruments"
                                    -> a FAIR VALUE
```

The two entities are being asked the same question and answering with different
quantities. The filing never states `171,726` (the proposed "company value" is
`62,647 + 109,079`, a sum of two categories); and a Microsoft total built as
`(809) + (44)` would be a sum of fair values, which still would not be a notional.
**Substituting values cannot make this case sound.** Its metric has to name the
quantity — *notional amount of foreign exchange derivatives* — so the same thing is
retrieved for every entity.

### `rank-005` — a notional amount against an OCI gain

```
Q: "Rank … by Interest rate contracts in FY2025: Apple, The Coca-Cola Company,
    Microsoft."                       metric: "Interest rate contracts"

Apple      gold  12,875  aapl p42  "The notional amounts of the Company's outstanding
                                    derivative instruments … Interest rate contracts"
                                    -> a NOTIONAL AMOUNT
Coca-Cola  gold      16  ko p84    table "Gain (Loss) Recognized in OCI"
                                    -> a GAIN in other comprehensive income
Microsoft  gold       0  (type not established)
```

`16` is not a mis-scoped `13,674`: an OCI gain and a notional principal are different
quantities. The verifier's `13,674` is the right *kind* of number to put beside Apple's
`12,875`, which confirms the metric must be re-specified — this is a fixture redesign,
not a scope correction.

### `crossdiff-003` — the metric is the word "Total"

```
Q: "What is the difference in Total between Apple and Tesla in FY2025?"
                                      metric: "Total"

Apple  gold   9,683   Total of the Federal grouping in the income tax note
Tesla  gold  13,279   Total of the accrued liabilities note
```

The question has no coherent subject. "Total" of what? The two golds are totals of
unrelated concepts, so the difference the case computes is meaningless, and the
verifier's proposed replacements (`$20,719` and `$54,941`) are a tax provision and
total liabilities — still unrelated. This case cannot be repaired by any value
substitution; it needs a metric, and probably a different question.

### The general defect

The pattern across all three is the same and it is worth naming, because it predicts
where else it will occur: **a metric string that is a generic word — `Total`,
`Current`, `Foreign exchange contracts`, `Interest rate contracts`, `United States` —
matches facts in several different tables, so the fixture retrieves a different
quantity for each entity.** `Foreign exchange contracts` exists in both a notional
table and a fair-value table; `Total` exists in every note.

So the migration has two different jobs, and conflating them would ship cases that look
repaired but are still incoherent:

| fix | slots |
|---|---|
| the metric is coherent; the gold took a component of it — **substitute the value** | compare-001, compare-003, compare-005, compare-009, compare-010, crossdiff-004 |
| the metric is coherent and the gold is already right — **no change** | rank-002 |
| the metric is incoherent across entities — **re-specify the metric, then re-pick every value** | compare-004, rank-005, crossdiff-003 |

`crossdiff-003` is the strongest case for retirement rather than repair, since no
single coherent quantity is visible in its question.

No fixture has been changed. This is recorded so the migration is one deliberate batch
rather than eleven independent edits — and so that "10 WRONG_SCOPE_GOLD" is not read
as ten interchangeable value substitutions, which is how it was reported.
