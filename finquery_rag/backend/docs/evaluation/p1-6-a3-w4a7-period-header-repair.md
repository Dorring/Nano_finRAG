# P1.6-A3-W4-A7 — the period-header predicate, replaced rather than retuned

Diagnosis and one fix. The fix is in `period_binding_shadow.py::_is_period_header_cell`.

```
python verify_period_header_repair.py --out artifacts/evaluation/p1-6-a3-w4a7-repair
python emission_admission_shadow.py   --out artifacts/evaluation/p1-6-a3-w4a7-shadow
python audit_valid_losses.py --delta .../p1-6-a3-w4a7-shadow/emission-admission-shadow.json ...
```

## What was replaced

```
date extracted + remaining prose <= 12 characters  ->  a period header
```

W4-A6 falsified the second half: it refused 876 source-grounded cells, `For the Year Ended
September 30, 2025` among them. The twelve had no semantic content to begin with, so
widening it to 20 or 50 would only move the same arbitrary line. It is gone, and what
replaces it asks the question the length rule was approximating:

```
what follows the date   must be nothing but a units caption, or the cell is a sentence
                        that mentions a date
what precedes it        must *introduce* a period -- `For the Year Ended`, `held at`,
                        `as of`.  `... Chief Executive Officer February 20, 2026` does
                        not introduce one
and not an event        a present participle before the locator makes the date the date
                        of a happening, not a reporting point
and not an aside        a bracket opened before the date must contain a locator exactly
```

The last two are the part a length rule can never reach, and they are why the fix is not
just a bigger number:

```
financial instruments held   at  Dec. 31, 2025      a state at a time
PSUs                vesting  on  March 25, 2026     an event that happens

Age (at December 31, 2025)                          a period
... (as previously reported as of Dec. 31, 2023)    a qualification
```

Same shape either way. The oracle is the two family lists W4-A6 read out of the filings —
ten positives and nine negatives, copied from real cells, kept together in
`tests/test_period_binding_shadow.py` precisely so a predicate that passes the positives by
accepting everything fails the negatives.

## The repair against the audit

```
                            before   after
cells audited                 1649     749    -900
A  V2_PRODUCER_GAP             876       0    -876
B  LEGACY_FALSE_POSITIVE        28      37      +9
C  VALID_BUT_OUT_OF_SCOPE      171     171       0
D  SOURCE_AMBIGUOUS            574     541     -33
```

**`V2_PRODUCER_GAP = 0`.** The entry predicate W4-B was gated on is met.

**B and C held, which is the part that matters as much.** B's nine new cells are the two
families the repair now refuses and the audit had never seen before — `Date: October 31,
2025` and `Dated: February 27, 2025`, which the old predicate bound as periods. They are
the date line of the same signature block that `President and Chief Financial Officer
February 20, 2026` is the name line of, so they are adjudicated into that family, and the
extension of the table is recorded rather than folded in.

**One family moved out of D by binding, and it is worth naming.** `Summary of TSRU and PTU
information as of December 31, 2024 (a), (b) :` — 30 cells filed as ambiguous — is now a
period the producer accepts. It is lexically identical to `Fair value purchase price
allocation as of May 1, 2023`, which is class A, so no deterministic predicate can separate
them: my filing one as A and the other as D was a judgement about the note, not about the
cell. It is bound now, and the judgement that it should not be is not one a predicate can
carry.

## The change against the binder

```
                            before   after
unchanged_admitted           18923   19925    +1002
producer_loss                16785   15783    -1002
net_loss (stored regression)  8369    7367    -1002
rule_gain                    14802   14802        0
producer_gain                    0       0        0
unchanged_withheld           13740   13740        0
```

The prediction was +876/-876. It came out **+1002/-1002**: the repair also bound 519 columns
that had no binding at all, and cells in those columns move with it. Same shape, larger than
the single cause predicted.

## Monotonicity — not met, and every violation is a refusal

```
columns newly bound            519
columns unbound by the repair   27
columns whose period CHANGED   645
columns unchanged             5700
row bindings lost / gained     0 / 0
```

**27 columns that bound before the repair do not bind after it.** The first version was 30,
and three of those were a bug in the repair itself — an unclosed-bracket rule that refused
`Age (at December 31, 2025)`, a legitimate period expression. The verification found it;
that is what the verification is for.

The remaining 27 are two families, and neither is recall lost:

```
 9  `Date: October 31, 2025` / `Dated: February 27, 2025`    a signature date -- the old
                                                             predicate bound it as a period
18  `Balance, January 1, 2022` / `Nonvested, December 31,     a *row label* the legacy
    2023` / `Outstanding, December 31, 2023`                 header selector treats as a
                                                             header (the W4-A3 geometry)
```

Refusing both is, on the reading in this document, correct — a signature date is not a
reporting period and a row label is not a column header. **But that is a reading, and the
gate said strict monotonicity.** So it is reported rather than assumed, and W4-B does not
proceed on the strength of it.

The 645 period changes are a different kind of movement and are not refusals at all: they
are `2025 -> 2025-12-31`, `2024 -> 2024-09-30` — a column that bound at YEAR granularity
now binding at DAY, because the repair makes more month-day cells eligible for the
`pending` join. That is a refinement on 645 columns, and it is the *second-order* effect of
the fix: the repair changed which cells set `pending`, which changes what rule 2 joins.
It has not been verified column by column, and it should be before W4-B.

## State

```
V2_PRODUCER_GAP      = 0      met
UNEXPLAINED_VALID_LOSS = 541  all registered as ambiguous, none absorbed
strict monotonicity          not met: 27 refusals, 645 refinements
```
