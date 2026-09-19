# P1.6-A5 — canonical concepts make the whole stratum answerable

The measurement the switch decision rests on.  Every entity of all 20 re-derived
cross-entity cases, resolved through `concept_alignment` against the iXBRL of its filing.

```
entity resolution   RESOLVED 47 / 47   AMBIGUOUS 0   NO_COMPANY_LEVEL_FACT 0
case status         ANSWERABLE 20 / 20
expected-value      6 / 6 matched
```

Against the baseline this replaces:

| | store coordinates | canonical concepts |
|---|---|---|
| cross-entity cases usable | 7 / 20 | **20 / 20** |
| ambiguous coordinates | 1,433 / 7,887 (18.2%) | 0 |
| concepts absent outright | 3 cases (`Apple Operating income`, `Apple Total liabilities`, `NVIDIA Total assets`) | 0 |

The three cases the store could not answer at all — `compare-010`, `rank-005`,
`crossdiff-005` — resolve first time, and the two whose store values came from the wrong
table (`compare-003`'s Tesla total assets, `crossdiff-004`'s Coca-Cola long-term debt) land
on the balance-sheet figure.

## The candidate ordering is load-bearing, not cosmetic

Pooling the candidate concepts instead of applying them in order would leave **8 of the 47
entity resolutions ambiguous**:

```
Tesla  net_income            3794   NetIncomeLoss   parent-only
                             3855   ProfitLoss      consolidated

Coca-Cola net_income        13107   NetIncomeLoss
                            13137   ProfitLoss

Pfizer net_income            8031   NetIncomeLoss
                             8062   ProfitLoss

Tesla  comprehensive_income  4825   vs  4886
```

All four are the same shape: the parent-only and the consolidated figure sit at the same
period, both undimensioned, differing by the noncontrolling interest. Without the ordered
preference the coordinate would again hold two legitimate values, and the operand guard
would be right to refuse it — which is the defect this whole line of work has been
removing. The alignment's ordering is what makes the coordinate identify one value.

It is worth being explicit about what that means: the resolution is only as good as the
ordering, and the ordering is a **decision** (the benchmark asks about the consolidated
figure) recorded in `concept_alignment.py`, not a derivation. Changing it changes two of
the cross-entity golds, as noted there.

## What this does and does not establish

**Does:** the eight filings can be rebuilt as `(concept, dimensions, period, unit, value)`;
the canonical quantities resolve to one company-level fact per entity; and the 20-case
stratum is answerable against that.

**Does not:** that the system under test will answer them. This measures the *fixture's*
answerability, not the runtime's ability — the runtime still retrieves from the current
fact store, which is unchanged. Switching the store is the next step and a separate one,
which is why nothing here has been written.

Also not established: whether the canonical values agree with the *presented* tables in
every case. Six were checked against hand-read values and matched; the remaining 41
resolve to a single undimensioned fact but have not been read back out of a page. That is
the same standard applied to the earlier fixture work — resolution is not verification —
and it should be closed before the store is switched.

## Status

No store changed, no fixture changed, retrieval unchanged. Baseline archived at
`artifacts/evaluation/p1-6-a-scoping/coordinate-ambiguity-survey.json`; this measurement at
`artifacts/evaluation/p1-6-a5-canonical/canonical-ambiguity.json`.
