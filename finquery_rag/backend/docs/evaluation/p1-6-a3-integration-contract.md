# P1.6-A3 — integration contract: period, temporal kind, admission

Short by design. It exists to stop one failure before A3-3 runs: a legal, fully traceable
period being silently dropped because the source did not also state a temporal shape.

## Three responsibilities, currently two

```
PeriodBindingV2      source truth about WHEN
TemporalKind         source truth about temporal semantics
EmissionAdmission    whether the fact is specified enough to enter Store V2
```

`TemporalKind` and `EmissionAdmission` are one thing today, and the rule

```
temporal_kind not in {point, duration, comparison}  ->  drop the fact
```

is where they were welded together.

## What that costs

Coca-Cola's equity statement, settled in A3-1d:

```
normalized_period = 2025
granularity       = YEAR
temporal_kind     = YEAR / UNRESOLVED      the source states no `years ended`
provenance        = complete               source cell named, scope named, method named
```

Under the welded rule this fact is dropped, and dropped **for the same reason a table of
junk is dropped** — which is wrong twice over. It is a real fact with a real, checkable
period; and dropping it is extraction performing a refusal, which is the thing P1.6 has
spent its whole length separating out.

## The boundary

```
a fact enters Store V2 when its period is sufficiently specified to be USED
a fact is USED only when the consumer's period is COMPATIBLE
```

```
YEAR(2025)   usable for  FY2025        not usable for  2025-12-31
DAY(2025-12-31)  usable for  FY2025, 2025-12-31
UNRESOLVED period                        enters only as a fact with no period claim
```

Admission asks **is this specified enough to be true**. The resolver asks **is this
compatible with what was asked**. Refusal belongs to the second, and only there.

## What a fact must carry

`AtomicFact` keeps the provenance rather than flattening it, or the work of A3-1 is lost
one layer down and Store V1's problem recurs:

```
normalized_period
period_granularity
period_binding_method
period_source_cells
period_target_scope
temporal_kind
```

## Consequences for A3-3

A3-3 changes `TemporalKind`, and under this boundary **it cannot change what is admitted**.
Correcting NVDA's and Tesla's `bucket` will not, on its own, put a single fact into the
store — admission has already moved off `temporal_kind`. That is the intended shape: the
two repairs stay separately attributable, and A3-3 cannot re-drop Coca-Cola's equity
statement on its way through.

## Status

Design only. Nothing implemented; A3-3 is diagnosed and shadowed next.
