# P1.6-A2B-11 — slot resolution against Store V2

Step 3, scoped to what one filing can demonstrate. Store V2 holds JPMorganChase, so **no
benchmark case is complete here** — every case needs its other entities too. What one filing
can settle is the slot, which is where the question has been all along.

## The result

```
ok   JPMorganChase  Net income                  FY2025
       -> 57,048  under 'Year ended December 31 / Total / 2025'
       (23 segment values present and not chosen)

--   JPMorganChase  Diluted earnings per share  FY2025  NO_COMPANY_LEVEL_FACT
--   JPMorganChase  Comprehensive income        FY2025  NO_COMPANY_LEVEL_FACT
--   JPMorganChase  Interest expense            FY2025  NO_COMPANY_LEVEL_FACT
```

`Net income` resolves to the firm figure with **23 segments present and rejected** — the
segment table's `Corporate 4,520` among them. That is `compare-009`'s question answered at
the retrieval layer: the legacy store holds both values under one coordinate and cannot
separate them; Store V2 separates them by the filing's own column header, and the resolver
picks the company one.

The three refusals are correct behaviour rather than failures. **No confident pick was
made**, which is the property that matters.

## What the refusals are

The company-level column is **not always headed `Total`**. In JPMorganChase's
`Selected income statement data` table the firm column is headed by the company name or by
the year, so a rule keyed on `Total | Consolidated | Firm` finds no company column and
refuses rather than picking one of the others:

```
Diluted earnings per share   20.02  under 'Financial performance of JPMorganChase / ...'
Comprehensive income       65,214  under 'Year ended December 31 / 2025 / Other comprehe...'
Interest expense           97,898  under '... / Revenue'  and  '... / Interest incom...'
```

Those values are the right ones — `97,898` is the interest expense verified in P1.6-0D,
`65,214` the comprehensive income from the canonical work — so the facts are present and
only the naming is short. Broadening the rule is the next step, and it is a design question
rather than a patch: "what heads a company column" has more answers than one, and a rule
loose enough to catch them all will start catching segment columns too.

## It found a real defect on the way

The first run resolved `Interest expense` to **95,640** — which is `Total noninterest
expense`. The metric was matched as a substring, and `interest expense` is a substring of
`noninterest expense`, so the slot resolved to a different line item with nothing to
indicate anything was wrong.

That is the same failure this whole line of work exists to remove — a confident answer from
the wrong evidence — arriving from a direction nothing had yet checked. Matching is now on
word boundaries.

It is worth noting that the acceptance framing caught it: a resolver that returned the best
available candidate would have looked successful here.

## What this does and does not establish

**Establishes:** given `(entity, metric, period)` and a filing whose firm column is named,
Store V2 resolves to the company's figure and rejects the segment values that share its
coordinate. The dimension `compare-009` needs survives from the source to the retrieval
layer and is used there.

**Does not establish:** that a benchmark case can run. Seven cases involve JPMorganChase and
every one of them needs other entities that this store does not hold. The scoped unit is the
slot, and the case follows the full rollout.

## Status

One new script. No fixture, denominator, legacy store, view, index or production behaviour
changed. Store V2 and the legacy store both untouched by this step.
