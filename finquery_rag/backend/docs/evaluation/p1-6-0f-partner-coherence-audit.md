# P1.6-0F — a comparison is only coherent if *both* entities are right

The P1.6-0D verification classified the **slots P1.6-0C had flagged**. But a
comparison, ranking or difference is coherent only when **every** entity in the case
carries the right quantity. Cases whose partner entity was never flagged therefore went
unchecked, and three of them are wrong.

This blocks the v5 → v6 migration: applying the verified values as they stand would fix
one side of several cases and leave the other side wrong, producing fixtures that look
regression-tested and are not.

## Per case, both entities

| case | flagged entity | partner entity | partner correct? |
|---|---|---|---|
| compare-001 | Coca-Cola `$5,678` | Tesla `$47,627` | **yes** — see below |
| compare-003 | Tesla `$13,292` | Coca-Cola `42,178` | yes (ko p87 consolidated income statement `Gross profit $ 42,178`) |
| compare-005 | Apple `$11,487` | JPMorganChase `$840` | **not checked** (jpm p147 `Current $ 840 $ 462`, a component-style line) |
| compare-009 | JPMorganChase `$4,520` | Apple `112,010` | yes |
| compare-010 | Microsoft `$5,424` | Apple `44,452` | **no** — see below |
| crossdiff-003 | Apple `9,683` | Tesla `$13,279` | incoherent (metric is `Total`) |
| crossdiff-004 | Coca-Cola `850` | Visa `19,602` | yes (v p74 balance sheet `Long-term debt 19,602 20,836`) |
| rank-002 | Tesla `$6,411` | Apple `(34,550)`, Microsoft `32,488` | **no — sign error** |
| rank-005 | Coca-Cola `16` | Apple `12,875`, Microsoft `0` | incoherent (different quantity types) |

## `compare-001` — the verified verdict is wrong, and so is the correction below it

> **Corrected by `p1-6-0f-coherence-pass.md`.** The analysis in this section concluded
> that `compare-001`'s gold is correct because both values are "United States"
> jurisdiction figures. The full pass found they are jurisdiction figures **of different
> concepts**: Tesla's `47,627` is US *revenue* (page 129, "revenues by geographic area")
> while Coca-Cola's `5,678` is US *pretax income* (page 103, "Income before income taxes
> consisted of the following"). The comparison is incoherent after all — not for the
> reason the verification gave, but incoherent. This section is kept for the record; the
> coherence pass supersedes it.

The question is *"Which company had a higher **United States** in FY2025, The Coca-Cola
Company or Tesla?"* and the stored metric is `United States`. Both golds are that
jurisdiction's figure:

```
ko   p103   United States $ 5,678  $ 2,499  $ 1,991   International 10,320 … Total $ 15,998
tsla p129   United States $ 47,627 $ 47,725 $ 45,235
```

`5,678` and `47,627` are the same quantity for the same scope, so **the comparison is
coherent and the gold is correct**. The verification's proposed `$15,998` is Coca-Cola's
*company total* — the answer to a different question. Applying it would break a case that
works.

This is the same shape as `rank-002`'s Tesla operand: a correct value sitting at an
ambiguous coordinate. The guard refusing it is still right; the gold is not wrong.

## `compare-010` — the partner is wrong too

`Other current liabilities` matches both a note's sub-line and the balance-sheet total:

```
aapl p43   Other current liabilities 44,452   44,024   Total other current liabilities $ 66,387
aapl p43   (same note) Income taxes payable $ 13,016 …
msft p71   Other current liabilities $ 5,424  $ 3,580        (within a lease note)
msft p40   Other current liabilities 25,020   19,185         (balance sheet)
```

Only Microsoft was flagged and corrected. **Apple's `44,452` is a component too** — the
balance-sheet figure is `66,387`. Fixing Microsoft alone would compare a total against a
component.

## `rank-002` — a sign error that flips the expected answer

This one overturns an earlier conclusion.

```
gold        Apple "(34,550)"          -> parsed as −34,550
expected_ranking  ["Microsoft", "Tesla", "Apple"]

aapl p27    Research and development $ 34,550   10 %
aapl p32    Research and development   34,550   31,370   29,915
msft p38    Research and development   32,488   29,510   27,195
tsla p58    Research and development $ 6,411    $ 4,540    $ 3,969
```

Apple's FY2025 R&D is **+34,550**, positive, on both the income statement and the
selected-financial-data page. The gold states it parenthesised, and `_decimal` reads
parentheses as negative.

With the correct sign the true ranking is **`Apple > Microsoft > Tesla`**
(34,550 > 32,488 > 6,411), not `Microsoft > Tesla > Apple`. So:

- the runtime's released answer `Microsoft > Tesla > Apple` is **wrong**, not
  coincidentally right;
- it was **scored correct** because the gold's `expected_ranking` was derived from the
  mis-signed value.

That is a false-release mask: a wrong answer marked correct by a defective gold. It also
means the earlier reading of this case — "a correct answer from an ungrounded operand,
so the guard's cost is zero" — is not right either. The operand was ungrounded *and* the
answer was wrong; the guard refusing it is correct on both counts, and the case should
be counted as a defect caught rather than a release lost.

`rank-002` was the case that motivated the grounded-release metric, so this is worth
noting: the metric caught the ungrounded operand, and this catches what the metric could
not see — that the gold itself was what made the answer look right.

## What this changes

The eleven-slot verdict list is **not sufficient input to the migration**. Before any
fixture is written, each case needs a coherence pass over every entity, and the
corrections are:

| case | action |
|---|---|
| compare-001 | **no change** — the verified verdict is wrong; the gold is correct |
| compare-003, compare-009, crossdiff-004 | proceed with the verified substitution |
| compare-005 | check JPMorganChase `840` before touching Apple |
| compare-010 | correct **both**: Apple `66,387` and Microsoft `25,020` |
| rank-002 | fix the **sign** on Apple (`+34,550`) and re-derive `expected_ranking` to `["Apple","Microsoft","Tesla"]` |
| crossdiff-003, rank-005, compare-004 | as proposed in `p1-6-0e-metric-replacements.md` |

No fixture has been changed. The migration stays frozen until this pass is signed off,
because the corrections above are not all of the same kind: two are value substitutions,
one is a sign fix that changes the ground-truth *answer*, and one is a verdict being
withdrawn.
