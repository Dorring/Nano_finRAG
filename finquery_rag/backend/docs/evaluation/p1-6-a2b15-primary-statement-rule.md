# P1.6-A2B-15 — the primary-statement rule works; three ambiguities remain

Carrying `statement_type` into the store and keying company-level on it rather than on the
column name. Measured on 47 slots across the 20 cross-entity cases.

## Before and after

```
                             column-name rule   primary-statement rule
RESOLVED_COMPANY_LEVEL                  4                  10
NO_COMPANY_LEVEL_FACT                  40                  29
AMBIGUOUS_COMPANY_LEVEL                 0                   3
NO_FACT                                 3                   5
```

The rule change moved resolution from 4 to 10 and cut the dominant refusal from 40 to 29.
Then a second defect surfaced inside it and was fixed, taking ambiguity from 8 to 3.

## The ten resolved, all correct

```
compare-002  Visa          20,058     compare-010  Microsoft  128,528
compare-004  Microsoft    101,832     crossdiff-002 Microsoft 101,832
compare-006  JPMorganChase 65,214     rank-001     Visa        20,058
rank-002     Microsoft     32,488     rank-003     Microsoft  104,075
rank-003     JPMorganChase 65,214     rank-005     Microsoft  128,528
```

Every one matches a value verified against a filing earlier in this work, and every one came
from an `INCOME_STATEMENT` — which is the rule doing exactly what it was written to do.

## The defect the change surfaced

With the primary-statement rule in place, `compare-009 JPMorganChase Net income` went
**AMBIGUOUS with six values**, and the reason was the metric matcher:

```
_matches_metric("net income per share: Basic", "net income")   -> True
```

Word-boundary matching allows anything to follow the metric as long as a space does, so
`Net income per share`, `Net income attributable to …` and several segment rows all counted
as `Net income`. The primary statement holds all of them, so the slot saw six competing
company-level values.

Matching is now **equality after stripping a unit caption** — `Net income (in millions)`
matches, `Net income per share` does not. Ambiguity fell 8 → 3.

That is the second defect to arrive through this one function, after the substring bug that
resolved `Interest expense` to `Total noninterest expense`. **Both were found by an
acceptance check and neither by reading the code**, and both looked like a working resolver
from the inside. The docstring now says so.

## The three that remain, and why they are the real question

```
compare-007  JPMorganChase  Net income      ['18245','27761','4520','57048','6522']
compare-009  JPMorganChase  Net income      ['18245','27761','4520','57048','6522']
crossdiff-004 The Coca-Cola Company  Long-term debt  ['-705','11648','97']
```

For JPMorganChase, `57,048` — the verified figure — **is among the values**, and so are four
others. Several rows or several tables classified `INCOME_STATEMENT` state a different
number under the same label and period. So the ambiguity is no longer about naming; it is
about **which table is the consolidated statement**, and that is a question the store cannot
currently answer.

That is the honest next step, and it is worth stating what it needs: either a finer
statement classification than `INCOME_STATEMENT` (JPMorganChase reports the same statement
in more than one place), or a tie-break on table identity. Neither is a naming rule.

## What is established

- `statement_type` travels from the parsed table to the store record, and `UNKNOWN` (10,046
  of JPMorganChase's 11,713) is treated as "not a primary statement", never as "probably
  one".
- Keying on it resolves ten slots to values that match the filings, where the column-name
  rule resolved four.
- The resolver still refuses rather than guessing — 29 + 3 + 5 slots made no confident pick.

## Status

No fixture, denominator, legacy store, view, index or production behaviour changed. Store V2
is a new file and nothing reads it in production.
