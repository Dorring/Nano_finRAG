# P1.6-0E — replacement metrics for the three incoherent cases

Three of the eleven contested golds are not mis-scoped. Their **metric string matches
different quantities in different tables**, so each entity's gold is a different kind of
number and no value substitution repairs the case (see `p1-6-0d-reconciliation.md`).
This proposes a replacement metric for each, with every value quoted from the filing.

The rule a replacement must satisfy: **name one quantity that every entity in the case
actually reports at company level.** Where a value has to be derived rather than quoted,
that is stated explicitly — a constructed number presented as source truth is how these
cases went wrong in the first place.

---

## `rank-005` — `Interest rate contracts` → `Interest rate contracts notional amount`

The old metric matched a notional amount for Apple and an OCI gain for Coca-Cola.

| entity | value | source |
|---|---|---|
| Apple | `12,875` | aapl p42, under `Derivative instruments designated as accounting hedges` |
| The Coca-Cola Company | `13,674` | ko p83, "The total notional values of derivatives that were designated and qualified as fair value hedges of this type were **$13,674** million and $12,628 million" |
| Microsoft | `1,150` | msft p58, notional table, `Interest rate contracts purchased 1,150` under `Designated as Hedging Instruments` |

**All three are quoted, not derived**, and all three are the *designated/hedging* IR
notional — Apple's line sits under "designated as accounting hedges", Coca-Cola's under
"designated and qualified as fair value hedges", Microsoft's under "Designated as Hedging
Instruments". The scope is consistent.

Expected answer: `The Coca-Cola Company > Apple > Microsoft` (13,674 > 12,875 > 1,150).

**Caveat to confirm:** Coca-Cola also uses cross-currency interest rate swaps. Confirm
`13,674` is the interest-rate-swap notional and that cross-currency swaps are disclosed
separately, so the metric does not silently include them.

---

## `crossdiff-003` — `Total` → `Total liabilities`

The old metric was the literal word `Total`, which matched a Federal tax sub-total for
Apple and an accrued-liabilities total for Tesla. The question — *"What is the difference
in Total between Apple and Tesla in FY2025?"* — has no coherent subject.

| entity | value | source |
|---|---|---|
| Apple | `285,508` | aapl p34 (consolidated balance sheet), `Total liabilities 285,508 308,030` |
| Tesla | `54,941` | tsla p68 (consolidated balance sheet), `Total liabilities 54,941 48,390` |

**Both are quoted as a single line on the balance sheet**, at identical scope, with no
derivation. This is the cleanest of the three.

Question becomes: *"What is the difference in Total liabilities between Apple and Tesla in
FY2025?"* Expected answer: `230,567` (285,508 − 54,941).

---

## `compare-004` — `Foreign exchange contracts` → `Foreign exchange contracts notional amount`

The old metric matched a notional amount for Apple and a *fair value* for Microsoft
(`Fair Values of Derivative Instruments`, `Designated as Hedging Instruments`).

| entity | value | source |
|---|---|---|
| Apple | `171,726` | aapl p42: designated `62,647` + not designated `109,079` |
| Microsoft | `58,521` | msft p58 notional table: purchased `15,214` + sold `43,307` |

**Both values are derived sums, and this is the one case where that is unavoidable.**
The two filings break the same quantity along different axes — Apple by hedge
designation, Microsoft by direction — and neither states a combined total. So there is no
common *stated* line.

That does not make the case unusable, but it changes what it tests: the question must
explicitly ask for **total** notional across all contracts, and the derived value must be
recorded as derived. If a constructed total is not wanted, this case has no repairable
metric — the alternatives are to re-pose it over a quantity both filings state as one
line, or retire it.

Expected answer: `Apple > Microsoft` (171,726 > 58,521).

**Caveat to confirm:** both filings may disclose additional FX derivative categories
(e.g. cross-currency swaps) that belong in the total but sit outside these two tables.
Confirm the two addends are the complete set for each company before fixing the value.

---

## Summary of the migration

One batch, three kinds of edit. Every value below was read from the filing at the cited
page; nothing is applied to any fixture until this is confirmed.

### 1. Metric unchanged — substitute the company-level value (6 slots)

| case | entity | gold | → company value | source |
|---|---|---|---|---|
| compare-001 | The Coca-Cola Company | `$ 5,678` | `$ 15,998` | ko p103 |
| compare-003 | Tesla | `$ 13,292` | `$ 17,094` | tsla p129 / p57 |
| compare-005 | Apple | `$ 11,487` | `$ 22,058` | aapl p43 |
| compare-010 | Microsoft | `$ 5,424` | `$ 25,020` | msft p71 |
| crossdiff-004 | The Coca-Cola Company | `850` | `$ 42,119` | ko p76 |
| compare-009 | JPMorganChase | `$ 4,520` | `$ 57,048` | jpm p341 |

`compare-009` is included: its verdict was settled at column level in P1.6-0D (gold sits
under the `Corporate` column, the company figure under `Total`), so it is the same kind
of edit as the rest.

### 2. Metric unchanged — no edit (1 slot)

| case | entity | value |
|---|---|---|
| rank-002 | Tesla | `$ 6,411` — confirmed company-level R&D on both the income statement (p70) and MD&A (p58) |

### 3. Metric re-specified — every value re-picked (3 cases)

| case | metric | values |
|---|---|---|
| rank-005 | `Interest rate contracts` → `Interest rate contracts notional amount` | Apple `12,875` (unchanged) · Coca-Cola `13,674` · Microsoft `1,150` |
| crossdiff-003 | `Total` → `Total liabilities` | Apple `285,508` · Tesla `54,941` |
| compare-004 | `Foreign exchange contracts` → `Foreign exchange contracts notional amount` | Apple `171,726` · Microsoft `58,521` *(both derived)* |

Note that these differ from the verifier's proposed values in two places, because the
metric changes and the value must follow it:

- **crossdiff-003 Apple**: the verifier proposed `$ 20,719` (the income-tax provision),
  which is coherent for Apple but has no counterpart in Tesla. Under `Total liabilities`
  the value is `285,508`.
- **rank-005 Microsoft**: the gold is currently `0`; under the notional metric it is
  `1,150`. Apple's `12,875` happens to be already correct, but for the new reason.

### Denominator

The cross-entity stratum stays at its full 20 cases. Nothing is dropped: the eleven
contested operands are resolved rather than excluded, and the three re-specified cases
remain gradeable once their metric names one quantity. This is a corrected 20, not a
shrunk one.

### Provenance of each value — what has been re-read, and what has not

These are not all equally established, and the migration should treat them so:

**Re-read directly from the filing during this work** (the block or line was quoted in
this session): compare-001 `$15,998`, compare-003 `$17,094`, compare-005 `$11,487` and
its `Federal / State / Foreign` components summing to `22,058`, compare-009 `$57,048`,
crossdiff-003 Apple `9,683` → and, for the new metric, `285,508` / `54,941`,
rank-005 all three, compare-004 both sides, rank-002 `6,411`.

**Taken from the verification report and not independently re-read**: compare-010
Microsoft `$25,020`, crossdiff-004 Coca-Cola `$42,119`. Both cite a balance-sheet page
and are plausible, but they are the only two values in the migration resting on a single
reader. Re-read those two pages before applying them.

**Caveats outstanding on the re-specified cases** — confirm before fixing values:
Coca-Cola's `13,674` excludes cross-currency swaps; the two addends in `compare-004` are
the complete set of FX derivative categories for each company.

