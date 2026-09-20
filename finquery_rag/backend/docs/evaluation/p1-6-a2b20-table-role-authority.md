# P1.6-A2B-20 — `table_role` at the resolver

One variable. The store gains `table_role` and `table_eligibility`; the resolver gains a
switch for which field decides that a table may speak for the company. Entity, metric and
period matching, the company-level column rule, the refusals, the fixture, the retrieval,
the Binder and production are untouched, and **the old behaviour is reproduced from the
same store** by asking for the old rule. Any difference between the two runs is
attributable to this one choice.

```
SEC HTML
  -> table_eligibility            (906 data tables / 592 page-layout scaffolds)
    -> statement_type + table_role
      -> AtomicFact
        -> Store V2
          -> 47-slot resolver
```

**Proof that nothing else moved.** The new store has 26,977 records and the old one has
26,977, with the *same multiset of `table_fragment_id`* — the record set is identical and
the only difference is the two added fields. Determinism still holds per filing.

```
  aapl_fy2025   Apple                  records   1095   authoritative   512
  jpm_fy2025    JPMorganChase          records  11713   authoritative   362
  ko_fy2025     The Coca-Cola Company  records   1936   authoritative   150
  msft_fy2025   Microsoft              records   1313   authoritative   297
  nvda_fy2025   NVIDIA                 records   1833   authoritative   352
  pfe_fy2024    Pfizer                 records   4500   authoritative   587
  tsla_fy2025   Tesla                  records   2069   authoritative   466
  v_fy2025      Visa                   records   2518   authoritative   626
```

## The gate

```
wrong_scope            0  PASS     resolved from a table with no authority
wrong_metric           0  PASS     resolving row label is not the metric
value_mismatch         0  PASS     disagrees with a value verified from the filing
dangerous_authority    0  PASS     resolved from an oracle-known dangerous table
scaffold_authority     0  PASS     resolved from page layout
```

The same checks under the old rule, so that "the new one is clean" is a comparison rather
than an assertion:

```
old rule:  wrong_scope 0    dangerous_authority 2
             compare-002 / Visa   from table_03d7f57a78224e363dfa1e97 (INCOME_STATEMENT)
             rank-001    / Visa   from table_03d7f57a78224e363dfa1e97 (INCOME_STATEMENT)
```

**The old rule promoted a table the oracle calls dangerous, twice.** `compare-002` and
`rank-001` are both Visa `Net income`, and the resolution came out of a table the oracle
adjudicated as a known-dangerous negative. The new rule refuses it.

## The re-decomposition

```
                     before                after
RESOLVED_COMPANY_LEVEL     10  ->   37
NO_COMPANY_LEVEL_FACT      29  ->    5
AMBIGUOUS_COMPANY_LEVEL     3  ->    0
NO_FACT                     5  ->    5
                          ---      ---
                           47       47
```

```
27  CAPABILITY_GAIN
19  UNCHANGED
 1  REFUSAL_RETYPED
 0  REGRESSION
 0  SAFER_FAIL_CLOSED
```

**No slot lost a resolution it should have kept.** The ten that were already resolved are
still resolved, with the same values.

### The two pre-registered transitions

Both landed exactly where they were predicted to, which is the point of having written
them down first:

```
compare-009  JPMorganChase  Net income     AMBIGUOUS -> RESOLVED   57,048
compare-007  JPMorganChase  Net income     AMBIGUOUS -> RESOLVED   57,048
crossdiff-004  Coca-Cola    Long-term debt AMBIGUOUS -> NO_COMPANY_LEVEL_FACT
```

JPMorganChase resolved because the competing candidates were statements that
`statement_type` could not order and `table_role` can. Coca-Cola moved to a refusal
because the old ambiguity was **a hedging note carrying `BALANCE_SHEET`** — the ambiguity
was the wrong authority, and removing it turns a confident-but-unfounded tie into an
honest refusal. As pre-registered, that is the fix working, not a regression.

### Values, spot-checked against the filings

Six values were verified against the filings earlier in this work and carried into the
resolver as `EXPECTED`. **Five of them are now reachable and all five match** — the sixth,
Apple's diluted earnings per share, is one of the `NO_FACT` slots below and was already
unreachable. Four more were checked directly against the raw HTML for this report:

```
TSLA  Net income           3,855      KO   Operating income      13,762
NVDA  Net income          72,880      V    Comprehensive income  20,614
```

All ten slots that were already resolved kept their values exactly.

## What is left, and why

```
compare-001  Coca-Cola    Net income               NO_FACT
rank-001     Coca-Cola    Net income               NO_FACT
compare-003  Coca-Cola    Total assets             NO_COMPANY_LEVEL_FACT
crossdiff-005 Coca-Cola   Total assets             NO_COMPANY_LEVEL_FACT
crossdiff-004 Coca-Cola   Long-term debt           NO_COMPANY_LEVEL_FACT
compare-007  Pfizer       Net income               NO_COMPANY_LEVEL_FACT
rank-004     Microsoft    Interest expense         NO_COMPANY_LEVEL_FACT
compare-005  Apple        Diluted earnings per share  NO_FACT
compare-008  Apple        Diluted earnings per share  NO_FACT
crossdiff-004 Visa        Long-term debt           NO_FACT
```

Three different causes, and **none of them is authority**:

- **`NO_FACT` (5), unchanged as pre-registered.** `table_role` is an authority field; it
  must not conjure a metric candidate that the metric matching never found. Coca-Cola
  states `Net Income Attributable to Shareowners of The Coca-Cola Company` and Pfizer
  states `Net income before allocation to noncontrolling interests` — both are the
  company's net income, and neither is the exact string the fixture asks for. This is
  `A_METRIC_ALIAS` from A2B-17 and belongs to metric canonicalization.
- **Coca-Cola `Total assets` / `Long-term debt` — the blocker below.** There is no
  authoritative Coca-Cola balance sheet in the store to resolve from.
- **Microsoft `Interest expense`.** Microsoft's income statement has no such row; the
  figure is disclosed in the notes. Refusing is correct.

## The blocker this experiment surfaced

**Nine of the 42 oracle-verified primary statements produce no store records at all.**

```
jpm_fy2025    BALANCE_SHEET   rows=38   tagged facts=108   table_0018c3ef848be10fdcd8b92e
jpm_fy2025    CASH_FLOW       rows=57   tagged facts=138   table_17bc704310122bf9b3b7192e
ko_fy2025     BALANCE_SHEET   rows=41   tagged facts=78    table_94851517fdc0fb6a1bd3c6ec
ko_fy2025     CASH_FLOW       rows=41   tagged facts=102   table_17ada59b3fed9f1095aea8d2
ko_fy2025     EQUITY          rows=47   tagged facts=108   table_e326ed2dfd397d8fee4ff547
nvda_fy2025   CASH_FLOW       rows=45   tagged facts=108   table_6c99239456fa18089655d5d5
pfe_fy2024    EQUITY          rows=38   tagged facts=126   table_9390c2c2c43ffbaf6b2e00b4
tsla_fy2025   CASH_FLOW       rows=55   tagged facts=111   table_0c3408f9ad8e47507745228b
v_fy2025      BALANCE_SHEET   rows=56   tagged facts=108   table_f219f7ebfed37b4355982043
```

It is **not caused by this change** — the same nine tables produce zero records in the
A2B-12 store, and the record set is identical. It is what caps Coca-Cola at two
authoritative tables out of five, and it is why three Coca-Cola slots above cannot resolve.

The loss is **after** the parse layer, not in it:

```
                        rows  cells   cells with a value   store records
aapl  balance sheet      42    504            237                100
ko    balance sheet      41    369            274                  0
ko    income statement   22    264            192                108
```

Coca-Cola's balance sheet yields *more* populated cells and *more* iXBRL anchors than
Apple's, and the store takes none of them. `build_canonical_fact_store` reports only 75
`missing_period` skips for the whole of Coca-Cola, so the cells are not being skipped at
the record filter — they are not reaching `atomic_facts` from `build_semantic_corpus` at
all. The exact filter is not identified here, and identifying it is a separate
investigation: this step was scoped to one variable and fixing a second would spend the
attribution.

## What this does establish, and what it does not

**Established.** The data-layer fix converts into resolver behaviour. 10 → 37 resolved,
every one of them from a table the oracle read and verified, with 0 wrong-scope, 0
wrong-metric, 0 value mismatch, 0 dangerous authority and 0 scaffold authority. Two of the
old resolutions came from a table the oracle calls dangerous and no longer do. Both
pre-registered transitions landed as predicted, including the one that is a *refusal*.

**Not established.** The 37 is a ceiling set by what the store can currently see, not by
authority: nine primary statements are invisible to it and three of the remaining ten
unresolved slots are blocked on that, not on anything `table_role` decides. And the
classifier's positive rule has still not been tested on a ninth filing — that stands, and
it is why the next state is `benchmark-validated`, not `production default`.

## Status

Resolver behaviour is unchanged except when `authority=AUTHORITY_TABLE_ROLE` is passed
explicitly; the default is still the old rule, so nothing in production moved. Evidence at
`artifacts/evaluation/p1-6-a2b20-store-v2-role/` — `store-v2.jsonl`, `table-roles.json`,
`authority-experiment.json`.

Next: **P1.6-A benchmark seal** on this store, then validation on a filing outside these
eight, then the production switch.
