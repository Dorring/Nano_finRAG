# P1.7 — FINAL HANDOFF

Everything below is either a measurement or is marked as retracted. **Nothing in
this document is an inference from the P1.7 working session's intermediate
reasoning**; where a conclusion was proposed and later overturned, it is in the
last section and must not be built on.

This file is the only P1.7 state. Do not continue from the session's transcript.

---

## 1. Verified facts

### Frozen runtime, 120 cases, pinned plans, Gate ON

```
Release Coverage      50/95   = 52.6%
Released Accuracy     48/50   = 96.0%
Incorrect Release     0
Correct Refusal       25/25   = 100%
Citation Precision    92.4%   (60/72 = 83.3% if resolved by physical id)
Citation Recall       98.5%   (60/62 = 96.8% if resolved by physical id)
```

Both citation figures are reported on purpose. Physical ids name *where* a
number was printed; the store holds one quantity once per table that repeats it
(20,394 records for 11,657 logical facts), so the logical figure is the one that
measures what was cited.

```
Tests            5131 passed / 142 skipped
Runtime          frozen  (no change during P1.7's audit work)
Fail-closed      unchanged
Store            frozen, untouched
Benchmark        frozen, untouched
```

### Retrieval benchmark, 75 scorable cases (benchmark path, not E2E)

```
                  R@5        R@10       R@20
shipped RRF     50.7%      58.0%      65.3%
structured      85.333%    93.333%    95.333%
```

```
Reranker: benchmark-positive, production gain not demonstrated, DEFAULT DISABLED
```

Its production gain was not demonstrated because `PROMOTED_INTO_WINDOW = 0`:
every promotion landed inside the window the Binder was already reading.

### Seals

```
repo HEAD        beae6c76749460dfc247f59c7bb7f559d2809ade
branch           feat/nf-v3-interview-final

gold-evidence-v1.jsonl     3d2a0c5b7839656ce1414923ab90d03b844bc5c5fc17d82dc5e68224f466eb05
canonical-eval-v1.jsonl    227f0341d94ab8d4b9e7ee033feaa9b40e86ec32c3137a74d2931657e9281ece
plan-fixtures-v8.jsonl     7463b45a5bfc309d0a5ab29cdb15b3f070c13c8526ae5d1486ca67847dda362d

runtime fact store        0382c6a17f7065516c0edd0d2ca10d5ea6e0e0c7bb0dff68acae38134fc4db57
                          /disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl
```

---

## 2. Established architectural conclusions

### The gate is no longer the bottleneck — measured, not argued

Same pinned plans, store, retriever, binder, validator, finalizer; only the gate
changes:

```
                 GATE ON    GATE BYPASS
gate blocked        20           0
reached retrieval   55          75
gold in pool        52          71
gold bound          43          44
released         50 (52.6%)  51 (53.7%)
INCORRECT RELEASE    0           0
```

**Gate-only headroom ≈ 1 case.** Any admission policy caps at 53.7%.

### First-failure attribution, 45 unreleased answerable cases, `UNATTRIBUTED = 0`

```
SEMANTIC_ALIGNMENT   20      the gate
BINDING              17      gold in pool, not bound
VALIDATION            6
RETRIEVAL             1
SLOT_COMPLETENESS     1
```

The largest single loss is now `BINDING`: under bypass the pool holds gold for
71 of 75 and only 44 bind. That is where the coverage is.

### What was fixed during P1.7 (all four shipped, none a policy relaxation)

```
entity-less slot grounding    a slot with no entity was looked up as the empty
                              entity, so the source read as "no such row"   +3
BOUND status derivation       a provider claiming BOUND while naming its own
                              gaps had the whole response discarded, skipping
                              the targeted-slot repair loop                   +1
citation logical identity     citations resolved to the fact, not the page;
                              closed the citation-precision target
C0 ceiling re-measurement     the "52.6% ceiling" had been measured before the
                              two fixes above and was stale
```

---

## 3. Confirmed problem — the one that defines the next phase

`tv2f01-s2-sum-007` asks for **The Coca-Cola Company's** operating income across
FY2024 and FY2025.

```
CONSOLIDATED STATEMENTS OF INCOME          (Coca-Cola's own)
    Net Operating Revenues | $ | 47,941 | $ | 47,061 | $ | 45,754
    Operating Income       |    13,762 |     9,992 |    11,311

equity-method investee summary             ("A summary of financial information
                                            for our equity method investees")
    Net operating revenues  | $ | 102,800 | $ | 99,043
    Operating income        | $ |  13,426 | $ |  12,536

current gold:  13,426 + 12,536 = 25,962       <- from the investee table
Coca-Cola's own two-year sum:  13,762 + 9,992 = 23,754
```

Source: `raw_sec_html/KO/SEC_21344_000162828026010047/primary.html`
(`ko-20251231.htm`, period_end 2025-12-31).

**The Store has no way to say which table a row came from.** A note or investee
table's rows inherit the filing's `entity`, so:

- `entity == the question's company` **cannot** detect this class. Run across
  every comparable case with resolved gold it returns 40 consistent, 0
  mismatched — including this one. That result is the check being blind, **not**
  the attribution being right.
- The discriminator that does work is the tables' own scale: Coca-Cola's revenue
  is 47,941; the table the gold draws from reports 102,800.

The proposed correction is written up, with both source tables and its hashes, in
`benchmarks/tv2_canonical_v1/migrations/p1-7-0a-oracle-attribution.json` —
**status PROPOSED_NOT_APPLIED**. Two directions are defensible and are not
equivalent (re-point the gold at 23,754, or mark the case ambiguous and move it
to the abstention side); choosing between them is a benchmark decision.

### The population, measured without any classifier

```
MULTI_VALUE       32    the gold's coordinate holds several values in the store
SINGLE_VALUE      43
NO_FACTS          45    (abstention cases, no fact ids)
```

**32 of 120.** This is a *risk set* needing source adjudication. It does **not**
say the gold is wrong in any of them — only that the coordinate does not
determine the answer.

---

## 4. RETRACTED — DO NOT USE

Each of these was produced during P1.7, looked supported, and was overturned by
later evidence. Do not carry them forward.

```
- GOLD_WRONG_TABLE = 68/120
    Invalid. The classifier's _PRIMARY accepted only captions matching
    "consolidated statements of ...", so every legitimate product, segment and
    MD&A table was flagged. 68 was its false-positive rate.

- "52.6% is the hard ceiling for any gate policy"
    Obsolete. Measured before two bug fixes landed. Current figure is 53.7%.

- "entity consistency 40/40 proves the attribution is right"
    False. It proves the check is blind to this class. The wrong table inherits
    the right entity.

- "a table is the filer's own only if it is a consolidated statement"
    False. Apple's iPad revenue (28,023) sits in a 'Products and Services
    Performance' table and is entirely Apple's.

- "table totals matching the filer's totals can decide ownership"
    Circular / insufficient. The filer's total comes from the store, which is
    the thing that has already mis-attributed the note table; and legitimate
    tables such as Apple's product table state no total at all.

- "the 60.7% representation ceiling is the achievable result"
    Not established. It assumes the twelve recoverable cases' golds are correct;
    at least one (sum-007) demonstrably is not.

- "the 32 MULTI_VALUE cases are benchmark defects"
    False. They are a risk set. Defects confirmed by source: 1.

- "the sidecar failed / did not converge"
    Mis-stated. The sidecar's table attribution for both Coca-Cola rows is
    CORRECT against the source. What failed was the runtime wiring, and the
    repeated belief -- mine -- that the gold's 13,426 was Coca-Cola's number.
```

---

## 5. Next phase — P1.8 Benchmark Source-Truth Closure

Do not start a benchmark migration or a Structural Context Sidecar before this.

```
Read this handoff. Independently verify the frozen baseline and the hashes
above. Re-verify sum-007's two source tables. Then define the

    BENCHMARK AUTHORITY CONTRACT

covering: company-level questions; scoped disclosure questions; multi-value
coordinates; SOURCE_AMBIGUOUS; logical-table authority.

The question it must answer:

    when one filing reports several real values for the same entity / metric /
    period, what source-level evidence decides which value is eligible to be
    that question's gold?

Do not modify Runtime, Store, Benchmark, Gate, Binder, Validator, Retriever or
Sidecar until that contract exists. Do not use "reaching 60% coverage" as an
adjudication criterion. Do not inherit any retracted hypothesis above.
```
