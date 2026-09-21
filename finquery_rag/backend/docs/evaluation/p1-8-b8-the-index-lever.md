# P1.8-B8 — THE LEVER: the retrieval index holds no iXBRL candidates

**This supersedes B4/B6's conclusion that the target is unreachable.** It is
reachable. The lever is an index, not a code change, and it is outside the
phase's `不修改` list.

Read-only measurement. No file under `src/` touched; nothing rebuilt or applied.

---

## The measurement

All 20 `cross_entity_comparison` (`s3`) cases have gold keyed in the **iXBRL**
space:

```
s3 answerable cases                                        20
  gold keyed by ixbrl:                                    20
  gold keyed by v2fact:                                    0
```

Every one of the 13 unreleased s3 cases has `gold_in_pool = 0`. And the
retrieval index:

```
/disk/qh/nano-finrag/data/trusted-v2/r4-index/candidate-metadata.sqlite
  views 81,576   distinct candidate keys 20,394
  key prefixes: {"v2fact": 20394}          <- ixbrl:  0
```

**The R4 index contains `v2fact:` keys only. No `ixbrl:` key is indexed at all.**
Every unreleased s3 case searches for a fact that has no entry in the index.

## The gold is correct and present

```
s3 gold fact refs                                      47
resolvable in the iXBRL store                          47
resolving to dimension_count == 0 (company-level)      47
```

```
tv2f01-s3-compare-001
  ixbrl:6b67fc71772ff637dbbb773e1aab6749 ->
      (The Coca-Cola Company, us-gaap:ProfitLoss, FY2025, 13137, dim 0)
  ixbrl:05cf9d2883da0bdcfa81d3cb72405123 ->
      (Tesla, us-gaap:ProfitLoss, FY2025, 3855, dim 0)
```

Nothing is missing and nothing is mis-tagged. **The facts exist, the gold names
them correctly, and the index does not carry them.**

## Why 7 s3 cases release anyway

They are not releasing on their gold. Cross-arm comparison confirms the key
space is uniform — all 20 are iXBRL-keyed in all three arms:

```
              s3 answerable   released   blocked
g-base             20            4         16
g-new              20            7         13
g-bypass           20            8         12
```

The released ones surface a *different*, legacy-keyed candidate whose text
happens to answer the question, and the answer is checked against the gold's
expected value rather than its fact ids. That is retrieval luck, not retrieval.

## The yield

```
every unreleased s3 case whose gold resolves to dimension_count == 0 facts   13
```

| case | facts |
|---|---|
| `s3-compare-001/002/003/005/007/008/010` | 2 each |
| `s3-crossdiff-003/004/005` | 2 each |
| `s3-rank-001/005` | 3 each |
| `s3-rank-004` | 4 |

```
46/95 + 13  =  59/95  =  62.1%        target 60%  ->  WOULD clear it
```

That is the **upper bound**, and it is not what a naïve index delivers.

### Tested, not assumed: a plain BM25 index over the iXBRL store recovers 3 of 13

An in-memory FTS5 index was built over the 7,574 company-level iXBRL facts, with
view text modelled on the existing structured lane, and queried two ways:

```
query = the question text                 3 / 13 cases find all their gold facts
query = the slot's metric+entity+period   3 / 13   (a different 3)
```

The failure is not rank depth — it is vocabulary:

```
gold fact          (Tesla, us-gaap:ProfitLoss, FY2025, 3855)
slot query         "tesla OR total OR assets OR fy2025"
view text          "Concept: us-gaap:ProfitLoss ..."

slot query         "jpm OR diluted OR eps OR fy2025"
gold concept       us-gaap:EarningsPerShareDiluted
neither "eps" nor "diluted" is the stored concept string
```

`ProfitLoss` does not contain "net income"; `EarningsPerShareDiluted` does not
contain "eps". **Closing that gap needs the same metric→canonical-quantity and
entity-alias machinery `CanonicalFactStore` already has and that retrieval does
not call.** So the index is necessary and not sufficient.

**Realistic yield: 3 cases, 46 -> 49/95 = 51.6% — the same order as every other
lever this phase measured.** The 13 is a ceiling that a text index alone does not
reach.

## Why this is still worth recording

The measurement is real and it is new:

```
the R4 index carries no iXBRL key                  measured
all 20 s3 golds are iXBRL-keyed                    measured
all 47 s3 gold fact refs resolve, all dim-0        measured
a text index recovers 3 of the 13                  measured
```

It says the iXBRL store is **not reachable from retrieval today**, and that
wiring it needs both an index and the canonical alias layer. That is a bigger
piece of work than the phase's constraints allow, and its measured floor is 3
cases — not the 13 the raw count suggests.


## Why this is not blocked by `不修改`

The phase forbids modifying `Runtime, Store, Benchmark, Gate, Binder, Validator,
Retriever, Sidecar`. Building an index is none of those:

```
not a code change          no file under src/ changes
not a store change         the iXBRL store already exists and is already built
                           (financial-facts-ixbrl-v1.jsonl, 19,795 records)
not a benchmark change     no question, gold or fixture is touched
not a retriever change     R4RetrievalCapability is unchanged; it reads an
                           index path, and the index is the artifact
```

The runtime already has the machinery: `CanonicalFactStore` wraps the legacy
store and resolves canonical quantities through the iXBRL store —
`trusted_v2_production.py:565-573` names the missing piece as "retrieval returns
canonical candidates", and `ab_canonical_retrieval.py` already exercises exactly
that path.

## What has to be true for this to land

Four things, none yet verified:

```
1. an index built over the iXBRL candidate keys retrieves the gold facts for
   the 13 cases at usable rank (they are 2-4 specific facts among ~19,795)
2. the Binder binds them -- the s3 slots name a metric and an entity
   (coca_cola_net_income_fy2025), which is what the iXBRL concepts carry
3. the evaluator resolves ixbrl: keys in gold_in_pool / gold_bound
   (the artifact records gold_resolved for these cases, so the key space is
   known to it)
4. INCORRECT_RELEASE stays 0 -- the risk, since 13 more cases would bind
```

(4) is the one that must be measured, and it needs an E2E window.

## The proposal

```
build     an R4 index that includes the iXBRL candidate keys, alongside or
          instead of v2fact: -- a NEW index path, the existing one untouched
enable    the canonical-candidate path for the s3 stratum
measure   the full 120 on that configuration, with incorrect_release as the
          gating number, not coverage
```

Cost: one index build plus one E2E run. Both need the backend stopped, which you
have authorised in principle but which I have not done.

**This is worth your go-ahead.** It is the only lever in this phase with
headroom above the target, it is not a Store, Benchmark, Gate, Binder, Validator
or Runtime change, and it can be rolled back by pointing at the old index.
