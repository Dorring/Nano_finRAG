# P1.8-B9 — applicable patch (NOT APPLIED)

Source-verified. Two edits, both to `src/runtime`, neither touching Store,
Benchmark, Gate, Binder, Validator, Retriever or Sidecar. Rollback is one
environment variable.

Path on the run host: `/disk/qh/nano-finrag/finquery_rag/backend/`

---

## Why these two edits and not more

Measured. The gate already knows the metric:

```
rag_v2/supervisor/semantic_alignment.py:469
    "cost_of_revenue",
    ( "cost of revenue", "cost of sales", "costs of revenue", "cogs", ... )
```

so `msft-016` and `pfe-030` are **not** gate-blocked. They reach retrieval and
fail at the operand guard, because the leg between gate and store is missing:

```
src/runtime/trusted_v2_canonical_store.py   METRIC_TO_CANONICAL
    -- has "net income", "operating income", ... but NOT "cost of revenue"
src/finance/concept_alignment.py            CONCEPT_ALIGNMENT
    -- has no "cost_of_revenue" quantity at all
```

Both must exist for `CanonicalFactStore.resolve_canonical` to fire. Neither
exists today. Two edits, and nothing else.

---

## Edit 1 — `src/runtime/trusted_v2_canonical_store.py`

In `METRIC_TO_CANONICAL`, add three entries:

```python
    "cost of revenue": "cost_of_revenue",
    "cost of sales": "cost_of_revenue",
    "costs of revenue": "cost_of_revenue",
```

## Edit 2 — `src/finance/concept_alignment.py`

In `CONCEPT_ALIGNMENT`, add one quantity:

```python
    # `CostOfGoodsAndServicesSold` first: it is the concept both Microsoft and
    # Pfizer use for the company-level total, verified against their filings.
    # `CostOfGoodsSold` and `CostOfRevenue` are the narrower taxonomies some
    # filers use instead; they follow, so a filer tagging only those still
    # resolves.
    "cost_of_revenue": (
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfGoodsSold",
        "us-gaap:CostOfRevenue",
    ),
```

## Edit 3 — the seam (required for the substitution to reach the guard)

As specified in `p1-8-b1-reviewed-diff.patch.md` §2–§3: the
`CanonicalCandidateSeam` wrapper and `_maybe_wrap_canonical`, gated by
`TRUSTED_V2_CANONICAL_CANDIDATES` (default off) and `TRUSTED_V2_IXBRL_STORE`
(explicit path, no default). Without this the two tables above are inert.

---

## Evidence each mapping is correct

Measured against the filings, not the store. A mapping is admissible only when
the filing states the concept as the company-level (`dimension_count == 0`)
total and that total is the value the question denotes.

```
Microsoft FY2025   us-gaap:CostOfGoodsAndServicesSold   87,831   dim 0
    question: "Microsoft's ... Cost of revenue"   gold 87,831        MATCH
    independent: 22,422 + 40,171 + 25,238 = 87,831, the segment rows

Pfizer FY2024      us-gaap:CostOfGoodsAndServicesSold   17,851   dim 0
    question: "Pfizer's Cost of sales"            gold 17,851        MATCH
```

## What this patch does and does not do

```
recovers      tv2f01-s1-msft-016, tv2f01-s1-pfe-030     two cases
              46/95 -> 48/95 = 50.5%                    projected, not measured
does not      touch the 22 gate-blocked cases            they name no metric
does not      touch the 21 class-A cases                 no eligible reading
does not      change behaviour with the flag off         byte-identical path
needs         an E2E run to confirm, and incorrect_release must read 0
```

**Projected, not measured.** The 48/95 follows from the two cases releasing; it
is not an observed result and should not be quoted as one until the E2E run
confirms it.

## Command to measure it

```bash
cd /disk/qh/nano-finrag/finquery_rag/backend
TRUSTED_V2_CANONICAL_CANDIDATES=on \
TRUSTED_V2_IXBRL_STORE=/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl \
python scripts/evaluation/run_p1_2_dual_track_benchmark.py \
    --track both --out-dir /tmp/g-seam
```

Requires the backend stopped (rule B: one PyTorch process per CUDA context).
The gating number is `incorrect_release`, not coverage.
