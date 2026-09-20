# P1.6-A3-W1 — fresh-rebuilt baseline at `8ef4c45`

Recorded once, at the last point where the boundary is clean:

```
W1  contract introduced, no producer consumes it      <- here
W2  first time the new contract starts carrying real data
```

The earlier W1 evidence was an import assertion plus a diff test that would have passed for
the wrong reason in any checkout without the commit. That is not a baseline to start
populating a contract against, so the pipeline was **rebuilt from the source**, not compared
against the stored artifacts.

```
build_store_v2.py --apply          fresh
evaluate_table_role_authority.py   fresh
seal_table_authority_benchmark.py  recomputed, not restated
```

## The numbers

```
Store V2 records              26,977
47 slots   RESOLVED 37   NO_COMPANY_LEVEL_FACT 5   NO_FACT 5   AMBIGUOUS 0

wrong_scope 0   wrong_metric 0   value_mismatch 0
dangerous_authority 0   scaffold_authority 0
under the old rule, dangerous_authority 2     (unchanged)
```

## Hashes, byte-identical to the sealed artifacts

```
ed43ddba4678fcd98ec5253d21e830ce211c10f183512a93e9cba076b4f8e8ea  store-v2.jsonl
ca29cd40e95d0943e53a68d3148f06b3c433b5fce739bb113cf2e6f345d64e19  table-roles.json
```

```
seal: no drift: every sealed number was recomputed and matched
```

A rebuild from source reproducing the sealed bytes is stronger than the seal re-reading its
own artifacts: it shows the frozen numbers come from the code and the corpus, not from the
files having been left alone.

## What this is for

From here, a dual-output delta in W2 or W3 can be stated against **the `8ef4c45`
fresh-rebuilt baseline** rather than against an older artifact assumed unchanged.

W2 and W3 do not need a full rebuild each: while the old period path is still the only
authoritative one, dual-output tests and reachability guards carry the weight. The full
system regression belongs at W4 and W6, where admission and evaluation authority actually
change.
