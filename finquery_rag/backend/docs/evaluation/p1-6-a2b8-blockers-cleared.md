# P1.6-A2B-8 — both blockers cleared, and one of them was not a blocker

## Blocker 1: the duplicated `table_id` — real, and fixed

The cause was an lxml proxy-identity bug:

```python
order = {id(n): i for i, n in enumerate(root.iter())}   # one pass, keyed by id()
...
o = order.get(id(n), 0)                                  # a second pass, looked up
```

lxml creates element proxies on demand and frees them, so `id()` is **reused**. The second
`root.iter()` does not yield the same objects, and the lookup could return another
element's index — or the `0` default. Two different tables then derived the same
`table_id`, which is where the three duplicates and the downstream
`duplicate canonical candidate key` came from.

One pass, `o` taken from it directly:

```
             before   after
JPM          679 blocks, 676 distinct table_id, 3 duplicated
             679 blocks, 679 distinct table_id, 0 duplicated
MSFT         1,135 blocks, 0 duplicated block_id
```

And the hop it was blocking now completes:

```
C  store records           11,819
   with column_header      11,819   (100%)
   with ixbrl_anchors       6,417
   distinct candidate_key  11,819   <- unique
```

So A→B→C is intact end to end: 13,682 cell anchors → 11,993 atomic facts, all carrying
`column_header` → 11,819 store records, keys unique.

## Blocker 2: the 54% anchor coverage — not a defect

I had this down as something to clear. It is the filing's own tagging density, and the
evidence is exact:

```
HTML    ix:nonFraction 7,961  +  ix:nonNumeric 481  =  8,442
parse   ixbrl_fact_count                            =  8,442
```

**The parse captures every tag in the document.** So a cell without an anchor is a cell the
filer did not tag, not one the parser missed.

That the unanchored records are real financial rows is true and does not contradict it —
they are MD&A summaries and highlights:

```
Interest-earning assets / Trading assets …   351
Total net revenue                             92
Provision for credit losses                   92
Net interest income                           73
Net income                                    71
column headers: 'Selected income statement data', 'Selected metrics / As of …'
```

SEC filings tag the primary statements and the notes; the MD&A tables that repeat those
numbers usually carry no tags. So the coverage figure measures the filer's tagging, and
`ixbrl_anchors` is at **100% of tagged cells** — which is the claim that is actually
available, and the one worth making.

The same reading applies to the earlier MSFT figure: 1,710 anchors resolving 1,710, against
1,829 emitted facts whose 119 uncovered are 29 header metadata facts and 90 `...TextBlock`
wrappers.

## What this means for the anchor field

`ixbrl_anchors` is a **provenance link where the filer supplied one**, not a coverage
metric. Anything reading it must treat absence as "untagged", never as "not a fact" — the
untagged cells above are ordinary financial rows.

That is worth stating in the code, because a consumer that filters on the field's presence
would silently drop `Net income` from the MD&A.

## Status

Two source changes: the single-pass `make_blocks`, and the earlier `row_index`. No fixture,
denominator, store, view, index or production behaviour changed. Legacy store frozen.
