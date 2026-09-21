# P1.6-A3-3 — temporal-kind evidence scoping, 2×2

Diagnosis only. **No rule change.**

A3-2 traced why NVIDIA's and Tesla's cash flow statements classify every column `bucket`:
`_classify_column_temporal` reads `header_text + " " + cell_text`, and `_BUCKET_RE`'s
`rating` matches inside **ope·rating**. Two candidate causes, separable — so all four
combinations were measured rather than one being assumed.

```
scope   old   header_text + cell_text          the row prose is part of the kind
        new   column-local evidence only
regex   raw   as written
        safe  `rating`/`range`/`grade`/`tier` as whole tokens, not substrings
```

## The result

```
                     old_raw           new_raw            old_safe          new_safe
nvda CF  (failing)   bucket 18         bucket 18          category 3,       duration 18
                                                        duration 15
tsla CF  (failing)   comparison 3,     bucket 18          comparison 3,     duration 18
                     bucket 15                           duration 15
aapl CF  (control)   comparison 3,     non_temporal 3,    comparison 3,     non_temporal 3,
                     duration 15       duration 15        duration 15       duration 15
msft CF  (control)   bucket 1,         duration 13        segment 1,        duration 13
                     duration 12                          duration 12
```

```
nvda col 0   old_raw bucket('rating')   new_raw bucket('rating')   old_safe category('Other')   new_safe duration('2025-01-26')
tsla col 0   old_raw comparison('increase')  new_raw bucket('rating')  old_safe comparison('increase')  new_safe duration
msft col 0   old_raw bucket('rating')   new_raw duration('Year Ended')  old_safe segment('foreign')   new_safe duration
```

**The token-safe regex is the load-bearing change; input scoping as implemented is not.**

- **`new_raw` does not fix NVIDIA** — still `bucket 18`. Scoping alone changes nothing,
  because the string it scopes *to* is `column_headers[col]`, which is itself polluted
  (`'2025 / Cash flows from operating activities:'`). It makes Tesla *worse*: dropping the
  row prose removes the `increase` match, so the `rating` collision is what is left.
- **`old_safe` nearly fixes both** — NVIDIA to `duration 15`, Tesla to `duration 15`.
- **`new_safe` fixes both completely** — `duration 18`.
- **Both controls keep their `duration`** in all four variants, and Microsoft improves from
  `bucket 1` to `duration 13`.

## The caveat that matters

The scoping variant tested is the **weak** form: it uses the parser's own `column_headers`
string, which is already contaminated by the row-label column. So "scoping alone does not
help" is true **of this implementation**, not of the idea. A stronger column-local
definition — the raw header cells at that column, before the header path is assembled —
was **not** tested, and the A3-1a2 evidence says the pollution enters in the grid expansion
above it.

**So this pass settles the regex and does not settle the scope.** Reporting it the other way
round would have retired an architectural question on a weak implementation's evidence.

What the result does show about ordering: `new_safe` is the only variant that is clean on
the label columns (Apple's `comparison 3` and NVIDIA's `category 3` disappear), so scoping
is doing real work that the regex alone does not — just not the work that rescues these two
tables.

## The other collisions, now observed

Microsoft's column 0 is `segment('foreign')` under `old_safe` — **the second pattern A3-2
flagged as the same species, firing.** Apple's `comparison 3` is `Increase` in a row label.
Neither is a guess any more; both are measured.

## Status

Nothing changed. Evidence at `artifacts/evaluation/p1-6-a3-3-scoping/temporal-scoping.json`,
which carries all four variants per column for all five tables.
