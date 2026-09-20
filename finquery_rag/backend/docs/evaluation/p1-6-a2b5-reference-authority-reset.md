# P1.6-A2B-5 — Reference Authority Reset

A decision record, not a measurement. It exists because the last several steps kept
treating an artifact as a standard, and that artifact turns out to have no producer, no
parsed source and no reproducible corpus behind it.

## What was happening

Every comparison in P1.6-A2B was implicitly asking *"does this match the store?"* — the
1,543-vs-7,322 gap, the schema mismatch, the urge to find the missing producer. That framing
treats `financial-facts.jsonl` as the definition of correct, and it is not, because:

```
the parsed documents it was built from     absent
the corpus that can reproduce them         absent   (the on-disk corpus yields 0 facts)
the code that built it                     absent   (no file in the tree contains
                                                     `citation_origin` or
                                                     `deterministic_physical_source_descriptor`)
```

An artifact with none of its inputs and none of its producer available cannot define what a
new pipeline ought to produce. Chasing it would have kept producing the same error in a
new form: **numbers near the legacy store read as good.**

## The authority, stated once

```
AUTHORITATIVE
  SEC primary.html + Inline XBRL + the document's own DOM/table structure.
  The filing states its own tables, rows, columns and tagged facts; nothing
  here infers them.

DERIVED AND AUDITABLE
  parsed cells -> atomic facts -> Canonical Fact Store V2.
  Every step is in this repository, re-runnable, and checkable against the
  authoritative layer.

LEGACY COMPATIBILITY ARTIFACT
  the current financial-facts.jsonl, frozen below. Read-only.
  It supports exactly two things: historical metric comparability, and replay
  of the old system's behaviour. It does not define correctness for anything new.

FORBIDDEN
  using the legacy store's record count, schema or vocabulary as a correctness
  oracle for Store V2. Not its totals, not its field set, not its fact_type
  names, not "close to the old number" as a success signal.
```

The test to apply to any future claim: **can this be checked against the filing?** If the
only thing supporting it is agreement with the legacy store, it is not supported.

## Frozen identities

The legacy artifact and the benchmark that was measured against it, pinned so that
historical comparison stays possible without either being mutated:

```
financial-facts.jsonl
  sha256  0382c6a17f7065516c0edd0d2ca10d5ea6e0e0c7bb0dff68acae38134fc4db57
  size    50,469,576 bytes
  mtime   2026-09-09T06:10
  records 20,394
  fields  34
  fact_type  atomic 9,316 | narrative 6,516 | row_matrix 3,715 | bucket 438 | comparison 409

plan-fixtures-v8.jsonl    7463b45a5bfc309d0a5ab29cdb15b3f070c13c8526ae5d1486ca67847dda362d
gold-evidence-v1.jsonl    3d2a0c5b7839656ce1414923ab90d03b844bc5c5fc17d82dc5e68224f466eb05
canonical-eval-v1.jsonl   227f0341d94ab8d4b9e7ee033feaa9b40e86ec32c3137a74d2931657e9281ece
```

The store's hash matches the `fact_store_sha256` already recorded in the fixture manifest,
so the pin is consistent with what the benchmark was measured against.

**Neither deleted nor modified.** It stays as it is, read-only, and the search for its
producer stops here — not because the question is uninteresting, but because the answer
cannot change what is correct.

## The benchmark, meanwhile

Two tracks, kept apart rather than merged:

```
Legacy track          the orphan store, frozen fixtures, historical comparability
Source-grounded track SEC source + audited gold + Store V2, decides trusted correctness
```

Store V2 does **not** replace the benchmark store now. That happens once source-grounded
audit covers enough cases to migrate deliberately — and the migration will be a decision
with its own gate, not a side effect of a rebuild.

## What this buys

Stated as the engineering position it is:

> The legacy benchmark store cannot be traced to its source, so it was not used as a
> reconstruction oracle. It is frozen for historical comparison, while a reproducible
> Canonical Fact Store V2 is built from the filings' own SEC Inline XBRL and migrated
> against source-grounded, audited cases.

That is a stronger position than reproducing a JSONL whose generation is unknown, and it is
the one this project can actually defend.

## Next

**P1.6-A2B-6** — diagnose the current repository's pipeline, with the question reframed:

> Why do rows that are plainly financial in the SEC HTML — `Net income`, `Revenue`,
> R&D — fail to become `metric_row`, and why do 645 of 679 tables get no statement
> section?

Answered against the filing, not against any store. The architecture suspicion to check
first is whether `metric_row` is gated on `section_type != UNKNOWN` — because "I do not know
which statement this is" is not the same claim as "this is not a financial row", and a row
reading `Net income | 57048 | 58471 | 49552` is a numeric metric row whatever table it
came from.
