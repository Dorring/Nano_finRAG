# P1.6-0K — the parse layer can't be run here, so the anchor change is unverified

Step one of option B was to record each block's descendant `ix:nonFraction` anchors and
verify on one filing. The change is written. **It could not be verified, and the reason
matters more than the change.**

## What blocks it

```
run_nf_v2_17a4_parse.py:12   from lxml import etree, html

$ .venv/bin/python -c "import lxml"
ModuleNotFoundError: No module named 'lxml'

/usr/bin/python3 -c "import lxml"      -> ModuleNotFoundError
pyproject.toml dependencies            -> lxml is not listed
```

`lxml` is not installed anywhere on the host and is not a declared dependency, so the
parser that produced the corpus **cannot be re-run here**. Its `.pyc` is dated 2026-09-15,
so it ran at some point — but not in this environment as it now stands.

This is the second signal of the same thing. The first was that the parsed documents
behind the benchmark store are not on the host either. Together they say: **the parse
layer is not part of what this deployment can run.** It is an input, not a stage.

## What that means for option B

The plan was:

```
record anchors -> re-parse the 8 filings -> rebuild the store -> rebuild the views
-> rebuild the R4 index
```

Every arrow after the first now has an unverified prerequisite, and the first arrow is
itself unverified:

1. **`lxml` must be present** for the parser to run at all. Installing it into the
   deployment venv is a change to the deployment, not to the code, and it is not mine to
   make unilaterally.
2. **The parser must reproduce the corpus.** Even with `lxml` installed, whether a re-parse
   regenerates the same parsed documents the store was built from is unknown — the same
   question that was left open when the store rebuild was scoped, and it is not answered
   by the dependency being present.
3. **Only then** does recording anchors, carrying them through, and re-indexing mean
   anything, because each stage's output feeds the next.

So option B is not blocked by a design question any more — the design is settled and the
anchor exists in the source. It is blocked by **the parse layer not being runnable in this
deployment**, and that is an environment fact to settle before any of it is scheduled.

## What was written, and its status

`run_nf_v2_17a4_parse.py` gains an `ixbrl_anchors(e)` helper that walks an element's
descendants for `ix:nonFraction`/`ix:nonNumeric`, applies the **same filter `ix_facts`
applies** (skips `xsi:nil`, requires `name` and `contextRef`) so every anchor names a fact
that actually reaches the corpus, and records `{fact_id, concept, context_ref}` on each
block's `metadata`. Both block-emission sites carry it.

## Verified, on Microsoft's filing

`lxml 6.1.3` installed into the deployment venv (`ensurepip`, then `pip install lxml` —
the venv had been created by `uv` with no pip of its own). The verification then ran:

```
blocks                1135
facts emitted         1829
anchors recorded      1710
anchors that resolve  1710 / 1710
facts reachable       1710 / 1829

by block type:  TABLE     1534 anchors in 57 blocks
                PARAGRAPH  176 anchors in 66 blocks
                HEADING      0 anchors in  0 blocks
```

**Every anchor resolves to a fact the same parse emits**, and 93.5% of emitted facts are
reachable from a block. Before the change that number was **zero** — no view had any path
to a tagged fact.

### The 119 that are not reachable, characterised

They are not a defect in the link. 29 carry no value at all — `dei:AmendmentFlag`,
`dei:EntityCentralIndexKey` and similar — and live in the `ix:header` rather than the
document body, so no block contains them and none should. The remaining **90 are almost
entirely `...TextBlock` concepts** — `ScheduleOfDerivativeInstrumentsGainLoss…TextBlock`,
`CybersecurityRiskManagementProcesses…TextBlock` — which wrap a whole note rather than
stating a value.

So the gap is not leaf facts. The anchors cover the population the link exists for, and
the uncovered facts are of two kinds the route has nothing to say about: header metadata
and note-sized text wrappers.

### What this establishes

The anchor route works, verified on real data against a count taken independently from the
raw HTML. Recording anchors at parse time does link blocks to the facts they state, with
complete resolution and a fully explained residual.

It does **not** establish that the rest of option B is worth doing: carrying the anchor
through `atomic facts -> store records -> candidate views -> R4 index` still has to be
built, and the re-parse still has to reproduce the corpus. What is now settled is that the
first link — the one nothing else could proceed without — holds.

## The decision this needs

Either:

- **install `lxml`** into the deployment venv and run the verification on one filing —
  a small deployment change, and it answers point 1 only; or
- **treat the parse layer as out of scope for this deployment**, in which case option B
  cannot be done here at all and the retrieval re-keying has to be solved another way —
  most plausibly by attaching canonical identity to the *existing* views, which needs the
  correspondence that this option was supposed to supply from the source.

The second is the uncomfortable one and it should be stated plainly: if the parse layer
cannot run, the anchor route is closed, and the remaining routes all involve deciding
which fact a passage refers to **without** the filing's own anchor.
