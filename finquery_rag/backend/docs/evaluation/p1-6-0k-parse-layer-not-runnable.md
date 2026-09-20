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

**Unverified.** It imports cleanly and is consistent with `ix_facts` by construction, but
the reconciliation it exists to pass — anchors resolving to emitted facts, and every
emitted fact reachable from a block — has not been run, because nothing that imports this
module can run here.

`verify_parse_anchors.py` is the check, written and ready: it reports anchors per block
type, anchors that fail to resolve, and facts not reachable from any block. It needs the
same `lxml`.

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
