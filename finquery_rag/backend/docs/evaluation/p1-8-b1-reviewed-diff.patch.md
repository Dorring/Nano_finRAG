# P1.8-B1 — REVIEWED DIFF (NOT APPLIED)

A switchable canonical-candidate seam. **Not applied, not written to `src/`.**

This file is deliberately named `.patch.md` so it is not picked up by
`git apply` or `patch(1)`. To evaluate it, read it; to build it, copy the blocks
below into the named files.

Read `p1-8-b1-canonical-retrieval-headroom.md` first. **The measurement says
this change is worth 2–3 cases** (46/95 → 48–49/95). It is written because the
brief asked for it, not because it is the right next move. §5 of that report
names where the coverage actually is.

---

## 0. The one decision that blocks the whole patch

Every other detail below is mechanical. This one is a judgement and it is why
the patch is not applied.

The production store loader rejects any record without a citation:

```
trusted_v2_production.py:379
    if self.require_citation_id and citation_id is None:
        raise TrustedV2ProductionConfigurationError(
            f"fact store record {index} is missing structured citation_id")
```

iXBRL records have no `citation_id`, no `page`, no `table_fragment_id` — their
provenance is `document_id` + `context_ref`. `CanonicalFactStore._as_fact`
sets `"citation_id": None`, so **canonical facts cannot enter the production
store at all today.**

The patch therefore has to synthesise an address for a canonical candidate.
There is no way around it, and there are only two shapes:

```
(a) NEW CITATION SPACE   citation:ix:<sha256(document_id|context_ref)>
    Honest: it names a filing and an XBRL context, which is what the iXBRL
    layer actually has.  It is NOT the physical citation space the benchmark
    measures, so citation P/R would have to be redefined for these cases.

(b) KEEP THE LEGACY ADDRESS
    The canonical fact supplies the VALUE; the legacy fact that was already
    bound supplies the CITATION.  Citation P/R is untouched.
    Requires the legacy record to still be in the packet, which it is -- the
    seam substitutes the bound value, not the evidence set.
```

**(b) is recommended.** It preserves the citation metrics, it keeps
`require_citation_id` intact as a fail-closed invariant, and it makes the seam's
blast radius "which value is bound" rather than "what a citation means". The
patch below is written for (b) and marks the two places (a) would differ.

If the answer is (a), the patch is materially larger and the citation contract
needs its own review first. **Do not proceed on this patch until that is settled.**

---

## 1. Files touched

```
src/runtime/trusted_v2_production.py     +1 wrapper factory, +1 env flag   (~70 lines)
src/runtime/trusted_v2_canonical_seam.py NEW                                (~150 lines)
tests/runtime/test_canonical_seam.py     NEW                                (~120 lines)
```

Nothing else. No store rebuild, no index rebuild, no benchmark change, no gate,
binder, validator or finalizer change, no fail-closed semantic change.

---

## 2. `src/runtime/trusted_v2_canonical_seam.py` (new)

```python
"""Substitute a canonical candidate where the legacy store cannot pick one.

Contracts this seam exists to honour (see the Benchmark Authority Contract):
  * a question's coordinate denotes the company-level fact unless the question
    text names the scope;
  * company-level == an XBRL fact whose context carries no dimensions;
  * a coordinate that still holds several values stays ambiguous and stays
    refused.

So the seam resolves ONLY on exactly one undimensioned fact.  Zero or several
=> the legacy answer is returned untouched.  It cannot make a coordinate less
ambiguous than it is, which is what keeps the fail-closed guard honest.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

from .trusted_v2_canonical_store import CanonicalFactStore

#: Read once.  Unset or "off" => the seam is inert and behaviour is byte-identical.
ENV_FLAG = "TRUSTED_V2_CANONICAL_CANDIDATES"


def seam_enabled(environ: Mapping[str, str] | None = None) -> bool:
    env = os.environ if environ is None else environ
    return str(env.get(ENV_FLAG, "")).strip().casefold() in {"1", "true", "on", "yes"}


def _identity(record: Mapping[str, Any]) -> tuple:
    """The tuple a legacy and a canonical record must share to be the same fact.

    This is `logical_fact_id`'s payload, kept identical on purpose: two
    different identity functions would let the seam resolve a substitution the
    citation layer did not agree with.
    """

    def fold(value: Any) -> str:
        return " ".join(str(value if value is not None else "").casefold().split())

    return tuple(
        fold(record.get(field))
        for field in ("entity", "metric", "period", "value", "unit", "scale", "currency")
    )


class CanonicalCandidateSeam:
    """Wrap a store and swap in a canonical candidate where one is determined.

    Delegates everything.  Only `facts_at_coordinate` can differ, and only in
    the direction of *fewer* candidates.
    """

    def __init__(self, legacy: Any, canonical: CanonicalFactStore) -> None:
        self._legacy = legacy
        self._canonical = canonical

    # --- the seam -----------------------------------------------------------

    def facts_at_coordinate(self, entity: Any, metric: Any, period: Any):
        legacy_facts = self._legacy.facts_at_coordinate(entity, metric, period)
        resolved = self._canonical.resolve_canonical(
            entity, self._canonical.canonical_quantity(metric) or "", period
        )
        # (b): no canonical quantity, or not uniquely determined -> unchanged.
        if not resolved:
            return legacy_facts
        canonical_fact = resolved[0]

        # The canonical value is authoritative ONLY if the legacy packet already
        # carries a record stating the same quantity, so the citation keeps
        # pointing at a physical row.  Otherwise the substitution would bind a
        # value with no address to cite.
        support = [
            fact for fact in legacy_facts
            if _identity(fact)[3:] == _identity(canonical_fact)[3:]
        ][:1]
        if not support:
            return legacy_facts

        # Substitution, never an append: the candidate budget is unchanged.
        return tuple(support)

    # --- delegation ---------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        return getattr(self._legacy, name)
```

### What this deliberately does not do

- It does not read the question text. Scope questions (class S) are **not**
  recognised, so a scoped question will not be resolved by this seam — it will
  keep failing closed, as it does today. Adding scope recognition is the change
  that would make this worth building, and it is not in this patch.
- It does not touch `candidate_keys`, `materialize` or `iter_records`, so the
  legacy key space still governs retrieval, citations and the gold's `v2fact:`
  ids. That is what makes rollback total.

---

## 3. `src/runtime/trusted_v2_production.py` — the wiring

One factory, inserted beside `_load_production_fact_store`, and one call-site
change where the store is constructed.

```python
def _maybe_wrap_canonical(
    store: Any,
    *,
    ixbrl_path: Path | str | None,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Wrap `store` in the canonical seam when the flag is on and iXBRL exists.

    Returns `store` unchanged in every other case -- no flag, no iXBRL file, or
    a store that does not expose `facts_at_coordinate`.  The default path is
    therefore the current one, byte for byte.
    """

    from .trusted_v2_canonical_seam import CanonicalCandidateSeam, seam_enabled

    if not seam_enabled(environ):
        return store
    if ixbrl_path is None or not Path(ixbrl_path).is_file():
        return store
    if not hasattr(store, "facts_at_coordinate"):
        return store

    canonical = CanonicalFactStore(store, ixbrl_path)
    if not canonical.canonical_ready:
        return store
    return CanonicalCandidateSeam(store, canonical)
```

and at the existing construction site:

```python
    store = StructuredFactStore(fact_path)
    store = _maybe_wrap_canonical(
        store,
        ixbrl_path=_env(environ, "TRUSTED_V2_IXBRL_STORE", None),
    )
    return store
```

`TRUSTED_V2_IXBRL_STORE` names the iXBRL file explicitly. It is **not**
defaulted, on purpose: defaulting it would make the seam's behaviour depend on a
file appearing on disk, which is not a property a deployment should inherit
silently.

---

## 4. `tests/runtime/test_canonical_seam.py` (new) — the properties to prove

Each of these is a claim in the report. If any fails, the seam is not safe.

```python
def test_flag_off_is_byte_identical(store, ixbrl):
    """Unset flag -> the wrapper is never constructed."""
    assert _maybe_wrap_canonical(store, ixbrl_path=ixbrl, environ={}) is store


def test_already_unique_coordinate_is_unchanged(store, ixbrl):
    """A coordinate with one legacy value keeps exactly that value."""
    seam = CanonicalCandidateSeam(store, CanonicalFactStore(store, ixbrl))
    before = store.facts_at_coordinate("Apple", "Net income", "FY2025")
    assert seam.facts_at_coordinate("Apple", "Net income", "FY2025") == before


def test_seam_never_increases_a_candidate_set(store, ixbrl):
    """Substitution, never append -- the budget invariant."""
    seam = CanonicalCandidateSeam(store, CanonicalFactStore(store, ixbrl))
    for entity, metric, period in coordinates_of_interest:
        assert len(seam.facts_at_coordinate(entity, metric, period)) <= \
               len(store.facts_at_coordinate(entity, metric, period))


def test_ambiguous_coordinate_still_refuses(store, ixbrl):
    """Coca-Cola / Operating income / FY2024 has two legacy values and no
    canonical quantity for `Operating income` maps to one undimensioned fact
    *of the legacy packet's* -- so it must still return both, and the
    CONFLICTING_VALUES guard must still fire."""
    ...
    assert coordinate_status(bound, seam.facts_at_coordinate(...)) \
           is CoordinateStatus.CONFLICTING_VALUES


def test_no_support_no_substitution(store, ixbrl):
    """If the legacy packet carries no record stating the canonical quantity,
    the legacy answer is returned untouched -- nothing is bound uncitable."""


def test_citation_identity_is_preserved(store, ixbrl):
    """(b): the returned record is a legacy record, so citation_id, page and
    table_fragment_id are unchanged from what the store would have returned."""
```

The fourth is the load-bearing one. It is the property that keeps
`INCORRECT_RELEASE = 0`, and it is the one to run first against the 9
`EVIDENCE_CONFLICT` cases before any wider measurement.

---

## 5. Rollback

```
unset TRUSTED_V2_CANONICAL_CANDIDATES   ->  old code path, old store
```

No store, index or artifact is migrated by this patch. `_maybe_wrap_canonical`
returns the unwrapped store whenever the flag is off, and nothing else in the
runtime learns the seam exists. Rollback is one environment variable and a
restart, with no data step.

---

## 6. What this patch does not fix, stated plainly

```
recovers        at most 3 cases, of which 2 are defensible
                46/95 -> 48-49/95 = 50.5-51.6%
does not reach  57/95; the 60% target is not reachable by this change
does not touch  the 22 gate-blocked cases, which never reach retrieval
does not handle scoped questions (class S), which stay refused
```

**If the goal is 60%, this patch is not the lever.** The gate's residual 22
refusals are 4x larger and unmeasured; that is where the next round should look.

### 6.1 As written, this patch is a no-op — and that is the real finding

The seam delegates resolution to `CanonicalFactStore.resolve_canonical`, which
resolves through `METRIC_TO_CANONICAL` and `CONCEPT_ALIGNMENT`. Checked against
the three nominally recoverable cases:

```
msft-016  metric "Cost of revenue"  ->  not in METRIC_TO_CANONICAL  ->  seams to None
pfe-030   metric "Cost of sales"    ->  not in METRIC_TO_CANONICAL  ->  seams to None
jpm-009   metric "Total lending-…"  ->  not in METRIC_TO_CANONICAL  ->  seams to None
```

`METRIC_TO_CANONICAL` lists twelve quantities and none of these three is among
them, by the module's own deliberate design. So enabling the flag today changes
nothing at all: `resolve_canonical` returns `[]` for every one of the nine
`EVIDENCE_CONFLICT` cases, the seam returns the legacy candidates untouched, and
the guard refuses exactly as before.

**The plumbing is not the work.** The work is the metric-authority table — for
each metric, which canonical quantity (or which scope) it denotes. That is a
statement about what a question means, so it is a benchmark decision, and it is
the thing the Benchmark Authority Contract requires be settled before any code
moves. Two of the three are defensible from the standard taxonomy:

```
"cost of revenue"  -> us-gaap:CostOfGoodsAndServicesSold   (also an arithmetic
"cost of sales"    -> us-gaap:CostOfGoodsAndServicesSold    identity in the
                                                           source: 87,831 =
                                                           22,422 + 40,171 + 25,238)
```

The third has no defensible mapping and should stay refused.

**So the honest reading of this patch is:** it prepares a seam that is worth
2–3 cases once a mapping table exists, and the mapping table is worth 2–3 cases
once a seam exists. Neither is worth much without the other, and together they
are still half of what the gate lever is worth.
