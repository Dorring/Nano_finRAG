"""H2A-2D-4B: the resolvers' callers, and what an alias disagreement means.

`_structured_evidence_identity` and `_identity` accept several field names. The
question 2D-4B asks is not how many names they accept, but whether every accepted
name denotes **one semantic identity domain** -- and whether their callers know
which domain they are holding.

    Resolver                              Intended domain   Accepted fields
    ---------------------------------------------------------------------------
    coordinator._structured_evidence_identity   evidence     evidence_id, fact_id,
                                                             candidate_id,
                                                             candidate_key, chunk_id
    provenance._identity                        evidence     evidence_id, fact_id,
                                                             candidate_id

Both have exactly one production caller, and both callers hold admitted evidence
packets:

- `coordinator.py:93`, inside `_structured_citations`, projecting public sources;
- `provenance.py:64`, inside `build_claim_provenance`, projecting claim support.

So neither is an identity-domain *matcher*. They are SAME-DOMAIN FIELD-ALIAS
RESOLVERS: evidence-domain records, resolved across the field names successive
producers have used for an evidence identity.

`fact_id` was the open question, and the contract answers it rather than the
name: the fact store emits `"fact_id": evidence_id`
(`trusted_v2_canonical_fact_store.py:225`), so it is a historical alias of the
evidence identity, not a physical-fact domain of its own.

The last section pins alias *disagreement*, which is a different question from
cross-domain substitution and was left implicit until now.
"""

from __future__ import annotations

import pytest

from src.runtime.trusted_v2_coordinator import _structured_evidence_identity
from src.runtime.trusted_v2_provenance import _identity

#: Authored evidence-domain values. Distinct, so a disagreement is visible.
EVIDENCE_A = "evidence:aaaaaaaaaaaaaaaa"
EVIDENCE_B = "evidence:bbbbbbbbbbbbbbbb"

RESOLVERS = (_structured_evidence_identity, _identity)


# --- one domain, several field names ------------------------------------------


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_every_accepted_field_is_an_evidence_identity_alias(resolve) -> None:
    """Each name in the ladder carries an evidence-domain value.

    `fact_id` is included deliberately: the fact store fills it with the evidence
    id, so it aliases rather than naming a separate physical-fact domain.
    """

    for field in ("evidence_id", "fact_id", "candidate_id", "candidate_key", "chunk_id"):
        if field in ("candidate_key", "chunk_id") and resolve is _identity:
            continue
        assert resolve({field: EVIDENCE_A}) == EVIDENCE_A, field


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_a_disagreement_resolves_to_the_canonical_field(resolve) -> None:
    """Alias disagreement is not left to list order.

    Two same-domain aliases carrying different values is a malformed record, not
    a precedence puzzle.  `evidence_id` is the canonical field name and wins;
    what matters is that the outcome is *stated* here rather than depending on
    which name happens to appear first in a tuple.

    This is a different question from cross-domain substitution: both values are
    evidence-domain, so no domain confusion is possible.  What could go wrong is
    silent disagreement -- a record whose two identity fields disagree being read
    as whichever one the resolver reached first.
    """

    record = {"evidence_id": EVIDENCE_A, "fact_id": EVIDENCE_B}

    assert resolve(record) == EVIDENCE_A
    assert resolve(record) != EVIDENCE_B


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_agreement_is_unremarkable(resolve) -> None:
    """The ordinary case: producers fill aliases with the same value."""

    assert resolve({"evidence_id": EVIDENCE_A, "fact_id": EVIDENCE_A}) == EVIDENCE_A


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_an_empty_record_resolves_to_nothing(resolve) -> None:
    """Absence is absence -- not a fabricated identity."""

    assert resolve({}) == ""
    assert resolve({"evidence_id": "   "}) == ""


# --- the domain boundary still holds ------------------------------------------


@pytest.mark.parametrize("resolve", RESOLVERS)
def test_a_cross_domain_value_is_still_not_evidence_identity(resolve) -> None:
    """The 2D-4 property, re-asserted where the resolvers are actually called.

    `citation:` and `physical:` share the digest suffix with the evidence id for
    the same underlying fact, and must still not resolve as evidence identity.
    """

    assert resolve({"citation_id": "citation:aaaaaaaaaaaaaaaa"}) == ""
    assert resolve({"physical_id": "physical:aaaaaaaaaaaaaaaa"}) == ""
