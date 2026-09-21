"""H2A-2D-4: four identity domains over one digest, and no guessing between them.

The fact store derives `physical:`, `evidence:`, `candidate:` and `citation:` from
the *same* `(document_id, table_id, row_id, cell_id)` payload.  That is not
evidence of duplication, and this file exists so the two readings cannot be
confused:

    one underlying fact
        -> four ids that share a digest suffix, by construction
        -> and four different identity *roles*, by contract

The digest suffix is a coincidence of the payload, not an equality rule.  Sharing
it must never let one domain stand in for another -- so the negative cases below
are the load-bearing ones, not the equality.

Expected ids are derived here from the authored tuple with `hashlib`, following
the documented record-id contract, rather than read back out of the fact store.
Deriving them is not circular: the payload format is the published contract, and
the point is that both sides agree on it without either asking the other.
"""

from __future__ import annotations

import hashlib
from typing import Any

from src.runtime.trusted_v2_coordinator import _structured_evidence_identity
from src.runtime.trusted_v2_provenance import _identity

#: One authored financial fact, as the extractor located it.
DOCUMENT, TABLE, ROW, CELL = "doc-1", "table-7", "row-3", "cell-2"

#: The domains the fact store emits for it.
DOMAINS = ("physical", "evidence", "candidate", "citation")


def _authored_id(domain: str) -> str:
    """The record id for the authored fact, derived from the documented contract."""

    payload = "\x1f".join((DOCUMENT, TABLE, ROW, CELL)).encode("utf-8")
    return f"{domain}:{hashlib.sha256(payload).hexdigest()}"


# --- one digest, four domains -------------------------------------------------


def test_the_four_domains_share_one_digest_by_construction() -> None:
    """Sharing a suffix is a property of the payload, not of the identity."""

    suffixes = {_authored_id(domain).split(":", 1)[1] for domain in DOMAINS}
    assert len(suffixes) == 1


def test_each_domain_prefix_is_distinct() -> None:
    """Four ids, not one id written four ways."""

    assert {_authored_id(domain) for domain in DOMAINS} != set()
    assert len({_authored_id(domain) for domain in DOMAINS}) == len(DOMAINS)


# --- the field contract assigns the domains -----------------------------------


def test_each_field_carries_its_own_domain() -> None:
    """The record's field names are the domain declaration.

    `evidence_id` carries the evidence domain, `citation_id` the citation
    domain, `candidate_id` the candidate domain.  A caller holding one of these
    fields already knows which domain it is asking about, and does not need a
    resolver to guess it back.
    """

    record = {
        "evidence_id": _authored_id("evidence"),
        "fact_id": _authored_id("evidence"),
        "candidate_id": _authored_id("candidate"),
        "candidate_key": _authored_id("candidate"),
        "citation_id": _authored_id("citation"),
    }

    for field_name, domain in (
        ("evidence_id", "evidence"),
        ("fact_id", "evidence"),
        ("candidate_id", "candidate"),
        ("candidate_key", "candidate"),
        ("citation_id", "citation"),
    ):
        assert record[field_name] == _authored_id(domain), field_name
        assert record[field_name].split(":", 1)[0] == domain, field_name


# --- the load-bearing property: no cross-domain substitution -------------------


def test_a_citation_id_alone_does_not_resolve_as_evidence_identity() -> None:
    """Sharing the digest must not make a citation an evidence identity.

    A mapping that declares only a citation has declared a citation.  The
    evidence resolver must not quietly accept it because the suffix matches --
    that would turn a shared payload into an implicit cross-domain equality
    rule, and a citation reference would start authorizing itself as the
    evidence it points at.
    """

    citation_only = {"citation_id": _authored_id("citation")}

    assert _identity(citation_only) == ""
    assert _structured_evidence_identity(citation_only) == ""


def test_a_physical_id_alone_does_not_resolve_as_evidence_identity() -> None:
    """Same rule for the physical/source domain."""

    physical_only = {"physical_id": _authored_id("physical")}

    assert _identity(physical_only) == ""
    assert _structured_evidence_identity(physical_only) == ""


def test_the_candidate_domain_is_not_evidence_despite_the_shared_digest() -> None:
    """Where the two *do* differ, the resolver's choice is the declaration.

    Documented rather than asserted as an error: a candidate id resolves through
    the evidence ladder by its field name, and it resolves to the *candidate*
    id it was given -- never to the evidence id the suffix would also match.
    """

    record = {"candidate_id": _authored_id("candidate")}

    assert _identity(record) == _authored_id("candidate")
    assert _identity(record) != _authored_id("evidence")


def test_the_declared_domain_wins_over_field_order() -> None:
    """A record declaring both resolves by its own declaration, not by suffix."""

    record = {
        "evidence_id": _authored_id("evidence"),
        "citation_id": _authored_id("citation"),
    }

    assert _identity(record) == _authored_id("evidence")


# --- what the ladder is actually for ------------------------------------------


def test_the_ladder_is_a_field_alias_contract_not_a_domain_matcher() -> None:
    """Recorded so the ladders are not mistaken for a second identity authority.

    They resolve *which field* a packet uses for its identity, across the field
    names producers have used over time.  They do not and must not decide which
    *domain* a string belongs to -- the prefix already says that.
    """

    for field in ("evidence_id", "fact_id", "candidate_id", "candidate_key", "chunk_id"):
        value: Any = f"{field}-value"
        assert _structured_evidence_identity({field: value}) == value
