"""The single authority for what evidence fields may cross a model boundary.

Before this module there was one allowlisted projection (the Binder's) and one
ungoverned path (the Specialist's, which received whole evidence packets).  Two
consequences, both observed:

* the same evidence had two disclosure policies, so "can a model see the
  extracted source text?" had two answers depending on which model was asked;
* nothing decided the question.  The Binder's safety came from an allowlist in
  ``binder_fact_view``; the Specialist's came from the *shape* of a packet dict
  -- its prompt asks for ``ev["source_text"]`` at top level, and the text is
  nested under ``metadata``, so the branch is dead.  A safety property that
  holds by accident stops holding the moment the shape changes.

This module makes the answer a decision rather than an accident.  The rule is
deny by default: a field reaches a model only if the profile for that boundary
names it.  A field added to the evidence contract is invisible everywhere until
someone adds it to a profile deliberately.

What this module does *not* do: it does not define the evidence domain model,
does not fetch or admit evidence, and does not decide binding.  It answers one
question -- which fields, for which boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from typing import Any

__all__ = [
    "EvidenceDisclosureProfile",
    "UnknownDisclosureProfile",
    "allowed_fields",
    "project",
]


class UnknownDisclosureProfile(ValueError):
    """Raised when a caller asks for a profile this authority does not define."""


class EvidenceDisclosureProfile(str, Enum):
    """The model boundaries that may receive evidence.

    Only boundaries that exist are listed.  A profile invented for symmetry
    would be a field list nobody validates against a real prompt.
    """

    BINDER = "BINDER"
    SPECIALIST = "SPECIALIST"


#: Deny by default: these are the *only* fields each boundary may see.
#:
#: BINDER keeps its established surface.  It is deliberately unchanged -- it
#: selects one fact from many and needs the typed financial and table
#: provenance identifiers to do so.
_BINDER_FIELDS: tuple[str, ...] = (
    # identity and provenance
    "fact_id",
    "evidence_id",
    "candidate_id",
    "candidate_key",
    "citation_id",
    "physical_source_id",
    "source_id",
    "document_id",
    "pdf_page",
    "page",
    "table_fragment_id",
    "logical_table_id",
    "table_id",
    "row_id",
    "column_id",
    "cell_id",
    # typed financial and table semantics
    "entity",
    "issuer",
    "company",
    "ticker",
    "metric",
    "normalized_metric",
    "raw_metric",
    "metric_path",
    "metric_paths",
    "period",
    "normalized_period",
    "raw_period",
    "periods",
    "value",
    "parsed_numeric_value",
    "raw_value",
    "currency",
    "unit",
    "scale",
    "normalized_scale",
    "raw_scale",
    "scope",
    "scope_label",
    "segment_label",
    "row_label",
    "row_path",
    "row_hierarchy",
    "column_label",
    "column_header_path",
    "multi_level_column_headers",
    "table_title",
    "statement_title",
    "statement_type",
    "section_title",
    "section_path",
    # required by the binding validator, which refuses to default it
    "provenance_complete",
)

#: SPECIALIST is the field set its prompt actually reads, audited field by field
#: from ``LocalSpecialistGenerator.render_prompt``:
#:
#:   citation_id, metric (normalized_metric fallback), period, value, unit,
#:   currency, scale, scope, document_id
#:
#: Two things the prompt *asks for* are deliberately absent, and the difference
#: matters:
#:
#: * ``source_text``.  The prompt has an ``Evidence: {ev['source_text']}`` branch
#:   that has never fired on this path, because the text lives under
#:   ``metadata`` and the branch reads the top level.  So the specialist does not
#:   receive source text today, and adding an excerpt here would broaden model
#:   exposure under the banner of unifying it.  If a generator is later shown to
#:   need narrative evidence, ``evidence_excerpt`` gets added here as a named,
#:   bounded, provenance-carrying field -- and this comment is where that
#:   decision gets recorded.
#: * ``page``.  The prompt reads ``ev.get("page")`` at top level and falls back
#:   to 1; the real value is again nested.  Propagating it is H2A-1C's job
#:   (provenance lineage), and it is added here when it survives that far --
#:   not before, or the profile would name a field that is always absent.
_SPECIALIST_FIELDS: tuple[str, ...] = (
    "evidence_id",
    "citation_id",
    "metric",
    "normalized_metric",
    "period",
    "value",
    "unit",
    "currency",
    "scale",
    "scope",
    "document_id",
)


PROFILE_FIELDS: Mapping[EvidenceDisclosureProfile, tuple[str, ...]] = {
    EvidenceDisclosureProfile.BINDER: _BINDER_FIELDS,
    EvidenceDisclosureProfile.SPECIALIST: _SPECIALIST_FIELDS,
}


def allowed_fields(profile: EvidenceDisclosureProfile | str) -> tuple[str, ...]:
    """The fields this boundary may see, for tests and for debug output."""

    try:
        return PROFILE_FIELDS[EvidenceDisclosureProfile(profile)]
    except (KeyError, ValueError) as exc:
        raise UnknownDisclosureProfile(
            f"no disclosure profile named {profile!r}; expected one of "
            f"{[item.value for item in EvidenceDisclosureProfile]}"
        ) from exc


def project(
    evidence: Mapping[str, Any],
    *,
    profile: EvidenceDisclosureProfile | str,
    resolve: Callable[[Mapping[str, Any], str], Any] | None = None,
) -> dict[str, Any]:
    """Project one authoritative evidence object onto a model-facing view.

    ``resolve`` supplies the value for a permitted field.  The Binder needs a
    structural lookup that walks nested table context; the Specialist reads the
    packet directly.  The *policy* -- which fields, and the fact that the list is
    closed -- lives here once, so the two cannot drift into separate rules.

    Fields the profile does not name are dropped, including fields the evidence
    contract gains later.  That is the point of the allowlist: an unlisted field
    is invisible rather than leaked.
    """

    fields = allowed_fields(profile)
    lookup = resolve or (lambda fact, name: fact.get(name))
    return {
        field: value
        for field in fields
        if (value := lookup(evidence, field)) is not None
    }


def fields_present(
    evidence: Mapping[str, Any],
    *,
    profile: EvidenceDisclosureProfile | str,
) -> Sequence[str]:
    """Which permitted fields this evidence actually carries.

    Debug-level only: names, never values, so a disclosure question can be
    answered without putting evidence content into a trace.
    """

    return tuple(sorted(project(evidence, profile=profile)))
