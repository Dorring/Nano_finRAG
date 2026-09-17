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
    "NESTED_FIELD_POLICY",
    "EvidenceDisclosureProfile",
    "UnknownDisclosureProfile",
    "allowed_fields",
    "project",
]


class UnknownDisclosureProfile(ValueError):
    """Raised when a caller asks for a profile this authority does not define."""


class DisclosureArtifactType(str, Enum):
    """The kinds of authoritative object a profile can be asked to project.

    One authority, one role per boundary, one allowlist per artifact type.
    Introducing a second authority for calculations would recreate the split
    this module exists to remove: the question "what may this model see" would
    again have two answers depending on which authority you asked.
    """

    EVIDENCE = "EVIDENCE"
    CALCULATION = "CALCULATION"


class EvidenceDisclosureProfile(str, Enum):
    """The model boundaries that may receive evidence.

    Only boundaries that exist are listed.  A profile invented for symmetry
    would be a field list nobody validates against a real prompt.
    """

    BINDER = "BINDER"
    SPECIALIST = "SPECIALIST"
    V1_ANSWER = "V1_ANSWER"
    INGEST_TABLE = "INGEST_TABLE"


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


#: The V1 rollback runtime answers from assembled retrieval context rather than
#: from evidence packets, so it is a different authoritative source type with a
#: different role.  It gets its own profile rather than borrowing BINDER or
#: SPECIALIST, whose field semantics do not describe a retrieved chunk.
#:
#: Audited field by field from ``ContextBuilder.build`` -- the only consumer on
#: this path -- rather than from the retrieval object's full schema.  Chunk text
#: is legitimately required here: V1 answering reads the retrieved content, so
#: ``content`` is allowed deliberately.  What is *not* allowed is the chunk
#: object, or its ``metadata`` dictionary wholesale.
#:
#: The audit had to be widened once, and the reason is worth keeping: reading
#: ``ContextBuilder.build`` alone is not enough.  Its output
#: (``last_context_evidence``) has a *second* consumer --
#: ``EvidenceItem.from_chunk`` -- whose ``document_name`` drives the
#: answerability check, which decides whether the user-visible answer carries a
#: "could not verify these documents" suffix.  An allowlist built from the
#: formatter alone silently changed V1's answers.  The fields below are the
#: union of what the formatter and that consumer read.
#:
#: ``score``, ``rerank_score`` and the identity fields are retrieval internals,
#: and they are permitted only because removing them would change V1 behaviour.
#: This phase governs disclosure; it does not narrow function.
_V1_ANSWER_FIELDS: tuple[str, ...] = (
    "content",
    "doc_id",
    "chunk_id",
    "document_name",
    "doc_name",
    "page",
    "content_type",
    "type",
    "score",
    "rerank_score",
    "metadata",
)

#: Fields permitted *inside* a nested container.  A profile that names a
#: container without naming its keys would be an exclude-list by another route,
#: and a key added to chunk metadata later would cross models by default -- the
#: exact failure this module exists to prevent.
#:
#: ``parent_excerpt`` is required: ``_merge_parent_context_chunks`` expands a
#: child hit to its parent section text, and that text becomes the content the
#: answer model reads.  It is retrieved document text, sourced from the same
#: document as the chunk, so it is permitted for the same reason ``content`` is.
_V1_ANSWER_METADATA_FIELDS: tuple[str, ...] = (
    "type",
    "page",
    "parent_id",
    "section_path",
    "child_hit_count",
    "table_num",
    "parent_excerpt",
    # read by ``EvidenceItem.from_chunk``, which the answerability check depends on
    "document_name",
    "doc_name",
    "filename",
    "content_type",
)

#: Document ingestion, which runs before any evidence exists.  This is the one
#: boundary whose legitimate input *is* raw document text -- the model's whole
#: job is to clean an extracted table, and it cannot do that without the table
#: and its surrounding page text.
#:
#: So the profile permits raw text, and that is the point: the decision is now
#: written down and auditable rather than being an omission nobody examined.  A
#: reader asking "who decided a model may see raw page text?" gets an answer
#: instead of a shrug.  ``page_text`` and ``table_markdown`` are the only inputs
#: the prompt interpolates; anything else on this path stays out.
_INGEST_TABLE_FIELDS: tuple[str, ...] = (
    "page_text",
    "table_markdown",
    "page_num",
)

NESTED_FIELD_POLICY: Mapping[EvidenceDisclosureProfile, Mapping[str, tuple[str, ...]]] = {
    EvidenceDisclosureProfile.V1_ANSWER: {"metadata": _V1_ANSWER_METADATA_FIELDS},
}


PROFILE_FIELDS: Mapping[EvidenceDisclosureProfile, tuple[str, ...]] = {
    EvidenceDisclosureProfile.BINDER: _BINDER_FIELDS,
    EvidenceDisclosureProfile.SPECIALIST: _SPECIALIST_FIELDS,
    EvidenceDisclosureProfile.V1_ANSWER: _V1_ANSWER_FIELDS,
    EvidenceDisclosureProfile.INGEST_TABLE: _INGEST_TABLE_FIELDS,
}


#: Calculation payloads, governed by the same authority and the same role.
#:
#: The audit found `_call_specialist` applying two policies in one call: the
#: evidence items beside it were projected, while the calculation payload was
#: `CalculationResult.to_dict()` -- the *internal diagnostics* form, which
#: carries each operand's full `source_text` and the raw `error_message`, unlike
#: `to_public_dict()` which substitutes a bounded excerpt.
#:
#: Three fields, audited from `LocalSpecialistGenerator.render_prompt`, which
#: reads `value`, `unit` and `operation` and nothing else.  `to_public_dict()` is
#: deliberately *not* used here: "safe for a public response" and "safe for this
#: model" are different questions asked by different consumers, and conflating
#: them is how a payload ends up governed by whoever wrote the other one.
CALCULATION_PROFILE_FIELDS: Mapping[EvidenceDisclosureProfile, tuple[str, ...]] = {
    EvidenceDisclosureProfile.SPECIALIST: (
        "operation",
        "unit",
        "value",
    ),
}


def project_calculation(
    calculation: Mapping[str, Any],
    *,
    profile: EvidenceDisclosureProfile | str,
) -> dict[str, Any] | None:
    """Project an authoritative calculation payload onto a model-facing view.

    Deny by default, like the evidence projection.  A profile with no entry for
    calculations gets ``None`` -- not the payload, and not a partial guess -- so
    a boundary that has not been considered cannot receive one by omission.
    """

    resolved = EvidenceDisclosureProfile(profile)
    permitted = CALCULATION_PROFILE_FIELDS.get(resolved)
    if permitted is None:
        return None
    view: dict[str, Any] = {}
    for field in permitted:
        value = calculation.get(field)
        if value is None:
            continue
        # Same rule as the evidence projection: a permitted name is not a licence
        # for a structure stored under it.
        if isinstance(value, (Mapping, list, tuple, set, frozenset)):
            continue
        view[field] = value
    return view


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
    nested = NESTED_FIELD_POLICY.get(EvidenceDisclosureProfile(profile), {})
    lookup = resolve or (lambda fact, name: fact.get(name))
    view: dict[str, Any] = {}
    for field in fields:
        value = lookup(evidence, field)
        if value is None:
            continue
        permitted = nested.get(field)
        if permitted is None:
            # Deny by default governs *structure*, not only names.  A permitted
            # field name is not a licence for whatever is stored under it: a
            # mapping or sequence there would carry keys nobody approved, so it
            # is dropped unless this profile explicitly governs that container.
            if isinstance(value, (Mapping, list, tuple, set, frozenset)):
                continue
            view[field] = value
            continue
        # A governed container is filtered key by key -- allowing ``metadata``
        # whole would let any key added to it later reach a model without anyone
        # deciding it should -- and only one level deep, since a permitted key
        # whose value is itself a structure is the same hole one level down.
        if not isinstance(value, Mapping):
            continue
        view[field] = {
            key: item
            for key, item in value.items()
            if key in permitted
            and not isinstance(item, (Mapping, list, tuple, set, frozenset))
        }
    return view


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
