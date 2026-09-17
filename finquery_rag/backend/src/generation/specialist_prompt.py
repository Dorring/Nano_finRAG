"""The Local Financial Specialist's model-facing prompt renderer.

This is the prompt builder that sits downstream of the disclosure projection:
it receives already-projected evidence views and renders the text a model is
asked to answer from.  It is deliberately separate from
``local_specialist_generator``, which imports ``torch`` and the nanochat model
runtime at module scope.

Why the separation is load-bearing rather than cosmetic: ``conftest.py``
excludes a test module from collection entirely when an optional runtime it
needs is missing (``OPTIONAL_RUNTIME_MODULES``), so an assertion about *the
prompt a model sees* written next to ``import torch`` would silently not run on
any checkout without torch -- reporting green while testing nothing.  The
renderer is pure string assembly and needs no model, so it lives here and is
asserted on directly.

The contract rendered is ``FinancialGenerationViewV1``.

H2A-3B2 removed every fabricated default from this renderer.  The rule, frozen:

    missing information must never be converted into invented domain information

Five expressions broke it -- ``scale or "1"``, ``document_id or "filing"``,
``metric or "Metric"``, ``period or "Period"`` and ``scope or metric`` -- and
they were worse than an omission in a way worth stating plainly.  A reader can
tell ``Unit: not specified`` from a real unit; there is no such thing as a
metric named ``Metric``, so a model shown it has been handed a *value* where the
record had none.  The last one was the worst of the five: ``scope or metric``
answered an unknown scope with an assertion that scope equals metric, which is
not a placeholder at all but a new fact about the world.

This renderer states absence and invents nothing.  Every field is written
through ``_render``, so "the model is shown the evidence, or an explicit
statement that it is not there" is a property of the renderer rather than the
habit of whoever wrote each line.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["ABSENT", "render_specialist_prompt"]

#: How a field the evidence does not carry is written into the prompt.
#:
#: One marker for every field, and the choice is not stylistic.  The frozen
#: schema (``data/grounding_alignment/v1/financial-generation-view-v1.md``) is a
#: fixed set of lines per evidence block, and the canonical renderer for that
#: schema -- ``rag_v2/generation/financial_view_v1.py``, SHA-pinned to it --
#: already answers this question the same way, in its ``_value`` helper, for
#: every field including metric, period and scope.  Two renderers of one contract
#: spelling absence differently is a divergence to argue about; inventing a
#: *value* is not, which is why the five F10 expressions were defects rather than
#: a style the other renderer happened to disagree with.
#:
#: The alternative the audit considered was omitting the line instead.  That is
#: honest too, but it makes the block's *shape* depend on the data, so a reading
#: of the prompt can no longer distinguish a field that was absent from a
#: renderer version that stopped emitting it -- and answer rule 6 asks the model
#: to notice missing evidence, which a stated absence serves and a vanished line
#: does not.
ABSENT = "not specified"


def _stated(item: Mapping[str, Any], name: str) -> Any:
    """The field exactly as the evidence states it, or ``None`` if it does not.

    Absence arrives two ways -- an absent key, and the empty string -- and this
    treats both as absent rather than as *present and empty*.  The disclosure
    projection drops ``None`` outright, so in production only the first shape
    occurs; the second is here because "the evidence says the metric is the
    empty string" is not a thing anyone recorded, and rendering it verbatim
    would put a blank where a statement belongs.

    Nothing is converted.  ``0`` survives as ``0``, which is the B0 finding: a
    truthiness test cannot tell a real zero from an absent value, and the page
    field is where that cost a fact.
    """

    value = item.get(name)
    if value is None:
        return None
    return value if str(value).strip() else None


def _render(value: Any) -> str:
    """A field as it is written into the prompt: its value, or stated absence.

    This does not choose a value; it chooses only how to spell *no value*.  Every
    line in the block goes through it.
    """

    return ABSENT if value is None else str(value).strip()


def render_specialist_prompt(
    question: str,
    evidence_items: Sequence[Mapping[str, Any]],
    calculation_result: Mapping[str, Any] | None = None,
) -> str:
    """Render the specialist prompt adhering strictly to FinancialGenerationViewV1.

    ``evidence_items`` are disclosure-projected views, never whole evidence
    packets.  This function reads fields; it does not decide which fields exist,
    so it cannot widen model exposure on its own -- adding a field here has no
    effect until the SPECIALIST disclosure profile allows it to arrive.
    """

    lines = [f"[QUESTION]\n{question.strip()}\n", "[VERIFIED EVIDENCE]\n"]

    for i, ev in enumerate(evidence_items, start=1):
        cite_id = ev.get("citation_id", f"E{i}")
        if not re.match(r"^E\d+$", cite_id):
            cite_id = f"E{i}"

        # The one fallback left, and it is not the same kind of thing as the
        # five this phase removed: ``normalized_metric`` is a second field the
        # evidence contract really carries and this profile really admits, not a
        # literal invented here.  The canonical renderer and the routing policy
        # both read the pair the same way, and dropping it would replace a
        # present metric with "not specified".  Written out rather than as
        # ``or`` so that the distinction is visible at the point of use: what
        # was removed is *inventing* a value, not *reading* a declared field.
        metric = _stated(ev, "metric")
        if metric is None:
            metric = _stated(ev, "normalized_metric")

        # ``Source`` names where the evidence came from, in two components, and
        # each is stated independently -- so `doc-1:7`, `doc-1:not specified`
        # and `not specified:7` all say exactly what is known.  When neither
        # component is known the line is the single marker, which is what the
        # canonical renderer emits for it.
        document = _stated(ev, "document_id")
        page = _stated(ev, "page")
        source = (
            ABSENT if document is None and page is None
            else f"{_render(document)}:{_render(page)}"
        )

        lines.append(f"[{cite_id}]")
        lines.append(f"Metric: {_render(metric)}")
        lines.append(f"Period: {_render(_stated(ev, 'period'))}")
        lines.append(f"Scope: {_render(_stated(ev, 'scope'))}")
        lines.append(f"Value: {_render(_stated(ev, 'value'))}")
        lines.append(f"Unit: {_render(_stated(ev, 'unit'))}")
        lines.append(f"Currency: {_render(_stated(ev, 'currency'))}")
        lines.append(f"Scale: {_render(_stated(ev, 'scale'))}")
        lines.append(f"Source: {source}")

        if "source_text" in ev and ev["source_text"]:
            lines.append(f"Evidence: {ev['source_text']}")
        lines.append("")

    if calculation_result:
        c1_val = str(calculation_result.get("value", "")).strip()
        c1_unit = calculation_result.get("unit", "")
        c1_op = calculation_result.get("operation", "calculated_metric")
        lines.append("[VERIFIED CALCULATION]\n")
        lines.append("[C1]")
        lines.append(f"Operation: {c1_op}")
        lines.append(f"Value: {c1_val} {c1_unit}".strip())
        lines.append("")

    lines.append("[ANSWER RULES]")
    lines.append("1. Use only the verified evidence and calculation above.")
    lines.append("2. Do not introduce outside financial knowledge.")
    lines.append(
        "3. Preserve supplied numbers, periods, units, currencies and scales exactly."
    )
    lines.append("4. Do not recalculate canonical calculation results.")
    lines.append("5. Cite factual claims using the supplied [E#] / [C#] IDs.")
    lines.append(
        "6. If required evidence is missing, explicitly state that the provided evidence is insufficient."
    )
    lines.append("7. Answer concisely.")

    return "\n".join(lines)
