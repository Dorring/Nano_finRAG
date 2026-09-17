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
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["render_specialist_prompt"]


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

        metric = ev.get("metric") or ev.get("normalized_metric") or "Metric"
        period = ev.get("period") or "Period"
        value = str(ev.get("value", "")).strip()
        unit = ev.get("unit") or "not specified"
        currency = ev.get("currency") or "not specified"
        scale = ev.get("scale") or "1"
        scope = ev.get("scope") or metric
        source_doc = ev.get("document_id") or "filing"

        # H2A-3B0.  This read ``ev.get("page") or 1``, and that expression
        # fabricated a page on *every* call: ``page`` was absent from the
        # SPECIALIST disclosure profile, so the lookup always missed and every
        # source line asserted page 1.  The model was being told, as
        # provenance, something nobody had established.
        #
        # Two errors were stacked in one expression, and they are worth keeping
        # apart.  ``or`` treats a genuine page 0 as absent, though the evidence
        # contract keeps 0 valid and absence preserved as absence -- a
        # truthiness test conflates the two.  And the fallback invented a
        # *specific* value rather than stating the absence, which is the
        # difference between an honest marker and a fabricated one: ``unit`` and
        # ``currency`` two lines above fall back to "not specified", which a
        # reader can tell from a real unit, while ``1`` is indistinguishable
        # from a page the extractor actually found.
        #
        # Absence is now rendered as absence, in the same idiom the neighbouring
        # fields already use.
        page = ev.get("page")
        page_label = "not specified" if page is None else str(page)

        lines.append(f"[{cite_id}]")
        lines.append(f"Metric: {metric}")
        lines.append(f"Period: {period}")
        lines.append(f"Scope: {scope}")
        lines.append(f"Value: {value}")
        lines.append(f"Unit: {unit}")
        lines.append(f"Currency: {currency}")
        lines.append(f"Scale: {scale}")
        lines.append(f"Source: {source_doc}:{page_label}")

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
