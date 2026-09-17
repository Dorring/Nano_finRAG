import os
import re
from typing import Any

import requests

from rag_v2.derived import (
    AdmittedDerivedArtifactV1,
    ModelDerivedArtifactV1,
    TransformationKindV1,
    verify_table_fidelity,
)
from rag_v2.evidence.disclosure import EvidenceDisclosureProfile, project


def format_table(table: Any) -> str:
    """Format a Camelot table as Markdown."""
    if table.df.empty or len(table.df) < 1:
        return ""

    # Camelot may embed newlines, carriage returns, or non-breaking spaces
    # inside individual dataframe cells.  Leaving them in place turns one
    # visual table row into several Markdown lines, disconnecting labels from
    # their numeric values downstream.  Normalise all whitespace within each
    # cell before serialising, while preserving the dataframe's columns.
    formatted_table = table.df.apply(
        lambda column: column.map(
            lambda value: re.sub(r"\s+", " ", str(value or "")).strip()
        )
    )

    try:
        final_table = formatted_table.rename(columns=formatted_table.iloc[0]).drop(formatted_table.index[0]).reset_index(drop=True)
    except Exception:
        final_table = formatted_table

    return final_table.to_markdown(index=False)


def is_usable_table_markdown(markdown: str) -> bool:
    """Reject parser output that looks like a flattened page, not a table.

    Camelot can return a non-empty dataframe after merging most of a visual
    table into one or two very long cells.  Treating that artifact as a valid
    table prevents the native PyMuPDF fallback from running and creates
    misleading row/cell evidence.  The check intentionally validates only
    structure and length; it does not depend on document names, page numbers,
    or financial terms.
    """
    rows = []
    for line in (markdown or "").splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if not cells or all(not cell for cell in cells):
            continue
        if all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells):
            continue
        rows.append(cells)

    if len(rows) < 2:
        return False
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        return False

    non_empty = [cell for row in rows for cell in row if cell]
    if not non_empty:
        return False
    # One giant flattened cell is not a reliable row/column structure. Keep
    # the threshold generous for narrative labels while failing malformed PDF
    # extraction that packs a whole table into a single field.
    if max(len(cell) for cell in non_empty) > 320:
        return False
    return True


def _read_tables(pdf_path: str, pages: str) -> list[Any]:
    """Best-effort Camelot extraction. Failures must not block PDF ingest."""

    # Imported here rather than at module scope.  Camelot is a heavy PDF library
    # that only the extraction functions need, and importing it at module scope
    # meant this module -- and ``ingest.py``, which imports it -- could not be
    # imported at all on a checkout without it.  The consequence was concrete:
    # ``enhance_table_with_context`` needs no PDF stack, and no test could
    # exercise it.  H2A-3B0 moved the specialist renderer out of the
    # torch-importing module for the same reason; a test that cannot run is not
    # a test, and the one place this module needed camelot is here.
    import camelot

    try:
        tables = camelot.read_pdf(pdf_path, pages=pages, flavor="stream", edge_tol=50, row_tol=10)
        if len(tables) > 0:
            return list(tables)
        print("Stream mode found 0 tables, falling back to lattice.")
    except Exception as exc:
        print(f"Stream mode failed: {exc}, falling back to lattice.")

    try:
        return list(camelot.read_pdf(pdf_path, pages=pages, flavor="lattice"))
    except Exception as exc:
        print(f"Lattice mode also failed: {exc}; continuing without table chunks.")
        return []


def _safe_table_bbox(table: Any) -> tuple | None:
    try:
        bbox = getattr(table, "bbox", None)
        if not bbox:
            return None
        return tuple(bbox)
    except Exception as exc:
        print(f"Skipping table bbox: {exc}")
        return None


def extract_tables_with_camelot(pdf_path: str, pages: str = "1-end") -> dict[int, list[dict]]:
    """
    Extract PDF tables with Camelot as a non-critical enhancement.

    Returns: {page_num: [{"md": table_markdown, "bbox": optional_bbox}, ...]}.
    A failure to detect tables should never fail document upload; text extraction
    in ingest.py remains the primary ingestion path.
    """
    tables_by_page: dict[int, list[dict]] = {}
    tables = _read_tables(pdf_path, pages)

    for table in tables:
        try:
            page_num = int(table.page)
            table_markdown = format_table(table=table)
        except Exception as exc:
            print(f"Skipping malformed table: {exc}")
            continue

        if not table_markdown or not is_usable_table_markdown(table_markdown):
            if table_markdown:
                print(f"Skipping structurally malformed table on page {page_num}.")
            continue

        tables_by_page.setdefault(page_num, []).append({
            "md": table_markdown,
            "bbox": _safe_table_bbox(table),
        })

    print(f"Extracted {sum(len(v) for v in tables_by_page.values())} usable tables in total")
    return tables_by_page


def _admission_record(
    candidate: ModelDerivedArtifactV1,
    admission: Any,
) -> dict[str, Any]:
    """The bounded record a chunk carries about how its content was admitted.

    H2A-3D.  Names, a reason and lineage; never content.  A reader of a chunk
    can now tell extracted source text from a verified representation of it,
    which is precisely the distinction the pipeline could not make before --
    model-written text and parser-produced text arrived under the same key with
    the same shape.

    ``location`` names the structured position a refusal turned on, and it is
    source-derived: a row label and a column header from the authoritative
    table, not the model's text.  An admission record should be able to say
    *where* without quoting what it refused.
    """

    return {
        "transformation": candidate.transformation.value,
        "admitted": bool(admission.admitted),
        "reason": admission.reason.value,
        "location": admission.location,
        "source_reference": candidate.source_reference,
        "artifact_id": candidate.artifact_id,
        "model_id": candidate.model_id,
    }


def enhance_table_with_context(
    table_md: dict,
    page_text: str,
    page_num: int,
    *,
    source_reference: str,
) -> dict:
    """Clean an extracted table with a model, and admit the result before using it.

    H2A-3D, closing F6.  What this used to do was take the ``CLEANED TABLE:``
    section of the model's answer and return it as ``content`` -- the same key,
    with the same shape, that the fallback returns.  From that point on nothing
    in the pipeline could tell model-produced text from parser-produced text, and
    the model's answer became embedded, BM25-indexed, expanded into
    ``parent_excerpt`` and read as evidence by every downstream consumer.

    Generation succeeding was being treated as admission succeeding.  The
    prompt's "Do NOT change numeric values" is a request, and a request is not a
    check; the ``cleaned_table = table_md["md"]`` default above it was a
    fallback for *failure*, never for *wrongness*.

    So the answer is now a **candidate** first.  It becomes a
    ``ModelDerivedArtifactV1``, a deterministic verifier compares its numeric
    content against the authoritative pre-model table, and only an admitted
    artifact's content leaves this function.  Everything else -- no key, a
    failed call, no ``CLEANED TABLE:`` section, or a failed verification --
    returns the authoritative table unchanged.

    ``content`` is therefore always safe to write into a chunk, and
    ``admission_record`` is the bounded record of which of the two it is.
    ``None`` there means no model produced content here at all, so what came
    back is source-original.

    What is deliberately *not* returned any more is the model's prose summary.
    It is unverifiable -- nothing in the source says what a description of the
    table ought to say -- and it used to be folded into both ``content`` and
    ``parent_excerpt``, which is the same defect one step removed: model text
    occupying a trusted field.  The prompt still asks for it; the answer is
    simply not used.

    ``source_reference`` names the authoritative artifact this table came from,
    so the admitted content keeps its lineage instead of arriving as a piece of
    text with no origin.
    """
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    nvidia_model = os.getenv("NVIDIA_MODEL_NAME", "meta/llama-3.1-8b-instruct")

    if not nvidia_api_key:
        print("NVIDIA_API_KEY not set, skipping table enhancement")
        return {"content": table_md["md"], "admission_record": None}

    url = "https://integrate.api.nvidia.com/v1/chat/completions"

    # The one boundary where raw document text is the legitimate input.  Routing
    # it through the disclosure authority does not strip anything -- it records
    # that this model may see page text, and why, so the question has an answer
    # instead of being an omission.  Anything not named in the profile cannot
    # be added to this prompt without failing here first.
    disclosed = project(
        {
            "page_text": page_text,
            "table_markdown": table_md["md"],
            "page_num": page_num,
        },
        profile=EvidenceDisclosureProfile.INGEST_TABLE,
    )
    page_text = disclosed["page_text"]
    table_md = {**table_md, "md": disclosed["table_markdown"]}
    page_num = disclosed["page_num"]

    prompt = f"""You are a data preprocessing assistant for a retrieval system operating on financial documents.

Rules:
1. Write a concise 2-3 sentence description of what the table represents
2. Clean the table: remove stray text, preserve headers and alignment
3. Do NOT change numeric values, dates, currencies, or units
4. Do NOT infer missing values or recompute totals

Page number: {page_num}

Table (markdown):
{table_md["md"]}

Surrounding text from the same page:
{page_text}

Return the result in this format:
TABLE SUMMARY:
<summary text>

CLEANED TABLE:
<markdown table>"""

    payload = {
        "model": nvidia_model,
        "temperature": 0.2,
        "top_p": 0.7,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_tokens": 1024,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Bearer {nvidia_api_key}",
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response_json = response.json()

        enhanced = response_json["choices"][0]["message"]["content"].strip()

        table_match = re.search(r"CLEANED TABLE:\s*(.*)", enhanced, re.DOTALL)
        parsed_table = table_match.group(1).strip() if table_match else ""

        if not parsed_table:
            # A model that answered without a table produced nothing to admit.
            # This is the same "no candidate" case as a missing key, not a
            # rejection -- there is nothing to reject.
            return {"content": table_md["md"], "admission_record": None}

        candidate = ModelDerivedArtifactV1(
            artifact_id=f"{source_reference}::table-cleaning",
            content=parsed_table,
            transformation=TransformationKindV1.TABLE_CLEANING,
            source_reference=source_reference,
            provider_id="nvidia",
            model_id=nvidia_model,
        )
        admission = verify_table_fidelity(candidate, table_md["md"])

    except Exception as exc:
        print(f"NVIDIA API table enhancement failed: {exc}")
        return {"content": table_md["md"], "admission_record": None}

    if admission.admitted:
        # The only way this constructor succeeds, and therefore the only way
        # model content leaves this function.
        admitted = AdmittedDerivedArtifactV1(candidate, admission)
        return {
            "content": admitted.content,
            "admission_record": _admission_record(candidate, admission),
        }

    print(
        f"Rejected a model-cleaned table on page {page_num}: "
        f"{admission.reason.value} at {admission.location or 'the table structure'}"
    )
    return {
        "content": table_md["md"],
        "admission_record": _admission_record(candidate, admission),
    }
