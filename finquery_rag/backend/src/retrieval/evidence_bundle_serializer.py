"""Evidence-safe candidate serialization for cross-encoder reranking."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class RerankEvidenceBundle:
    candidate_id: str
    document_name: str
    page: int | None
    section_path: tuple[str, ...] = ()
    block_type: str = "text"
    table_title: str | None = None
    table_headers: tuple[str, ...] = ()
    row_label: str | None = None
    row_text: str | None = None
    unit_context: str | None = None
    narrative_text: str | None = None
    parent_summary: str | None = None

def _text(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None

def build_evidence_bundle(candidate: dict[str, Any]) -> RerankEvidenceBundle:
    metadata = candidate.get("metadata") or {}
    block_type = str(metadata.get("type") or "text")
    section = metadata.get("section_path") or ()
    section = tuple(part.strip() for part in section.split(">") if part.strip()) if isinstance(section, str) else tuple(str(part) for part in section if str(part).strip())
    headers = metadata.get("table_headers") or metadata.get("headers") or ()
    headers = tuple(part.strip() for part in headers.split("|") if part.strip()) if isinstance(headers, str) else tuple(str(part) for part in headers if str(part).strip())
    content = str(candidate.get("content") or "").strip()
    return RerankEvidenceBundle(
        candidate_id=str(metadata.get("parent_id") or candidate.get("doc_id") or ""),
        document_name=str(metadata.get("doc_name") or metadata.get("filename") or ""),
        page=metadata.get("page"),
        section_path=section,
        block_type="table_row" if block_type == "table_cell" else block_type,
        table_title=_text(metadata.get("table_title")),
        table_headers=headers,
        row_label=_text(metadata.get("row_label")),
        row_text=_text(metadata.get("row_text") or (content if block_type == "table_row" else None)),
        unit_context=_text(metadata.get("unit_context") or metadata.get("unit")),
        narrative_text=None if block_type in {"table_row", "table_cell"} else content,
        parent_summary=_text(metadata.get("parent_title")),
    )

def serialize_for_reranking(bundle: RerankEvidenceBundle) -> str:
    fields = [
        ("DOCUMENT", bundle.document_name), ("PAGE", str(bundle.page) if bundle.page is not None else None),
        ("SECTION", " > ".join(bundle.section_path)), ("TYPE", bundle.block_type),
        ("TABLE", bundle.table_title), ("HEADERS", " | ".join(bundle.table_headers)),
        ("ROW_LABEL", bundle.row_label), ("ROW", bundle.row_text), ("UNIT", bundle.unit_context),
        ("TEXT", bundle.narrative_text), ("PARENT", bundle.parent_summary),
    ]
    return "\n".join(f"{label}: {value}" for label, value in fields if value)

def build_token_budgeted_text(bundle: RerankEvidenceBundle, tokenizer: Any | None = None, max_length: int = 1024) -> str:
    required = [("DOCUMENT", bundle.document_name), ("PAGE", str(bundle.page) if bundle.page is not None else None), ("SECTION", " > ".join(bundle.section_path)), ("TYPE", bundle.block_type), ("TABLE", bundle.table_title), ("HEADERS", " | ".join(bundle.table_headers)), ("ROW_LABEL", bundle.row_label), ("ROW", bundle.row_text), ("UNIT", bundle.unit_context)]
    optional = [("TEXT", bundle.narrative_text), ("PARENT", bundle.parent_summary)]
    output: list[str] = []
    for label, value in required + optional:
        if not value:
            continue
        proposed = "\n".join(output + [f"{label}: {value}"])
        tokens = len(proposed.split()) if tokenizer is None else len(tokenizer(proposed, add_special_tokens=False).get("input_ids", []))
        if tokens <= max_length or label in {"ROW_LABEL", "ROW", "HEADERS"}:
            output.append(f"{label}: {value}")
    return "\n".join(output)
