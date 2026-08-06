"""Gate 08 query construction, production identity mapping and pool safety."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.retrieval.candidate_identity import candidate_key, identity_from_candidate

from .query_plan_models import QueryPlan
from .shadow_index_reader import SearchHit, ShadowIndexReader


def stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_query(plan: QueryPlan, *, slot: Mapping[str, Any] | None = None, stage: str = "table") -> str:
    parts = [plan.raw_question]
    if plan.issuer:
        parts.append(str(plan.issuer))
    phrases = [str(slot.get("raw_metric_phrase"))] if slot else list(plan.metric_phrases)
    parts.extend(value for value in phrases if value)
    concepts = list(slot.get("concept_candidates") or ()) if slot else []
    if not concepts:
        concepts = [str(value) for item in plan.operand_slots for value in item.concept_candidates]
    parts.extend(concepts[:3])
    period = slot.get("period") if slot else None
    if period:
        parts.append(str(period))
    else:
        parts.extend(str(value) for value in plan.periods)
    if plan.statement_hint:
        parts.append(str(plan.statement_hint))
    if plan.operation:
        parts.append(str(plan.operation))
    if stage in {"fact", "row"} and slot:
        parts.append(str(slot.get("role") or ""))
    return " | ".join(str(value) for value in parts if str(value).strip())


def _norm(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _value_tokens(text: str) -> list[str]:
    return [token for token in re.findall(r"[-+]?\(?\d[\d, .]*\)?%?", text) if any(ch.isdigit() for ch in token)]


class ProductionCandidateMapper:
    """Map structural views to the same candidate identity used by production.

    The mapper reads only the frozen Raw BM25 store and the corpus filename
    mapping.  It never reads questions, labels, governance, or expected
    values.  A tie is deliberately returned as ambiguous rather than expanded
    into multiple strict identities.
    """

    def __init__(self, db_path: Path, corpus_path: Path, *, tenant_id: int = 1) -> None:
        self.db_path = Path(db_path)
        corpus = json.loads(Path(corpus_path).read_text(encoding="utf-8"))
        self.filename_to_document = {
            str(item["filename"]): str(item["document_id"]) for item in corpus.get("documents", [])
        }
        self.by_doc_page: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        self.by_key: dict[str, dict[str, Any]] = {}
        self._load(tenant_id)

    @staticmethod
    def _content_hash(content: str) -> str:
        return hashlib.sha256(str(content).replace("\r\n", "\n").replace("\r", "\n").encode()).hexdigest()

    def _load(self, tenant_id: int) -> None:
        connection = sqlite3.connect(f"file:{self.db_path.resolve().as_posix()}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT doc_id, content, metadata_json, user_id, doc_name FROM chunk_store WHERE user_id=?",
                (tenant_id,),
            )
            for doc_id, content, metadata_json, user_id, doc_name in rows:
                filename = str(doc_name or "")
                document_id = self.filename_to_document.get(filename)
                if not document_id:
                    continue
                try:
                    metadata = json.loads(metadata_json or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                block_type = str(metadata.get("type") or "text")
                if block_type == "table_cell":
                    continue
                page = metadata.get("page")
                try:
                    page_int = int(page)
                except (TypeError, ValueError):
                    continue
                item: dict[str, Any] = {
                    "doc_id": str(doc_id),
                    "evidence_id": str(doc_id),
                    "candidate_id": str(doc_id),
                    "content": str(content or ""),
                    "metadata": metadata,
                    "tenant_id": int(user_id),
                    "document_id": document_id,
                    "benchmark_document_id": document_id,
                    "filename": filename,
                    "page": page_int,
                    "block_type": block_type,
                    "parent_row_id": metadata.get("parent_row_id"),
                    "content_hash": self._content_hash(str(content or "")),
                }
                try:
                    item["candidate_key"] = candidate_key(identity_from_candidate(item))
                except Exception:
                    continue
                key = str(item["candidate_key"])
                self.by_key[key] = item
                self.by_doc_page[(document_id, page_int)].append(item)
        finally:
            connection.close()
        for candidates in self.by_doc_page.values():
            candidates.sort(key=lambda item: (str(item.get("block_type")), str(item.get("doc_id"))))

    def _view_features(self, view: Mapping[str, Any]) -> tuple[str, list[str], str]:
        text = str(view.get("retrieval_text") or "")
        metadata = view.get("metadata") or {}
        metric = str(metadata.get("metric_path") or "")
        if not metric:
            metric_match = re.search(r"^Metric:\s*(.+)$", text, re.MULTILINE)
            metric = metric_match.group(1).strip() if metric_match else ""
        values = _value_tokens(text)
        reported = re.search(r"(?:Reported Value|Value|Raw Value):\s*([^\n]+)", text, re.IGNORECASE)
        if reported:
            values = _value_tokens(reported.group(1)) + values
        return metric, list(dict.fromkeys(values)), text

    def map_view(self, view: Mapping[str, Any]) -> dict[str, Any]:
        metadata = view.get("metadata") or {}
        document_id = str(metadata.get("document_id") or "")
        pages = metadata.get("pdf_pages") or []
        try:
            page = int(pages[0])
        except (IndexError, TypeError, ValueError):
            return {"strict_candidate_status": "unmapped", "candidate_key": None, "mapping_score": 0.0}
        metric, values, text = self._view_features(view)
        metric_tokens = [token for token in _norm(metric).split() if len(token) > 2]
        unit_type = str(view.get("unit_type") or "")
        desired_types = {"table_row"} if unit_type in {"row", "atomic_fact", "comparison_fact", "bucket_fact"} else {"text", "table_row", "table"}
        candidates = self.by_doc_page.get((document_id, page), [])
        scored: list[tuple[float, dict[str, Any]]] = []
        for candidate in candidates:
            content = _norm(candidate.get("content"))
            score = 0.0
            if str(candidate.get("block_type")) in desired_types:
                score += 3.0
            if metric_tokens:
                score += min(5.0, sum(token in content.split() for token in metric_tokens) * 1.5)
                if _norm(metric) and _norm(metric) in content:
                    score += 4.0
            raw_content = str(candidate.get("content") or "")
            for value in values[:8]:
                numeric = re.sub(r"[^0-9]+", "", value)
                if numeric and numeric in re.sub(r"[^0-9]+", "", raw_content):
                    score += 2.0
            if unit_type == "section" and str(candidate.get("block_type")) == "text":
                score += 2.0
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda item: (-item[0], str(item[1]["candidate_key"])))
        if not scored:
            return {"strict_candidate_status": "unmapped", "candidate_key": None, "mapping_score": 0.0}
        best_score, best = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else -1.0
        # A strict source must be unique.  Tied same-page rows remain a
        # navigation trace and are never expanded to all identities.
        if len(scored) > 1 and best_score <= second_score + 0.01:
            return {
                "strict_candidate_status": "ambiguous",
                "candidate_key": None,
                "mapping_score": best_score,
                "candidate_tie_count": sum(abs(score - best_score) <= 0.01 for score, _ in scored),
            }
        return {
            "strict_candidate_status": "unique",
            "candidate_key": str(best["candidate_key"]),
            "mapping_score": best_score,
            "candidate_page": page,
            "candidate_block_type": best.get("block_type"),
        }


def hit_record(reader: ShadowIndexReader, hit: SearchHit, *, route: str, slot_id: str | None = None) -> dict[str, Any]:
    view = reader.view(hit.retrieval_view_id) or {}
    return {
        "retrieval_view_id": hit.retrieval_view_id,
        "unit_type": hit.unit_type,
        "route": route,
        "slot_id": slot_id,
        "bm25_rank": hit.bm25_rank,
        "dense_rank": hit.dense_rank,
        "fused_rank": hit.fused_rank,
        "bm25_score": hit.bm25_score,
        "dense_score": hit.dense_score,
        "rrf_score": hit.rrf_score,
        "metadata": view.get("metadata", {}),
    }


def raw_candidate_hash(candidates: Iterable[Mapping[str, Any]]) -> str:
    return stable_hash([
        {
            "candidate_key": item.get("candidate_key"),
            "stage_rank": item.get("stage_rank"),
            "score": item.get("score"),
        }
        for item in candidates
    ])


def merge_raw_protected(
    raw_candidates: list[dict[str, Any]],
    structured_records: list[dict[str, Any]],
    *,
    structured_k: int = 40,
) -> dict[str, Any]:
    raw_keys = [str(item.get("candidate_key")) for item in raw_candidates]
    raw_set = set(raw_keys)
    seen = set(raw_set)
    residual: list[dict[str, Any]] = []
    for item in structured_records:
        key = str(item.get("original_candidate_identity") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        residual.append(item)
        if len(residual) >= structured_k:
            break
    combined = list(raw_candidates) + [
        {
            "candidate_key": item["original_candidate_identity"],
            "structured_rank": index,
            "structured": item,
        }
        for index, item in enumerate(residual, 1)
    ]
    return {
        "raw_full_pool_before": raw_candidates,
        "raw_full_pool_after": combined[: len(raw_candidates)],
        "structured_residual": residual,
        "combined_pool": combined,
        "raw_candidate_loss": raw_keys != [str(item.get("candidate_key")) for item in combined[: len(raw_candidates)]],
        "raw_candidate_hash_before": raw_candidate_hash(raw_candidates),
        "raw_candidate_hash_after": raw_candidate_hash(combined[: len(raw_candidates)]),
    }
