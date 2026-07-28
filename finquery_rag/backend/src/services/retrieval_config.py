"""Retrieval model configuration helpers.

This module centralizes embedding/reranker settings without loading models.
It keeps CI and preflight checks offline while making production overrides
explicit.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any


DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
DEFAULT_RERANKER = "heuristic"


def get_embedding_model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL).strip() or DEFAULT_EMBEDDING_MODEL


def get_reranker_name() -> str:
    return (os.getenv("RERANKER_PROVIDER") or os.getenv("RAG_RERANKER") or DEFAULT_RERANKER).strip() or DEFAULT_RERANKER


def get_reranker_model() -> str | None:
    value = os.getenv("RERANKER_MODEL_NAME") or os.getenv("RAG_RERANKER_MODEL")
    if value is None:
        return None
    value = value.strip()
    return value or None


def build_retrieval_model_config() -> dict[str, Any]:
    """Return non-secret retrieval model config and validation warnings."""
    embedding_model = get_embedding_model_name()
    reranker = get_reranker_name()
    reranker_model = get_reranker_model()
    errors: list[str] = []
    warnings: list[str] = []

    embedding_path_exists = _path_exists_if_local(embedding_model)
    reranker_model_path_exists = _path_exists_if_local(reranker_model)

    if reranker == "cross-encoder" and not reranker_model:
        errors.append("RAG_RERANKER_MODEL is required when RAG_RERANKER=cross-encoder")
    if _looks_like_local_path(embedding_model) and not embedding_path_exists:
        errors.append("EMBEDDING_MODEL_NAME points to a missing local path")
    if reranker_model and _looks_like_local_path(reranker_model) and not reranker_model_path_exists:
        errors.append("RAG_RERANKER_MODEL points to a missing local path")
    if not _looks_like_local_path(embedding_model):
        warnings.append("embedding model is a remote/model-hub name; ensure it is cached or downloads are allowed")
    if reranker in {"cross-encoder", "bge_v2_m3", "bge-v2-m3"} and reranker_model and not _looks_like_local_path(reranker_model):
        warnings.append("cross-encoder reranker model is a remote/model-hub name; prefer a local path for offline deployment")

    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "embedding_model": embedding_model,
        "embedding_model_is_local_path": _looks_like_local_path(embedding_model),
        "embedding_model_path_exists": embedding_path_exists,
        "reranker": reranker,
        "reranker_model_configured": bool(reranker_model),
        "reranker_model": reranker_model,
        "reranker_model_is_local_path": _looks_like_local_path(reranker_model),
        "reranker_model_path_exists": reranker_model_path_exists,
    }


def _looks_like_local_path(value: str | None) -> bool:
    if not value:
        return False
    return (
        value.startswith((".", "/", "\\", "~"))
        or ":\\" in value
        or ":/" in value
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
    )


def _path_exists_if_local(value: str | None) -> bool | None:
    if not _looks_like_local_path(value):
        return None
    return Path(value).expanduser().exists()
