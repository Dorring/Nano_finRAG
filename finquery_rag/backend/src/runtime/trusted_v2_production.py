"""Canonical production construction for the Trusted Financial Runtime V2.

The TV2 component factories intentionally use dependency injection.  That is
useful for component tests, but it also leaves a deployment seam: an
application process needs one *real* factory that constructs the complete V2
graph.  This module is that seam.

The builder is deliberately strict.  It requires provisioned R4 indexes,
structured fact materialization, provider credentials, and the verified local
specialist checkpoint.  It never creates a deterministic test provider, uses
the legacy V1 retriever, or silently falls back to V1.  Deployments may still
override it with ``TRUSTED_V2_RUNTIME_BUILDER=module:callable`` when their
assets live outside this repository.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
import sqlite3
import threading
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.evidence import BailianBinderProvider, SemanticBinderService
from rag_v2.supervisor import (
    APIProvider,
    BailianProvider,
    SupervisorService,
    UnknownSemanticPolicy,
)

from .runtime_contract import FinancialQueryRequest
from .trusted_v2_adapter import TrustedFinancialRuntimeV2
from .trusted_v2_binder import SemanticEvidenceEvaluationCapability
from .trusted_v2_calculation import DeterministicCalculationCapability
from .trusted_v2_capabilities import TrustedV2CapabilityPorts
from .trusted_v2_factory import build_trusted_v2_runtime
from .trusted_v2_generation import (
    DeterministicFactRenderer,
    LocalSpecialistGenerationAdapter,
    TrustedV2GenerationCapability,
)
from .trusted_v2_r4 import CandidateDirectR4Policy, R4RetrievalCapability
from .trusted_v2_validation import TrustedReleaseValidationCapability


class TrustedV2ProductionConfigurationError(RuntimeError):
    """Raised when the canonical V2 production graph is not constructible."""


_R4_BM25_LANES = (
    "candidate_raw_bm25",
    "candidate_structured_bm25",
)
_R4_DENSE_LANES = (
    "candidate_raw_dense",
    "candidate_structured_dense",
)
_R4_LANES = _R4_BM25_LANES + _R4_DENSE_LANES


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _text(value)
        if text:
            return text
    return None


def _stable_unique(values: Iterable[Any]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _text(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return tuple(result)


def _env(environ: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    value = environ.get(name, default)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_env(environ: Mapping[str, str], name: str) -> str:
    value = _env(environ, name)
    if not value:
        raise TrustedV2ProductionConfigurationError(
            f"missing required V2 configuration: {name}"
        )
    return value


def _path_env(
    environ: Mapping[str, str],
    name: str,
    *,
    directory: bool,
) -> Path:
    raw = _required_env(environ, name)
    path = Path(raw).expanduser()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise TrustedV2ProductionConfigurationError(
            f"{name} must point to an existing {kind}: {path}"
        )
    return path.resolve()


def _bool_env(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    value = _env(environ, name)
    if value is None:
        return default
    normalized = value.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise TrustedV2ProductionConfigurationError(
        f"{name} must be a boolean (true/false)"
    )


def _int_env(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int = 0,
) -> int:
    value = _env(environ, name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TrustedV2ProductionConfigurationError(
            f"{name} must be an integer"
        ) from exc
    if parsed < minimum:
        raise TrustedV2ProductionConfigurationError(
            f"{name} must be >= {minimum}"
        )
    return parsed


def _float_env(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    value = _env(environ, name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TrustedV2ProductionConfigurationError(
            f"{name} must be a number"
        ) from exc
    if minimum is not None and parsed < minimum:
        raise TrustedV2ProductionConfigurationError(
            f"{name} must be >= {minimum}"
        )
    return parsed


def _read_json_rows(path: Path) -> list[Mapping[str, Any]]:
    """Read a fact registry without executing code or accepting Gold hints."""

    try:
        opener = gzip.open if path.name.casefold().endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as handle:
            if path.name.casefold().endswith(".jsonl") or path.name.casefold().endswith(
                ".jsonl.gz"
            ):
                rows: list[Mapping[str, Any]] = []
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise TrustedV2ProductionConfigurationError(
                            f"fact store line {line_number} is not valid JSON"
                        ) from exc
                    if not isinstance(value, Mapping):
                        raise TrustedV2ProductionConfigurationError(
                            f"fact store line {line_number} must be an object"
                        )
                    rows.append(value)
                return rows
            payload = json.load(handle)
    except OSError as exc:
        raise TrustedV2ProductionConfigurationError(
            f"could not read V2 fact store: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise TrustedV2ProductionConfigurationError(
            f"fact store is not valid JSON: {path}"
        ) from exc

    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, Mapping):
        rows = None
        for key in ("facts", "records", "items", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                rows = candidate
                break
        if rows is None:
            # A mapping keyed by candidate ID is convenient for small
            # deployments.  It is still normalized through the same strict
            # record validation below.
            rows = []
            for candidate_key, value in payload.items():
                if not isinstance(value, Mapping):
                    raise TrustedV2ProductionConfigurationError(
                        "fact store mapping values must be objects"
                    )
                record = dict(value)
                record.setdefault("candidate_key", str(candidate_key))
                rows.append(record)
    else:
        raise TrustedV2ProductionConfigurationError(
            "V2 fact store must contain an array or object"
        )

    normalized_rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise TrustedV2ProductionConfigurationError(
                f"fact store record {index} must be an object"
            )
        normalized_rows.append(row)
    return normalized_rows


class StructuredFactStore:
    """Read-only candidate-key -> structured FinancialFact materializer.

    R4 indexes intentionally store retrieval views, not trusted numeric fact
    objects.  The production graph therefore needs a separately provisioned
    registry.  This store keeps that boundary explicit and rejects records
    whose provenance cannot support the Binder and release validator.
    """

    def __init__(self, path: Path | str, *, require_citation_id: bool = True) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise TrustedV2ProductionConfigurationError(
                f"V2 fact store does not exist: {self.path}"
            )
        self.require_citation_id = bool(require_citation_id)
        self._by_candidate: dict[str, dict[str, Any]] = {}
        self._load()
        if not self._by_candidate:
            raise TrustedV2ProductionConfigurationError(
                f"V2 fact store contains no materializable candidates: {self.path}"
            )
        self.sha256 = _sha256_file(self.path)

    @staticmethod
    def _candidate_keys(record: Mapping[str, Any]) -> tuple[str, ...]:
        values: list[Any] = []
        for key in ("candidate_key", "candidate_id", "candidate_keys", "candidate_ids"):
            value = record.get(key)
            if isinstance(value, (list, tuple, set)):
                values.extend(value)
            elif value is not None:
                values.append(value)
        return _stable_unique(values)

    def _normalize_record(self, raw: Mapping[str, Any], index: int) -> dict[str, Any]:
        record = copy.deepcopy(dict(raw))
        candidate_keys = self._candidate_keys(record)
        if not candidate_keys:
            raise TrustedV2ProductionConfigurationError(
                f"fact store record {index} is missing candidate_key/candidate_id"
            )

        evidence_id = _first_text(record.get("evidence_id"), record.get("fact_id"))
        if evidence_id is None:
            raise TrustedV2ProductionConfigurationError(
                f"fact store record {index} is missing evidence_id/fact_id"
            )
        record["evidence_id"] = evidence_id
        record.setdefault("fact_id", evidence_id)

        citation_id = _first_text(record.get("citation_id"))
        if citation_id is None:
            raw_citations = record.get("citation_ids")
            if isinstance(raw_citations, (list, tuple)) and len(raw_citations) == 1:
                citation_id = _first_text(raw_citations[0])
        if self.require_citation_id and citation_id is None:
            raise TrustedV2ProductionConfigurationError(
                f"fact store record {index} is missing structured citation_id"
            )
        if citation_id is not None:
            record["citation_id"] = citation_id

        if record.get("provenance_complete") is not True:
            raise TrustedV2ProductionConfigurationError(
                f"fact store record {index} must set provenance_complete=true"
            )
        record["provenance_complete"] = True

        # Normalize names emitted by the sealed FinancialFactV1 artifacts.  No
        # values or IDs are inferred from answer text.
        aliases = {
            "metric": ("normalized_metric", "raw_metric"),
            "period": ("normalized_period", "raw_period"),
            "scale": ("normalized_scale", "raw_scale"),
            "value": ("parsed_numeric_value", "raw_value"),
            "page": ("pdf_page",),
            "source_id": ("physical_source_id", "document_id"),
            "source_text": ("evidence_text", "content", "raw_content", "row_label"),
        }
        for canonical, alternatives in aliases.items():
            existing = record.get(canonical)
            if existing is not None and (
                not isinstance(existing, str) or existing.strip()
            ):
                continue
            for key in alternatives:
                value = record.get(key)
                if value is None or (isinstance(value, str) and not value.strip()):
                    continue
                # Preserve typed numeric/page values; the binder/calculator
                # consumes structured fields and must not receive textified
                # numbers merely because an alias was used.
                record[canonical] = copy.deepcopy(value)
                break

        if not _first_text(
            record.get("source_id"),
            record.get("physical_source_id"),
            record.get("document_id"),
            record.get("cell_id"),
        ):
            raise TrustedV2ProductionConfigurationError(
                f"fact store record {index} is missing physical source identity"
            )

        record["candidate_key"] = candidate_keys[0]
        record.setdefault("candidate_id", candidate_keys[0])
        record["candidate_ids"] = list(candidate_keys)
        return record

    def _load(self) -> None:
        for index, raw in enumerate(_read_json_rows(self.path), 1):
            record = self._normalize_record(raw, index)
            for candidate_key in record["candidate_ids"]:
                existing = self._by_candidate.get(candidate_key)
                if existing is not None:
                    raise TrustedV2ProductionConfigurationError(
                        f"fact store has ambiguous duplicate candidate key: {candidate_key}"
                    )
                self._by_candidate[candidate_key] = record

    @property
    def candidate_count(self) -> int:
        return len(self._by_candidate)

    @property
    def candidate_keys(self) -> tuple[str, ...]:
        return tuple(self._by_candidate)

    def materialize(self, candidate_key: str) -> Mapping[str, Any]:
        key = _text(candidate_key)
        record = self._by_candidate.get(key)
        if record is None:
            raise KeyError(f"candidate_not_materialized:{key}")
        materialized = copy.deepcopy(record)
        materialized["candidate_key"] = key
        materialized.setdefault("candidate_id", key)
        return materialized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_r4_index(index_dir: Path | str) -> dict[str, Any]:
    """Validate the four-lane R4 index before it enters the runtime graph."""

    root = Path(index_dir).expanduser().resolve()
    if not root.is_dir():
        raise TrustedV2ProductionConfigurationError(
            f"R4 index directory does not exist: {root}"
        )
    metadata_path = root / "candidate-metadata.sqlite"
    if not metadata_path.is_file():
        raise TrustedV2ProductionConfigurationError(
            f"R4 index metadata is missing: {metadata_path}"
        )

    lane_files: dict[str, str] = {}
    for lane in _R4_BM25_LANES:
        path = root / lane / "bm25" / "index.sqlite"
        if not path.is_file():
            raise TrustedV2ProductionConfigurationError(
                f"R4 BM25 lane is missing: {path}"
            )
        lane_files[lane] = str(path)
    for lane in _R4_DENSE_LANES:
        for filename in ("ids.json", "vectors.npy"):
            path = root / lane / "dense" / filename
            if not path.is_file():
                raise TrustedV2ProductionConfigurationError(
                    f"R4 dense lane is missing: {path}"
                )
        lane_files[lane] = str(root / lane / "dense")

    try:
        with sqlite3.connect(
            f"file:{metadata_path.as_posix()}?mode=ro", uri=True
        ) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(view_metadata)")
            }
            required_columns = {
                "lane",
                "view_id",
                "candidate_key",
                "view_type",
                "retrieval_text",
                "document_id",
                "metadata_json",
            }
            if not required_columns <= columns:
                missing = sorted(required_columns - columns)
                raise TrustedV2ProductionConfigurationError(
                    "R4 metadata schema is missing columns: " + ", ".join(missing)
                )
            row_count = int(
                connection.execute("SELECT COUNT(*) FROM view_metadata").fetchone()[0]
            )
            lanes = {
                str(row[0])
                for row in connection.execute("SELECT DISTINCT lane FROM view_metadata")
            }
    except sqlite3.Error as exc:
        raise TrustedV2ProductionConfigurationError(
            f"R4 metadata cannot be opened read-only: {metadata_path}"
        ) from exc
    if row_count <= 0:
        raise TrustedV2ProductionConfigurationError(
            f"R4 metadata contains no retrieval views: {metadata_path}"
        )
    missing_lanes = sorted(set(_R4_LANES) - lanes)
    if missing_lanes:
        raise TrustedV2ProductionConfigurationError(
            "R4 metadata is missing lanes: " + ", ".join(missing_lanes)
        )
    return {
        "index_dir": str(root),
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "row_count": row_count,
        "lanes": lane_files,
    }


def _provider_common(
    environ: Mapping[str, str],
    prefix: str,
    *,
    fallback_prefix: str | None = None,
) -> tuple[str, str, str]:
    def value(name: str) -> str:
        direct = _env(environ, f"{prefix}{name}")
        if direct:
            return direct
        if fallback_prefix:
            fallback = _env(environ, f"{fallback_prefix}{name}")
            if fallback:
                return fallback
        raise TrustedV2ProductionConfigurationError(
            f"missing required V2 configuration: {prefix}{name}"
        )

    return value("BASE_URL"), value("API_KEY"), value("MODEL")


def _build_supervisor(environ: Mapping[str, str]) -> SupervisorService:
    provider_name = (_env(environ, "V2_SUPERVISOR_PROVIDER", "bailian") or "bailian").casefold()
    base_url, api_key, model_name = _provider_common(environ, "V2_SUPERVISOR_")
    temperature = _float_env(environ, "V2_SUPERVISOR_TEMPERATURE", 0.0, minimum=0.0)
    enable_thinking = _bool_env(environ, "V2_SUPERVISOR_ENABLE_THINKING", False)
    try:
        if provider_name == "bailian":
            provider = BailianProvider(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                enable_thinking=enable_thinking,
                temperature=temperature,
                max_tokens=_int_env(environ, "V2_SUPERVISOR_MAX_TOKENS", 512, minimum=1),
                timeout=_float_env(environ, "V2_SUPERVISOR_TIMEOUT_SECONDS", 180.0, minimum=0.1),
                max_retries=0,
            )
        elif provider_name == "api":
            provider = APIProvider(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=_int_env(environ, "V2_SUPERVISOR_MAX_TOKENS", 512, minimum=1),
                timeout=_float_env(environ, "V2_SUPERVISOR_TIMEOUT_SECONDS", 120.0, minimum=0.1),
                provider_role="supervisor",
                model_role="strong_general_llm",
                structured_output=True,
            )
        else:
            raise TrustedV2ProductionConfigurationError(
                "V2_SUPERVISOR_PROVIDER must be 'bailian' or 'api'"
            )
    except TrustedV2ProductionConfigurationError:
        raise
    except Exception as exc:
        raise TrustedV2ProductionConfigurationError(
            f"could not construct V2 Supervisor provider '{provider_name}'"
        ) from exc
    return SupervisorService(provider)


def _build_binder(environ: Mapping[str, str]) -> SemanticBinderService:
    provider_name = (_env(environ, "V2_BINDER_PROVIDER", "bailian") or "bailian").casefold()
    if provider_name != "bailian":
        raise TrustedV2ProductionConfigurationError(
            "V2_BINDER_PROVIDER must be 'bailian'; no generic Binder provider is registered"
        )
    base_url, api_key, model_name = _provider_common(
        environ,
        "V2_BINDER_",
        fallback_prefix="V2_SUPERVISOR_",
    )
    try:
        provider = BailianBinderProvider(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            enable_thinking=_bool_env(environ, "V2_BINDER_ENABLE_THINKING", False),
            temperature=_float_env(environ, "V2_BINDER_TEMPERATURE", 0.0, minimum=0.0),
            timeout=_float_env(environ, "V2_BINDER_TIMEOUT_SECONDS", 180.0, minimum=0.1),
            max_retries=0,
        )
    except Exception as exc:
        raise TrustedV2ProductionConfigurationError(
            "could not construct V2 Semantic Binder provider"
        ) from exc
    return SemanticBinderService(provider)


def _build_specialist(environ: Mapping[str, str]) -> LocalSpecialistGenerationAdapter:
    checkpoint = _path_env(
        environ,
        "TRUSTED_V2_SPECIALIST_CHECKPOINT",
        directory=False,
    )
    try:
        # Importing this module imports torch and the NanoChat model runtime;
        # keep it lazy so contract/unit tests remain CPU-safe.
        from src.generation.local_specialist_generator import LocalSpecialistGenerator

        specialist = LocalSpecialistGenerator(
            checkpoint_path=checkpoint,
            device=_env(environ, "TRUSTED_V2_SPECIALIST_DEVICE"),
            max_new_tokens=_int_env(
                environ,
                "TRUSTED_V2_SPECIALIST_MAX_NEW_TOKENS",
                128,
                minimum=1,
            ),
            temperature=_float_env(
                environ,
                "TRUSTED_V2_SPECIALIST_TEMPERATURE",
                0.0,
                minimum=0.0,
            ),
        )
        specialist.load()
    except TrustedV2ProductionConfigurationError:
        raise
    except Exception as exc:
        raise TrustedV2ProductionConfigurationError(
            "could not load the configured V2 Financial Specialist checkpoint"
        ) from exc
    return LocalSpecialistGenerationAdapter(specialist)


def _build_budget(environ: Mapping[str, str]) -> AdaptiveRAGBudgetV1:
    return AdaptiveRAGBudgetV1(
        max_replan_rounds=_int_env(environ, "V2_MAX_REPLANS", 2, minimum=0),
        max_total_tool_calls=_int_env(environ, "V2_MAX_TOOL_CALLS", 5, minimum=1),
        max_same_tool_retry=_int_env(
            environ,
            "V2_MAX_SAME_TOOL_RETRIES",
            1,
            minimum=0,
        ),
        max_identical_query_retry=_int_env(
            environ,
            "V2_MAX_IDENTICAL_QUERY_RETRIES",
            0,
            minimum=0,
        ),
    )


@dataclass
class TrustedV2RuntimeResources:
    """Process-scoped resources shared by per-request V2 runtimes."""

    index_reader: Any
    fact_store: StructuredFactStore
    supervisor: SupervisorService
    binder: SemanticBinderService
    specialist: LocalSpecialistGenerationAdapter
    budget: AdaptiveRAGBudgetV1
    config_fingerprint: str
    index_manifest: Mapping[str, Any]


def _configuration_fingerprint(environ: Mapping[str, str]) -> str:
    names = (
        "TRUSTED_V2_R4_INDEX_DIR",
        "TRUSTED_V2_FACT_STORE_PATH",
        "TRUSTED_V2_SPECIALIST_CHECKPOINT",
        "TRUSTED_V2_SPECIALIST_DEVICE",
        "TRUSTED_V2_SPECIALIST_MAX_NEW_TOKENS",
        "TRUSTED_V2_SPECIALIST_TEMPERATURE",
        "V2_SUPERVISOR_PROVIDER",
        "V2_SUPERVISOR_BASE_URL",
        "V2_SUPERVISOR_API_KEY",
        "V2_SUPERVISOR_MODEL",
        "V2_SUPERVISOR_ENABLE_THINKING",
        "V2_SUPERVISOR_TEMPERATURE",
        "V2_SUPERVISOR_MAX_TOKENS",
        "V2_SUPERVISOR_TIMEOUT_SECONDS",
        "V2_BINDER_PROVIDER",
        "V2_BINDER_BASE_URL",
        "V2_BINDER_API_KEY",
        "V2_BINDER_MODEL",
        "V2_BINDER_ENABLE_THINKING",
        "V2_BINDER_TEMPERATURE",
        "V2_BINDER_TIMEOUT_SECONDS",
        "V2_MAX_REPLANS",
        "V2_MAX_TOOL_CALLS",
        "V2_MAX_SAME_TOOL_RETRIES",
        "V2_MAX_IDENTICAL_QUERY_RETRIES",
    )
    values: dict[str, str | None] = {}
    for name in names:
        value = _env(environ, name)
        if name.endswith("API_KEY") and value:
            value = hashlib.sha256(value.encode("utf-8")).hexdigest()
        values[name] = value
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_trusted_v2_production_configuration(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run a cheap, side-effect-free deployment preflight.

    Provider clients and the specialist model are not loaded here.  This
    function only verifies the paths and index/fact contracts needed before a
    request can construct the real graph; the full builder performs the
    expensive dependency load once and caches it.
    """

    env = environ if environ is not None else os.environ
    index_dir = _path_env(env, "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    fact_path = _path_env(env, "TRUSTED_V2_FACT_STORE_PATH", directory=False)
    checkpoint = _path_env(env, "TRUSTED_V2_SPECIALIST_CHECKPOINT", directory=False)
    index_manifest = inspect_r4_index(index_dir)
    fact_store = StructuredFactStore(fact_path)
    # Validate that the declared provider family and all endpoint/model values
    # are present without instantiating network clients.
    supervisor_provider = (_env(env, "V2_SUPERVISOR_PROVIDER", "bailian") or "bailian").casefold()
    if supervisor_provider not in {"bailian", "api"}:
        raise TrustedV2ProductionConfigurationError(
            "V2_SUPERVISOR_PROVIDER must be 'bailian' or 'api'"
        )
    _provider_common(env, "V2_SUPERVISOR_")
    binder_provider = (_env(env, "V2_BINDER_PROVIDER", "bailian") or "bailian").casefold()
    if binder_provider != "bailian":
        raise TrustedV2ProductionConfigurationError(
            "V2_BINDER_PROVIDER must be 'bailian'"
        )
    _provider_common(env, "V2_BINDER_", fallback_prefix="V2_SUPERVISOR_")
    return {
        "config_fingerprint": _configuration_fingerprint(env),
        "r4_index": index_manifest,
        "fact_store_path": str(fact_path),
        "fact_count": fact_store.candidate_count,
        "specialist_checkpoint": str(checkpoint),
        "specialist_checkpoint_sha256": _sha256_file(checkpoint),
    }


_RESOURCE_CACHE: MutableMapping[str, TrustedV2RuntimeResources] = {}
_RESOURCE_LOCK = threading.Lock()


def clear_trusted_v2_production_cache() -> None:
    """Clear cached process resources (primarily useful for tests/reloads)."""

    with _RESOURCE_LOCK:
        resources = list(_RESOURCE_CACHE.values())
        _RESOURCE_CACHE.clear()
    for resource in resources:
        closer = getattr(resource.index_reader, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass
        for provider in (
            getattr(resource.supervisor, "provider", None),
            getattr(resource.binder, "provider", None),
        ):
            closer = getattr(provider, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:
                    pass


def _load_resources(environ: Mapping[str, str]) -> TrustedV2RuntimeResources:
    index_dir = _path_env(environ, "TRUSTED_V2_R4_INDEX_DIR", directory=True)
    fact_path = _path_env(environ, "TRUSTED_V2_FACT_STORE_PATH", directory=False)
    index_manifest = inspect_r4_index(index_dir)
    fact_store = StructuredFactStore(fact_path)
    index_reader = None
    try:
        from src.pdf_retrieval_v4.candidate_view_index import CandidateViewIndexReader

        index_reader = CandidateViewIndexReader(index_dir)
        supervisor = _build_supervisor(environ)
        binder = _build_binder(environ)
        specialist = _build_specialist(environ)
        return TrustedV2RuntimeResources(
            index_reader=index_reader,
            fact_store=fact_store,
            supervisor=supervisor,
            binder=binder,
            specialist=specialist,
            budget=_build_budget(environ),
            config_fingerprint=_configuration_fingerprint(environ),
            index_manifest=index_manifest,
        )
    except TrustedV2ProductionConfigurationError:
        if index_reader is not None:
            index_reader.close()
        raise
    except Exception as exc:
        if index_reader is not None:
            index_reader.close()
        raise TrustedV2ProductionConfigurationError(
            "could not construct the canonical Trusted V2 production resources"
        ) from exc


def _cached_resources(environ: Mapping[str, str]) -> TrustedV2RuntimeResources:
    key = _configuration_fingerprint(environ)
    with _RESOURCE_LOCK:
        cached = _RESOURCE_CACHE.get(key)
        if cached is not None:
            return cached
        resources = _load_resources(environ)
        _RESOURCE_CACHE[key] = resources
        return resources


def _document_scope(request: FinancialQueryRequest) -> tuple[str, ...]:
    raw = request.request_metadata.get("document_names", ())
    if isinstance(raw, str):
        return (raw.strip(),) if raw.strip() else ()
    if not isinstance(raw, Iterable):
        return ()
    return _stable_unique(raw)


def build_trusted_v2_runtime_for_request(
    engine: Any,
    request: FinancialQueryRequest,
    *,
    resources: TrustedV2RuntimeResources | None = None,
) -> TrustedFinancialRuntimeV2:
    """Build one real ``TrustedFinancialRuntimeV2`` for a financial request.

    ``engine`` is accepted for the shared ``FinancialQARuntime`` builder
    signature but is intentionally unused: V2 owns its R4 index and never
    calls the legacy V1 retriever.  Expensive clients/models are process
    cached; request-scoped R4 policy and capability wrappers keep document
    scope and trace state isolated.
    """

    del engine
    if not isinstance(request, FinancialQueryRequest):
        raise TypeError("request must be a FinancialQueryRequest")
    env = os.environ
    resources = resources or _cached_resources(env)
    if not isinstance(resources, TrustedV2RuntimeResources):
        raise TypeError("resources must be TrustedV2RuntimeResources")

    document_scope = _document_scope(request)
    try:
        from src.pdf_retrieval_v4.candidate_direct_retriever import CandidateDirectRetriever

        retriever = CandidateDirectRetriever(resources.index_reader)
        policy = CandidateDirectR4Policy(
            retriever,
            materializer=resources.fact_store.materialize,
            document_scope=document_scope,
        )
        retrieval = R4RetrievalCapability(
            policy,
            document_scope=document_scope,
        )
        evidence = SemanticEvidenceEvaluationCapability(resources.binder)
        capabilities = TrustedV2CapabilityPorts(
            retrieval=retrieval,
            evidence_evaluator=evidence,
            calculation=DeterministicCalculationCapability(),
            generation=TrustedV2GenerationCapability(
                routing_policy=None,
                renderer=DeterministicFactRenderer(),
                specialist=resources.specialist,
            ),
            release_validator=TrustedReleaseValidationCapability(),
        )
        return build_trusted_v2_runtime(
            resources.supervisor,
            capabilities=capabilities,
            budget=resources.budget,
            # A production direct-fact request must not proceed when its
            # metric is not explicitly recognized by the alignment gate.
            # Generic operation-only calculation prompts remain compatible.
            unknown_semantic_policy=UnknownSemanticPolicy.STRICT_DIRECT_FACT,
        )
    except TrustedV2ProductionConfigurationError:
        raise
    except Exception as exc:
        raise TrustedV2ProductionConfigurationError(
            "could not build the request-scoped Trusted V2 runtime graph"
        ) from exc


__all__ = [
    "StructuredFactStore",
    "TrustedV2ProductionConfigurationError",
    "TrustedV2RuntimeResources",
    "build_trusted_v2_runtime_for_request",
    "clear_trusted_v2_production_cache",
    "inspect_r4_index",
    "validate_trusted_v2_production_configuration",
]
