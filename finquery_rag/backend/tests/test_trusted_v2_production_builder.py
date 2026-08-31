"""Tests for the canonical, dependency-injected Trusted V2 production builder."""

from __future__ import annotations

import gzip
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

import pytest

from rag_v2.adaptive import AdaptiveRAGBudgetV1
from rag_v2.evidence import SemanticBinderService
from rag_v2.supervisor import DeterministicFallbackProvider, SupervisorService
from src.runtime import (
    FinancialQueryRequest,
    StructuredFactStore,
    TrustedFinancialRuntimeV2,
    TrustedV2ProductionConfigurationError,
    TrustedV2RuntimeResources,
    build_trusted_v2_runtime_for_request,
    inspect_r4_index,
    validate_trusted_v2_production_configuration,
)
from src.runtime.trusted_v2_generation import LocalSpecialistGenerationAdapter


def _fact(candidate_key: str = "candidate:1") -> dict[str, Any]:
    return {
        "candidate_key": candidate_key,
        "evidence_id": "evidence-1",
        "citation_id": "citation-1",
        "provenance_complete": True,
        "physical_source_id": "annual-report-2024",
        "normalized_metric": "Revenue",
        "normalized_period": "FY2024",
        "parsed_numeric_value": 391_035,
        "unit": "USD",
        "normalized_scale": "million",
        "pdf_page": 12,
    }


def test_structured_fact_store_reads_gzipped_jsonl_and_normalizes_sealed_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "facts.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(json.dumps(_fact()) + "\n")

    store = StructuredFactStore(path)
    materialized = store.materialize("candidate:1")

    assert store.candidate_count == 1
    assert materialized["evidence_id"] == "evidence-1"
    assert materialized["citation_id"] == "citation-1"
    assert materialized["metric"] == "Revenue"
    assert materialized["period"] == "FY2024"
    assert materialized["value"] == 391_035
    assert materialized["source_id"] == "annual-report-2024"
    assert materialized["page"] == 12

    # Materialization is isolated from the process-scoped registry.
    materialized["metric"] = "mutated"
    assert store.materialize("candidate:1")["metric"] == "Revenue"


def test_structured_fact_store_rejects_duplicate_or_incomplete_provenance(
    tmp_path: Path,
) -> None:
    duplicate_path = tmp_path / "duplicate.json"
    duplicate_path.write_text(
        json.dumps([_fact(), _fact("candidate:1")]),
        encoding="utf-8",
    )
    with pytest.raises(
        TrustedV2ProductionConfigurationError,
        match="ambiguous duplicate candidate key",
    ):
        StructuredFactStore(duplicate_path)

    incomplete = _fact()
    incomplete.pop("citation_id")
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps([incomplete]), encoding="utf-8")
    with pytest.raises(
        TrustedV2ProductionConfigurationError,
        match="citation_id",
    ):
        StructuredFactStore(incomplete_path)


def _minimal_r4_index(root: Path, *, all_lanes: bool = True) -> None:
    (root / "candidate-metadata.sqlite").parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(root / "candidate-metadata.sqlite") as connection:
        connection.execute(
            "CREATE TABLE view_metadata ("
            "lane TEXT, view_id TEXT, candidate_key TEXT, view_type TEXT, "
            "retrieval_text TEXT, document_id TEXT, metadata_json TEXT)"
        )
        lanes = (
            "candidate_raw_bm25",
            "candidate_structured_bm25",
            "candidate_raw_dense",
            "candidate_structured_dense",
        )
        if not all_lanes:
            lanes = lanes[:1]
        connection.executemany(
            "INSERT INTO view_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (lane, f"view:{lane}", "candidate:1", "fact", "Revenue", "doc-1", "{}")
                for lane in lanes
            ],
        )
        connection.commit()

    for lane in (
        "candidate_raw_bm25",
        "candidate_structured_bm25",
        "candidate_raw_dense",
        "candidate_structured_dense",
    ):
        if "bm25" in lane:
            path = root / lane / "bm25" / "index.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"sqlite fixture")
        else:
            directory = root / lane / "dense"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "ids.json").write_text("[]", encoding="utf-8")
            (directory / "vectors.npy").write_bytes(b"npy fixture")


def test_inspect_r4_index_requires_all_four_candidate_lanes(tmp_path: Path) -> None:
    index_dir = tmp_path / "r4"
    _minimal_r4_index(index_dir)

    manifest = inspect_r4_index(index_dir)

    assert manifest["row_count"] == 4
    assert set(manifest["lanes"]) == {
        "candidate_raw_bm25",
        "candidate_structured_bm25",
        "candidate_raw_dense",
        "candidate_structured_dense",
    }

    incomplete_dir = tmp_path / "incomplete-r4"
    _minimal_r4_index(incomplete_dir, all_lanes=False)
    with pytest.raises(
        TrustedV2ProductionConfigurationError,
        match="metadata is missing lanes",
    ):
        inspect_r4_index(incomplete_dir)


def test_preflight_fails_closed_when_production_assets_are_not_provisioned() -> None:
    with pytest.raises(
        TrustedV2ProductionConfigurationError,
        match="TRUSTED_V2_R4_INDEX_DIR",
    ):
        validate_trusted_v2_production_configuration({})


def test_preflight_accepts_complete_layout_without_network(tmp_path: Path) -> None:
    index_dir = tmp_path / "r4"
    _minimal_r4_index(index_dir)
    fact_path = tmp_path / "facts.json"
    fact_path.write_text(json.dumps([_fact()]), encoding="utf-8")
    checkpoint = tmp_path / "specialist.pt"
    checkpoint.write_bytes(b"checkpoint fixture")

    report = validate_trusted_v2_production_configuration(
        {
            "TRUSTED_V2_R4_INDEX_DIR": str(index_dir),
            "TRUSTED_V2_FACT_STORE_PATH": str(fact_path),
            "TRUSTED_V2_SPECIALIST_CHECKPOINT": str(checkpoint),
            "V2_SUPERVISOR_PROVIDER": "api",
            "V2_SUPERVISOR_BASE_URL": "https://example.invalid/v1",
            "V2_SUPERVISOR_API_KEY": "test-secret",
            "V2_SUPERVISOR_MODEL": "test-model",
            "V2_BINDER_PROVIDER": "bailian",
        }
    )

    assert report["r4_index"]["row_count"] == 4
    assert report["fact_count"] == 1
    assert report["specialist_checkpoint_sha256"]
    assert len(report["config_fingerprint"]) == 64


class _BinderProvider:
    provider_name = "test-binder"
    model_name = "test-binder"
    last_call = None


class _SpecialistBackend:
    def generate(
        self,
        question: str,
        evidence_items: list[dict[str, Any]],
        calculation_result: Mapping[str, Any] | None = None,
    ) -> str:
        return "test candidate"


def test_builder_constructs_request_scoped_v2_graph_with_injected_resources(
    tmp_path: Path,
) -> None:
    facts_path = tmp_path / "facts.json"
    facts_path.write_text(json.dumps([_fact()]), encoding="utf-8")
    fact_store = StructuredFactStore(facts_path)

    # The deterministic provider is deliberately confined to this test.  The
    # production environment builder accepts only configured remote providers.
    provider = DeterministicFallbackProvider({})
    resources = TrustedV2RuntimeResources(
        index_reader=object(),
        fact_store=fact_store,
        supervisor=SupervisorService(provider),
        binder=SemanticBinderService(_BinderProvider()),
        specialist=LocalSpecialistGenerationAdapter(_SpecialistBackend()),
        budget=AdaptiveRAGBudgetV1(),
        config_fingerprint="test-fingerprint",
        index_manifest={"row_count": 1},
    )
    request = FinancialQueryRequest(
        request_id="req-builder",
        user_id="user-builder",
        session_id="session-builder",
        original_query="What about last year?",
        standalone_query="What was Apple FY2023 revenue?",
        query_as_resolved=True,
        request_metadata={"conversation_history": [{"role": "user", "content": "old"}]},
    )

    runtime = build_trusted_v2_runtime_for_request(
        None,
        request,
        resources=resources,
    )

    assert isinstance(runtime, TrustedFinancialRuntimeV2)
    assert runtime.coordinator.supervisor is resources.supervisor
    assert runtime.coordinator.capabilities.retrieval is not None
    assert runtime.coordinator.capabilities.evidence_evaluator is not None
    assert runtime.coordinator.capabilities.calculation is not None
    assert runtime.coordinator.capabilities.generation is not None
    assert runtime.coordinator.capabilities.release_validator is not None

    retrieval = runtime.coordinator.capabilities.retrieval
    assert retrieval.policy.materializer.__self__ is fact_store
    assert retrieval.policy.materializer.__func__ is StructuredFactStore.materialize
    # The request-scoped graph carries only the document scope; raw context is
    # filtered by FinancialQueryRequest/V2ExecutionRequest before execution.
    assert retrieval.document_scope == ()
