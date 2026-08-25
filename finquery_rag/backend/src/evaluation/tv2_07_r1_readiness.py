"""TV2-07R1 canonical production-readiness preflight and execution helpers.

This module deliberately does not change any financial runtime.  It verifies
that a new held-out readiness set, frozen corpus/index, and configured
Specialist checkpoint are eligible for a formal run.  The actual run remains
blind: labels are loaded only by the post-execution scorer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluation.tv2_07_readiness import (
    TV2IntegratedEvaluationRunner,
    TV2ReadinessDecision,
    TV2ReadinessLabel,
    TV2ReadinessMetrics,
    TV2ReadinessQuery,
    _hash,
    _now,
    _read_jsonl,
    build_tv2_07_manifest,
    load_tv2_07_dataset,
    write_tv2_07_artifacts,
)

CANONICAL_PROVENANCE = frozenset({
    "fresh_company_held_out",
    "untouched_frozen_eval",
})
REJECTED_DATASET_MARKERS = (
    "tv2_07_production_readiness",
    "nf-v2-17-fresh-blind-eval",
    "nf-v2-10-final-trusted-e2e",
)
REQUIRED_CATEGORY_GROUPS = {
    "direct_fact": {"direct_fact", "fact"},
    "multi_evidence": {"multi_evidence", "cross_source", "cross_page"},
    "calculation": {"calculation", "unsupported_calculation"},
    "qualitative": {"qualitative_synthesis", "qualitative"},
    "negative": {
        "no_answer",
        "insufficient_evidence",
        "evidence_conflict",
        "unsupported_calculation",
    },
    "traps": {
        "wrong_period_trap",
        "wrong_row_trap",
        "unit_scale_trap",
        "wrong_unit_scale_trap",
    },
    "recovery": {"retrieval_recovery", "repair_once", "validator_rejection"},
    "multi_turn": {"multi_turn", "multi_turn_context", "contextual_query"},
}


def _read_json(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    with candidate.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{candidate} must contain a JSON object")
    return value


def _canonical_category(value: str) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _sha_file(path: str | Path) -> str | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    digest = hashlib.sha256()
    with candidate.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha(repo_path: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _path_is_rejected(path: str | Path) -> bool:
    normalized = str(path).replace("\\", "/").casefold()
    return any(marker in normalized for marker in REJECTED_DATASET_MARKERS)


def _check(value: bool, reason: str, checks: dict[str, Any], reasons: list[str]) -> None:
    checks[reason] = value
    if not value:
        reasons.append(reason)


@dataclass(frozen=True)
class TV2R1PreflightReport:
    """Machine-readable eligibility result before a formal readiness run."""

    status: str
    ready_to_run: bool
    case_count: int
    answerable_cases: int
    multi_turn_cases: int
    category_counts: dict[str, int]
    checks: dict[str, Any]
    blocking_reasons: tuple[str, ...]
    corpus_manifest: dict[str, Any]
    index_manifest: dict[str, Any]
    model_manifest: dict[str, Any]
    runtime_manifest: dict[str, Any]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": "TV2-07R1",
            "status": self.status,
            "ready_to_run": self.ready_to_run,
            "case_count": self.case_count,
            "answerable_cases": self.answerable_cases,
            "multi_turn_cases": self.multi_turn_cases,
            "category_counts": dict(self.category_counts),
            "checks": copy.deepcopy(self.checks),
            "blocking_reasons": list(self.blocking_reasons),
            "corpus_manifest": copy.deepcopy(self.corpus_manifest),
            "index_manifest": copy.deepcopy(self.index_manifest),
            "model_manifest": copy.deepcopy(self.model_manifest),
            "runtime_manifest": copy.deepcopy(self.runtime_manifest),
            "generated_at": self.generated_at,
        }


def _empty_report(
    *,
    repo_path: str | Path,
    reasons: Sequence[str],
    checks: Mapping[str, Any] | None = None,
) -> TV2R1PreflightReport:
    return TV2R1PreflightReport(
        status="PENDING",
        ready_to_run=False,
        case_count=0,
        answerable_cases=0,
        multi_turn_cases=0,
        category_counts={},
        checks=dict(checks or {}),
        blocking_reasons=tuple(dict.fromkeys(str(item) for item in reasons)),
        corpus_manifest={},
        index_manifest={},
        model_manifest={},
        runtime_manifest={
            "git_sha": _git_sha(repo_path),
            "python_version": platform.python_version(),
            "dependency_lock_sha256": _sha_file(Path(repo_path) / "uv.lock"),
        },
        generated_at=_now(),
    )


def build_tv2_07_r1_preflight(
    *,
    repo_path: str | Path,
    queries_path: str | Path,
    labels_path: str | Path,
    corpus_freeze_path: str | Path,
    index_config_path: str | Path,
    index_build_path: str | Path,
    index_integrity_path: str | Path,
    model_manifest_path: str | Path,
    raw_corpus_manifest_path: str | Path | None = None,
    parsed_corpus_manifest_path: str | Path | None = None,
    wiring_queries_path: str | Path | None = None,
    wiring_labels_path: str | Path | None = None,
    min_cases: int = 100,
    verify_index_paths: bool = False,
) -> TV2R1PreflightReport:
    """Validate all formal-run prerequisites without invoking V1/V2."""

    if min_cases <= 0:
        raise ValueError("min_cases must be positive")
    checks: dict[str, Any] = {}
    reasons: list[str] = []
    paths = {
        "queries": Path(queries_path),
        "labels": Path(labels_path),
        "corpus_freeze": Path(corpus_freeze_path),
        "index_config": Path(index_config_path),
        "index_build": Path(index_build_path),
        "index_integrity": Path(index_integrity_path),
        "model_manifest": Path(model_manifest_path),
    }
    optional_manifest_paths = {
        "raw_corpus_manifest": (
            Path(raw_corpus_manifest_path)
            if raw_corpus_manifest_path is not None
            else None
        ),
        "parsed_corpus_manifest": (
            Path(parsed_corpus_manifest_path)
            if parsed_corpus_manifest_path is not None
            else None
        ),
    }
    for name, candidate in optional_manifest_paths.items():
        if candidate is not None:
            _check(candidate.is_file(), f"{name}_exists", checks, reasons)
    for name, candidate in paths.items():
        _check(candidate.is_file(), f"{name}_exists", checks, reasons)
    if reasons:
        return _empty_report(repo_path=repo_path, reasons=reasons, checks=checks)

    try:
        queries, labels = load_tv2_07_dataset(queries_path, labels_path)
    except (OSError, TypeError, ValueError) as exc:
        return _empty_report(
            repo_path=repo_path,
            reasons=["canonical_dataset_invalid", type(exc).__name__],
            checks={**checks, "canonical_dataset_valid": False},
        )

    _check(
        not _path_is_rejected(queries_path) and not _path_is_rejected(labels_path),
        "canonical_dataset_path_not_consumed_or_wiring",
        checks,
        reasons,
    )
    _check(
        len(queries) >= min_cases,
        "canonical_case_count_at_least_minimum",
        checks,
        reasons,
    )
    provenance = {
        query.dataset_provenance for query in queries
    } | {label.dataset_provenance for label in labels}
    _check(
        bool(provenance) and provenance.issubset(CANONICAL_PROVENANCE),
        "canonical_provenance_is_fresh_or_untouched",
        checks,
        reasons,
    )
    _check(
        all(query.dataset_provenance == label.dataset_provenance for query, label in zip(
            sorted(queries, key=lambda item: item.case_id),
            sorted(labels, key=lambda item: item.case_id),
        )),
        "query_label_provenance_matches",
        checks,
        reasons,
    )
    category_counts = Counter(_canonical_category(query.category) for query in queries)
    for group, accepted in REQUIRED_CATEGORY_GROUPS.items():
        _check(
            any(category in accepted for category in category_counts),
            f"category_group_{group}",
            checks,
            reasons,
        )
    _check(
        any(not label.answerable or not label.expected_release for label in labels),
        "negative_cases_present",
        checks,
        reasons,
    )
    _check(
        any(query.input_turns for query in queries),
        "multi_turn_input_present",
        checks,
        reasons,
    )

    if wiring_queries_path and Path(wiring_queries_path).is_file():
        wiring_queries = [
            TV2ReadinessQuery.from_dict(row)
            for row in _read_jsonl(wiring_queries_path)
        ]
        canonical_ids = {query.case_id for query in queries}
        wiring_ids = {query.case_id for query in wiring_queries}
        _check(
            not canonical_ids.intersection(wiring_ids),
            "canonical_case_ids_disjoint_from_wiring_fixture",
            checks,
            reasons,
        )
        _check(
            _hash([query.to_dict() for query in queries])
            != _hash([query.to_dict() for query in wiring_queries]),
            "canonical_query_hash_distinct_from_wiring_fixture",
            checks,
            reasons,
        )
    else:
        _check(False, "wiring_fixture_reference_available", checks, reasons)
    if wiring_labels_path and Path(wiring_labels_path).is_file():
        wiring_labels = [
            TV2ReadinessLabel.from_dict(row)
            for row in _read_jsonl(wiring_labels_path)
        ]
        _check(
            _hash([label.to_dict() for label in labels])
            != _hash([label.to_dict() for label in wiring_labels]),
            "canonical_label_hash_distinct_from_wiring_fixture",
            checks,
            reasons,
        )
    else:
        _check(False, "wiring_label_fixture_reference_available", checks, reasons)

    try:
        corpus = _read_json(corpus_freeze_path)
        raw_manifest_rows = (
            _read_jsonl(optional_manifest_paths["raw_corpus_manifest"])
            if optional_manifest_paths["raw_corpus_manifest"] is not None
            else []
        )
        parsed_manifest_rows = (
            _read_jsonl(optional_manifest_paths["parsed_corpus_manifest"])
            if optional_manifest_paths["parsed_corpus_manifest"] is not None
            else []
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _empty_report(
            repo_path=repo_path,
            reasons=["corpus_manifest_invalid", type(exc).__name__],
            checks={**checks, "corpus_manifest_valid": False},
        )
    if optional_manifest_paths["raw_corpus_manifest"] is not None:
        _check(
            bool(raw_manifest_rows),
            "raw_corpus_manifest_non_empty",
            checks,
            reasons,
        )
    if optional_manifest_paths["parsed_corpus_manifest"] is not None:
        _check(
            bool(parsed_manifest_rows),
            "parsed_corpus_manifest_non_empty",
            checks,
            reasons,
        )
    if raw_manifest_rows and corpus.get("primary_documents") is not None:
        _check(
            len(raw_manifest_rows) == int(corpus["primary_documents"]),
            "raw_corpus_manifest_document_count_matches",
            checks,
            reasons,
        )
    if parsed_manifest_rows and raw_manifest_rows:
        _check(
            {
                row.get("document_id")
                for row in parsed_manifest_rows
                if row.get("document_id")
            }
            == {
                row.get("document_id")
                for row in raw_manifest_rows
                if row.get("document_id")
            },
            "parsed_raw_document_ids_match",
            checks,
            reasons,
        )
    try:
        index_config = _read_json(index_config_path)
        index_build = _read_json(index_build_path)
        index_integrity = _read_json(index_integrity_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _empty_report(
            repo_path=repo_path,
            reasons=["index_manifest_invalid", type(exc).__name__],
            checks={**checks, "index_manifest_valid": False},
        )
    _check(
        corpus.get("production_v1_modified") is False
        and corpus.get("gold_evidence_generated") is False
        and corpus.get("questions_generated") is False,
        "corpus_frozen_without_production_or_gold_mutation",
        checks,
        reasons,
    )
    _check(
        bool(corpus.get("searchable_corpus_sha"))
        and bool(corpus.get("searchable_manifest_sha")),
        "corpus_searchable_identity_present",
        checks,
        reasons,
    )
    _check(
        index_config.get("production_v1_overwritten") is False
        and index_config.get("searchable_corpus_sha")
        == corpus.get("searchable_corpus_sha"),
        "index_config_matches_frozen_corpus",
        checks,
        reasons,
    )
    _check(
        index_build.get("production_indices_modified") is False
        and index_build.get("fts", {}).get("built") is True
        and index_build.get("dense", {}).get("built") is True
        and index_build.get("hybrid", {}).get("built") is True,
        "index_build_is_frozen_and_complete",
        checks,
        reasons,
    )
    _check(
        index_integrity.get("duplicate_index_ids") == 0
        and index_integrity.get("metadata_schema_failures") == 0
        and index_integrity.get("missing_indexed_chunks") == 0
        and index_integrity.get("orphan_index_entries") == 0
        and float(index_integrity.get("provenance_complete_percent", 0)) == 100.0
        and index_integrity.get("searchable_corpus_sha")
        == corpus.get("searchable_corpus_sha"),
        "index_integrity_is_clean",
        checks,
        reasons,
    )
    if verify_index_paths:
        for branch in ("dense", "fts"):
            branch_path = index_build.get(branch, {}).get("path")
            _check(
                bool(branch_path) and Path(branch_path).exists(),
                f"{branch}_index_path_exists",
                checks,
                reasons,
            )

    try:
        model = _read_json(model_manifest_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return _empty_report(
            repo_path=repo_path,
            reasons=["model_manifest_invalid", type(exc).__name__],
            checks={**checks, "model_manifest_valid": False},
        )
    checkpoint_path = model.get("checkpoint_path")
    expected_checkpoint_sha = model.get("checkpoint_sha256")
    actual_checkpoint_sha = _sha_file(checkpoint_path) if checkpoint_path else None
    _check(bool(checkpoint_path), "specialist_checkpoint_path_declared", checks, reasons)
    _check(
        bool(expected_checkpoint_sha) and actual_checkpoint_sha == expected_checkpoint_sha,
        "specialist_checkpoint_hash_matches",
        checks,
        reasons,
    )
    _check(
        model.get("role") == "LOCAL_FINANCIAL_SPECIALIST_GENERATOR",
        "specialist_role_matches",
        checks,
        reasons,
    )
    _check(
        model.get("decoding", {}).get("temperature") in (0, 0.0)
        and model.get("decoding", {}).get("no_thinking_mode") is True
        and model.get("decoding", {}).get("no_cot") is True,
        "specialist_decoding_is_frozen",
        checks,
        reasons,
    )
    _check(
        bool(model.get("precision")) and bool(model.get("device")),
        "specialist_device_precision_declared",
        checks,
        reasons,
    )

    runtime_manifest = {
        "git_sha": _git_sha(repo_path),
        "python_version": platform.python_version(),
        "dependency_lock_sha256": _sha_file(Path(repo_path) / "uv.lock"),
        "runtime_config_sha256": _hash({
            "corpus_sha": corpus.get("searchable_corpus_sha"),
            "index_config_sha": _sha_file(index_config_path),
            "model_manifest_sha": _sha_file(model_manifest_path),
        }),
    }
    _check(bool(runtime_manifest["git_sha"]), "runtime_git_sha_present", checks, reasons)
    _check(
        bool(runtime_manifest["dependency_lock_sha256"]),
        "dependency_lock_hash_present",
        checks,
        reasons,
    )
    manifest_hashes = {
        "corpus_freeze_sha256": _sha_file(corpus_freeze_path),
        "index_config_sha256": _sha_file(index_config_path),
        "index_build_sha256": _sha_file(index_build_path),
        "index_integrity_sha256": _sha_file(index_integrity_path),
        "readiness_queries_sha256": _sha_file(queries_path),
        "readiness_labels_sha256": _sha_file(labels_path),
        "specialist_checkpoint_sha256": actual_checkpoint_sha,
    }
    ready = not reasons
    return TV2R1PreflightReport(
        status="READY_TO_RUN" if ready else "PENDING",
        ready_to_run=ready,
        case_count=len(queries),
        answerable_cases=sum(label.answerable for label in labels),
        multi_turn_cases=sum(bool(query.input_turns) for query in queries),
        category_counts=dict(category_counts),
        checks=checks,
        blocking_reasons=tuple(dict.fromkeys(reasons)),
        corpus_manifest={
            **corpus,
            "manifest_path": str(corpus_freeze_path),
            "manifest_sha256": manifest_hashes["corpus_freeze_sha256"],
            "raw_document_manifest_path": (
                str(optional_manifest_paths["raw_corpus_manifest"])
                if optional_manifest_paths["raw_corpus_manifest"] is not None
                else None
            ),
            "raw_document_manifest_sha256": (
                _sha_file(optional_manifest_paths["raw_corpus_manifest"])
                if optional_manifest_paths["raw_corpus_manifest"] is not None
                else None
            ),
            "raw_document_count": len(raw_manifest_rows),
            "parsed_document_manifest_path": (
                str(optional_manifest_paths["parsed_corpus_manifest"])
                if optional_manifest_paths["parsed_corpus_manifest"] is not None
                else None
            ),
            "parsed_document_manifest_sha256": (
                _sha_file(optional_manifest_paths["parsed_corpus_manifest"])
                if optional_manifest_paths["parsed_corpus_manifest"] is not None
                else None
            ),
            "parsed_document_count": len(parsed_manifest_rows),
        },
        index_manifest={
            "config": index_config,
            "build": index_build,
            "integrity": index_integrity,
            "manifest_paths": {
                "config": str(index_config_path),
                "build": str(index_build_path),
                "integrity": str(index_integrity_path),
            },
            "manifest_hashes": {
                key: value
                for key, value in manifest_hashes.items()
                if key.startswith("index_")
            },
        },
        model_manifest={
            **model,
            "manifest_path": str(model_manifest_path),
            "manifest_sha256": _sha_file(model_manifest_path),
            "actual_checkpoint_sha256": actual_checkpoint_sha,
        },
        runtime_manifest={
            **runtime_manifest,
            "manifest_hashes": manifest_hashes,
        },
        generated_at=_now(),
    )


def build_tv2_07_r1_manifest(
    *,
    repo_path: str | Path,
    queries: Sequence[TV2ReadinessQuery],
    labels: Sequence[TV2ReadinessLabel],
    preflight: TV2R1PreflightReport,
    runtime_config: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    manifest = build_tv2_07_manifest(
        repo_path=repo_path,
        queries=queries,
        labels=labels,
        runtime_config=runtime_config,
        corpus_hash=preflight.corpus_manifest.get("searchable_corpus_sha"),
        # The checkpoint was already hashed by preflight.  Avoid a second
        # multi-gigabyte read during manifest construction.
        model_checkpoint=None,
    )
    manifest["model_checkpoint"] = preflight.model_manifest.get("checkpoint_path")
    manifest["model_checkpoint_sha256"] = preflight.model_manifest.get(
        "actual_checkpoint_sha256"
    )
    manifest.update({
        "evaluation_name": "TV2-07R1 Canonical Production Readiness",
        "evaluation_phase": "TV2-07R1",
        "runtime_git_sha": preflight.runtime_manifest.get("git_sha"),
        "evaluation_runner_sha": _sha_file(
            Path(repo_path) / "src/evaluation/tv2_07_r1_readiness.py"
        ),
        "runtime_config_sha": manifest.get("runtime_config_hash"),
        "dependency_lock_sha": preflight.runtime_manifest.get(
            "dependency_lock_sha256"
        ),
        "corpus_manifest_sha": preflight.corpus_manifest.get("manifest_sha256"),
        "raw_corpus_manifest_sha": preflight.corpus_manifest.get(
            "raw_document_manifest_sha256"
        ),
        "parsed_corpus_manifest_sha": preflight.corpus_manifest.get(
            "parsed_document_manifest_sha256"
        ),
        "index_manifest_sha": _hash(preflight.index_manifest),
        "index_config_sha": preflight.runtime_manifest.get(
            "manifest_hashes", {}
        ).get("index_config_sha256"),
        "index_build_sha": preflight.runtime_manifest.get(
            "manifest_hashes", {}
        ).get("index_build_sha256"),
        "index_integrity_sha": preflight.runtime_manifest.get(
            "manifest_hashes", {}
        ).get("index_integrity_sha256"),
        "readiness_set_sha": manifest.get("evaluation_set_sha256"),
        "specialist_checkpoint_id": preflight.model_manifest.get("checkpoint_path"),
        "specialist_checkpoint_hash": preflight.model_manifest.get(
            "actual_checkpoint_sha256"
        ),
        "random_seed": random_seed,
        "preflight_status": preflight.status,
        "production_runtime": "V1",
        "v2_authority": "OFF",
        "canary": "NOT_STARTED",
    })
    return manifest


def finalize_r1_decision(
    metrics: TV2ReadinessMetrics,
    preflight: TV2R1PreflightReport,
    *,
    quality_reviewed: bool = False,
    latency_reviewed: bool = False,
    qualitative_reviewed: bool = False,
) -> TV2ReadinessMetrics:
    """Apply R1's explicit review prerequisites without inventing thresholds."""

    reasons = list(metrics.quality_hold_reasons)
    if not preflight.ready_to_run:
        reasons.append("CANONICAL_PREFLIGHT_PENDING")
    if not quality_reviewed:
        reasons.append("QUALITY_REVIEW_PENDING")
    if not latency_reviewed:
        reasons.append("LATENCY_REVIEW_PENDING")
    if preflight.multi_turn_cases and not qualitative_reviewed:
        reasons.append("QUALITATIVE_REVIEW_PENDING")
    reasons = tuple(dict.fromkeys(reasons))
    if any(metrics.hard_gate_counts.values()):
        decision = TV2ReadinessDecision.BLOCKED_FOR_SAFETY
    elif reasons:
        decision = TV2ReadinessDecision.HOLD_FOR_QUALITY
    else:
        decision = TV2ReadinessDecision.READY_FOR_CANARY
    return replace(metrics, quality_hold_reasons=reasons, decision=decision)


def write_tv2_07_r1_pending_artifacts(
    output_dir: str | Path,
    *,
    preflight: TV2R1PreflightReport,
    repo_path: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(
        json.dumps(preflight.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    decision = {
        "decision": "HOLD_FOR_QUALITY",
        "reason": "canonical readiness evaluation not established",
        "blocking_reasons": list(preflight.blocking_reasons),
        "production_runtime": "V1",
        "v2_production_authority": "OFF",
        "v2_canary": "NOT_STARTED",
    }
    (output / "decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "evaluation_phase": "TV2-07R1",
        "git_sha": _git_sha(repo_path),
        "preflight_status": preflight.status,
        "readiness_evaluation_executed": False,
        "production_runtime": "V1",
        "v2_authority": "OFF",
        "canary": "NOT_STARTED",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_tv2_07_r1_artifacts(
    output_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    preflight: TV2R1PreflightReport,
    queries: Sequence[TV2ReadinessQuery],
    labels: Sequence[TV2ReadinessLabel],
    scored_cases: Sequence[Mapping[str, Any]],
    metrics: TV2ReadinessMetrics,
    qualitative_review: Mapping[str, Any] | None = None,
) -> None:
    output = Path(output_dir)
    write_tv2_07_artifacts(
        output,
        manifest=manifest,
        queries=queries,
        labels=labels,
        scored_cases=scored_cases,
        metrics=metrics,
        runtime_manifest={
            "production_runtime": "V1",
            "evaluation_runtime": "TrustedFinancialRuntimeV2",
            "v2_authority": "OFF",
            "gold_evidence_injection": False,
            "preflight_status": preflight.status,
        },
    )
    (output / "preflight.json").write_text(
        json.dumps(preflight.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "corpus-manifest.json").write_text(
        json.dumps(preflight.corpus_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "qualitative-review.json").write_text(
        json.dumps(
            dict(qualitative_review or {
                "status": "PENDING",
                "reviewed_cases": 0,
            }),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_tv2_07_r1_dataset(
    queries_path: str | Path,
    labels_path: str | Path,
) -> tuple[list[TV2ReadinessQuery], list[TV2ReadinessLabel]]:
    if _path_is_rejected(queries_path) or _path_is_rejected(labels_path):
        raise ValueError("TV2-07R1 rejects consumed or wiring dataset paths")
    queries, labels = load_tv2_07_dataset(queries_path, labels_path)
    provenance = {
        query.dataset_provenance for query in queries
    } | {label.dataset_provenance for label in labels}
    if not provenance.issubset(CANONICAL_PROVENANCE):
        raise ValueError(
            "TV2-07R1 only accepts fresh_company_held_out or untouched_frozen_eval"
        )
    return queries, labels


__all__ = [
    "CANONICAL_PROVENANCE",
    "REQUIRED_CATEGORY_GROUPS",
    "TV2R1PreflightReport",
    "TV2IntegratedEvaluationRunner",
    "build_tv2_07_r1_manifest",
    "build_tv2_07_r1_preflight",
    "finalize_r1_decision",
    "load_tv2_07_r1_dataset",
    "write_tv2_07_r1_artifacts",
    "write_tv2_07_r1_pending_artifacts",
]
