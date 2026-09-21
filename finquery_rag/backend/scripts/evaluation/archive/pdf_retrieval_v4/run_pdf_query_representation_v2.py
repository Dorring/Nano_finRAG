"""Build and evaluate the PDF Query Representation V2 concept-only gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.evaluation.run_pdf_retrieval_v2_lite import _write
from src.evaluation.pdf_query_representation_v2 import (
    canonical_key,
    char_score,
    concept_family,
    fixed_rrf,
    generic_aliases,
    natural_phrase,
    normalize_label,
    ranks,
    stable_id,
    token_bm25_scores,
)
from src.retrieval.embedding_provider import ExistingMiniLMEmbeddingProvider
from src.services.retrieval_config import get_embedding_model_name

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/pdf-query-representation-v2"
BENCHMARK_ISSUERS = {"microsoft", "apple", "nvidia", "jpmorgan", "tesla", "coca-cola", "visa", "pfizer"}
NO_ANSWER_PHRASES = ("lunar mining royalties", "quantum battery inventory", "Martian payroll costs", "underwater data center assets", "teleportation revenue")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _issuer(document_id: str) -> str:
    return document_id.split("_fy", 1)[0].replace("_", " ").title()


def _registry(views: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for view in views:
        label = normalize_label(str(view["metric"]))
        if len(label) < 3:
            continue
        grouped[canonical_key(label)].append(view)
    records = []
    for key, items in sorted(grouped.items()):
        labels = sorted({normalize_label(str(item["metric"])) for item in items})
        documents = sorted({str(item["document_id"]) for item in items})
        label_lineage = {
            label: sorted({str(item["document_id"]) for item in items if normalize_label(str(item["metric"])) == label})
            for label in labels
        }
        canonical = min(labels, key=lambda label: (len(label.split()), len(label), label))
        aliases = sorted({alias for label in labels for alias in generic_aliases(label)})
        records.append({
            "concept_id": stable_id("financial-concept", key),
            "canonical_label": canonical,
            "labels": labels,
            "generic_aliases": aliases,
            "label_lineage": label_lineage,
            "concept_family": concept_family(canonical),
            "source_document_ids": documents,
            "source_lineage": ["existing_pdf_row_label"],
            "local_xbrl_label_status": "not_available",
        })
    return records


def _holdout_registry(registry: list[dict[str, object]], holdout_document: str) -> list[dict[str, object]]:
    output = []
    for item in registry:
        labels = [
            label
            for label, sources in item["label_lineage"].items()
            if any(source != holdout_document for source in sources)
        ]
        if not labels:
            continue
        canonical = min(labels, key=lambda label: (len(label.split()), len(label), label))
        aliases = sorted({alias for label in labels for alias in generic_aliases(label)})
        output.append({**item, "canonical_label": canonical, "labels": labels, "generic_aliases": aliases})
    return output


def _queries(registry: list[dict[str, object]], views: list[dict[str, object]], per_issuer: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    concept_by_key = {canonical_key(str(item["canonical_label"])): item for item in registry}
    available: dict[str, list[tuple[dict[str, object], dict[str, object]]]] = defaultdict(list)
    for view in views:
        concept = concept_by_key.get(canonical_key(str(view["metric"])))
        if concept is not None:
            available[str(view["document_id"])].append((view, concept))
    queries = []
    for document_id, items in sorted(available.items()):
        unique: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
        for view, concept in items:
            if len(concept["source_document_ids"]) >= 2:
                unique.setdefault(str(concept["concept_id"]), (view, concept))
        base = sorted(unique.values(), key=lambda pair: (str(pair[1]["concept_family"]), str(pair[1]["canonical_label"])))
        chosen = [base[index % len(base)] for index in range(per_issuer)]
        for index, (view, concept) in enumerate(chosen):
            phrase, method = natural_phrase(str(concept["canonical_label"]), index)
            matching_ids = [
                item["concept_id"]
                for item in registry
                if phrase in item.get("generic_aliases", [])
            ]
            periods = list(view.get("table_period_tokens") or [])
            period = str(periods[0]) if periods else None
            question = f"What was {_issuer(document_id)}'s {phrase}"
            if period:
                question += f" in fiscal {period.removeprefix('FY')}"
            question += "?"
            queries.append({
                "query_id": stable_id("pdf-query-v2", document_id, concept["concept_id"], index),
                "issuer": _issuer(document_id),
                "document_id": document_id,
                "natural_question": question,
                "metric_phrase": phrase,
                "intended_concept_family": concept["concept_family"],
                "allowed_concept_ids": matching_ids or [concept["concept_id"]],
                "forbidden_nearby_concepts": [],
                "period": period,
                "statement_hint": None,
                "development_target_candidate_key": view["candidate_key"],
                "paraphrase_method": method,
                "review_status": "ai_assisted_pending_human_review",
                "human_reviewed": False,
            })
    controls = []
    for document_id in sorted(available):
        for phrase in NO_ANSWER_PHRASES:
            controls.append({"query_id": stable_id("pdf-query-v2-no-answer", document_id, phrase), "document_id": document_id, "metric_phrase": phrase, "expected_route": "no_answer", "review_status": "ai_assisted_pending_human_review", "human_reviewed": False})
    return queries, controls


def _resolve(
    phrase: str,
    registry: list[dict[str, object]],
    query_vector: np.ndarray,
    concept_vectors: np.ndarray,
) -> dict[str, object]:
    labels = [" ".join([str(item["canonical_label"]), *item.get("generic_aliases", [])]) for item in registry]
    exact = [1.0 if normalize_label(phrase) in [*item["labels"], *item.get("generic_aliases", [])] else 0.0 for item in registry]
    bm25 = token_bm25_scores(phrase, labels)
    chars = [char_score(phrase, label) for label in labels]
    dense = np.asarray(concept_vectors) @ np.asarray(query_vector)
    score = fixed_rrf([ranks(exact), ranks(bm25), ranks(chars), ranks(dense.tolist())])
    order = sorted(range(len(registry)), key=lambda index: (-score[index], index))[:3]
    phrase_terms = set(normalize_label(phrase).split())
    best_overlap = max(
        (len(phrase_terms & set(normalize_label(label).split())) / max(1, len(phrase_terms)) for label in labels),
        default=0,
    )
    # Require two-thirds phrase support so a real generic noun (for example
    # "revenue") cannot validate an otherwise unsupported compound metric.
    lexical_supported = max(exact, default=0) > 0 or best_overlap >= (2 / 3) or max(chars, default=0) >= 0.75
    selected = [registry[index] for index in order] if lexical_supported else []
    return {
        "resolution_status": "resolved" if selected else "no_supported_concept",
        "concept_candidates": [{"concept_id": item["concept_id"], "rank": rank + 1, "concept_family": item["concept_family"], "canonical_label": item["canonical_label"]} for rank, item in enumerate(selected)],
        "signals_used": ["exact_normalization", "token_bm25", "character_trigram", "existing_short_text_embedding", "fixed_rrf_k60"],
    }


def run(args: argparse.Namespace) -> int:
    views_path = args.runtime_dir / "pdf-v2-lite-retrieval-views.json"
    payload = json.loads(views_path.read_text(encoding="utf-8"))
    views = payload["views"]
    documents = sorted({str(view["document_id"]) for view in views})
    assert not any(any(name in document.casefold() for name in BENCHMARK_ISSUERS) for document in documents)
    registry = _registry(views)
    queries, controls = _queries(registry, views, args.per_issuer)
    provider = ExistingMiniLMEmbeddingProvider(model_name_or_path=get_embedding_model_name(), device=args.device)
    all_phrases = [str(item["metric_phrase"]) for item in [*queries, *controls]]
    query_vectors = provider.encode_queries(all_phrases)
    fold_registries = {document: _holdout_registry(registry, document) for document in documents}
    fold_vectors = {
        document: provider.encode_documents([str(item["canonical_label"]) for item in fold_registry])
        for document, fold_registry in fold_registries.items()
    }
    results = []
    top1 = top3 = wrong_family = 0
    reciprocal = 0.0
    for query, query_vector in zip(queries, query_vectors[: len(queries)], strict=True):
        fold_registry = fold_registries[str(query["document_id"])]
        resolution = _resolve(
            str(query["metric_phrase"]),
            fold_registry,
            query_vector,
            fold_vectors[str(query["document_id"])],
        )
        candidates = resolution["concept_candidates"]
        ids = [item["concept_id"] for item in candidates]
        allowed = set(query["allowed_concept_ids"])
        hit_ranks = [index + 1 for index, item in enumerate(ids) if item in allowed]
        top1 += int(bool(ids) and ids[0] in allowed)
        top3 += int(bool(hit_ranks))
        reciprocal += 1 / hit_ranks[0] if hit_ranks else 0
        wrong_family += int(bool(candidates) and candidates[0]["concept_family"] != query["intended_concept_family"])
        results.append({**query, **resolution})
    control_vectors = query_vectors[len(queries) :]
    intrusions = sum(
        bool(
            _resolve(
                str(item["metric_phrase"]),
                fold_registries[str(item["document_id"])],
                vector,
                fold_vectors[str(item["document_id"])],
            )["concept_candidates"]
        )
        for item, vector in zip(controls, control_vectors, strict=True)
    )
    count = len(queries)
    metrics = {
        "query_count": count,
        "top_1_accuracy": top1 / count if count else 0,
        "top_3_recall": top3 / count if count else 0,
        "mrr": reciprocal / count if count else 0,
        "wrong_concept_family_rate": wrong_family / count if count else 0,
        "no_answer_concept_intrusion_rate": intrusions / len(controls) if controls else 0,
    }
    gate_passed = count >= 120 and metrics["top_1_accuracy"] >= 0.70 and metrics["top_3_recall"] >= 0.85 and metrics["wrong_concept_family_rate"] <= 0.05 and metrics["no_answer_concept_intrusion_rate"] <= 0.05
    split = {"strategy": "three_fold_leave_one_issuer_out", "documents": [{"document_id": document, "issuer": _issuer(document), "fold": index + 1} for index, document in enumerate(documents)], "split_frozen_before_query_generation": True, "frozen_transfer_documents": sorted(BENCHMARK_ISSUERS)}
    _write(args.out_dir / "document-split-manifest.json", split)
    _write(args.out_dir / "concept-registry.json", {"schema": "pdf-query-representation-v2/concept-registry/v1", "concept_count": len(registry), "records": registry})
    _write(args.out_dir / "natural-query-set.json", {"query_count": len(queries), "no_answer_count": len(controls), "queries": queries, "no_answer_controls": controls})
    _write(args.out_dir / "concept-resolution-results.json", {"metrics": metrics, "records": results})
    _write(args.out_dir / "query-variant-manifest.json", {"variants": ["raw_query", "top_1_canonical_query", "raw_plus_top_3_concept_query"], "executed_in_gate": ["concept_resolution_only"], "query_level_rrf": {"k": 60, "weights": "uniform", "parameter_search": False}})
    decision = "concept_resolution_gate_passed" if gate_passed else "concept_resolution_quality_insufficient"
    _write(args.out_dir / "next-gate.json", {"decision": decision, "next_gate": "query_only_hybrid_shadow" if gate_passed else "stop_query_representation_v2", "production_switch_allowed": False})
    acceptance = {"schema": "pdf-query-representation-v2/acceptance/v1", "runtime_views_sha256": _sha(views_path), "document_split_sha256": _sha(args.out_dir / "document-split-manifest.json"), "concept_registry_sha256": _sha(args.out_dir / "concept-registry.json"), "natural_query_set_sha256": _sha(args.out_dir / "natural-query-set.json"), "concept_metrics": metrics, "concept_gate_passed": gate_passed, "hybrid_retrieval_runs": 0, "development_target_identity_reads_for_posthoc_scoring": len(queries), "frozen_72_question_reads": 0, "frozen_gold_source_reads": 0, "expected_value_reads": 0, "model_training_calls": 0, "answer_generation_calls": 0, "production_index_writes": 0, "production_behavior_changed": False, "production_switch_allowed": False, "frozen_transfer_allowed": False, "human_reviewed": False, "review_mode": "ai_assisted_independent_development_annotation", "decision": decision}
    _write(args.out_dir / "acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--per-issuer", type=int, default=40)
    parser.add_argument("--device", default=os.getenv("PDF_QUERY_V2_EMBEDDING_DEVICE", "cpu"))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
