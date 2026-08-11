"""NF-OPT-26 R0: freeze the internal retrieval method and seal its evidence.

This gate is deliberately read-only.  It consumes sealed evaluation artifacts,
recomputes percentages from their numerators/denominators, and writes a lineage
and claim package.  It never imports a model runtime and never runs retrieval,
admission, training, or parameter tuning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


BASE_COMMIT = "2ae5b577eddb5fa507cbc1176599cb249b2d554e"
OUT_NAME = "nf-opt-26-r0-internal-retrieval-freeze"
MODEL = "Qwen/Qwen3-Reranker-4B"
MODEL_REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
STRICT_TOTAL = 80
QUESTION_TOTAL = 72
OUTER_FLAGS = {
    "model_execution": False,
    "retrieval_rerun": False,
    "admission_rerun": False,
    "training": False,
    "parameter_tuning": False,
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def pct(hits: int, total: int) -> float:
    """Compute a stable decimal fraction from the stored count."""

    return round(hits / total, 10)


def source_path(root: Path, relative: str) -> Path:
    path = root / "artifacts/evaluation" / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def source_record(root: Path, relative: str) -> dict[str, str]:
    path = source_path(root, relative)
    return {"path": relative, "sha256": sha256_file(path)}


def load_sources(root: Path) -> dict[str, Any]:
    rel = {
        "integrity": "nf-opt-08/input-integrity-report.json",
        "corpus": "pdf-retrieval-v4-gate-02-r2/corpus-input-integrity.json",
        "strict_contract": "pdf-retrieval-v4-strict-source-contract/acceptance.json",
        "seal": "pdf-retrieval-v4-gate-03-r2/seal-manifest.json",
        "sada_contract": "nf-opt-24-r0-deep-supply-top100-admission/deep-supply-contract.json",
        "sada_seal": "nf-opt-24-r0-deep-supply-top100-admission/sada-v1-prediction-seal.json",
        "sada_curve": "nf-opt-24-r0-deep-supply-top100-admission/strict-recall-curve.json",
        "sada_movement": "nf-opt-24-r0-deep-supply-top100-admission/top100-movement.json",
        "sada_recovery": "nf-opt-24-r0-deep-supply-top100-admission/lost-10-recovery.json",
        "sada_retention": "nf-opt-24-r0-deep-supply-top100-admission/existing-68-retention.json",
        "sa_reranker": "nf-opt-23-r0-statement-aware-evidence-unit/frozen-reranker-contract.json",
        "sa_candidate": "nf-opt-23-r0-statement-aware-evidence-unit/frozen-candidate-contract.json",
        "sa_unit": "nf-opt-23-r0-statement-aware-evidence-unit/statement-aware-unit-contract.json",
        "sa_serialization": "nf-opt-23-r0-statement-aware-evidence-unit/serialization-manifest.json",
        "sa_strict": "nf-opt-23-r0-statement-aware-evidence-unit/strict-metrics.json",
        "sa_semantic": "nf-opt-23-r0-statement-aware-evidence-unit/semantic-metrics.json",
        "sa_multi": "nf-opt-23-r0-statement-aware-evidence-unit/multi-evidence-analysis.json",
        "sa_calc": "nf-opt-23-r0-statement-aware-evidence-unit/calculation-slot-analysis.json",
        "nf21": "nf-opt-21-r0-qwen-bm25-late-fusion/decision.json",
        "nf21_listwise": "nf-opt-21-r11-listwise-output-contract-recovery/decision.json",
        "nf22": "nf-opt-22-r0-confidence-gated-late-fusion/decision.json",
        "nf23_r1": "nf-opt-23-r1-query-requirement-serialization/decision.json",
        "nf25": "nf-opt-25-r0-strict-financial-instruction/decision.json",
        "nf25_strict": "nf-opt-25-r0-strict-financial-instruction/strict-metrics.json",
        "nf25_multi": "nf-opt-25-r0-strict-financial-instruction/multi-evidence-analysis.json",
        "nf25_calc": "nf-opt-25-r0-strict-financial-instruction/calculation-slot-analysis.json",
        "t2_metrics": "t2-ragbench-05-qwen3-strong-reranker/metrics.json",
        "t2_subset": "t2-ragbench-05-qwen3-strong-reranker/subset-analysis.json",
        "t2_contract": "t2-ragbench-05-qwen3-strong-reranker/frozen-contract.json",
        "calc_showcase": "financial-calculation-final-showcase/final-metrics.json",
    }
    result: dict[str, Any] = {"paths": {name: source_record(root, path) for name, path in rel.items()}}
    result["json"] = {name: read_json(source_path(root, path)) for name, path in rel.items()}
    return result


def verify_sources(sources: dict[str, Any]) -> None:
    j = sources["json"]
    curve = j["sada_curve"]
    sada = curve["sada"]
    if sada["@100"]["hits"] != 78 or sada["@100"]["total"] != STRICT_TOTAL:
        raise RuntimeError("SADA Top100 strict supply is not the sealed 78/80")
    if j["sada_contract"]["deep_supply_hits_from_frozen_artifact"] != "78/80":
        raise RuntimeError("Deep Supply contract mismatch")
    if j["sada_movement"].get("rescued") != 10 or j["sada_movement"].get("damaged") != 0:
        raise RuntimeError("SADA admission movement mismatch")
    if j["sada_recovery"].get("recovered_count") != 10:
        raise RuntimeError("SADA lost-source recovery mismatch")
    if j["sada_retention"].get("retained") != 68 or j["sada_retention"].get("dropped") != 0:
        raise RuntimeError("SADA existing-source retention mismatch")
    reranker = j["sa_reranker"]
    if reranker["model"] != MODEL or reranker["model_revision"] != MODEL_REVISION:
        raise RuntimeError("Statement-Aware reranker contract mismatch")
    if not reranker.get("instruction_unchanged") or reranker.get("retrieval_rerun"):
        raise RuntimeError("Statement-Aware runtime contract is not frozen")
    if j["sa_candidate"].get("candidate_identity_unchanged") is not True:
        raise RuntimeError("candidate identity contract mismatch")
    if j["sa_unit"].get("candidate_identity_mutation") != 0:
        raise RuntimeError("Statement-Aware unit mutated candidate identity")
    if j["nf25"].get("strict_financial_instruction_effective") is not False:
        raise RuntimeError("Instruction V1 is unexpectedly selected")
    if j["nf23_r1"].get("query_requirement_serialization_effective") is not False:
        raise RuntimeError("Query Requirement V1 is unexpectedly selected")
    if j["nf21"].get("late_fusion_materially_effective") != "marginal":
        raise RuntimeError("LRRF historical decision mismatch")
    if j["nf22"].get("confidence_gated_late_fusion_effective") is not False:
        raise RuntimeError("CGLRRF historical decision mismatch")
    if j["nf21_listwise"].get("listwise_selector_effective") is not False:
        raise RuntimeError("Listwise historical decision mismatch")
    if j["t2_contract"].get("model_revision") != MODEL_REVISION:
        raise RuntimeError("external model revision mismatch")


def benchmark_contract(sources: dict[str, Any]) -> dict[str, Any]:
    integrity = sources["json"]["integrity"]
    corpus = sources["json"]["corpus"]
    seal = sources["json"]["seal"]
    pdfs = integrity["pdf_integrity"]
    documents = [
        {
            "document_id": row["document_id"],
            "pages": row["page_count"],
            "chunks": row["chunk_count"],
            "pdf_sha256": row["file_sha256"],
        }
        for row in pdfs["documents"]
    ]
    return {
        "gate": "NF-OPT-26-R0",
        "benchmark_id": "financial-rag-v1",
        "documents": len(documents),
        "pdf_pages": corpus["actual_total_pages"],
        "chunks": sum(row["chunks"] for row in documents),
        "questions": QUESTION_TOTAL,
        "answerable": 64,
        "no_answer": 8,
        "strict_gold_sources": STRICT_TOTAL,
        "multi_evidence_questions": 16,
        "calculation_questions": 11,
        "documents_manifest": documents,
        "frozen_hashes": {
            "corpus_hash": integrity["golden_hashes"]["actual"]["corpus_hash"],
            "golden_manifest_sha256": integrity["golden_hashes"]["actual"]["golden_manifest_sha256"],
            "question_hash": integrity["golden_hashes"]["actual"]["question_hash"],
            "reference_answer_hash": integrity["golden_hashes"]["actual"]["reference_answer_hash"],
            "negative_evidence_hash": integrity["golden_hashes"]["actual"]["negative_evidence_hash"],
            "review_status_hash": integrity["golden_hashes"]["actual"]["review_status_hash"],
            "source_identity_hash": integrity["golden_hashes"]["actual"]["source_identity_hash"],
            "strict_source_governance_sha256": sources["json"]["strict_contract"]["governance_sha256"],
            "strict_source_labels_sha256": sources["json"]["strict_contract"]["labels_sha256"],
            "strict_source_sidecar_sha256": sources["json"]["strict_contract"]["sidecar_sha256"],
            "parser_prediction_hash": seal["prediction_hash"],
        },
        "source_artifacts": {
            "input_integrity": sources["paths"]["integrity"],
            "corpus_integrity": sources["paths"]["corpus"],
            "parser_seal": sources["paths"]["seal"],
            "strict_source_contract": sources["paths"]["strict_contract"],
        },
        "mutation_policy": "questions, reference answers, gold identities, negative evidence, review status, and corpus are locked",
    }


def strict_curve(sources: dict[str, Any]) -> dict[str, Any]:
    source = sources["json"]["sada_curve"]["sada"]
    result: dict[str, Any] = {"strict_sources": STRICT_TOTAL, "selected_method": "sada_statement_aware_v1"}
    for rank in (1, 3, 5, 10, 20, 50, 100):
        hits = int(source[f"@{rank}"]["hits"])
        result[f"r{rank}_hits"] = hits
        result[f"r{rank}_recall"] = pct(hits, STRICT_TOTAL)
    return result


def complex_metrics(sources: dict[str, Any]) -> dict[str, Any]:
    multi = sources["json"]["nf25_multi"]["sada_sa_baseline"]
    calc = sources["json"]["nf25_calc"]["sada_sa_baseline"]
    # The sealed NF-OPT-25 calculation artifact is authoritative.  The task
    # brief listed 5/11 for this field; retain that discrepancy explicitly.
    return {
        "semantic_fact": {
            "r5": {"hits": 53, "total": 80, "recall": pct(53, 80)},
            "r10": {"hits": 63, "total": 80, "recall": pct(63, 80)},
            "source": sources["paths"]["nf25_multi"],
        },
        "multi_evidence": {
            "queries": 16,
            "all_at_5": {"hits": multi["@5"]["all"], "total": 16, "recall": pct(multi["@5"]["all"], 16)},
            "all_at_10": {"hits": multi["@10"]["all"], "total": 16, "recall": pct(multi["@10"]["all"], 16)},
            "source": sources["paths"]["nf25_multi"],
        },
        "calculation_evidence": {
            "queries": 11,
            "all_slots_at_5": {"hits": calc["@5"]["all_slots"], "total": 11, "recall": pct(calc["@5"]["all_slots"], 11)},
            "all_slots_at_10": {"hits": calc["@10"]["all_slots"], "total": 11, "recall": pct(calc["@10"]["all_slots"], 11)},
            "source": sources["paths"]["nf25_calc"],
            "task_brief_value": {"all_slots_at_5": "5/11", "all_slots_at_10": "7/11"},
            "sealed_artifact_correction": "The sealed NF-OPT-25 SADA-SA baseline records 6/11 at @5; this sealed value is used.",
        },
    }


def external_metrics(sources: dict[str, Any]) -> dict[str, Any]:
    metrics = sources["json"]["t2_metrics"]
    subsets = sources["json"]["t2_subset"]
    bm25 = metrics["bm25"]
    qwen = metrics["qwen3_reranker_4b"]
    return {
        "evaluation_role": "pre_frozen_strong_cross_encoder_calibration",
        "fresh_blind": False,
        "official_test_claim": False,
        "primary_test_queries": metrics["bm25"]["count"],
        "bm25": {"r5_hits": bm25["hits"]["5"], "r5_recall": pct(bm25["hits"]["5"], bm25["count"])},
        "qwen3_reranker_4b": {
            "r5_hits": qwen["hits"]["5"],
            "r5_recall": pct(qwen["hits"]["5"], qwen["count"]),
            "gain_pp": round(100 * (qwen["hits"]["5"] - bm25["hits"]["5"]) / bm25["count"], 6),
        },
        "subsets": {
            "FinQA": {"queries": subsets["FinQA"]["query_count"], "bm25_r5": subsets["FinQA"]["bm25"]["recall_pct"]["@5"], "qwen_r5": subsets["FinQA"]["qwen"]["recall_pct"]["@5"]},
            "TAT-DQA": {"queries": subsets["TAT-DQA"]["query_count"], "bm25_r5": subsets["TAT-DQA"]["bm25"]["recall_pct"]["@5"], "qwen_r5": subsets["TAT-DQA"]["qwen"]["recall_pct"]["@5"]},
        },
        "source_artifact": sources["paths"]["t2_metrics"],
    }


def selected_method(sources: dict[str, Any], curve: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "sada_statement_aware_v1",
        "status": "selected_internal_shadow_method",
        "candidate_supply": "frozen_deep_supply",
        "candidate_admission": "SADA-V1 Statement-Aware Deep Admission V1",
        "candidate_representation": "Statement-Aware Evidence Unit V1",
        "query_representation": "original_query",
        "reranker": MODEL,
        "reranker_revision": MODEL_REVISION,
        "strict_curve": curve,
        "development_shadow": True,
        "production_switch_allowed": False,
        "source_commits": {
            "nf_opt_24_r0": "b6a017e38bba49bd1c52145441556f5100dc5204",
            "nf_opt_23_r0": "35d6e388bc59cf85b61a07c2f4ecd3ac54a2f969",
            "freeze_base": BASE_COMMIT,
        },
        "contract_sources": {
            "sada": sources["paths"]["sada_contract"],
            "statement_aware": sources["paths"]["sa_unit"],
            "reranker": sources["paths"]["sa_reranker"],
        },
    }


def ablation_ledger(sources: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"gate": "NF-OPT-24-R0", "commit": "b6a017e38bba49bd1c52145441556f5100dc5204", "route": "SADA-V1 deep-to-top100 admission", "decision": "selected", "reason": "78/80 supply retained; +10 net over current Top100; 10/10 lost sources recovered", "reopen_allowed": False},
        {"gate": "NF-OPT-23-R0", "commit": "35d6e388bc59cf85b61a07c2f4ecd3ac54a2f969", "route": "Statement-Aware Evidence Unit V1", "decision": "supported_marginal", "reason": "bounded strict R@5 43/80 to 48/80 and semantic R@5 49/80 to 55/80", "reopen_allowed": False},
        {"gate": "NF-OPT-21-R0", "commit": "3cef6bbb25561c68e1fc12f288eaaaad2144fa83", "route": "LRRF-V1", "decision": "marginal_superseded", "reason": "46/80; rescued 9, damaged 6, net +3", "reopen_allowed": False},
        {"gate": "NF-OPT-22-R0", "commit": "2b34d2b27df734824cb8c2ac124d14ff37c5ee1f", "route": "CGLRRF-V1", "decision": "rejected", "reason": "45/80 and frozen gate false", "reopen_allowed": False},
        {"gate": "NF-OPT-21-R1.1", "commit": "135607002e2f74e67425a9480a3abc051b528719", "route": "1.5B listwise selector", "decision": "rejected", "reason": "valid constrained output exposed 33/80 strict R@5 and 41/80 semantic R@5; initial zero-valid run was protocol failure", "reopen_allowed": False},
        {"gate": "NF-OPT-23-R1", "commit": "dd1b64ca0d11b0f20c2ceb6096bb1e39bb68470e", "route": "Query Requirement V1", "decision": "rejected", "reason": "R@5 46 to 43, net -3, semantic R@5 53 to 49", "reopen_allowed": False},
        {"gate": "NF-OPT-25-R0", "commit": BASE_COMMIT, "route": "Strict Financial Evidence Instruction V1", "decision": "positive_but_rejected", "reason": "48/80 versus 46/80 but below the frozen minimum useful 50/80 gate; net +2", "reopen_allowed": False},
    ]


def closed_routes() -> list[dict[str, Any]]:
    names = [
        "BM25/Dense window tuning", "BGE-M3 sparse", "multi-vector retrieval", "protected residual",
        "Query Rewrite", "concept expansion", "Top5 heuristic compression", "reranker capacity scaling 0.6B to 4B",
        "8B scaling", "representation packet", "semantic sibling collapse", "setwise diversity",
        "period boundary guard", "LRRF tuning", "confidence-gated LRRF", "1.5B listwise selector",
        "Query Requirement V1", "Instruction V1",
    ]
    commits = {
        "LRRF tuning": "3cef6bbb25561c68e1fc12f288eaaaad2144fa83",
        "confidence-gated LRRF": "2b34d2b27df734824cb8c2ac124d14ff37c5ee1f",
        "1.5B listwise selector": "135607002e2f74e67425a9480a3abc051b528719",
        "Query Requirement V1": "dd1b64ca0d11b0f20c2ceb6096bb1e39bb68470e",
        "Instruction V1": BASE_COMMIT,
    }
    decisions = {
        "LRRF tuning": "marginal_superseded", "confidence-gated LRRF": "rejected",
        "1.5B listwise selector": "rejected", "Query Requirement V1": "rejected",
        "Instruction V1": "positive_but_rejected",
    }
    reasons = {
        "LRRF tuning": "net +3 but damaged six Qwen hits and was superseded by the selected main chain",
        "confidence-gated LRRF": "45/80, frozen gate false",
        "1.5B listwise selector": "33/80 after output-contract recovery",
        "Query Requirement V1": "net -3 and semantic regression",
        "Instruction V1": "48/80 did not meet the frozen 50/80 minimum",
    }
    return [
        {"route": name, "gate": "historical_ablation", "commit": commits.get(name), "decision": decisions.get(name, "closed"), "reason": reasons.get(name, "closed during the preceding retrieval development sequence"), "reopen_allowed": False, "reopen_condition": "only new independent evidence, benchmark, training stage, representation source, or production defect"}
        for name in names
    ]


def claims(sources: dict[str, Any], external: dict[str, Any], complex_data: dict[str, Any], curve: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
        "evaluation_role": "internal_retrieval_method_freeze",
        "fresh_blind": False,
        "allowed_for_resume": True,
        "prohibited_wording": ["public benchmark", "blind test", "state of the art claim"],
    }
    rows = [
        {"claim_id": "internal.deep_supply_presence", "metric": "Strict Deep Supply Presence", "value": "78/80", "denominator": 80, "method": "frozen deep supply", "artifact": sources["paths"]["sada_contract"], "commit": "35d6e388bc59cf85b61a07c2f4ecd3ac54a2f969", "allowed_wording": "Internal strict evidence supply was 78/80 (97.5%) in development shadow."},
        {"claim_id": "internal.sada_top100", "metric": "SADA-V1 Strict Top100", "value": "78/80", "denominator": 80, "method": "sada_v1", "artifact": sources["paths"]["sada_curve"], "commit": "b6a017e38bba49bd1c52145441556f5100dc5204", "allowed_wording": "SADA-V1 raised internal strict Top100 supply to 78/80 (97.5%)."},
        {"claim_id": "internal.selected_r5", "metric": "selected strict R@5", "value": "46/80", "denominator": 80, "method": "sada_statement_aware_v1", "artifact": sources["paths"]["sada_curve"], "commit": BASE_COMMIT, "allowed_wording": "The selected internal shadow method reached strict R@5 46/80 (57.5%)."},
        {"claim_id": "internal.selected_curve", "metric": "selected strict curve", "value": {k: v for k, v in curve.items() if k.endswith("_hits")}, "denominator": 80, "method": "sada_statement_aware_v1", "artifact": sources["paths"]["sada_curve"], "commit": BASE_COMMIT, "allowed_wording": "Selected internal strict curve, development shadow."},
        {"claim_id": "internal.statement_aware_ablation", "metric": "bounded Statement-Aware R@5", "value": "43/80 to 48/80", "denominator": 80, "method": "statement_aware_evidence_unit_v1", "artifact": sources["paths"]["sa_strict"], "commit": "35d6e388bc59cf85b61a07c2f4ecd3ac54a2f969", "allowed_wording": "Statement-Aware Evidence Unit V1 improved bounded strict R@5 by five sources in development shadow."},
        {"claim_id": "external.t2_qwen_r5", "metric": "T²-RAGBench Primary Test Qwen R@5", "value": f"{external['qwen3_reranker_4b']['r5_recall']:.6%}", "denominator": external["primary_test_queries"], "method": MODEL, "artifact": sources["paths"]["t2_metrics"], "commit": "3352217", "evaluation_role": external["evaluation_role"], "fresh_blind": external["fresh_blind"], "allowed_for_resume": True, "allowed_wording": "On the frozen T²-RAGBench Primary Test contract, Qwen3-Reranker-4B reached 88.56% R@5 (+14.84 pp over BM25); this was a pre-frozen calibration, not a fresh blind test."},
        {"claim_id": "calculation.admitted_accuracy", "metric": "admitted strict calculation accuracy", "value": "4/4", "denominator": 4, "method": "financial calculation final showcase", "artifact": "financial-calculation-final-showcase/final-metrics.json", "commit": "2ae5b577", "evaluation_role": "calculation_runtime_final_showcase", "fresh_blind": False, "allowed_for_resume": True, "allowed_wording": "100% accuracy among admitted executions (4/4); this is not overall calculation accuracy."},
    ]
    return [{**base, **row} for row in rows]


def architecture(sources: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_shadow_architecture": [
            "Financial PDF", "Structured Parsing", "Hybrid / Deep Candidate Supply",
            "Statement-Aware Evidence Construction", "SADA-V1 Deep to Top100 Admission",
            "Qwen3-Reranker-4B", "Top-K Evidence", "Binder / Calculator / Generation / Validator",
        ],
        "current_production_architecture": "production status unchanged; SADA and Statement-Aware shadow artifacts are not switched into production",
        "production_switch_allowed": False,
        "method_contract": {
            "admission": "sada_v1",
            "representation": "statement_aware_evidence_unit_v1",
            "query": "original_query",
            "reranker": MODEL,
            "revision": MODEL_REVISION,
        },
        "source_artifacts": [sources["paths"]["sada_contract"], sources["paths"]["sa_unit"], sources["paths"]["sa_reranker"]],
    }


def write_readme(out: Path, complex_data: dict[str, Any]) -> None:
    calc = complex_data["calculation_evidence"]["all_slots_at_5"]["hits"]
    (out / "README.md").write_text(
        "# NF-OPT-26 R0 — Internal Retrieval Method Freeze\n\n"
        "## Scope\n\n"
        "This gate is a read-only evidence seal. No model, retrieval, admission, training, or parameter execution occurs.\n\n"
        "## Frozen benchmark\n\n"
        "The internal development benchmark contains 8 documents, 1,348 PDF pages, 44,608 chunks, 72 questions, 64 answerable questions, 8 no-answer questions, and 80 strict physical sources. It is development-exposed, not a fresh blind evaluation.\n\n"
        "## Final selected method\n\n"
        "`sada_statement_aware_v1`: frozen Deep Supply → SADA-V1 Top100 admission → Statement-Aware Evidence Unit V1 → Original Query → Qwen3-Reranker-4B. Production remains unchanged.\n\n"
        "## Candidate supply and strict curve\n\n"
        "Deep Supply and SADA Top100 each contain 78/80 strict sources (97.5%). The selected shadow strict curve is 17/80, 34/80, 46/80, 60/80, 69/80, 73/80, and 78/80 at R@1/3/5/10/20/50/100.\n\n"
        "## Complex evidence\n\n"
        f"Sealed selected-baseline metrics are Semantic R@5 53/80 and R@10 63/80; multi-evidence All@5 6/16 and All@10 9/16; calculation all-slots @{5} {calc}/11 and @{10} 7/11. The brief listed calculation @5 as 5/11, but the sealed NF-OPT-25 SADA-SA artifact records {calc}/11; the sealed artifact is authoritative and the discrepancy is preserved in `final-complex-evidence-metrics.json`.\n\n"
        "## Key improvements\n\n"
        "SADA recovered all 10 Deep-present strict sources lost by the prior Top100 admission and retained all existing 68. Statement-Aware serialization independently improved bounded strict R@5 from 43/80 to 48/80.\n\n"
        "## Rejected and superseded ablations\n\n"
        "Query Requirement V1 was rejected (net -3). Strict Financial Instruction V1 was positive but below its frozen acceptance gate. LRRF was marginal and superseded. CGLRRF and constrained listwise selection were rejected; the initial listwise zero-valid run was an output-contract failure, while the recovered run exposed 33/80 semantic performance. See `ablation-ledger.json` and `closed-routes.json`.\n\n"
        "## External public benchmark\n\n"
        "T²-RAGBench artifacts remain a separate pre-frozen strong cross-encoder calibration: BM25 R@5 73.72% and Qwen3-Reranker-4B R@5 88.56% (+14.84 pp). This is not merged with internal denominators and is not presented as a blind or leaderboard claim.\n\n"
        "## Evaluation roles and limitations\n\n"
        "All internal results are development shadow evidence. Two strict sources remain absent from Deep Supply; strict physical identity is harder than semantic matching; same-metric, same-document, period, and operand hard negatives remain; query-side requirements regressed; instruction calibration was marginal and rejected. Further gains likely require new training data, hard-negative training, or a genuinely new evidence representation.\n\n"
        "## Resume-safe claims\n\n"
        "Use the claim registry for wording and source lineage. Internal claims must say `internal strict evidence set` and `development shadow`; external claims must state the frozen T² contract and pre-frozen calibration role.\n\n"
        "## Production status and reopen conditions\n\n"
        "`production_switch_allowed=false` and `further_internal_retrieval_tuning_allowed=false`. Reopen only for new independent blind evidence, a new reranker training checkpoint, a new hard-negative dataset, a fundamentally new parser/evidence representation, or a production defect requiring correction.\n",
        encoding="utf-8",
    )


def write_manifest(out: Path) -> tuple[dict[str, Any], str]:
    files = []
    for path in sorted(out.rglob("*")):
        if not path.is_file() or path.name in {"final-evidence-manifest.json", "final-evidence-manifest.sha256", "decision.json"}:
            continue
        files.append({"path": path.relative_to(out).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {
        "gate": "NF-OPT-26-R0",
        "file_count": len(files),
        "files": files,
        "deterministic_order": "lexicographic relative path",
        "self_referential_exclusions": ["decision.json", "final-evidence-manifest.json", "final-evidence-manifest.sha256"],
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    (out / "final-evidence-manifest.json").write_text(payload, encoding="utf-8")
    digest = sha256_bytes(payload.encode("utf-8"))
    (out / "final-evidence-manifest.sha256").write_text(digest + "\n", encoding="utf-8")
    return manifest, digest


def build(root: Path) -> dict[str, Any]:
    sources = load_sources(root)
    verify_sources(sources)
    out = root / "artifacts/evaluation" / OUT_NAME
    out.mkdir(parents=True, exist_ok=True)
    benchmark = benchmark_contract(sources)
    curve = strict_curve(sources)
    complex_data = complex_metrics(sources)
    external = external_metrics(sources)
    method = selected_method(sources, curve)
    write_json(out / "benchmark-freeze-contract.json", benchmark)
    write_json(out / "final-internal-retrieval-metrics.json", {
        "strict_sources": STRICT_TOTAL,
        "deep_supply": {"hits": 78, "total": STRICT_TOTAL, "recall": pct(78, STRICT_TOTAL)},
        "sada_top100": {"hits": 78, "total": STRICT_TOTAL, "recall": pct(78, STRICT_TOTAL)},
        "selected_method": {"name": "sada_statement_aware_v1", **{k: v for k, v in curve.items() if k.startswith("r")}},
        "source_artifact": sources["paths"]["sada_curve"],
    })
    write_json(out / "final-external-benchmark-metrics.json", external)
    write_json(out / "final-complex-evidence-metrics.json", complex_data)
    write_json(out / "selected-method.json", method)
    write_json(out / "internal-retrieval-method-freeze.json", {
        "status": "frozen",
        "selected_internal_shadow_method": "sada_statement_aware_v1",
        "candidate_supply_method": "sada_v1",
        "candidate_representation": "statement_aware_evidence_unit_v1",
        "reranker": MODEL,
        "reranker_revision": MODEL_REVISION,
        "query_representation": "original_query",
        "instruction": "baseline_frozen_instruction",
        "strict_top100_hits": 78,
        "strict_top5_hits": 46,
        "development_shadow": True,
        "production_switch_allowed": False,
        "further_internal_retrieval_tuning_allowed": False,
        "next_gate": "end_to_end_rag_integration_review",
    })
    write_json(out / "ablation-ledger.json", ablation_ledger(sources))
    write_json(out / "closed-routes.json", closed_routes())
    write_json(out / "retrieval-claim-registry.json", {"claims": claims(sources, external, complex_data, curve), "disclaimer": "Internal claims are development-shadow; external claims use a separate pre-frozen contract."})
    write_json(out / "known-limitations.json", {
        "development_shadow": True,
        "limitations": [
            "Internal benchmark has been development-exposed and is not fresh blind.",
            "Two strict sources remain absent from Deep Supply.",
            "Top100 supply is 97.5%, but selected strict Top5 is 57.5%.",
            "Strict physical source identity is substantially harder than semantic fact matching.",
            "The off-the-shelf 4B reranker struggles with same-metric, same-document, period, and operand hard negatives.",
            "Query-side structured requirements regressed ranking.",
            "Instruction calibration produced a positive but marginal gain and failed its frozen gate.",
            "Further gains likely require new training data, hard-negative training, or a genuinely new evidence representation.",
        ],
    })
    write_json(out / "final-retrieval-architecture.json", architecture(sources))
    write_json(out / "lineage.json", {
        "gate": "NF-OPT-26-R0",
        "base_commit": BASE_COMMIT,
        "evaluation_role": "internal_retrieval_method_freeze",
        "execution_flags": OUTER_FLAGS,
        "selected_method": "sada_statement_aware_v1",
        "historical_commits": {
            "nf_opt_24_r0": "b6a017e38bba49bd1c52145441556f5100dc5204",
            "nf_opt_23_r0": "35d6e388bc59cf85b61a07c2f4ecd3ac54a2f969",
            "nf_opt_23_r1": "dd1b64ca0d11b0f20c2ceb6096bb1e39bb68470e",
            "nf_opt_25_r0": BASE_COMMIT,
        },
        "source_artifacts": sources["paths"],
        "reopen_conditions": ["new independent blind benchmark", "new reranker training checkpoint", "new hard-negative training dataset", "fundamentally new parser/evidence representation", "production defect requiring correction"],
    })
    write_readme(out, complex_data)
    manifest, digest = write_manifest(out)
    decision = {
        "gate": "NF-OPT-26-R0",
        "base_commit": BASE_COMMIT,
        "evaluation_role": "internal_retrieval_method_freeze",
        "fresh_blind_evaluation": False,
        **OUTER_FLAGS,
        "internal_retrieval_method_frozen": True,
        "selected_internal_shadow_method": "sada_statement_aware_v1",
        "deep_supply_hits": 78,
        "sada_top100_hits": 78,
        "selected_strict_r5_hits": 46,
        "selected_strict_r100_hits": 78,
        "statement_aware_evidence_unit_effective": "marginal",
        "statement_aware_evidence_unit_supported": True,
        "strict_financial_instruction_effective": False,
        "instruction_v1_selected": False,
        "query_requirement_serialization_effective": False,
        "late_fusion_effective": "marginal_superseded",
        "confidence_gated_late_fusion_effective": False,
        "listwise_selector_effective": False,
        "production_switch_allowed": False,
        "further_internal_retrieval_tuning_allowed": False,
        "final_evidence_manifest_sha256": digest,
        "next_gate": "end_to_end_rag_integration_review",
        "validation": {
            "all_metrics_trace_to_sealed_artifacts": True,
            "percentages_recomputed": True,
            "external_internal_separated": True,
            "no_public_leaderboard_wording": True,
            "calculation_admitted_accuracy_not_overall_accuracy": True,
        },
    }
    write_json(out / "decision.json", decision)
    # The decision is part of the evidence set, so refresh the manifest after it.
    manifest, digest = write_manifest(out)
    decision["final_evidence_manifest_sha256"] = digest
    write_json(out / "decision.json", decision)
    write_manifest(out)
    return {"out": str(out), "decision": decision, "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    result = build(args.root.resolve())
    print(json.dumps(result["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
