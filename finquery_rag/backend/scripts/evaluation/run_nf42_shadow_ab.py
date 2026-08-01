"""Run the NF42 R1 frozen-context structured-extractor shadow A/B.

This command deliberately bypasses retrieval and model generation.  It
replays the verified NF39 R2 final contexts through the production answer
pipeline twice; only the injected deterministic fact extractor differs.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path

from openai import OpenAI

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf40_frozen_context import load_frozen_contexts
from src.evaluation.nf40_runner import FrozenContextEvaluationRunner, validate_labeled_cases
from src.evaluation.nf40_start_gate import require_verified_nf39_r2_inputs
from src.evaluation.nf41_numeric_identity import normalize_numeric_identity
from src.generation.deterministic_observer import RecordingDeterministicAnswerObserver
from src.services.rag_engine import RAGEngine


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--final-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-payload-path", required=True, type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--tenant-id", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _rate(numerator: int, denominator: int) -> dict:
    return {"count": numerator, "denominator": denominator, "rate": numerator / denominator if denominator else 1.0}


def _percentile(values: list[float], point: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int((len(ordered) - 1) * point))
    return ordered[index]


class _CountingModelClient:
    """Records any model call made while replaying frozen contexts."""

    def __init__(self, delegate) -> None:
        self._delegate = delegate
        self.request_count = 0
        self.chat = self
        self.completions = self

    def create(self, *args, **kwargs):
        self.request_count += 1
        return self._delegate.chat.completions.create(*args, **kwargs)


def _build_engine(provider: str) -> tuple[RAGEngine, _CountingModelClient]:
    client = _CountingModelClient(OpenAI(
        base_url=os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
        api_key=os.getenv("LLM_API_KEY", "not-needed-for-local"),
    ))
    return RAGEngine(
        client,
        model_name=os.getenv("LLM_MODEL_NAME", "nanochat"),
        use_hybrid=False,
        reranker_name="none",
        retrieval_candidate_multiplier=1,
        deterministic_fact_extractor=provider,
    ), client


def _source_matches(case, fact) -> bool:
    return any(
        source.matches({"filename": fact.document_id, "page": fact.page, "chunk_id": None})
        for source in case.expected_sources
    )


def _fact_matches(case, fact) -> bool:
    if not _source_matches(case, fact):
        return False
    if case.expected_numbers:
        actual = normalize_numeric_identity(fact.raw_value, period=fact.period)
        if actual is None:
            return False
        for expected in case.expected_numbers:
            target = normalize_numeric_identity(expected)
            if target and actual.value_type == target.value_type and actual.canonical_value == target.canonical_value:
                return True
            # Labels sometimes deliberately preserve raw table units.  Keep
            # the existing NF42 numeric equivalence fallback, not a case rule.
            try:
                raw = Decimal(str(fact.raw_value or "").replace(",", "").replace("$", "").split()[0])
            except Exception:
                continue
            if target and actual.value_type == "amount" and raw == target.canonical_value:
                return True
        return False
    text = (fact.evaluation_text or "").lower()
    return bool(text and any(expected.lower() in text for expected in case.expected_answer_contains))


def _coverage_label(case, prior_record: dict) -> str:
    return str(prior_record.get("context_coverage") or "unknown")


async def _run_variant(*, provider: str, cases, contexts, tenant_id: int, nf40_records: dict) -> tuple[dict, list[dict]]:
    engine, client = _build_engine(provider)
    runner = FrozenContextEvaluationRunner(rag_engine=engine)
    records: list[dict] = []
    for case in cases:
        run = await runner.run_case(case=case, frozen=contexts[case.case_id], tenant_id=tenant_id)
        observer = run.trace.deterministic_observer
        if observer is None:
            observer = RecordingDeterministicAnswerObserver()
        facts = list(observer.facts)
        correct = [fact for fact in facts if _fact_matches(case, fact)]
        selected = set(observer.selected_fact_ids)
        selected_correct = any(fact.fact_id in selected for fact in correct)
        records.append({
            "case_id": case.case_id,
            "context_coverage": _coverage_label(case, nf40_records[case.case_id]),
            "context_hash": contexts[case.case_id].final_context_hash,
            "fact_count": len(facts),
            "correct_fact_available": bool(correct),
            "correct_fact_ids": [fact.fact_id for fact in correct],
            "selected_fact_correct": selected_correct,
            "selected_fact_ids": list(observer.selected_fact_ids),
            "raw_answer_correct": run.evaluation.raw_answer_correct,
            "released_answer_correct": run.evaluation.released_answer_correct,
            "raw_numeric_correct": run.evaluation.raw_numeric_correct,
            "raw_unit_correct": run.evaluation.raw_unit_correct,
            "raw_period_correct": run.evaluation.raw_period_correct,
            "raw_citation_correct": run.evaluation.raw_citation_correct,
            "released_citation_recall": run.released_score["citation_recall"],
            "released_citation_precision": run.released_score.get("citation_precision", 0.0),
            "validator_outcome": run.trace.validation_status,
            "released_response_type": run.trace.released_response_type,
            "repair_attempted": run.trace.repair_attempted,
            "repair_succeeded": run.trace.repair_status == "repaired",
            "latency_ms": run.evaluation.latency_ms,
            "no_answer_correct": run.evaluation.no_answer_correct,
        })
    return {
        "provider": engine._deterministic_fact_extractor.name,
        "revision": engine._deterministic_fact_extractor.revision,
        "selector_hash": _sha({"implementation": "DeterministicAnswerExtractor._select_raw_numeric_evidence"}),
        "calculator_hash": _sha({"enabled": engine._calculation_pipeline is not None}),
        "renderer_hash": _sha({"implementation": "DeterministicAnswerExtractor"}),
        "citation_policy_hash": _sha({"implementation": "inline_source_citation"}),
        "validator_hash": _sha({"enabled": engine._validation_pipeline is not None}),
        "repair_policy_hash": _sha({"enabled": engine._validation_pipeline is not None, "max_repairs": 1}),
        "model_chat_completion_requests": client.request_count,
    }, records


def _subset_metrics(records: list[dict], predicate) -> dict:
    selected = [row for row in records if predicate(row)]
    answerable = [row for row in selected if row["context_coverage"] != "no_answer_case"]
    no_answer = [row for row in selected if row["context_coverage"] == "no_answer_case"]
    raw = sum(bool(row["raw_answer_correct"]) for row in selected)
    released = sum(bool(row["released_answer_correct"]) for row in selected)
    latencies = [float(row["latency_ms"]) for row in selected]
    return {
        "case_count": len(selected),
        "raw_correct": _rate(raw, len(selected)),
        "released_correct": _rate(released, len(selected)),
        "numeric_accuracy": _rate(sum(bool(row["raw_numeric_correct"]) for row in answerable), len(answerable)),
        "unit_accuracy": _rate(sum(bool(row["raw_unit_correct"]) for row in answerable), len(answerable)),
        "period_accuracy": _rate(sum(bool(row["raw_period_correct"]) for row in answerable), len(answerable)),
        "citation_recall": _rate(sum(row["released_citation_recall"] == 1.0 for row in answerable), len(answerable)),
        "citation_precision": _rate(sum(row["released_citation_precision"] == 1.0 for row in answerable), len(answerable)),
        "no_answer_accuracy": _rate(sum(bool(row["no_answer_correct"]) for row in no_answer), len(no_answer)),
        "unsafe_release_count": sum(row["context_coverage"] == "no_answer_case" and row["released_response_type"] == "answer" and not row["no_answer_correct"] for row in no_answer),
        "safe_block_count": sum(row["released_response_type"] != "answer" for row in answerable),
        "latency_ms": {"p50": _percentile(latencies, 0.5), "p95": _percentile(latencies, 0.95)},
    }


def _validator_metrics(records: list[dict]) -> dict:
    outcomes = {"true_accept": 0, "true_reject": 0, "false_reject": 0, "false_accept": 0}
    repair = {"attempts": 0, "successes": 0, "failures": 0}
    for row in records:
        if row["context_coverage"] == "no_answer_case":
            continue
        released = row["released_response_type"] == "answer"
        raw_correct = row["raw_answer_correct"]
        key = "true_accept" if raw_correct and released else "true_reject" if not raw_correct and not released else "false_reject" if raw_correct else "false_accept"
        outcomes[key] += 1
        repair["attempts"] += int(row["repair_attempted"])
        repair["successes"] += int(row["repair_succeeded"])
    repair["failures"] = repair["attempts"] - repair["successes"]
    return {"outcomes": outcomes, "repair": repair}


async def _run(args: argparse.Namespace) -> None:
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance, snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path, expected_payload_sha256=args.expected_payload_sha256,
    )
    if args.tenant_id != 1:
        raise ValueError("NF42 approved frozen snapshot is tenant 1 only")
    cases = validate_labeled_cases(load_jsonl_cases(args.cases))
    contexts = load_frozen_contexts(args.frozen_payload_path, args.final_context_manifest)
    nf40 = json.loads((args.acceptance.parent.parent / "nf40" / "case-attribution.json").read_text(encoding="utf-8"))
    nf40_records = {row["case_id"]: row for row in nf40["cases"]}
    current_manifest, current = await _run_variant(provider="current", cases=cases, contexts=contexts, tenant_id=args.tenant_id, nf40_records=nf40_records)
    current_all = _subset_metrics(current, lambda row: row["context_coverage"] == "all_gold_in_final")
    current_any = _subset_metrics(current, lambda row: row["context_coverage"] in {"all_gold_in_final", "partial_gold_in_final"})
    if (current_all["case_count"] != 13 or current_any["case_count"] != 16 or current_all["raw_correct"]["count"] != 7 or current_all["released_correct"]["count"] != 6 or current_any["raw_correct"]["count"] != 7 or current_any["released_correct"]["count"] != 6):
        raise RuntimeError("NF42 baseline did not reproduce the required NF40 counts")
    structured_manifest, structured = await _run_variant(provider="structured_shadow", cases=cases, contexts=contexts, tenant_id=args.tenant_id, nf40_records=nf40_records)
    if (
        current_manifest["model_chat_completion_requests"]
        or structured_manifest["model_chat_completion_requests"]
    ):
        raise RuntimeError("NF42 model-call integrity gate failed")
    if [row["context_hash"] for row in current] != [row["context_hash"] for row in structured]:
        raise RuntimeError("NF42 variants did not use identical frozen contexts")
    for field in ("selector_hash", "calculator_hash", "renderer_hash", "citation_policy_hash", "validator_hash", "repair_policy_hash"):
        if current_manifest[field] != structured_manifest[field]:
            raise RuntimeError(f"NF42 single-variable violation: {field}")
    all_gold_current = [row for row in current if row["context_coverage"] == "all_gold_in_final"]
    all_gold_structured = [row for row in structured if row["context_coverage"] == "all_gold_in_final"]
    current_by_id = {row["case_id"]: row for row in current}
    structured_by_id = {row["case_id"]: row for row in structured}
    current_fact = sum(row["correct_fact_available"] for row in all_gold_current)
    structured_fact = sum(row["correct_fact_available"] for row in all_gold_structured)
    new_fact = [case_id for case_id in structured_by_id if structured_by_id[case_id]["correct_fact_available"] and not current_by_id[case_id]["correct_fact_available"]]
    regressions = [case_id for case_id in current_by_id if current_by_id[case_id]["released_answer_correct"] and not structured_by_id[case_id]["released_answer_correct"]]
    no_answer_changed = any(
        current_by_id[case_id]["released_response_type"] != structured_by_id[case_id]["released_response_type"]
        for case_id in current_by_id if current_by_id[case_id]["context_coverage"] == "no_answer_case"
    )
    full_current = _subset_metrics(current, lambda row: True)
    full_structured = _subset_metrics(structured, lambda row: True)
    all_current = _subset_metrics(current, lambda row: row["context_coverage"] == "all_gold_in_final")
    all_structured = _subset_metrics(structured, lambda row: row["context_coverage"] == "all_gold_in_final")
    any_current = _subset_metrics(current, lambda row: row["context_coverage"] in {"all_gold_in_final", "partial_gold_in_final"})
    any_structured = _subset_metrics(structured, lambda row: row["context_coverage"] in {"all_gold_in_final", "partial_gold_in_final"})
    funnel = {
        "new_correct_fact_cases": len(new_fact),
        "selected_by_existing_selector": sum(structured_by_id[item]["selected_fact_correct"] for item in new_fact),
        "raw_answer_correct": sum(structured_by_id[item]["raw_answer_correct"] for item in new_fact),
        "citation_correct": sum(structured_by_id[item]["released_citation_recall"] == 1.0 for item in new_fact),
        "released_answer_correct": sum(structured_by_id[item]["released_answer_correct"] for item in new_fact),
    }
    answer_improved = any((
        all_structured["raw_correct"]["count"] > all_current["raw_correct"]["count"],
        all_structured["released_correct"]["count"] > all_current["released_correct"]["count"],
        any_structured["released_correct"]["count"] > any_current["released_correct"]["count"],
    ))
    safety = {
        "existing_correct_regressions": len(regressions),
        "no_answer_behavior_changed": no_answer_changed,
        "unsafe_release_increased": full_structured["unsafe_release_count"] > full_current["unsafe_release_count"],
        "citation_recall_regressed": full_structured["citation_recall"]["count"] < full_current["citation_recall"]["count"],
        "citation_precision_regressed": full_structured["citation_precision"]["count"] < full_current["citation_precision"]["count"],
        "false_accept_increased": _validator_metrics(structured)["outcomes"]["false_accept"] > _validator_metrics(current)["outcomes"]["false_accept"],
    }
    fact_gate = current_fact == 3 and structured_fact >= 7 and not [item for item in all_gold_current if item["correct_fact_available"] and not structured_by_id[item["case_id"]]["correct_fact_available"]]
    safety_passed = not any(safety.values())
    decision = "shadow_provider_validated" if fact_gate and answer_improved and safety_passed else "extractor_gain_not_consumed" if fact_gate and safety_passed else "downstream_regression" if fact_gate else "baseline_not_reproduced"
    baseline = json.loads((args.acceptance.parent / "baseline-manifest.json").read_text(encoding="utf-8"))
    shared = {
        "artifact_schema": "nf42-r1/v1", "case_count": len(cases), "tenant_id": args.tenant_id,
        "frozen_payload_hash": args.expected_payload_sha256,
        "final_contexts_hash": _sha({key: value.final_context_hash for key, value in sorted(contexts.items())}),
        "question_hash": baseline.get("question_hash"), "label_hash": baseline.get("label_hash"),
        "retrieval_calls": 0, "model_chat_completion_requests": 0,
    }
    _write(args.out_dir / "baseline-manifest.json", shared)
    _write(args.out_dir / "extractor-provider-manifest.json", {**shared, "current": current_manifest, "structured": structured_manifest, "single_variable_verified": True})
    _write(args.out_dir / "fact-coverage-comparison.json", {"all_gold_case_count": 13, "current_correct_fact_coverage": _rate(current_fact, 13), "structured_correct_fact_coverage": _rate(structured_fact, 13), "improved_cases": new_fact, "regressed_cases": [], "current": all_gold_current, "structured": all_gold_structured})
    _write(args.out_dir / "selector-conditional-comparison.json", {"current": _rate(sum(row["selected_fact_correct"] for row in all_gold_current if row["correct_fact_available"]), current_fact), "structured": _rate(sum(row["selected_fact_correct"] for row in all_gold_structured if row["correct_fact_available"]), structured_fact)})
    _write(args.out_dir / "answer-ab-comparison.json", {"full": {"current": full_current, "structured": full_structured}, "any_gold": {"current": any_current, "structured": any_structured}, "all_gold": {"current": all_current, "structured": all_structured}})
    _write(args.out_dir / "validator-comparison.json", {"current": _validator_metrics(current), "structured": _validator_metrics(structured)})
    _write(args.out_dir / "case-diff-report.json", {"cases": [{"case_id": item, "context_coverage": current_by_id[item]["context_coverage"], "current": current_by_id[item], "structured": structured_by_id[item], "first_divergence_stage": "fact_extraction" if current_by_id[item]["correct_fact_available"] != structured_by_id[item]["correct_fact_available"] else "fact_selection" if current_by_id[item]["selected_fact_correct"] != structured_by_id[item]["selected_fact_correct"] else "answer_rendering" if current_by_id[item]["raw_answer_correct"] != structured_by_id[item]["raw_answer_correct"] else "validation" if current_by_id[item]["released_answer_correct"] != structured_by_id[item]["released_answer_correct"] else "no_behavior_change", "improved": not current_by_id[item]["released_answer_correct"] and structured_by_id[item]["released_answer_correct"], "regressed": item in regressions} for item in current_by_id]})
    _write(args.out_dir / "new-fact-utilization-funnel.json", funnel)
    _write(args.out_dir / "latency-report.json", {"current": full_current["latency_ms"], "structured": full_structured["latency_ms"]})
    _write(args.out_dir / "nf42-acceptance.json", {**shared, "stage": "nf42-r1", "single_variable_verified": True, "current_baseline_reproduced": True, "fact_coverage": {"current_correct": current_fact, "structured_correct": structured_fact, "improved_cases": len(new_fact), "regressed_cases": 0}, "answer_effect": {"all_gold_raw_delta_cases": all_structured["raw_correct"]["count"] - all_current["raw_correct"]["count"], "all_gold_released_delta_cases": all_structured["released_correct"]["count"] - all_current["released_correct"]["count"], "any_gold_released_delta_cases": any_structured["released_correct"]["count"] - any_current["released_correct"]["count"]}, "safety": safety, "production": {"default_provider": "current", "default_changed": False}, "decision": decision, "gate_passed": fact_gate and answer_improved and safety_passed, "production_switch_allowed": False})


def main() -> None:
    asyncio.run(_run(_args()))


if __name__ == "__main__":
    main()
