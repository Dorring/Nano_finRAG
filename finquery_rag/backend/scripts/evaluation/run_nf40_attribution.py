"""Run a single, local-only NF40 frozen-context attribution evaluation."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path

from openai import OpenAI

from src.evaluation.evaluation import load_jsonl_cases
from src.evaluation.nf40_frozen_context import load_frozen_contexts
from src.evaluation.nf40_runner import FrozenContextEvaluationRunner, validate_labeled_cases
from src.evaluation.nf40_start_gate import require_verified_nf39_r2_inputs
from src.services.rag_engine import RAGEngine


def _sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--acceptance", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--final-context-manifest", required=True, type=Path)
    parser.add_argument("--frozen-payload-path", required=True, type=Path)
    parser.add_argument("--expected-payload-sha256", required=True)
    parser.add_argument("--tenant-id", required=True, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--runtime-dir", required=True, type=Path)
    return parser.parse_args()


def _build_frozen_context_engine(client: OpenAI) -> RAGEngine:
    """Build only the answer-pipeline dependencies required by NF40.

    Retrieval is never reached with a verified frozen context. Avoiding the
    optional reranker makes this command independent of its model cache and
    prevents an unrelated remote download before the evaluation begins.
    """
    return RAGEngine(
        client,
        model_name=os.getenv("LLM_MODEL_NAME", "nanochat"),
        use_hybrid=False,
        reranker_name="none",
        retrieval_candidate_multiplier=1,
    )


async def _run(args: argparse.Namespace) -> None:
    # All gates execute before engine/model construction and before outputs.
    require_verified_nf39_r2_inputs(
        acceptance_path=args.acceptance,
        snapshot_manifest_path=args.snapshot_manifest,
        frozen_payload_path=args.frozen_payload_path,
        expected_payload_sha256=args.expected_payload_sha256,
    )
    cases = validate_labeled_cases(load_jsonl_cases(args.cases))
    contexts = load_frozen_contexts(args.frozen_payload_path, args.final_context_manifest)
    if args.tenant_id != 1:
        raise ValueError("NF40 frozen snapshot is approved only for tenant 1")

    client = OpenAI(
        base_url=os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
        api_key=os.getenv("LLM_API_KEY", "not-needed-for-local"),
    )
    engine = _build_frozen_context_engine(client)
    runner = FrozenContextEvaluationRunner(rag_engine=engine)
    runs, metrics = await runner.run(cases=cases, contexts=contexts, tenant_id=args.tenant_id)

    baseline = json.loads((args.acceptance.parent / "baseline-manifest.json").read_text(encoding="utf-8"))
    final_contexts_hash = _sha256_json({case_id: context.final_context_hash for case_id, context in sorted(contexts.items())})
    model = {
        "model_name": engine.model_name,
        "endpoint_identity": os.getenv("LLM_API_BASE_URL", "http://127.0.0.1:8500/v1"),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": engine.max_new_tokens,
    }
    run_manifest = {
        "artifact_schema": "nf40/v1",
        "question_hash": baseline.get("question_hash"),
        "label_hash": baseline.get("label_hash"),
        "candidate_pool_hash": baseline.get("candidate_pool_hash"),
        "final_contexts_hash": final_contexts_hash,
        "generator": model,
        "prompt_hash": _sha256_json({"gateway": type(engine._llm_gateway).__name__}),
        "context_template_hash": _sha256_json({"renderer": "nf39-r2/context-renderer/v1"}),
        "calculator_config_hash": _sha256_json({"enabled": engine._calculation_pipeline is not None}),
        "validator_config_hash": _sha256_json({"enabled": engine._validation_pipeline is not None}),
        "repair_policy_hash": _sha256_json({"enabled": engine._validation_pipeline is not None, "max_repairs": 1}),
        "final_top_k": 5,
        "case_count": len(cases),
    }
    public_rows = [run.public_record for run in runs]
    _write_json(args.out_dir / "baseline-manifest.json", baseline)
    _write_json(args.out_dir / "frozen-context-manifest.json", {"artifact_schema": "nf40/v1", "final_contexts_hash": final_contexts_hash, "cases": [{key: value for key, value in row.items() if key != "rendered_content"} for row in public_rows]})
    _write_json(args.out_dir / "run-manifest.json", run_manifest)
    _write_json(args.out_dir / "pipeline-stage-summary.json", metrics)
    _write_json(args.out_dir / "case-attribution.json", {"artifact_schema": "nf40/v1", "cases": public_rows})
    _write_json(args.out_dir / "conditional-metrics.json", metrics)
    _write_json(args.out_dir / "nf40-acceptance.json", {"artifact_schema": "nf40/v1", "completed": True, "production_behavior_changed": False, "case_count": len(cases)})
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    private_path = args.runtime_dir / "nf40-raw-trace.jsonl"
    private_path.write_text("".join(json.dumps(run.private_record, ensure_ascii=False) + "\n" for run in runs), encoding="utf-8")
    private_path.chmod(stat.S_IRUSR | stat.S_IWUSR)


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
