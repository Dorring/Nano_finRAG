"""Generate and evaluate one split of the native structured-fact benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.structured_fact_v2 import parse_structured_fact_query, structured_fact_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/structured-fact-v2-benchmark"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _stable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(str(row["fact_identity"]).encode()).hexdigest())


def build_cases(facts: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    split_facts = [row for row in facts if row["split"] == split and row["normalized_value"] is not None]
    by_issuer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in split_facts:
        by_issuer[str(fact["issuer"])].append(fact)
    cases: list[dict[str, Any]] = []
    for issuer in sorted(by_issuer):
        unique: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        by_concept: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for fact in by_issuer[issuer]:
            unique[(str(fact["concept"]), str(fact["period_end"]))].append(fact)
        eligible = [rows[0] for rows in unique.values() if len(rows) == 1]
        for fact in eligible:
            by_concept[str(fact["concept"])].append(fact)
        single_pool = _stable_rows(eligible)
        selected_singles: list[dict[str, Any]] = []
        used_concepts: set[str] = set()
        for fact in single_pool:
            concept = str(fact["concept"])
            if concept in used_concepts:
                continue
            selected_singles.append(fact)
            used_concepts.add(concept)
            if len(selected_singles) == 20:
                break
        if len(selected_singles) != 20:
            raise ValueError(f"issuer lacks 20 unique single facts: {issuer}")
        for index, fact in enumerate(selected_singles):
            cases.append(
                {
                    "case_id": f"{split}:single:{hashlib.sha256((issuer + str(index) + fact['fact_identity']).encode()).hexdigest()}",
                    "case_type": "single_fact",
                    "issuer": issuer,
                    "metric": fact["label"],
                    "periods": [fact["period_end"]],
                    "question": f"According to {issuer}'s Form 10-K, what was {fact['label']} for the period ended {fact['period_end']}?",
                    "gold_fact_identities": [fact["fact_identity"]],
                }
            )
        pair_pool = []
        for concept, rows in by_concept.items():
            periods = {str(row["period_end"]): row for row in rows}
            if len(periods) >= 2:
                chosen = [periods[key] for key in sorted(periods, reverse=True)[:2]]
                pair_pool.append((concept, chosen))
        pair_pool.sort(key=lambda item: hashlib.sha256((issuer + item[0]).encode()).hexdigest())
        if len(pair_pool) < 10:
            raise ValueError(f"issuer lacks 10 complete period pairs: {issuer}")
        for index, (_concept, pair) in enumerate(pair_pool[:10]):
            periods = sorted(str(row["period_end"]) for row in pair)
            cases.append(
                {
                    "case_id": f"{split}:pair:{hashlib.sha256((issuer + str(index) + pair[0]['concept']).encode()).hexdigest()}",
                    "case_type": "period_pair",
                    "issuer": issuer,
                    "metric": pair[0]["label"],
                    "periods": periods,
                    "question": f"According to {issuer}'s Form 10-K, what was {pair[0]['label']} for the periods ended {periods[0]} and {periods[1]}?",
                    "gold_fact_identities": [row["fact_identity"] for row in pair],
                }
            )
    issuers = sorted(by_issuer)
    for index in range(10):
        issuer = issuers[index % len(issuers)]
        cases.append(
            {
                "case_id": f"{split}:no-answer:{index:02d}",
                "case_type": "no_answer",
                "issuer": issuer,
                "metric": f"nonexistent lunar reserve metric {index}",
                "periods": ["2099-12-31"],
                "question": f"According to {issuer}'s Form 10-K, what was nonexistent lunar reserve metric {index} for 2099?",
                "gold_fact_identities": [],
            }
        )
    return sorted(cases, key=lambda row: str(row["case_id"]))


def evaluate(cases: list[dict[str, Any]], facts: list[dict[str, Any]], *, split: str) -> dict[str, Any]:
    corpus = [row for row in facts if row["split"] == split]
    records = []
    for case in cases:
        scored = []
        parsed_query = parse_structured_fact_query(str(case["question"]))
        if parsed_query is not None:
            query_issuer, query_metric, query_periods = parsed_query
            for fact in corpus:
                score = structured_fact_score(
                    query_issuer=query_issuer,
                    query_metric=query_metric,
                    query_periods=query_periods,
                    fact=fact,
                )
                if score is not None:
                    scored.append((score, str(fact["fact_identity"])))
        ranked = [identity for _score, identity in sorted(scored, key=lambda item: (-item[0], item[1]))[:5]]
        gold = set(case["gold_fact_identities"])
        hit_count = len(gold.intersection(ranked))
        records.append(
            {
                "case_id": case["case_id"],
                "case_type": case["case_type"],
                "gold_count": len(gold),
                "hit_count_at_5": hit_count,
                "all_gold_at_5": bool(gold) and hit_count == len(gold),
                "abstained": not ranked,
                "query_parsed": parsed_query is not None,
                "ranked_fact_identities": ranked,
            }
        )
    answerable = [row for row in records if row["gold_count"]]
    singles = [row for row in answerable if row["case_type"] == "single_fact"]
    pairs = [row for row in answerable if row["case_type"] == "period_pair"]
    no_answer = [row for row in records if row["case_type"] == "no_answer"]
    gold_count = sum(row["gold_count"] for row in answerable)
    hit_count = sum(row["hit_count_at_5"] for row in answerable)
    return {
        "case_count": len(records),
        "answerable_case_count": len(answerable),
        "no_answer_case_count": len(no_answer),
        "gold_fact_count": gold_count,
        "strict_fact_hit_count_at_5": hit_count,
        "strict_fact_recall_at_5": hit_count / gold_count,
        "single_fact_hit_count_at_5": sum(row["all_gold_at_5"] for row in singles),
        "single_fact_case_count": len(singles),
        "single_fact_recall_at_5": sum(row["all_gold_at_5"] for row in singles) / len(singles),
        "complete_pair_count_at_5": sum(row["all_gold_at_5"] for row in pairs),
        "pair_case_count": len(pairs),
        "complete_pair_recall_at_5": sum(row["all_gold_at_5"] for row in pairs) / len(pairs),
        "no_answer_correct_count": sum(row["abstained"] for row in no_answer),
        "no_answer_accuracy": sum(row["abstained"] for row in no_answer) / len(no_answer),
        "records": records,
    }


def run(args: argparse.Namespace) -> int:
    facts = _read_jsonl(args.facts)
    cases = build_cases(facts, split=args.split)
    results = evaluate(cases, facts, split=args.split)
    args.case_output.parent.mkdir(parents=True, exist_ok=True)
    args.case_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cases),
        encoding="utf-8",
    )
    config = {
        "schema": "structured-financial-fact-v2/retrieval-config/v1",
        "split": args.split,
        "top_k": 5,
        "document_match": "exact issuer",
        "metric_match": "exact normalized concept-label token set",
        "period_match": "exact period_end",
        "weights": {"document_and_metric": 9.0, "period": 3.0},
        "alias_table": None,
        "embedding_model": None,
        "reranker": None,
        "parameter_scan_performed": False,
        "retrieval_input": "question_text_only",
        "case_slot_fields_read_by_retriever": False,
    }
    acceptance = {
        "schema": "structured-financial-fact-v2/benchmark-acceptance/v1",
        "split": args.split,
        "fact_corpus_sha256": _sha(args.facts),
        "case_file_sha256": _sha(args.case_output),
        "case_count": results["case_count"],
        "gold_fact_count": results["gold_fact_count"],
        "frozen_72_question_reads": 0,
        "prior_development_query_reads": 0,
        "case_slot_field_reads_by_retriever": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "structured_fact_v2_development_measured" if args.split == "development" else "structured_fact_v2_holdout_measured_once",
    }
    _write(args.out_dir / f"{args.split}-benchmark-contract.json", {"case_count": len(cases), "case_types": {kind: sum(row["case_type"] == kind for row in cases) for kind in ("single_fact", "period_pair", "no_answer")}, "runtime_cases_committed": False})
    _write(args.out_dir / f"{args.split}-retrieval-configuration.json", config)
    _write(args.out_dir / f"{args.split}-retrieval-results.json", results)
    _write(args.out_dir / f"{args.split}-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "holdout"), required=True)
    parser.add_argument("--case-output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
