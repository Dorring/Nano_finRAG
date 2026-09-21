"""NF-OPT-17 Gate B: create independent table-backed hard-negative annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_17 import (
    FinancialFact,
    build_hard_negative_annotation,
    metric_label_for_concept,
    normalize_excerpt,
    parse_context_period,
    validate_generated_annotation,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GATE_A_MANIFEST = ROOT / "artifacts/evaluation/nf-opt-17/development-corpus-manifest.json"
DEFAULT_OUT = ROOT / "artifacts/evaluation/nf-opt-17-gate-b"
DEFAULT_RUNTIME_DIR = ROOT / ".runtime" / "nf-opt-17-dev-corpus"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tag_name(tag: object) -> str:
    return str(getattr(tag, "name", "") or "").casefold()


def _named_child_text(tag: Any, suffix: str) -> str:
    child = tag.find(lambda item: _tag_name(item).endswith(suffix))
    return str(child.get_text(" ", strip=True) if child else "")


def _contexts(soup: Any) -> dict[str, dict[str, str]]:
    parsed: dict[str, dict[str, str]] = {}
    for context in soup.find_all(lambda item: _tag_name(item) == "xbrli:context"):
        context_id = str(context.get("id") or "")
        if not context_id:
            continue
        if context.find(lambda item: _tag_name(item).endswith("explicitmember") or _tag_name(item).endswith("typedmember")):
            continue
        values = {
            "start": _named_child_text(context, "startdate"),
            "end": _named_child_text(context, "enddate"),
            "instant": _named_child_text(context, "instant"),
        }
        if parse_context_period(values) is not None:
            parsed[context_id] = values
    return parsed


def _is_ix_fact(tag: object) -> bool:
    return _tag_name(tag) == "ix:nonfraction"


def _extract_document_facts(document: Mapping[str, Any], *, runtime_dir: Path) -> list[FinancialFact]:
    """Extract only table-backed, dimensionality-free US-GAAP facts."""
    from bs4 import BeautifulSoup

    relative_name = str(document["runtime_filename"])
    path = runtime_dir / relative_name
    content = path.read_bytes()
    if _sha(content) != str(document["content_sha256"]):
        raise ValueError(f"runtime source hash mismatch: {relative_name}")
    soup = BeautifulSoup(content, "lxml")
    contexts = _contexts(soup)
    facts: list[FinancialFact] = []
    for table_index, table in enumerate(soup.find_all("table")):
        for row_index, row in enumerate(table.find_all("tr")):
            excerpt = normalize_excerpt(row.get_text(" ", strip=True))
            if not excerpt:
                continue
            for tag in row.find_all(_is_ix_fact):
                if tag.find_parent("table") is not table:
                    continue
                concept = str(tag.get("name") or "")
                metric = metric_label_for_concept(concept)
                context_id = str(tag.get("contextref") or "")
                fact_id = str(tag.get("id") or "")
                if not metric or not fact_id or context_id not in contexts:
                    continue
                parsed_period = parse_context_period(contexts[context_id])
                if parsed_period is None:
                    continue
                period_end, period_kind = parsed_period
                facts.append(
                    FinancialFact(
                        source_cik=str(document["cik"]),
                        accession_number=str(document["accession_number"]),
                        primary_document=str(document["primary_document"]),
                        issuer=str(document["issuer"]),
                        fact_id=fact_id,
                        concept=concept,
                        metric=metric,
                        context_id=context_id,
                        period_end=period_end,
                        period_kind=period_kind,
                        table_index=table_index,
                        row_index=row_index,
                        evidence_excerpt=excerpt,
                    )
                )
    return facts


def _select_annotations(facts: list[FinancialFact], *, target: int) -> list[dict[str, Any]]:
    """Select source-diverse annotations with two structural hard negatives."""
    row_concept: dict[tuple[int, int, str], list[FinancialFact]] = defaultdict(list)
    table_period: dict[tuple[int, str, str], list[FinancialFact]] = defaultdict(list)
    for fact in facts:
        row_concept[(fact.table_index, fact.row_index, fact.concept)].append(fact)
        table_period[(fact.table_index, fact.period_end, fact.period_kind)].append(fact)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in row_concept.values():
        by_period: dict[str, list[FinancialFact]] = defaultdict(list)
        for fact in group:
            by_period[fact.period_end].append(fact)
        if len(by_period) < 2:
            continue
        current_period = max(by_period)
        prior_period = max(period for period in by_period if period != current_period)
        positive = sorted(by_period[current_period], key=lambda fact: fact.fact_id)[0]
        wrong_period = sorted(by_period[prior_period], key=lambda fact: fact.fact_id)[0]
        table_options = table_period[(positive.table_index, positive.period_end, positive.period_kind)]
        wrong_metric_options = [
            fact
            for fact in table_options
            if fact.metric != positive.metric and fact.row_index != positive.row_index
        ]
        if not wrong_metric_options:
            continue
        wrong_metric = sorted(
            wrong_metric_options,
            key=lambda fact: (fact.metric, fact.row_index, fact.fact_id),
        )[0]
        annotation = build_hard_negative_annotation(
            positive=positive,
            wrong_period=wrong_period,
            wrong_metric=wrong_metric,
        )
        validate_generated_annotation(annotation)
        candidates[positive.metric].append(annotation)

    for rows in candidates.values():
        rows.sort(
            key=lambda record: (
                int(record["positive_candidate"]["table_index"]),
                int(record["positive_candidate"]["row_index"]),
                str(record["annotation_id"]),
            )
        )
    selected: list[dict[str, Any]] = []
    seen_questions: set[str] = set()
    offsets = {metric: 0 for metric in candidates}
    while len(selected) < target:
        progressed = False
        for metric in sorted(candidates):
            offset = offsets[metric]
            rows = candidates[metric]
            while offset < len(rows) and rows[offset]["question"] in seen_questions:
                offset += 1
            offsets[metric] = offset
            if offset >= len(rows):
                continue
            row = rows[offset]
            offsets[metric] += 1
            selected.append(row)
            seen_questions.add(str(row["question"]))
            progressed = True
            if len(selected) == target:
                break
        if not progressed:
            break
    if len(selected) != target:
        raise ValueError(f"only {len(selected)} valid annotations; target is {target}")
    return sorted(selected, key=lambda record: str(record["annotation_id"]))


def _record_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    positives = record["positive_candidate"]
    negatives = record["hard_negatives"]
    return {
        "annotation_id": record["annotation_id"],
        "query_sha256": _sha(str(record["question"]).encode("utf-8")),
        "issuer": record["issuer"],
        "source_document": record["source_document"],
        "positive_candidate_key": positives["candidate_key"],
        "positive_candidate_content_sha256": positives["candidate_content_sha256"],
        "negative_candidates": [
            {
                "negative_type": negative["negative_type"],
                "candidate_key": negative["candidate"]["candidate_key"],
                "candidate_content_sha256": negative["candidate"]["candidate_content_sha256"],
            }
            for negative in negatives
        ],
    }


def run(args: argparse.Namespace) -> int:
    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    documents = list(source_manifest["documents"])
    if len(documents) != 4 or not all(document.get("downloaded") for document in documents):
        raise ValueError("Gate B requires exactly four hash-verified Gate A development documents")
    by_document: dict[str, list[dict[str, Any]]] = {}
    for document in documents:
        facts = _extract_document_facts(document, runtime_dir=args.runtime_dir)
        by_document[str(document["cik"])] = _select_annotations(facts, target=args.target_per_document)
    annotations = [record for cik in sorted(by_document) for record in by_document[cik]]
    expected_count = len(documents) * args.target_per_document
    if len(annotations) != expected_count:
        raise ValueError("annotation count does not match frozen per-document target")
    for annotation in annotations:
        validate_generated_annotation(annotation)

    args.annotation_output.parent.mkdir(parents=True, exist_ok=True)
    args.annotation_output.write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in annotations),
        encoding="utf-8",
    )
    runtime_bytes = args.annotation_output.read_bytes()
    negative_type_counts: dict[str, int] = defaultdict(int)
    for annotation in annotations:
        for negative in annotation["hard_negatives"]:
            negative_type_counts[str(negative["negative_type"])] += 1
    manifest_records = [_record_manifest(annotation) for annotation in annotations]
    manifest = {
        "schema": "nf-opt-17/development-hard-negative-manifest/v1",
        "annotation_count": len(annotations),
        "per_document_count": {cik: len(records) for cik, records in sorted(by_document.items())},
        "runtime_annotation_sha256": _sha(runtime_bytes),
        "runtime_annotation_committed": False,
        "records": manifest_records,
    }
    quality = {
        "annotation_count": len(annotations),
        "positive_count": len(annotations),
        "hard_negative_count": sum(negative_type_counts.values()),
        "negative_type_counts": dict(sorted(negative_type_counts.items())),
        "unique_query_count": len({str(annotation["question"]) for annotation in annotations}),
        "unique_positive_candidate_count": len(
            {str(annotation["positive_candidate"]["candidate_key"]) for annotation in annotations}
        ),
        "human_review_status": "not_reviewed",
        "expected_answer_stored": False,
    }
    acceptance = {
        "schema": "nf-opt-17/gate-b/acceptance/v1",
        "source_manifest_sha256": source_manifest["source_manifest_sha256"],
        "development_document_count": len(documents),
        "annotation_count": len(annotations),
        "hard_negative_count": sum(negative_type_counts.values()),
        "frozen_benchmark_question_or_label_reads": 0,
        "expected_answer_stored": False,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "independent_hard_negative_development_set_generated",
        "next_gate": "nf-opt-17-gate-c-independent_annotation_review",
    }
    _write(args.out_dir / "development-annotation-manifest.json", manifest)
    _write(args.out_dir / "hard-negative-quality-report.json", quality)
    _write(args.out_dir / "next-gate.json", {
        "decision": acceptance["decision"],
        "next_gate": acceptance["next_gate"],
        "production_switch_allowed": False,
    })
    _write(args.out_dir / "nf-opt-17-gate-b-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_GATE_A_MANIFEST)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME_DIR)
    parser.add_argument("--annotation-output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-per-document", type=int, default=20)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
