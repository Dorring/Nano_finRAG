"""Extract native Inline XBRL facts from the frozen Structured Fact V2 corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.evaluation.nf_opt_17 import parse_context_period
from src.evaluation.structured_fact_v2 import (
    NativeFinancialFact,
    fact_identity,
    normalize_concept_label,
    parse_numeric_value,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "artifacts/evaluation/structured-fact-v2-gate-b"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tag_name(tag: object) -> str:
    return str(getattr(tag, "name", "") or "").casefold()


def _child_text(tag: Any, suffix: str) -> str:
    child = tag.find(lambda item: _tag_name(item).endswith(suffix))
    return str(child.get_text(" ", strip=True) if child else "")


def _contexts(soup: Any) -> dict[str, dict[str, str]]:
    contexts = {}
    for context in soup.find_all(lambda item: _tag_name(item) == "xbrli:context"):
        context_id = str(context.get("id") or "")
        values = {
            "start": _child_text(context, "startdate"),
            "end": _child_text(context, "enddate"),
            "instant": _child_text(context, "instant"),
        }
        if context_id and parse_context_period(values):
            contexts[context_id] = values
    return contexts


def _units(soup: Any) -> dict[str, str]:
    units = {}
    for unit in soup.find_all(lambda item: _tag_name(item) == "xbrli:unit"):
        unit_id = str(unit.get("id") or "")
        measure = _child_text(unit, "measure")
        if unit_id and measure:
            units[unit_id] = measure
    return units


def _statement_title(table: Any) -> str | None:
    caption = table.find("caption")
    if caption and caption.get_text(" ", strip=True):
        return " ".join(caption.get_text(" ", strip=True).split())[:240]
    for previous in table.find_all_previous(limit=20):
        text = " ".join(previous.get_text(" ", strip=True).split())
        if text and "statement" in text.casefold() and len(text) <= 240:
            return text
    return None


def extract_document_facts(document: dict[str, Any], *, runtime_dir: Path) -> list[NativeFinancialFact]:
    from bs4 import BeautifulSoup

    path = runtime_dir / str(document["runtime_filename"])
    if _sha(path) != document["content_sha256"]:
        raise ValueError(f"source hash mismatch: {document['primary_document']}")
    soup = BeautifulSoup(path.read_bytes(), "lxml")
    contexts = _contexts(soup)
    units = _units(soup)
    document_id = f"sec:{document['cik']}:{document['accession_number']}"
    facts: list[NativeFinancialFact] = []
    seen: set[str] = set()
    for table in soup.find_all("table"):
        statement = _statement_title(table)
        for tag in table.find_all(lambda item: _tag_name(item) == "ix:nonfraction"):
            concept = str(tag.get("name") or "")
            context_id = str(tag.get("contextref") or "")
            source_fact_id = str(tag.get("id") or "")
            if not concept or not context_id or not source_fact_id or context_id not in contexts:
                continue
            parsed_period = parse_context_period(contexts[context_id])
            if parsed_period is None:
                continue
            period_end, period_kind = parsed_period
            period_start = contexts[context_id].get("start") or None
            unit_ref = str(tag.get("unitref") or "") or None
            unit_measure = units.get(unit_ref or "")
            currency = unit_measure.split(":", 1)[-1] if unit_measure and "iso4217:" in unit_measure.casefold() else None
            normalized_value, scale_power = parse_numeric_value(
                tag.get_text(" ", strip=True),
                sign=str(tag.get("sign") or "") or None,
                scale=str(tag.get("scale") or "") or None,
            )
            identity = fact_identity(
                document_id=document_id,
                concept=concept,
                context_id=context_id,
                unit_ref=unit_ref,
                source_fact_id=source_fact_id,
            )
            if identity in seen:
                continue
            seen.add(identity)
            facts.append(
                NativeFinancialFact(
                    fact_identity=identity,
                    document_id=document_id,
                    issuer=str(document["issuer"]),
                    cik=str(document["cik"]),
                    accession_number=str(document["accession_number"]),
                    concept=concept,
                    label=normalize_concept_label(concept),
                    statement=statement,
                    context_id=context_id,
                    period_start=period_start,
                    period_end=period_end,
                    period_kind=period_kind,
                    unit_ref=unit_ref,
                    unit_measure=unit_measure,
                    currency=currency,
                    scale_power=scale_power,
                    raw_value=tag.get_text(" ", strip=True),
                    normalized_value=normalized_value,
                    source_fact_id=source_fact_id,
                )
            )
    return facts


def run(args: argparse.Namespace) -> int:
    manifest = json.loads(args.corpus_manifest.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if len(documents) != 6:
        raise ValueError("Gate B requires the six frozen documents")
    records = []
    counts: Counter[str] = Counter()
    for document in documents:
        facts = extract_document_facts(document, runtime_dir=args.runtime_dir)
        counts[f"split:{document['split']}"] += len(facts)
        counts[f"document:{document['cik']}"] += len(facts)
        records.extend({**fact.record(), "split": document["split"]} for fact in facts)
    if not records or len({row["fact_identity"] for row in records}) != len(records):
        raise ValueError("fact extraction produced no facts or duplicate identities")
    args.fact_output.parent.mkdir(parents=True, exist_ok=True)
    args.fact_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    field_coverage = {
        "fact_count": len(records),
        "by_split": {key.removeprefix("split:"): value for key, value in sorted(counts.items()) if key.startswith("split:")},
        "by_document": {key.removeprefix("document:"): value for key, value in sorted(counts.items()) if key.startswith("document:")},
        "label_present": sum(bool(row["label"]) for row in records),
        "period_present": sum(bool(row["period_end"]) for row in records),
        "unit_present": sum(bool(row["unit_measure"]) for row in records),
        "currency_present": sum(bool(row["currency"]) for row in records),
        "statement_present": sum(bool(row["statement"]) for row in records),
        "normalized_value_present": sum(row["normalized_value"] is not None for row in records),
    }
    acceptance = {
        "schema": "structured-financial-fact-v2/gate-b/acceptance/v1",
        "corpus_manifest_sha256": _sha(args.corpus_manifest),
        "runtime_fact_corpus_sha256": _sha(args.fact_output),
        "runtime_fact_corpus_committed": False,
        "fact_count": len(records),
        "fact_identity_collision_count": 0,
        "development_holdout_identity_overlap_count": 0,
        "frozen_72_question_reads": 0,
        "gold_field_reads": 0,
        "model_calls": 0,
        "answer_generation_calls": 0,
        "production_index_writes": 0,
        "production_behavior_changed": False,
        "production_switch_allowed": False,
        "decision": "native_ixbrl_fact_extraction_ready",
        "next_gate": "structured_fact_v2_query_and_benchmark_contract",
    }
    _write(args.out_dir / "fact-schema.json", {"schema": "structured-financial-fact-v2/fact/v1", "fields": list(NativeFinancialFact.__dataclass_fields__)})
    _write(args.out_dir / "fact-field-coverage.json", field_coverage)
    _write(args.out_dir / "fact-corpus-manifest.json", {"fact_count": len(records), "runtime_sha256": acceptance["runtime_fact_corpus_sha256"], "runtime_committed": False})
    _write(args.out_dir / "next-gate.json", {"decision": acceptance["decision"], "next_gate": acceptance["next_gate"], "production_switch_allowed": False})
    _write(args.out_dir / "structured-fact-v2-gate-b-acceptance.json", acceptance)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-manifest", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--fact-output", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
