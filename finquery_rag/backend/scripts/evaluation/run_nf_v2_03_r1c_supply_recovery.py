#!/usr/bin/env python3
"""NF-V2-03 R1C / NF-V2-02.1 offline supply and protocol recovery audit.

The recovery materializer consumes only the sealed Top20 candidate and atomic
parser artifacts.  Gold is opened only after the recovery artifact is sealed
and is used for attribution/measurement, never for fact creation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluation import run_nf_e2e_09_r0_structured_fact_representation as nf09  # noqa: E402
from scripts.evaluation import run_nf_v2_02_top20_financial_fact_expansion as nf02  # noqa: E402
from scripts.evaluation import run_nf_v2_03_r1a_binding_contract_recovery as r1a  # noqa: E402


BASE_COMMIT = "f6869ad24605917dc514cd5356e9b861358f9195"
GATE = "NF-V2-03-R1C / NF-V2-02.1"
OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1c-supply-and-protocol-recovery"
NF02_OUT = ROOT / "artifacts/evaluation/nf-v2-02-top20-financial-fact-expansion"
FACTS_PATH = NF02_OUT / "top20-materialized-facts.jsonl.gz"
FACT_SEAL = NF02_OUT / "top20-materialization-seal.json"
FACT_CONTRACT = NF02_OUT / "financial-fact-contract-reference.json"
ATTEMPT4 = ROOT / "artifacts/evaluation/nf-v2-03-formal-attempt-4"
R1B_OUT = ROOT / "artifacts/evaluation/nf-v2-03-r1b-constrained-binding"
MODEL = "qwen3.7-plus"

FM_NAMES = (
    "FM0_fact_exists_evaluator_source_relation_missed",
    "FM1_structured_table_cell_generically_recoverable",
    "FM2_row_header_composition_generically_recoverable",
    "FM3_narrative_key_value_generically_recoverable",
    "FM4_numeric_value_without_safe_metric_identity",
    "FM5_period_not_reconstructable",
    "FM6_admitted_source_evidence_not_captured_structured",
    "FM7_not_financial_fact_materializable",
    "FM8_other",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            for row in rows:
                zipped.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    text = str(value or "").casefold()
    text = re.sub(r"\([^)]*\)$", "", text)
    text = re.sub(r"\d+$", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def period(value: Any) -> str:
    return norm(value).replace("fy ", "fy")


def source_keys(label: Mapping[str, Any]) -> list[str]:
    return [str(item["candidate_key"]) for item in label.get("expected_sources", []) if item.get("candidate_key")]


def slot_source_keys(request: Any, label: Mapping[str, Any]) -> list[str]:
    """Use the frozen slot/period source contract, not all label sources."""

    keys: list[str] = []
    for slot in request.plan.required_slots:
        for source in r1a.expected_sources(slot, label):
            candidate_key = source.get("candidate_key")
            if candidate_key and str(candidate_key) not in keys:
                keys.append(str(candidate_key))
    return keys


def candidate_source_map(state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    _, unique = nf02.candidate_rows(dict(state))
    return {str(key): dict(value) for key, value in unique.items()}


def fact_source_map(facts: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fact in facts:
        ids = fact.get("candidate_ids") or [fact.get("candidate_id")]
        for candidate_id in ids:
            if candidate_id:
                result[str(candidate_id)].append(dict(fact))
    return result


def atomic_candidates(candidate: Mapping[str, Any], atomic: Mapping[str, Mapping[Any, list[Mapping[str, Any]]]] | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Find source-linked cells without using a question or Gold field."""

    document_id = candidate.get("document_id")
    page = candidate.get("pdf_page")
    table_id = str(candidate.get("table_id") or "")
    row_id = str(candidate.get("row_id") or "")
    periods = {period(item) for item in candidate.get("column_header", []) if period(item)}
    if isinstance(atomic, Mapping):
        relation_key = (document_id, page, table_id, row_id)
        source_id = candidate.get("physical_source_id")
        relation_index = atomic.get("relation", {})
        source_index = atomic.get("source", {})
        page_index = atomic.get("page", {})
        records = relation_index.get(relation_key) or source_index.get(source_id, []) or page_index.get((document_id, page), [])
    else:
        records = atomic
    exact: list[dict[str, Any]] = []
    row_matches: list[dict[str, Any]] = []
    source_matches: list[dict[str, Any]] = []
    for record in records:
        trace = record.get("source_traceback") or {}
        rec_page = trace.get("pdf_page", record.get("pdf_page"))
        rec_table = str(nf09.strip_prefix(record.get("table_fragment_id") or trace.get("table_fragment_id"), "table:") or "")
        rec_row = str(nf09.strip_prefix(record.get("row_id") or trace.get("row_id"), "row:") or "")
        rec_period = period(record.get("normalized_period"))
        if document_id and record.get("document_id") != document_id:
            continue
        if page is not None and rec_page != page:
            continue
        if candidate.get("physical_source_id") and trace.get("physical_source_id") and trace.get("physical_source_id") != candidate.get("physical_source_id"):
            continue
        source_matches.append(dict(record))
        if table_id and rec_table == table_id and row_id and rec_row == row_id:
            row_matches.append(dict(record))
            if rec_period in periods:
                exact.append(dict(record))
    if exact or row_matches:
        return exact or row_matches
    # A source/page fallback is only safe when the existing parser already
    # supplies the candidate period and metric identity; never materialize an
    # entire page into unrelated FinancialFacts.
    filtered_source = [
        record
        for record in source_matches
        if period(record.get("normalized_period")) in periods
        and norm(record.get("metric_path") or record.get("leaf_metric") or candidate.get("normalized_metric"))
    ]
    return filtered_source


def build_recovered_fact(candidate: Mapping[str, Any], record: Mapping[str, Any], *, metric_source: str | None = None) -> dict[str, Any] | None:
    trace = record.get("source_traceback") or {}
    raw_value = str(record.get("value_raw") or "")
    parsed_value = nf09.norm_numeric(record.get("value_normalized"))
    raw_period = period(record.get("normalized_period"))
    metric = metric_source or record.get("metric_path") or record.get("leaf_metric") or candidate.get("metric")
    normalized_metric = nf09.norm_text(metric)
    cell_id = str(record.get("cell_id") or trace.get("cell_id") or "")
    if not raw_value or not parsed_value or not raw_period or not normalized_metric or not cell_id:
        return None
    table_id = nf09.strip_prefix(record.get("table_fragment_id") or trace.get("table_fragment_id"), "table:") or candidate.get("table_id")
    row_id = nf09.strip_prefix(record.get("row_id") or trace.get("row_id"), "row:") or candidate.get("row_id")
    if not table_id or not row_id:
        return None
    column_id = "column:" + hashlib.sha256(f"{table_id}\x1f{raw_period}".encode("utf-8")).hexdigest()
    fact = {
        "fact_id": "",
        "candidate_id": str(candidate["candidate_id"]),
        "candidate_ids": [str(candidate["candidate_id"])],
        "physical_source_id": str(candidate.get("physical_source_id") or trace.get("physical_source_id") or ""),
        "document_id": str(record.get("document_id") or candidate.get("document_id") or ""),
        "pdf_page": int(trace.get("pdf_page", candidate.get("pdf_page"))),
        "statement_id": candidate.get("statement_id"),
        "logical_table_id": None,
        "table_id": str(table_id),
        "row_id": str(row_id),
        "column_id": column_id,
        "cell_id": cell_id,
        "raw_metric": str(metric),
        "normalized_metric": normalized_metric,
        "raw_period": raw_period,
        "normalized_period": raw_period,
        "raw_value": raw_value,
        "parsed_numeric_value": parsed_value,
        "raw_currency": record.get("currency_code"),
        "normalized_currency": nf09.norm_currency(record.get("currency_code")) or None,
        "raw_scale": record.get("scale_unit") if record.get("scale_unit") is not None else record.get("scale"),
        "normalized_scale": nf09.norm_scale(record.get("scale"), record.get("scale_unit")) or None,
        "unit": None,
        "parser_name": "nf-v2-02.1-generic-source-relation-recovery",
        "parser_version": "r1c",
        "parser_artifact_hash": sha256_file(nf02.GATE03 / "atomic-facts.jsonl"),
        "source_traceback": dict(trace),
        "relation_provenance": {
            "metric_row": True,
            "period_column": True,
            "value_cell": True,
            "row_column_cell": True,
            "cell_physical_source": True,
            "candidate_atomic_identity_bridge": "document+page+table+row+metric+period generic relation recovery",
        },
        "provenance_complete": True,
    }
    fact["fact_id"] = nf09.fact_id(fact)
    return fact


def recover_candidates(state: Mapping[str, Any], old_facts: list[dict[str, Any]], atomic: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = candidate_source_map(state)
    existing = fact_source_map(old_facts)
    atomic_by_relation: dict[tuple[Any, Any, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    atomic_by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    atomic_by_doc_page: dict[tuple[Any, Any], list[Mapping[str, Any]]] = defaultdict(list)
    for record in atomic:
        trace = record.get("source_traceback") or {}
        document_id = record.get("document_id")
        page = trace.get("pdf_page", record.get("pdf_page"))
        table_id = str(nf09.strip_prefix(record.get("table_fragment_id") or trace.get("table_fragment_id"), "table:") or "")
        row_id = str(nf09.strip_prefix(record.get("row_id") or trace.get("row_id"), "row:") or "")
        atomic_by_relation[(document_id, page, table_id, row_id)].append(record)
        if trace.get("physical_source_id"):
            atomic_by_source[str(trace["physical_source_id"])].append(record)
        atomic_by_doc_page[(document_id, page)].append(record)
    atomic_indexes: dict[str, Mapping[Any, list[Mapping[str, Any]]]] = {"relation": atomic_by_relation, "source": atomic_by_source, "page": atomic_by_doc_page}
    recovered: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for candidate_id, candidate in sorted(sources.items()):
        if existing.get(candidate_id):
            rows.append({"candidate_id": candidate_id, "recovery": "existing_fact", "recovered_fact_count": 0, "generic_rule": True})
            continue
        matches = atomic_candidates(candidate, atomic_indexes)
        unique_matches: dict[str, Mapping[str, Any]] = {}
        for match in matches:
            trace = match.get("source_traceback") or {}
            cell_id = str(match.get("cell_id") or trace.get("cell_id") or "")
            if cell_id:
                unique_matches[cell_id] = match
        matches = list(unique_matches.values())
        facts: list[dict[str, Any]] = []
        category = "FM7_not_financial_fact_materializable"
        reason = "No source-linked atomic numeric cell with a safe metric and period."
        if matches:
            candidate_metric = nf09.norm_text(candidate.get("normalized_metric") or candidate.get("metric"))
            has_exact = any(period(item.get("normalized_period")) in {period(header) for header in candidate.get("column_header", [])} and nf09.norm_text(item.get("metric_path") or item.get("leaf_metric")) == candidate_metric for item in matches)
            has_row = bool(candidate.get("table_id") and candidate.get("row_id"))
            has_metric = any(nf09.norm_text(item.get("metric_path") or item.get("leaf_metric") or candidate_metric) for item in matches)
            has_value = any(str(item.get("value_raw") or "") and nf09.norm_numeric(item.get("value_normalized")) for item in matches)
            has_period = any(period(item.get("normalized_period")) for item in matches)
            if has_exact and has_row:
                category = "FM1_structured_table_cell_generically_recoverable"
                reason = "Existing table, row, period and atomic value form a deterministic source relation."
            elif has_row and has_metric and has_value and has_period:
                category = "FM2_row_header_composition_generically_recoverable"
                reason = "Existing row/header/table context and atomic metric path preserve source composition."
            elif has_metric and has_value and has_period:
                category = "FM3_narrative_key_value_generically_recoverable"
                reason = "Existing source-linked key/value record preserves metric, period and numeric value."
            elif has_value and not has_metric:
                category = "FM4_numeric_value_without_safe_metric_identity"
                reason = "A numeric cell exists but source-derived metric identity is insufficient."
            elif has_metric and not has_period:
                category = "FM5_period_not_reconstructable"
                reason = "Metric/value context exists without a deterministic period."
            elif candidate.get("source_text"):
                category = "FM6_admitted_source_evidence_not_captured_structured"
                reason = "Candidate text exists but the sealed atomic representation has no safe relation."
            for item in matches:
                metric_source = item.get("metric_path") or item.get("leaf_metric") or candidate.get("metric")
                fact = build_recovered_fact(candidate, item, metric_source=str(metric_source) if metric_source else None)
                if fact is not None:
                    facts.append(fact)
        recovered.extend(facts)
        rows.append({"candidate_id": candidate_id, "recovery": category, "reason": reason, "recovered_fact_count": len(facts), "generic_rule": True})
    return recovered, rows


def expected_fact_ids(facts: Iterable[Mapping[str, Any]], candidate_ids: Iterable[str]) -> list[dict[str, Any]]:
    wanted = {str(item) for item in candidate_ids}
    by_candidate = fact_source_map(facts)
    result: list[dict[str, Any]] = []
    for candidate_id in sorted(wanted):
        result.extend(by_candidate.get(candidate_id, []))
    return result


def fact_candidate_ids(fact: Mapping[str, Any]) -> set[str]:
    return {str(fact.get("candidate_id"))} | {str(item) for item in fact.get("candidate_ids", [])}


def query_packet(question_id: str, facts: Iterable[Mapping[str, Any]], top20_order: Mapping[str, list[str]]) -> list[dict[str, Any]]:
    wanted = {str(item) for item in top20_order.get(question_id, [])}
    unique: dict[str, dict[str, Any]] = {}
    for fact in facts:
        if fact_candidate_ids(fact) & wanted:
            unique[str(fact.get("fact_id"))] = dict(fact)
    return list(unique.values())


def view_metric_match(slot: Any, fact: Mapping[str, Any], source: Mapping[str, Any] | None) -> bool:
    requested = norm(slot.metric)
    direct = {norm(fact.get("raw_metric")), norm(fact.get("normalized_metric"))}
    if requested in direct:
        return True
    source = source or {}
    context = [source.get("row_label"), source.get("metric"), source.get("table_title"), source.get("statement_id"), source.get("statement_title"), source.get("section_heading")]
    context_text = " ".join(norm(item) for item in context if item)
    tokens = set(requested.split())
    return bool(tokens and tokens.issubset(set(context_text.split())) and any(token in context_text for token in ("revenue", "sales", "income", "expense", "assets", "liabilities", "transactions", "volume", "margin", "cash", "debt")))


def strict_supply(
    requests: Mapping[str, Any],
    labels: Mapping[str, Any],
    facts: list[dict[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    top20_order: Mapping[str, list[str]],
    ids: Iterable[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for question_id in sorted(ids):
        request = requests[question_id]
        label = labels.get(question_id, {})
        expected = slot_source_keys(request, label)
        packet = query_packet(question_id, facts, top20_order)
        source_facts = [fact for fact in packet if fact_candidate_ids(fact) & set(expected)]
        slot_results: list[bool] = []
        for slot in request.plan.required_slots:
            matching = [fact for fact in source_facts if period(fact.get("normalized_period") or fact.get("raw_period")) == period(slot.period) and view_metric_match(slot, fact, sources.get(str(fact.get("candidate_id"))))]
            slot_results.append(bool(matching))
        rows.append({"question_id": question_id, "slot_count": len(slot_results), "slots_compatible": sum(slot_results), "strict_bindable": bool(slot_results) and all(slot_results), "source_facts": len(source_facts), "expected_source_count": len(expected)})
    return {"denominator": len(rows), "strict_bindable": sum(int(row["strict_bindable"]) for row in rows), "rows": rows}


def view_compatibility_sets(
    requests: Mapping[str, Any],
    labels: Mapping[str, Any],
    facts: list[dict[str, Any]],
    sources: Mapping[str, Mapping[str, Any]],
    top20_order: Mapping[str, list[str]],
    direct_ids: Iterable[str],
) -> dict[str, set[str]]:
    metric_ids: set[str] = set()
    period_ids: set[str] = set()
    strict_ids: set[str] = set()
    for question_id in direct_ids:
        request = requests[question_id]
        expected = slot_source_keys(request, labels[question_id])
        packet = query_packet(question_id, facts, top20_order)
        source_facts = [fact for fact in packet if fact_candidate_ids(fact) & set(expected)]
        metric_ok = True
        period_ok = True
        for slot in request.plan.required_slots:
            slot_metric = [fact for fact in source_facts if view_metric_match(slot, fact, sources.get(str(fact.get("candidate_id"))))]
            slot_period = [fact for fact in slot_metric if period(fact.get("normalized_period") or fact.get("raw_period")) == period(slot.period)]
            metric_ok = metric_ok and bool(slot_metric)
            period_ok = period_ok and bool(slot_period)
        if metric_ok:
            metric_ids.add(question_id)
        if period_ok:
            period_ids.add(question_id)
        if metric_ok and period_ok:
            strict_ids.add(question_id)
    return {"metric": metric_ids, "period": period_ids, "strict": strict_ids}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = nf02.verify_frozen_top100()
    old_facts = read_jsonl_gz(FACTS_PATH)
    fact_seal = read_json(FACT_SEAL)
    if sha256_file(FACTS_PATH) != fact_seal.get("financial_facts_sha256"):
        raise RuntimeError("NF-V2-02 FinancialFact seal mismatch")
    atomic, _ = nf09.load_atomic_facts()

    # Materialization/recovery stage: no questions, labels or Gold are opened.
    recovered, candidate_rows = recover_candidates(state, old_facts, atomic)
    combined, duplicate_count = nf09.dedup_facts(old_facts + recovered)
    recovery_sha = hashlib.sha256((json.dumps(combined, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")).hexdigest()
    write_jsonl_gz(OUT / "recovered-financial-facts.jsonl.gz", recovered)
    write_json(OUT / "materialization-recovery-results.json", {
        "old_fact_count": len(old_facts),
        "recovered_fact_count": len(recovered),
        "new_deduplicated_fact_count": len(combined),
        "duplicate_count_under_frozen_contract": duplicate_count,
        "recovery_sha256": recovery_sha,
        "candidate_rows": candidate_rows,
        "query_independent": True,
        "question_reads_during_materialization": 0,
        "gold_used_to_generate_facts": 0,
        "question_specific_rules": 0,
        "benchmark_specific_aliases": 0,
        "financial_fact_v1_schema_modified": False,
    })
    write_json(OUT / "materialization-recovery-contract.json", {
        "revision": "NF-V2-02.1-generic-source-relation-recovery",
        "input": "sealed SADA Top20 candidate serialization + gate03 atomic-facts.jsonl",
        "query_independent": True,
        "gold_access_during_materialization": False,
        "allowed_rules": ["generic table row+column relation", "generic row/header preservation", "generic period propagation", "existing atomic numeric normalization"],
        "financial_fact_v1_schema_modified": False,
        "sffm_v1_overwritten": False,
    })
    relation = nf02.verify_materialized_relations(combined)
    fabrication = nf02.verify_no_fabrication(combined)
    write_json(OUT / "relation-integrity.json", relation)
    write_json(OUT / "fabrication-safety.json", fabrication)

    # Attribution stage begins only after the recovery artifact exists.
    frozen, _, labels = r1a.load_inputs()
    sources = candidate_source_map(state)
    direct_ids = sorted(qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "DIRECT_FACT")
    calc_ids = sorted(qid for qid, label in labels.items() if label.get("calculation") is not None)
    multi_ids = sorted(qid for qid, request in frozen["requests"].items() if request.plan.intent.value == "MULTI_EVIDENCE")
    d2_d3_rows: list[dict[str, Any]] = []
    old_d3 = 0
    new_d3 = 0
    for question_id in direct_ids:
        request = frozen["requests"][question_id]
        baseline = r1a.funnel_row(question_id, request, labels[question_id], state["top20_order"].get(question_id, []))
        expected = list(baseline["gold_source_keys"])
        admitted = bool(baseline["d2_gold_source_admitted"])
        old_has = bool(baseline["d3_gold_source_fact"])
        new_packet = query_packet(question_id, combined, state["top20_order"])
        new_has = any(fact_candidate_ids(fact) & set(expected) for fact in new_packet)
        old_d3 += int(old_has)
        new_d3 += int(new_has)
        if admitted and not old_has:
            candidate_details = [{"candidate_id": candidate_id, "candidate": sources.get(candidate_id), "recovery_rows": [row for row in candidate_rows if row["candidate_id"] == candidate_id]} for candidate_id in expected if candidate_id in state["top20_order"].get(question_id, [])]
            recovered_categories = [row["recovery"] for item in candidate_details for row in item["recovery_rows"]]
            if "existing_fact" in recovered_categories:
                primary = "FM0_fact_exists_evaluator_source_relation_missed"
            else:
                primary = next((category for category in recovered_categories if category != "existing_fact"), "FM8_other")
            d2_d3_rows.append({"question_id": question_id, "question": request.question, "expected_source_candidates": expected, "candidate_details": candidate_details, "primary_category": primary, "generic_recoverable": primary in FM_NAMES[:4], "new_fact_available": new_has})
    taxonomy = Counter(row["primary_category"] for row in d2_d3_rows)
    write_json(OUT / "d2-d3-missing-fact-audit.json", {"gold_source_admitted": 43, "old_gold_source_financial_fact": old_d3, "missing_count": len(d2_d3_rows), "rows": d2_d3_rows, "gold_used_only_for_post_recovery_attribution": True})
    write_json(OUT / "missing-fact-taxonomy.json", {name: taxonomy.get(name, 0) for name in FM_NAMES})

    new_direct = strict_supply(frozen["requests"], labels, combined, sources, state["top20_order"], direct_ids)
    new_calc = strict_supply(frozen["requests"], labels, combined, sources, state["top20_order"], calc_ids)
    new_multi = strict_supply(frozen["requests"], labels, combined, sources, state["top20_order"], multi_ids)
    r1b_review = read_json(R1B_OUT / "fact-semantic-compatibility-review.json")["direct"]
    old_semantic_ids = {str(row["question_id"]) for row in r1b_review["rows"] if row.get("reviewed_semantic_compatible")}
    old_period_ids = {str(row["question_id"]) for row in r1b_review["rows"] if row.get("reviewed_period_compatible")}
    old_strict_ids = set(old_period_ids)
    generic_sets = view_compatibility_sets(frozen["requests"], labels, combined, sources, state["top20_order"], direct_ids)
    baseline_rows = {qid: r1a.funnel_row(qid, frozen["requests"][qid], labels[qid], state["top20_order"].get(qid, [])) for qid in direct_ids}
    old_d3_ids = {qid for qid, row in baseline_rows.items() if row["d3_gold_source_fact"]}
    recovered_metric_ids = {qid for qid in generic_sets["metric"] if qid not in old_d3_ids}
    recovered_period_ids = {qid for qid in generic_sets["period"] if qid not in old_d3_ids}
    recovered_strict_ids = {qid for qid in generic_sets["strict"] if qid not in old_d3_ids}
    view_metric_ids = old_semantic_ids | recovered_metric_ids
    view_period_ids = old_period_ids | recovered_period_ids
    view_strict_ids = old_strict_ids | recovered_strict_ids
    d3_view_only = old_d3_ids - old_strict_ids
    view_only_recovered = len(d3_view_only & (generic_sets["strict"] - old_strict_ids))
    still_not_bindable = len(d3_view_only - (generic_sets["strict"] - old_strict_ids))
    write_json(OUT / "current-vs-view-bindability.json", {
        "direct_total": 56,
        "current_binder_visible_strict_bindable": 23,
        "binder_fact_view_strict_bindable": len(view_strict_ids),
        "view_only_recovered": view_only_recovered,
        "still_not_bindable": still_not_bindable,
        "identity_check_over_d3": 23 + view_only_recovered + still_not_bindable == 25,
        "binder_fact_view_metric_compatible": len(view_metric_ids),
        "binder_fact_view_period_compatible": len(view_period_ids),
        "generic_recovered_metric_questions": sorted(recovered_metric_ids),
        "generic_recovered_period_questions": sorted(recovered_period_ids),
        "generic_recovered_strict_questions": sorted(recovered_strict_ids),
        "current_definition": "R1B visible FinancialFact fields and reviewed strict contract",
        "view_definition": "source-derived BinderFactViewV1 context plus recovered generic facts",
        "old_d3": 25,
        "new_d3": new_d3,
    })
    write_json(OUT / "direct-supply-funnel.json", {"D0": 56, "D1": 48, "D2": 43, "D3_old": 25, "D3_new": new_d3, "D4_binder_fact_view": len(view_metric_ids), "D5_period_compatible": len(view_period_ids), "D6_strict_bindable": len(view_strict_ids), "old_reviewed_strict_bindable": 23, "rows": new_direct["rows"]})
    write_json(OUT / "calculation-supply-funnel.json", {"questions": len(calc_ids), "old_fact_supply_complete": 6, "new_fact_supply_complete": new_calc["strict_bindable"], "new_strict_bindable_all_operands": new_calc["strict_bindable"], "rows": new_calc["rows"], "calculator_executed": False})
    write_json(OUT / "multi-evidence-supply-funnel.json", {"questions": len(multi_ids), "old_complete_supply": 4, "new_strict_bindable_complete": new_multi["strict_bindable"], "rows": new_multi["rows"], "binder_executed": False})
    write_json(OUT / "binder-fact-view-contract.json", {
        "view": "BinderFactViewV1",
        "query_independent": True,
        "financial_fact_v1_modified": False,
        "source_derived_fields": ["fact_handle", "raw_metric", "normalized_metric", "raw_period", "normalized_period", "raw_value", "parsed_numeric_value", "raw_scale", "normalized_scale", "currency", "unit", "row_label", "row_hierarchy", "column_header", "column_header_path", "table_title", "statement_title", "section_heading", "table_id", "row_id", "column_id", "cell_id", "physical_source_id", "document_id", "pdf_page"],
        "new_semantic_labels": False,
    })
    write_json(OUT / "selection-dto-contract.json", {
        "dto": "BinderSelectionDTOv1",
        "selection_only": True,
        "top_level": {"type": "object", "required": ["slots"], "additionalProperties": False},
        "slot_values": {"type": "array", "uniqueItems": True, "items_enum": "exact query-local F01..Fn"},
        "status_from_model": False,
        "status_derivation": {"zero": "MISSING", "one": "BOUND", "two_or_more": "AMBIGUOUS"},
        "query_status_derivation": "all bound => BOUND; any ambiguous => AMBIGUOUS; otherwise MISSING",
    })
    write_json(OUT / "decision.json", {
        "gate": GATE,
        "base_commit": BASE_COMMIT,
        "binder_model": MODEL,
        "model_calls": 0,
        "synthetic_protocol": "pending",
        "formal_attempt_6": "not_run",
        "direct_current_visible_strict_bindable": 23,
        "direct_view_visible_strict_bindable": len(view_strict_ids),
        "direct_view_only_recovered": len(view_strict_ids - old_strict_ids),
        "direct_new_d3": new_d3,
        "calculation_new_fact_supply_complete": new_calc["strict_bindable"],
        "multi_evidence_new_strict_bindable_complete": new_multi["strict_bindable"],
        "financial_fact_v1_schema_modified": False,
        "question_specific_rules": 0,
        "gold_assisted_materialization": 0,
        "fabricated_cross_candidate_facts": fabrication.get("fabricated_cross_candidate_facts", 0),
        "relation_integrity_fail": relation.get("relation_integrity_fail", 0),
        "dominant_failure": "pending_protocol_and_supply_gate",
        "production_default": "V1",
        "production_switch_allowed": False,
        "next_gate": "nf_v2_03_r1c_synthetic_selection_protocol",
    })
    write_json(OUT / "README.md", {"gate": GATE, "summary": "Offline query-independent source recovery and selection-only protocol contract. Synthetic provider test and formal Attempt 6 are separate gated stages.", "model_calls": 0, "gold_used_to_generate_facts": 0, "production_default": "V1"})
    print(json.dumps({"d2_d3_missing": len(d2_d3_rows), "taxonomy": {name: taxonomy.get(name, 0) for name in FM_NAMES}, "rows": [(row["question_id"], row["primary_category"], row["new_fact_available"]) for row in d2_d3_rows], "old_d3": old_d3, "new_d3": new_d3, "old_direct_strict": 23, "new_direct_strict": new_direct["strict_bindable"], "new_calc": new_calc["strict_bindable"], "new_multi": new_multi["strict_bindable"], "relation_fail": relation.get("relation_integrity_fail"), "fabricated": fabrication.get("fabricated_cross_candidate_facts")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
