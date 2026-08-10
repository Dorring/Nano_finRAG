#!/usr/bin/env python3
"""T2-04A.2 deterministic financial-structure signal audit.

Feature extraction is sealed before Train/Dev Gold target identities are read.
Primary Test and ConvFinQA targets are never loaded in this gate.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXPECTED_DATASET_ROWS = 23_088
DATASET_COMMIT = "adf7fe1541ac37351ce1142544d8e3b43010ed92"
PRIMARY_COUNTS = {"train": 15_314, "dev": 2_025, "test": 2_291}
PRIMARY_SUBSETS = ("FinQA", "TAT-DQA")
NATIVE_SPLITS = ("train", "dev", "test")
CANDIDATE_DEPTH = 50
KS = (1, 3, 5, 10, 20, 50)
METRIC_WORDS = {
    "assets", "cash", "cost", "debt", "expense", "flow", "income", "margin",
    "net", "profit", "revenue", "sales", "tax", "volume", "earnings",
    "liabilities", "equity", "transactions", "payments", "operating",
    "ebitda", "interest", "inventory", "capital", "dividend", "shares",
    "value", "values", "loss", "gain", "benefit", "depreciation",
}
OPERATIONAL_FEATURES = (
    "entity_normalized_match",
    "metric_normalized_match",
    "required_period_coverage_full",
    "metric_in_row_label",
    "period_in_table_header",
    "row_header_coherence",
    "operation_evidence_compatibility",
)
FEATURE_NAMES = (
    "entity_exact_match",
    "entity_normalized_match",
    "metric_exact_match",
    "metric_normalized_match",
    "period_any_match",
    "required_period_coverage",
    "contains_table",
    "metric_in_row_label",
    "metric_in_table_header",
    "period_in_table_header",
    "row_header_coherence",
    "multi_period_coverage",
    "operation_evidence_compatibility",
    "structure_match_count",
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                data = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                compressed.write(data)


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_tokens(value: Any) -> tuple[str, ...]:
    tokens = []
    for token in normalize_text(value).split():
        if token in {"the", "of", "for", "a", "an"}:
            continue
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and len(token) > 4:
            token = token[:-1]
        tokens.append(token)
    return tuple(sorted(tokens))


def normalize_entity(value: Any) -> str:
    suffixes = {
        "inc", "incorporated", "corporation", "corp", "company", "co",
        "ltd", "limited", "plc", "llc",
    }
    tokens = list(normalize_tokens(value))
    while tokens and tokens[-1] in suffixes:
        tokens.pop()
    return " ".join(tokens)


def extract_periods(value: Any) -> tuple[list[str], list[str]]:
    text = str(value or "")
    patterns = (
        r"(?i)\bfy\s*(?:19|20)\d{2}\b",
        r"(?i)\bfiscal\s+year\s+(?:19|20)\d{2}\b",
        r"(?i)\byears?\s+ended?(?:\s+\w+){0,5}\s+(?:19|20)\d{2}\b",
        r"(?i)\b(?:19|20)\d{2}\b",
        r"(?i)\bq[1-4]\b",
    )
    raw: list[str] = []
    for pattern in patterns:
        raw.extend(re.findall(pattern, text))
    normalized = []
    for item in raw:
        year = re.search(r"(?:19|20)\d{2}", item)
        if year:
            normalized.append(year.group(0))
        else:
            normalized.append(normalize_text(item).replace(" ", ""))
    unique = sorted(set(normalized), key=lambda item: (item.isdigit() is False, item))
    raw_unique = sorted(set(raw), key=normalize_text)
    return unique, raw_unique


def extract_metric_terms(value: Any) -> tuple[list[str], list[str]]:
    text = normalize_text(value)
    tokens = text.split()
    raw_terms: set[str] = set()
    normalized_terms: set[str] = set()
    for start in range(len(tokens)):
        for width in range(1, 5):
            end = start + width
            if end > len(tokens):
                continue
            phrase_tokens = tokens[start:end]
            if not any(token in METRIC_WORDS for token in phrase_tokens):
                continue
            raw_phrase = " ".join(phrase_tokens)
            raw_terms.add(raw_phrase)
            normalized_terms.add(" ".join(normalize_tokens(raw_phrase)))
    return sorted(raw_terms), sorted(normalized_terms)


def extract_operation(value: Any) -> str:
    text = normalize_text(value)
    if any(term in text for term in ("percentage change", "percent change", "percentage increase", "percent increase", "growth")):
        return "percentage_change"
    if "difference" in text or "net change" in text or "change between" in text:
        return "difference"
    if "average" in text or "per transaction" in text:
        return "average"
    if "ratio" in text or "divided by" in text:
        return "ratio"
    if "percentage" in text or "percent" in text:
        return "percentage"
    if "sum" in text or "combined" in text or "total of" in text:
        return "sum"
    if "compare" in text or "compared" in text or "versus" in text:
        return "comparison"
    if any(term in text for term in ("both", "each", "respectively")):
        return "multi_operand_other"
    return "direct_fact"


def extract_query(question: Any, company_name: Any) -> dict[str, Any]:
    text = str(question or "")
    periods, period_raw = extract_periods(text)
    raw_metrics, normalized_metrics = extract_metric_terms(text)
    entities = set()
    if company_name:
        entities.add(normalize_entity(company_name))
    prefix = text.split(":", 1)[0].strip()
    if prefix:
        entities.add(normalize_entity(prefix))
    entities.discard("")
    operation = extract_operation(text)
    multi_period = len(periods) > 1
    multi_operand = multi_period or operation != "direct_fact"
    return {
        "raw_question": text,
        "entities": sorted(entities),
        "raw_metric_terms": raw_metrics,
        "normalized_metric_terms": normalized_metrics,
        "periods": periods,
        "period_raw_values": period_raw,
        "currency_terms": sorted(set(re.findall(
            r"(?i)\b(?:usd|dollars?|euros?|gbp|eur)\b", text
        ))),
        "operation_intent": operation,
        "requires_multiple_periods": multi_period,
        "requires_multiple_operands": multi_operand,
        "empty_question": text == "",
    }


def metadata_paths(root: Path) -> list[tuple[str, str, Path]]:
    paths = [
        (subset, split, root / "data" / subset / split / "metadata.jsonl")
        for subset in ("FinQA", "TAT-DQA")
        for split in NATIVE_SPLITS
    ]
    paths.append(("ConvFinQA", "all", root / "data" / "ConvFinQA" / "turn_0.jsonl"))
    return paths


def load_query_inputs(root: Path) -> list[dict[str, Any]]:
    """Load only Train/Dev query fields; target context IDs are not accessed."""
    rows = []
    for subset in PRIMARY_SUBSETS:
        for split in ("train", "dev"):
            path = root / "data" / subset / split / "metadata.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    source = json.loads(line)
                    rows.append({
                        "query_id": str(source["id"]),
                        "subset": subset,
                        "split": split,
                        "question": source.get("question"),
                        "company_name": source.get("company_name"),
                    })
    if len(rows) != PRIMARY_COUNTS["train"] + PRIMARY_COUNTS["dev"]:
        raise RuntimeError(f"primary_query_input_count:{len(rows)}")
    if len({row["query_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate_primary_query_ids")
    return sorted(rows, key=lambda row: row["query_id"])


def load_frozen_candidates(
    prediction_root: Path, query_ids: set[str]
) -> dict[str, list[str]]:
    seal = json.loads(
        (prediction_root / "prediction-seal.json").read_text(encoding="utf-8")
    )
    if not seal.get("sealed") or seal.get("gold_scoring_reads_before_seal") != 0:
        raise RuntimeError("invalid_t2_01_seal")
    path = prediction_root / "bm25-predictions.jsonl.gz"
    if sha256_file(path) != seal["output_sha256"]["bm25"]:
        raise RuntimeError("bm25_prediction_mutated")
    candidates: dict[str, list[str]] = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query_id = str(row["query_id"])
            if query_id not in query_ids:
                continue
            ranked = row.get("ranked_contexts") or []
            ids = [str(item["context_id"]) for item in ranked]
            ranks = [int(item["rank"]) for item in ranked]
            if len(ids) < CANDIDATE_DEPTH or len(ids) != len(set(ids)):
                raise RuntimeError(f"candidate_depth_or_duplicate:{query_id}")
            if ranks[:CANDIDATE_DEPTH] != list(range(1, CANDIDATE_DEPTH + 1)):
                raise RuntimeError(f"candidate_rank_contract:{query_id}")
            candidates[query_id] = ids[:CANDIDATE_DEPTH]
    if set(candidates) != query_ids:
        raise RuntimeError("primary_candidate_identity_mismatch")
    return candidates


def load_candidate_contexts(
    root: Path, needed: dict[str, set[str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    """Read candidate corpus text only; never access query target fields."""
    contexts: dict[tuple[str, str], dict[str, Any]] = {}
    for subset, _split, path in metadata_paths(root):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                source = json.loads(line)
                context_id = str(source["context_id"])
                if context_id not in needed.get(subset, set()):
                    continue
                context = str(source.get("context") or "")
                key = (subset, context_id)
                existing = contexts.get(key)
                if existing is not None and existing["context"] != context:
                    raise RuntimeError(f"context_content_conflict:{subset}:{context_id}")
                if existing is None:
                    contexts[key] = {
                        "context_id": context_id,
                        "subset": subset,
                        "file_name": source.get("file_name"),
                        "context": context,
                    }
    missing = [
        (subset, context_id)
        for subset, ids in needed.items()
        for context_id in ids
        if (subset, context_id) not in contexts
    ]
    if missing:
        raise RuntimeError(f"candidate_context_missing:{len(missing)}")
    return contexts


def split_cells(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def separator_line(cells: list[str]) -> bool:
    return bool(cells) and all(
        not re.search(r"[A-Za-z0-9]", cell) and "-" in cell
        for cell in cells
    )


def parse_tables(text: str) -> list[dict[str, Any]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if "|" in line and len(split_cells(line)) >= 2:
            current.append(line)
        elif current:
            blocks.append(current)
            current = []
    if current:
        blocks.append(current)
    tables = []
    for raw_lines in blocks:
        data = [split_cells(line) for line in raw_lines]
        data = [cells for cells in data if not separator_line(cells)]
        if not data:
            continue
        header_cells = data[0] + (data[1] if len(data) > 1 else [])
        headers = [cell for cell in header_cells if cell]
        row_labels = []
        for cells in data[1:]:
            candidates = [
                cell for cell in cells
                if re.search(r"[A-Za-z]", cell)
                and not re.fullmatch(r"(?i)(?:fy\s*)?(?:19|20)\d{2}", cell.strip())
            ]
            if candidates:
                row_labels.append(candidates[0])
        header_periods, _raw = extract_periods(" ".join(headers))
        row_raw_metrics, row_norm_metrics = extract_metric_terms(
            " ".join(row_labels)
        )
        header_raw_metrics, header_norm_metrics = extract_metric_terms(
            " ".join(headers)
        )
        all_text = " ".join(" ".join(cells) for cells in data)
        table_raw_metrics, table_norm_metrics = extract_metric_terms(all_text)
        tables.append({
            "headers": sorted(set(headers), key=normalize_text),
            "row_labels": sorted(set(row_labels), key=normalize_text),
            "periods": header_periods,
            "raw_metric_terms": table_raw_metrics,
            "normalized_metric_terms": table_norm_metrics,
            "row_raw_metric_terms": row_raw_metrics,
            "row_normalized_metric_terms": row_norm_metrics,
            "header_raw_metric_terms": header_raw_metrics,
            "header_normalized_metric_terms": header_norm_metrics,
            "numeric_cell_count": sum(
                bool(re.search(r"\d", cell)) for cells in data for cell in cells
            ),
        })
    return tables


def extract_context(context_id: str, subset: str, text: str, file_name: Any) -> dict[str, Any]:
    tables = parse_tables(text)
    periods, period_raw = extract_periods(text)
    raw_metrics, normalized_metrics = extract_metric_terms(text)
    for table in tables:
        raw_metrics.extend(table["raw_metric_terms"])
        normalized_metrics.extend(table["normalized_metric_terms"])
    return {
        "context_id": context_id,
        "subset": subset,
        "file_name": file_name,
        "char_count": len(text),
        "context_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "normalized_text": normalize_text(text),
        "entities": [],
        "raw_metric_terms": sorted(set(raw_metrics)),
        "normalized_metric_terms": sorted(set(normalized_metrics)),
        "periods": periods,
        "period_raw_values": period_raw,
        "contains_table": bool(tables),
        "table_count": len(tables),
        "table_headers": sorted({
            header for table in tables for header in table["headers"]
        }, key=normalize_text),
        "row_labels": sorted({
            label for table in tables for label in table["row_labels"]
        }, key=normalize_text),
        "numeric_cell_count": sum(
            table["numeric_cell_count"] for table in tables
        ),
        "multi_period_table": any(len(table["periods"]) > 1 for table in tables),
        "tables": tables,
    }


def metric_match(
    query: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, bool]:
    q_raw = set(query["raw_metric_terms"])
    q_norm = set(query["normalized_metric_terms"])
    return (
        bool(q_raw & set(context["raw_metric_terms"])),
        bool(q_norm & set(context["normalized_metric_terms"])),
    )


def table_metric_period_match(
    query: dict[str, Any], context: dict[str, Any]
) -> tuple[bool, bool, float, bool]:
    q_raw = set(query["raw_metric_terms"])
    q_norm = set(query["normalized_metric_terms"])
    required = set(query["periods"])
    row_match = False
    header_match = False
    header_cover = 0.0
    coherent = False
    for table in context["tables"]:
        row_hit = bool(
            (q_raw & set(table["row_raw_metric_terms"]))
            or (q_norm & set(table["row_normalized_metric_terms"]))
        )
        header_hit = bool(
            (q_raw & set(table["header_raw_metric_terms"]))
            or (q_norm & set(table["header_normalized_metric_terms"]))
        )
        table_periods = set(table["periods"])
        cover = (
            len(required & table_periods) / len(required)
            if required else 0.0
        )
        row_match = row_match or row_hit
        header_match = header_match or header_hit
        header_cover = max(header_cover, cover)
        coherent = coherent or (row_hit and cover == 1.0)
    return row_match, header_match, header_cover, coherent


def build_features(
    query: dict[str, Any], context: dict[str, Any]
) -> dict[str, Any]:
    raw_context = context["normalized_text"]
    query_entities = set(query["entities"])
    entity_exact = int(any(
        entity and entity in raw_context for entity in query_entities
    ))
    entity_normalized = int(any(
        entity and all(token in raw_context.split() for token in entity.split())
        for entity in query_entities
    ))
    metric_exact, metric_normalized = metric_match(query, context)
    required = set(query["periods"])
    available = set(context["periods"])
    period_coverage = len(required & available) / len(required) if required else 0.0
    row_match, header_match, header_period_cover, coherent = table_metric_period_match(
        query, context
    )
    operation = query["operation_intent"]
    calculation_like = operation in {
        "difference", "sum", "ratio", "percentage",
        "percentage_change", "average", "comparison", "multi_operand_other",
    }
    if calculation_like:
        operation_compatible = int(
            metric_normalized and period_coverage == 1.0
            and (context["multi_period_table"] or len(required) > 1)
        )
    else:
        operation_compatible = int(metric_normalized or row_match or header_match)
    full_period = int(period_coverage == 1.0 and bool(required))
    multi_period = period_coverage if query["requires_multiple_periods"] else 0.0
    components = {
        "entity_normalized_match": entity_normalized,
        "metric_normalized_match": int(metric_normalized),
        "required_period_coverage_full": full_period,
        "metric_in_row_label": int(row_match),
        "period_in_table_header": header_period_cover,
        "row_header_coherence": int(coherent),
        "operation_evidence_compatibility": operation_compatible,
    }
    structure_count = sum(
        int(value > 0) for value in components.values()
    )
    return {
        "entity_exact_match": entity_exact,
        "entity_normalized_match": entity_normalized,
        "metric_exact_match": int(metric_exact),
        "metric_normalized_match": int(metric_normalized),
        "period_any_match": int(period_coverage > 0),
        "required_period_coverage": period_coverage,
        "contains_table": int(context["contains_table"]),
        "metric_in_row_label": int(row_match),
        "metric_in_table_header": int(header_match),
        "period_in_table_header": header_period_cover,
        "row_header_coherence": int(coherent),
        "multi_period_coverage": multi_period,
        "operation_evidence_compatibility": operation_compatible,
        "structure_match_count": structure_count,
        "period_requirement_present": bool(required),
        "query_operation": operation,
        "components": components,
    }


def score_ordered(
    query_ids: Iterable[str],
    ordered: dict[str, list[str]],
    gold: dict[str, str],
) -> dict[str, Any]:
    stats = {"count": 0, "hits": {str(k): 0 for k in KS}, "mrr": 0.0, "ndcg10": 0.0}
    for query_id in query_ids:
        stats["count"] += 1
        try:
            rank = ordered[query_id].index(gold[query_id]) + 1
        except ValueError:
            rank = None
        for cutoff in KS:
            if rank is not None and rank <= cutoff:
                stats["hits"][str(cutoff)] += 1
        if rank is not None and rank <= 5:
            stats["mrr"] += 1.0 / rank
        if rank is not None and rank <= 10:
            stats["ndcg10"] += 1.0 / math.log2(rank + 1.0)
    total = stats["count"]
    mrr = stats["mrr"] / total if total else 0.0
    ndcg10 = stats["ndcg10"] / total if total else 0.0
    return {
        "count": total,
        "hits": stats["hits"],
        "recall": {
            f"@{k}": f"{stats['hits'][str(k)]}/{total}" for k in KS
        },
        "recall_pct": {
            f"@{k}": round(stats["hits"][str(k)] / total * 100.0, 6)
            for k in KS
        },
        "mrr_at_5": mrr,
        "mrr_at_5_pct": round(mrr * 100.0, 6),
        "ndcg_at_10": ndcg10,
        "ndcg_at_10_pct": round(ndcg10 * 100.0, 6),
    }


def rank_feature_rows(
    feature_rows: list[dict[str, Any]], feature: str
) -> dict[str, list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["query_id"]].append(row)
    return {
        query_id: [
            row["candidate_context_id"]
            for row in sorted(
                rows,
                key=lambda item: (
                    -float(item["features"][feature]),
                    int(item["bm25_rank"]),
                    item["candidate_context_id"],
                ),
            )
        ]
        for query_id, rows in grouped.items()
    }


def bm25_order(feature_rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in feature_rows:
        grouped[row["query_id"]].append(row)
    return {
        query_id: [
            row["candidate_context_id"]
            for row in sorted(
                rows,
                key=lambda item: (int(item["bm25_rank"]), item["candidate_context_id"]),
            )
        ]
        for query_id, rows in grouped.items()
    }


def movement(
    baseline: dict[str, list[str]],
    reranked: dict[str, list[str]],
    gold: dict[str, str],
) -> dict[str, int]:
    rescued = damaged = unchanged = 0
    for query_id, gold_id in gold.items():
        before = gold_id in baseline[query_id][:5]
        after = gold_id in reranked[query_id][:5]
        if not before and after:
            rescued += 1
        elif before and not after:
            damaged += 1
        else:
            unchanged += 1
    return {
        "rescued_at_5": rescued,
        "damaged_at_5": damaged,
        "net_top5_gain": rescued - damaged,
        "unchanged_at_5": unchanged,
    }


def means(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def positive_rate(values: list[float]) -> float | None:
    return (
        sum(value > 0 for value in values) / len(values)
        if values else None
    )


def separability(
    feature_rows: list[dict[str, Any]], gold: dict[str, str]
) -> dict[str, Any]:
    result = {}
    for feature in FEATURE_NAMES:
        gold_values = [
            float(row["features"][feature])
            for row in feature_rows
            if gold.get(row["query_id"]) == row["candidate_context_id"]
        ]
        negative_values = [
            float(row["features"][feature])
            for row in feature_rows
            if gold.get(row["query_id"]) != row["candidate_context_id"]
        ]
        gold_mean = means(gold_values)
        negative_mean = means(negative_values)
        gold_rate = positive_rate(gold_values)
        negative_rate = positive_rate(negative_values)
        result[feature] = {
            "gold_candidate_count": len(gold_values),
            "hard_negative_count": len(negative_values),
            "gold_mean": gold_mean,
            "hard_negative_mean": negative_mean,
            "mean_delta": (
                gold_mean - negative_mean
                if gold_mean is not None and negative_mean is not None
                else None
            ),
            "gold_positive_rate": gold_rate,
            "hard_negative_positive_rate": negative_rate,
            "positive_rate_delta": (
                gold_rate - negative_rate
                if gold_rate is not None and negative_rate is not None
                else None
            ),
        }
    return result


def rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    output = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index + 1
        while end < len(indexed) and indexed[end][1] == indexed[index][1]:
            end += 1
        rank = (index + 1 + end) / 2.0
        for position in range(index, end):
            output[indexed[position][0]] = rank
        index = end
    return output


def spearman(values_x: list[float], values_y: list[float]) -> float | None:
    if len(values_x) < 2 or len(set(values_x)) < 2 or len(set(values_y)) < 2:
        return None
    rx = rank_values(values_x)
    ry = rank_values(values_y)
    mean_x = sum(rx) / len(rx)
    mean_y = sum(ry) / len(ry)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(rx, ry))
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in rx))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ry))
    return numerator / (denominator_x * denominator_y) if denominator_x and denominator_y else None


def rank_correlations(feature_rows: list[dict[str, Any]], gold: dict[str, str]) -> dict[str, Any]:
    gold_rows = defaultdict(list)
    for row in feature_rows:
        if gold.get(row["query_id"]) == row["candidate_context_id"]:
            gold_rows[row["query_id"]].append(row)
    result = {}
    for feature in FEATURE_NAMES:
        values = []
        ranks = []
        for rows in gold_rows.values():
            if not rows:
                continue
            row = rows[0]
            values.append(float(row["features"][feature]))
            ranks.append(-int(row["bm25_rank"]))
        result[feature] = {
            "query_count": len(values),
            "spearman_feature_vs_negative_bm25_rank": spearman(values, ranks),
        }
    return result


def feature_ranking_audit(
    feature_rows: list[dict[str, Any]],
    query_ids: list[str],
    gold: dict[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, list[str]]]]:
    baseline = bm25_order(feature_rows)
    baseline_metrics = score_ordered(query_ids, baseline, gold)
    output = {
        "bm25": baseline_metrics,
        "features": {},
        "candidate_depth": CANDIDATE_DEPTH,
    }
    scoped_gold = {query_id: gold[query_id] for query_id in query_ids}
    orders = {"bm25": baseline}
    for feature in FEATURE_NAMES:
        order = rank_feature_rows(feature_rows, feature)
        metrics = score_ordered(query_ids, order, gold)
        output["features"][feature] = {
            "metrics": metrics,
            "movement": movement(baseline, order, scoped_gold),
            "r_at_50_invariant": metrics["hits"]["50"] == baseline_metrics["hits"]["50"],
        }
        orders[feature] = order
    return output, orders


def subset_audit(
    feature_rows: list[dict[str, Any]],
    query_structures: dict[str, dict[str, Any]],
    gold: dict[str, str],
    ranking_audit: dict[str, Any],
) -> dict[str, Any]:
    output = {}
    for subset in PRIMARY_SUBSETS:
        ids = sorted(
            query_id for query_id, structure in query_structures.items()
            if structure["subset"] == subset and structure["split"] == "dev"
        )
        id_set = set(ids)
        subset_rows = [row for row in feature_rows if row["query_id"] in id_set]
        scoped_gold = {query_id: gold[query_id] for query_id in ids}
        baseline_order = bm25_order(subset_rows)
        baseline = score_ordered(ids, baseline_order, scoped_gold)
        feature_results = {}
        feature_scores = {}
        for feature in FEATURE_NAMES:
            order = rank_feature_rows(subset_rows, feature)
            metrics = score_ordered(ids, order, scoped_gold)
            feature_results[feature] = {
                "metrics": metrics,
                "movement": movement(baseline_order, order, scoped_gold),
                "r_at_50_invariant": (
                    metrics["hits"]["50"] == baseline["hits"]["50"]
                ),
            }
            feature_scores[feature] = metrics["recall_pct"]["@5"]
        best_feature = sorted(
            feature_scores, key=lambda feature: (-feature_scores[feature], feature)
        )[0]
        count_result = feature_results["structure_match_count"]
        output[subset] = {
            "query_count": len(ids),
            "bm25": baseline,
            "best_individual_feature": best_feature,
            "best_individual_feature_r_at_5_pct": feature_scores[best_feature],
            "features": feature_results,
            "structure_match_count": count_result["metrics"],
            "structure_match_count_movement": count_result["movement"],
        }
    return output


def query_type_audit(
    feature_rows: list[dict[str, Any]],
    query_structures: dict[str, dict[str, Any]],
    gold: dict[str, str],
) -> dict[str, Any]:
    output = {}
    for operation in sorted({
        structure["operation_intent"] for structure in query_structures.values()
    }):
        ids = sorted(
            query_id for query_id, structure in query_structures.items()
            if structure["operation_intent"] == operation
        )
        subset_rows = [row for row in feature_rows if row["query_id"] in set(ids)]
        if not subset_rows:
            continue
        baseline = score_ordered(ids, bm25_order(subset_rows), gold)
        output[operation] = {
            "query_count": len(ids),
            "bm25_r_at_5_pct": baseline["recall_pct"]["@5"],
            "metric_signal_availability": sum(
                row["features"]["metric_normalized_match"] > 0
                for row in subset_rows
            ) / len(subset_rows),
            "period_signal_availability": sum(
                row["features"]["period_any_match"] > 0
                for row in subset_rows
            ) / len(subset_rows),
            "row_header_signal_availability": sum(
                row["features"]["row_header_coherence"] > 0
                for row in subset_rows
            ) / len(subset_rows),
            "operation_compatibility_availability": sum(
                row["features"]["operation_evidence_compatibility"] > 0
                for row in subset_rows
            ) / len(subset_rows),
        }
    return output


def failure_cohorts(
    feature_rows: list[dict[str, Any]],
    query_structures: dict[str, dict[str, Any]],
    gold: dict[str, str],
) -> list[dict[str, Any]]:
    baseline = bm25_order(feature_rows)
    count_order = rank_feature_rows(feature_rows, "structure_match_count")
    by_query = defaultdict(list)
    for row in feature_rows:
        by_query[row["query_id"]].append(row)
    output = []
    for query_id in sorted(by_query):
        before = gold[query_id] in baseline[query_id][:5]
        after = gold[query_id] in count_order[query_id][:5]
        if before == after:
            continue
        original_rank = (
            baseline[query_id].index(gold[query_id]) + 1
            if gold[query_id] in baseline[query_id] else None
        )
        new_rank = (
            count_order[query_id].index(gold[query_id]) + 1
            if gold[query_id] in count_order[query_id] else None
        )
        lookup = {row["candidate_context_id"]: row for row in by_query[query_id]}
        top_competing = []
        for candidate_id in count_order[query_id][:5]:
            candidate = lookup[candidate_id]
            top_competing.append({
                "candidate_context_id": candidate_id,
                "bm25_rank": candidate["bm25_rank"],
                "features": candidate["features"],
            })
        output.append({
            "query_id": query_id,
            "subset": query_structures[query_id]["subset"],
            "question": query_structures[query_id]["raw_question"],
            "movement": "rescue" if after else "damage",
            "original_gold_rank": original_rank,
            "new_gold_rank": new_rank,
            "query_structure": query_structures[query_id],
            "gold_structure_features": lookup[gold[query_id]]["features"],
            "top_competing_candidates": top_competing,
        })
    return output[:200]


def coverage_report(
    query_structures: dict[str, dict[str, Any]],
    contexts: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    by_subset: dict[str, dict[str, int]] = {}
    for subset in PRIMARY_SUBSETS:
        query_items = [
            item for item in query_structures.values() if item["subset"] == subset
        ]
        query_total = len(query_items)
        by_subset[subset] = {
            "query_count": query_total,
            "entity_extraction_coverage": sum(bool(item["entities"]) for item in query_items) / query_total,
            "metric_extraction_coverage": sum(bool(item["normalized_metric_terms"]) for item in query_items) / query_total,
            "period_extraction_coverage": sum(bool(item["periods"]) for item in query_items) / query_total,
            "operation_extraction_coverage": sum(bool(item["operation_intent"]) for item in query_items) / query_total,
            "multi_period_detection_coverage": sum(item["requires_multiple_periods"] for item in query_items) / query_total,
            "empty_question_count": sum(item["empty_question"] for item in query_items),
        }
    context_by_subset = {}
    for subset in PRIMARY_SUBSETS:
        items = [item for (item_subset, _), item in contexts.items() if item_subset == subset]
        total = len(items)
        context_by_subset[subset] = {
            "unique_candidate_context_count": total,
            "entity_extraction_coverage": sum(bool(item["entities"]) for item in items) / total if total else 0.0,
            "metric_extraction_coverage": sum(bool(item["normalized_metric_terms"]) for item in items) / total if total else 0.0,
            "period_extraction_coverage": sum(bool(item["periods"]) for item in items) / total if total else 0.0,
            "table_detection_coverage": sum(item["contains_table"] for item in items) / total if total else 0.0,
            "row_label_extraction_coverage": sum(bool(item["row_labels"]) for item in items) / total if total else 0.0,
            "table_header_extraction_coverage": sum(bool(item["table_headers"]) for item in items) / total if total else 0.0,
        }
    return {"query": by_subset, "context": context_by_subset}


def load_train_dev_gold(root: Path, query_ids: set[str]) -> dict[str, str]:
    """This is the first target-identity read, after feature seal."""
    targets = {}
    for subset in PRIMARY_SUBSETS:
        for split in ("train", "dev"):
            path = root / "data" / subset / split / "metadata.jsonl"
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    source = json.loads(line)
                    query_id = str(source["id"])
                    if query_id in query_ids:
                        targets[query_id] = str(source["context_id"])
    if set(targets) != query_ids:
        raise RuntimeError("train_dev_gold_identity_mismatch")
    return targets


def main_protocol(
    dataset_root: Path,
    prediction_root: Path,
    protocol_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    if not protocol_root.exists():
        raise RuntimeError("t2_04a1_protocol_missing")
    protocol = json.loads(
        (protocol_root / "evaluation-protocol.json").read_text(encoding="utf-8")
    )
    if protocol["evaluation_protocol_accepted"] is not True:
        raise RuntimeError("evaluation_protocol_not_accepted")
    if protocol["primary_track"]["test_queries"] != PRIMARY_COUNTS["test"]:
        raise RuntimeError("primary_test_count_contract")
    query_inputs = load_query_inputs(dataset_root)
    query_ids = {row["query_id"] for row in query_inputs}
    query_structures = {
        row["query_id"]: {
            **extract_query(row["question"], row["company_name"]),
            "query_id": row["query_id"],
            "subset": row["subset"],
            "split": row["split"],
        }
        for row in query_inputs
    }
    candidates = load_frozen_candidates(prediction_root, query_ids)
    needed: dict[str, set[str]] = defaultdict(set)
    for row in query_inputs:
        needed[row["subset"]].update(candidates[row["query_id"]])
    contexts = load_candidate_contexts(dataset_root, needed)
    context_structures = {
        key: extract_context(
            context["context_id"],
            context["subset"],
            context["context"],
            context["file_name"],
        )
        for key, context in sorted(contexts.items())
    }
    query_rows = [
        {"query_id": query_id, **query_structures[query_id]}
        for query_id in sorted(query_structures)
    ]
    context_rows = [
        value for _key, value in sorted(context_structures.items())
    ]
    candidate_features = []
    for query_id in sorted(query_structures):
        structure = query_structures[query_id]
        for rank, context_id in enumerate(candidates[query_id], start=1):
            context = context_structures[(structure["subset"], context_id)]
            candidate_features.append({
                "query_id": query_id,
                "subset": structure["subset"],
                "split": structure["split"],
                "candidate_context_id": context_id,
                "bm25_rank": rank,
                "features": build_features(structure, context),
            })
    train_rows = [
        row for row in candidate_features
        if query_structures[row["query_id"]]["split"] == "train"
    ]
    dev_rows = [
        row for row in candidate_features
        if query_structures[row["query_id"]]["split"] == "dev"
    ]
    contract = {
        "gate": "T2-04A.2",
        "candidate_depth": CANDIDATE_DEPTH,
        "candidate_source": "T2-01 frozen BM25 Top50",
        "gold_independent": True,
        "feature_names": list(FEATURE_NAMES),
        "structure_match_count_components": list(OPERATIONAL_FEATURES),
        "structure_match_count_contract": "equal-weight positive component count; no tuning",
        "period_without_requirement": 0.0,
        "tie_break": ["feature_value_desc", "bm25_rank_asc", "context_id_asc"],
        "extractor": "FinancialStructureExtractor deterministic whole-context representation",
        "test_and_conv_locked": True,
    }
    write_json(output_root / "feature-contract.json", contract)
    write_jsonl_gz(output_root / "query-structure.jsonl.gz", query_rows)
    write_jsonl_gz(output_root / "context-structure.jsonl.gz", context_rows)
    write_jsonl_gz(output_root / "train-candidate-features.jsonl.gz", train_rows)
    write_jsonl_gz(output_root / "dev-candidate-features.jsonl.gz", dev_rows)
    feature_files = [
        "feature-contract.json",
        "query-structure.jsonl.gz",
        "context-structure.jsonl.gz",
        "train-candidate-features.jsonl.gz",
        "dev-candidate-features.jsonl.gz",
    ]
    feature_seal = {
        "gate": "T2-04A.2",
        "sealed": True,
        "feature_files": {
            name: sha256_file(output_root / name) for name in feature_files
        },
        "query_count": len(query_structures),
        "candidate_feature_count": len(candidate_features),
        "train_query_count": len({
            row["query_id"] for row in train_rows
        }),
        "dev_query_count": len({
            row["query_id"] for row in dev_rows
        }),
        "candidate_depth": CANDIDATE_DEPTH,
        "gold_reads_before_feature_seal": 0,
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
        "retrieval_rerun": False,
        "model_execution": False,
        "candidate_mutation": 0,
    }
    write_json(output_root / "feature-seal.json", feature_seal)

    # Only after the feature seal is written are Train/Dev targets loaded.
    gold = load_train_dev_gold(dataset_root, query_ids)
    dev_ids = sorted(
        query_id for query_id, structure in query_structures.items()
        if structure["split"] == "dev"
    )
    train_sep = separability(train_rows, gold)
    dev_sep = separability(dev_rows, gold)
    write_json(output_root / "train-feature-separability.json", train_sep)
    write_json(output_root / "dev-feature-separability.json", dev_sep)
    correlations = rank_correlations(dev_rows, gold)
    write_json(output_root / "feature-rank-correlation.json", correlations)
    dev_ranking, dev_orders = feature_ranking_audit(dev_rows, dev_ids, gold)
    write_json(output_root / "single-feature-dev-ranking.json", dev_ranking)
    write_json(
        output_root / "structure-count-dev-ranking.json",
        {
            "diagnostic_only": True,
            "feature": "structure_match_count",
            "metrics": dev_ranking["features"]["structure_match_count"]["metrics"],
            "movement": dev_ranking["features"]["structure_match_count"]["movement"],
            "bm25": dev_ranking["bm25"],
        },
    )
    subset = subset_audit(dev_rows, query_structures, gold, dev_ranking)
    write_json(output_root / "subset-analysis.json", subset)
    dev_query_structures = {
        query_id: query_structures[query_id] for query_id in dev_ids
    }
    query_types = query_type_audit(dev_rows, dev_query_structures, gold)
    write_json(output_root / "query-type-analysis.json", query_types)
    write_json(
        output_root / "rank-movement.json",
        {
            "candidate_depth": CANDIDATE_DEPTH,
            "bm25": dev_ranking["bm25"],
            "structure_match_count": dev_ranking["features"]["structure_match_count"],
            "all_features": {
                feature: payload["movement"]
                for feature, payload in dev_ranking["features"].items()
            },
        },
    )
    cohorts = failure_cohorts(dev_rows, query_structures, gold)
    write_jsonl_gz(output_root / "failure-cohorts.jsonl.gz", cohorts)
    coverage = coverage_report(query_structures, context_structures)
    write_json(output_root / "extraction-coverage.json", coverage)
    non_entity = [
        feature for feature in FEATURE_NAMES
        if feature not in {"entity_exact_match", "entity_normalized_match"}
    ]
    strongest = []
    for feature in non_entity:
        train_value = train_sep[feature]["mean_delta"]
        dev_value = dev_sep[feature]["mean_delta"]
        if train_value is not None and dev_value is not None and train_value > 0 and dev_value > 0:
            strongest.append({
                "feature": feature,
                "train_mean_delta": train_value,
                "dev_mean_delta": dev_value,
                "dev_positive_rate_delta": dev_sep[feature]["positive_rate_delta"],
                "dev_r_at_5_pct": dev_ranking["features"][feature]["metrics"]["recall_pct"]["@5"],
                "dev_net_top5_gain": dev_ranking["features"][feature]["movement"]["net_top5_gain"],
            })
    strongest.sort(
        key=lambda item: (
            -(item["dev_mean_delta"] + item["train_mean_delta"]),
            item["feature"],
        )
    )
    structure_nonentity = [
        item for item in strongest
        if item["feature"] in OPERATIONAL_FEATURES
    ]
    has_separability = bool(structure_nonentity)
    has_positive_movement = any(
        item["dev_net_top5_gain"] > 0 for item in structure_nonentity
    ) or dev_ranking["features"]["structure_match_count"]["movement"]["net_top5_gain"] > 0
    structure_signal_supported = bool(
        has_separability and has_positive_movement and structure_nonentity
    )
    decision = {
        "gate": "T2-04A.2",
        "base_commit": "4f5902b",
        "dataset_commit": DATASET_COMMIT,
        "primary_train_queries": PRIMARY_COUNTS["train"],
        "primary_dev_queries": PRIMARY_COUNTS["dev"],
        "primary_test_queries": PRIMARY_COUNTS["test"],
        "candidate_depth": CANDIDATE_DEPTH,
        "retrieval_rerun": False,
        "model_execution": False,
        "gold_reads_before_feature_seal": 0,
        "primary_test_gold_reads": 0,
        "convfinqa_gold_reads": 0,
        "query_metric_coverage": coverage["query"],
        "query_period_coverage": coverage["query"],
        "context_table_coverage": coverage["context"],
        "row_label_coverage": coverage["context"],
        "table_header_coverage": coverage["context"],
        "strongest_structure_signals": strongest[:10],
        "dev_bm25_recall_at_5": dev_ranking["bm25"]["recall_pct"]["@5"],
        "dev_structure_count_recall_at_5": dev_ranking["features"]["structure_match_count"]["metrics"]["recall_pct"]["@5"],
        "dev_structure_count_gain_pp": (
            dev_ranking["features"]["structure_match_count"]["metrics"]["recall_pct"]["@5"]
            - dev_ranking["bm25"]["recall_pct"]["@5"]
        ),
        "structure_signal_supported": structure_signal_supported,
        "next_gate": (
            "t2_04b_structure_aware_reranker"
            if structure_signal_supported
            else "t2_04_method_reconsideration"
        ),
        "test_structure_scoring": False,
        "convfinqa_structure_scoring": False,
        "feature_selection_on_test": False,
        "feature_weight_search": False,
        "decision_reason": [
            "Features were sealed before Train/Dev Gold target identity reads.",
            "Primary Test and ConvFinQA Gold were not read.",
            "Candidate universe remained exactly frozen BM25 Top50.",
            "Structure support requires non-entity signal separability and positive Dev movement.",
        ],
    }
    write_json(output_root / "decision.json", decision)
    write_json(
        output_root / "input-integrity.json",
        {
            "dataset_commit": DATASET_COMMIT,
            "protocol_sha256": sha256_file(
                protocol_root / "evaluation-protocol.json"
            ),
            "feature_seal_sha256": sha256_file(output_root / "feature-seal.json"),
            "gold_read_phase": "train_dev_after_feature_seal_only",
            "primary_test_gold_reads": 0,
            "convfinqa_gold_reads": 0,
            "retrieval_rerun": False,
            "model_execution": False,
            "candidate_mutation": 0,
        },
    )
    (output_root / "README.md").write_text(
        "# T2-04A.2 Financial Structure Signal Audit\n\n"
        "Deterministic query/context features were generated over the frozen "
        "BM25 Top50 candidate universe. Feature files were sealed before Train/"
        "Dev target context identities were read. Primary Test and ConvFinQA "
        "Gold were locked and not scored. Single-feature and equal-count "
        "rankings are diagnostics only; no model or weight search was run.\n",
        encoding="utf-8",
    )
    return decision


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--protocol-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    decision = main_protocol(
        args.dataset_root.resolve(),
        args.prediction_root.resolve(),
        args.protocol_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps(decision, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
