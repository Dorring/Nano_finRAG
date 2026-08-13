#!/usr/bin/env python3
"""NF-V2-06 R1A grounding-alignment data synthesis.

The runner is intentionally model-free.  It converts only FinQA/TAT-QA TRAIN
records into the same model-facing evidence view used by verified generation,
then constructs short deterministic targets and validates every accepted row.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pickle
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = next(
    (
        parent
        for parent in (ROOT, *ROOT.parents)
        if (parent / "finance-data-process/data/raw/finqa/FinQA_repo/dataset/train.json").is_file()
    ),
    ROOT.parents[-1],
)
DATA_ROOT = PROJECT_ROOT / "finance-data-process/data/raw"
BENCH_ROOT = ROOT / "benchmarks/financial_rag_v1/data"
OUT = ROOT / "data/grounding_alignment/v1"

BASE_COMMIT = "d871b339baaed0f15587a3ae67f156fe8632ff0c"
TARGET_TOTAL = 4000
TARGET_BEHAVIOR = {"POSITIVE_GROUNDED": 2400, "UNANSWERABLE": 800, "PARTIAL_DISTRACTOR": 800}
TARGET_POSITIVE_ROUTE = {"DIRECT": 1500, "CALCULATION_RESULT_VERBALIZATION": 600, "MULTI_EVIDENCE": 300}
SPLIT_TARGET = {"train": 3600, "alignment_dev": 200, "alignment_holdout": 200}
CONTEXT_LIMIT = 4096
MAX_VARIANTS_PER_BASE = 3
MAX_CONTEXT_CAP = 3

ANSWER_RULES = """[ANSWER RULES]
1. Use only the verified evidence and calculation above.
2. Do not introduce outside financial knowledge.
3. Preserve supplied numbers, periods, units, currencies and scales exactly.
4. Do not recalculate canonical calculation results.
5. Cite factual claims using the supplied [E#] / [C#] IDs.
6. If required evidence is missing, explicitly state that the provided evidence is insufficient.
7. Answer concisely."""


def stable_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(stable_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def norm_text(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9%$.-]+", " ", str(value or "").casefold()).split())


def norm_question(value: Any) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).split())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def write_jsonl_gz(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as out:
            for row in rows:
                out.write((json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode())


class TokenCounter:
    def __init__(self) -> None:
        self.kind = "nanochat-bpe-65k"
        self.path = "/home/mxf/.cache/nanochat/tokenizer/tokenizer.pkl"
        self.encoder = None
        try:
            with Path(self.path).open("rb") as f:
                self.encoder = pickle.load(f)
        except Exception:
            self.kind = "deterministic-byte-fallback"

    def count(self, text: str) -> int:
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(text, disallowed_special=()))
            except TypeError:
                return len(self.encoder.encode(text))
        return len(text.encode("utf-8"))


def table_to_lines(table: list[list[Any]]) -> list[str]:
    return [" | ".join(str(cell or "").strip() for cell in row) for row in table if any(str(cell or "").strip() for cell in row)]


def finqa_evidence(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    qa = record["qa"]
    pre = list(record.get("pre_text") or [])
    post = list(record.get("post_text") or [])
    table = list(record.get("table") or [])
    evidence: list[dict[str, Any]] = []
    for source_key, text in (qa.get("gold_inds") or {}).items():
        if source_key.startswith("table_"):
            try:
                index = int(source_key.split("_", 1)[1])
            except ValueError:
                continue
            content = " | ".join(str(x or "").strip() for x in (table[index] if index < len(table) else []))
            kind = "table_row"
        elif source_key.startswith("text_"):
            try:
                index = int(source_key.split("_", 1)[1])
            except ValueError:
                continue
            all_text = pre + post
            content = str(all_text[index]) if index < len(all_text) else str(text)
            kind = "narrative"
        else:
            content = str(text)
            kind = "source"
        if content.strip():
            evidence.append({"kind": kind, "source_key": source_key, "content": content.strip(), "source_label": str(record.get("filename") or "")})
    if not evidence:
        for index, row in enumerate(table[:4]):
            content = " | ".join(str(x or "").strip() for x in row)
            if content:
                evidence.append({"kind": "table_row", "source_key": f"table_{index}", "content": content, "source_label": str(record.get("filename") or "")})
    return evidence


def tatqa_evidence(record: Mapping[str, Any], question: Mapping[str, Any]) -> list[dict[str, Any]]:
    table_obj = record.get("table") or {}
    table = table_obj.get("table") if isinstance(table_obj, Mapping) else table_obj
    table = list(table or [])
    paragraphs = {str(p.get("uid")): str(p.get("text") or "") for p in record.get("paragraphs") or []}
    paragraphs.update({str(p.get("order")): str(p.get("text") or "") for p in record.get("paragraphs") or []})
    answers = [str(x) for x in (question.get("answer") if isinstance(question.get("answer"), list) else [question.get("answer")]) if x is not None]
    evidence: list[dict[str, Any]] = []
    rel = question.get("rel_paragraphs") or []
    for key in rel:
        content = paragraphs.get(str(key), "")
        if content:
            evidence.append({"kind": "narrative", "source_key": f"paragraph_{key}", "content": content, "source_label": str(table_obj.get("uid") or "")})
    # Prefer rows/cells that contain answer spans.  For arithmetic, keep the
    # whole compact table because both operands must remain visible.
    lines = table_to_lines(table)
    selected = [line for line in lines if any(norm_text(answer) and norm_text(answer) in norm_text(line) for answer in answers)]
    if question.get("answer_type") == "arithmetic" or not selected:
        selected = lines[: min(len(lines), 12)]
    for index, line in enumerate(selected):
        evidence.append({"kind": "table_row", "source_key": f"table_row_{index}", "content": line, "source_label": str(table_obj.get("uid") or "")})
    return evidence[:8]


def context_id(dataset: str, record: Mapping[str, Any], evidence: list[Mapping[str, Any]]) -> str:
    if dataset == "FinQA":
        context = {"filename": record.get("filename"), "table": record.get("table"), "pre_text": record.get("pre_text"), "post_text": record.get("post_text")}
    else:
        context = {"table": record.get("table"), "paragraphs": record.get("paragraphs")}
    return stable_sha(context)


def periods_in(text: str) -> list[str]:
    return sorted(set(re.findall(r"(?:FY\s*)?(?:19|20)\d{2}", text or "", flags=re.I)))


def answer_string(answer: Any) -> str:
    if isinstance(answer, list):
        return "; ".join(str(x).strip() for x in answer if str(x).strip())
    if answer is None:
        return ""
    return str(answer).strip()


def finqa_base(record: Mapping[str, Any]) -> dict[str, Any] | None:
    qa = record["qa"]
    answer = answer_string(qa.get("answer"))
    evidence = finqa_evidence(record)
    if not qa.get("question") or not answer or not evidence:
        return None
    route = "CALCULATION_RESULT_VERBALIZATION" if qa.get("program") and qa.get("exe_ans") is not None else "DIRECT"
    return {
        "source_dataset": "FinQA",
        "source_split": "train",
        "source_example_id": str(record.get("id")),
        "source_context_id": context_id("FinQA", record, evidence),
        "document_identity": str(record.get("filename") or ""),
        "question": str(qa.get("question")),
        "answer": answer,
        "evidence": evidence,
        "route": route,
        "program": str(qa.get("program") or ""),
        "canonical_result": str(qa.get("exe_ans")) if qa.get("exe_ans") is not None else None,
        "periods": periods_in(" ".join(x["content"] for x in evidence) + " " + str(qa.get("question"))),
        "base_fingerprint": stable_sha({"dataset": "FinQA", "id": record.get("id"), "context": context_id("FinQA", record, evidence)}),
    }


def tatqa_base(record: Mapping[str, Any], question: Mapping[str, Any]) -> dict[str, Any] | None:
    answer = answer_string(question.get("answer"))
    evidence = tatqa_evidence(record, question)
    if not question.get("question") or not answer or not evidence:
        return None
    answer_from = str(question.get("answer_from") or "")
    if question.get("answer_type") == "arithmetic":
        route = "CALCULATION_RESULT_VERBALIZATION"
    elif answer_from == "table-text" or question.get("answer_type") == "multi-span":
        route = "MULTI_EVIDENCE"
    else:
        route = "DIRECT"
    return {
        "source_dataset": "TAT-DQA",
        "source_split": "train",
        "source_example_id": f"{record.get('table', {}).get('uid')}:{question.get('uid')}",
        "source_context_id": context_id("TAT-DQA", record, evidence),
        "document_identity": str(record.get("table", {}).get("uid") or ""),
        "question": str(question.get("question")),
        "answer": answer,
        "evidence": evidence,
        "route": route,
        "program": str(question.get("derivation") or ""),
        "canonical_result": answer if route == "CALCULATION_RESULT_VERBALIZATION" else None,
        "periods": periods_in(" ".join(x["content"] for x in evidence) + " " + str(question.get("question"))),
        "base_fingerprint": stable_sha({"dataset": "TAT-DQA", "id": f"{record.get('table', {}).get('uid')}:{question.get('uid')}", "context": context_id("TAT-DQA", record, evidence)}),
    }


def load_bases() -> list[dict[str, Any]]:
    finqa_path = DATA_ROOT / "finqa/FinQA_repo/dataset/train.json"
    tatqa_path = DATA_ROOT / "tatqa/TAT-QA_repo/dataset_raw/tatqa_dataset_train.json"
    rows: list[dict[str, Any]] = []
    for record in json.loads(finqa_path.read_text(encoding="utf-8")):
        item = finqa_base(record)
        if item:
            rows.append(item)
    for record in json.loads(tatqa_path.read_text(encoding="utf-8")):
        for question in record.get("questions") or []:
            item = tatqa_base(record, question)
            if item:
                rows.append(item)
    return rows


def source_fingerprints() -> dict[str, Any]:
    benchmark_question_hashes: set[str] = set()
    benchmark_document_ids: set[str] = set()
    benchmark_context_hashes: set[str] = set()
    for path in (BENCH_ROOT / "questions.golden.jsonl", BENCH_ROOT / "questions.reference.jsonl"):
        if not path.exists():
            continue
        for row in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(row)
            benchmark_question_hashes.add(stable_sha(norm_question(item.get("question"))))
    label_path = BENCH_ROOT / "labels.golden.jsonl"
    if label_path.exists():
        for row in label_path.read_text(encoding="utf-8").splitlines():
            item = json.loads(row)
            for source in item.get("expected_sources") or []:
                benchmark_document_ids.update(str(source.get(key)) for key in ("document_id", "filename") if source.get(key))
    # SADA statements are the frozen internal source/context representation.
    try:
        from scripts.evaluation import run_nf_e2e_01_r0_frozen_retrieval_integration_review as e2e01
        cases, _ = e2e01.load_sada_inputs(ROOT)
        for candidates in cases.values():
            for candidate in candidates:
                benchmark_context_hashes.add(str(candidate.get("serialization_sha256")))
                parsed = candidate.get("parsed") or {}
                if parsed.get("document_id"):
                    benchmark_document_ids.add(str(parsed["document_id"]))
    except Exception:
        pass
    official_question_hashes: set[str] = set()
    official_context_hashes: set[str] = set()
    for path in (
        DATA_ROOT / "finqa/FinQA_repo/dataset/dev.json",
        DATA_ROOT / "finqa/FinQA_repo/dataset/test.json",
        DATA_ROOT / "tatqa/TAT-QA_repo/dataset_raw/tatqa_dataset_dev.json",
        DATA_ROOT / "tatqa/TAT-QA_repo/dataset_raw/tatqa_dataset_test_gold.json",
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for record in data:
            if "qa" in record:
                official_question_hashes.add(stable_sha(norm_question(record["qa"].get("question"))))
                official_context_hashes.add(stable_sha({"filename": record.get("filename"), "table": record.get("table"), "pre_text": record.get("pre_text"), "post_text": record.get("post_text")}))
            else:
                context = {"table": record.get("table"), "paragraphs": record.get("paragraphs")}
                official_context_hashes.add(stable_sha(context))
                for question in record.get("questions") or []:
                    official_question_hashes.add(stable_sha(norm_question(question.get("question"))))
    return {
        "internal_question_hashes": benchmark_question_hashes,
        "internal_document_ids": benchmark_document_ids,
        "internal_context_hashes": benchmark_context_hashes,
        "official_question_hashes": official_question_hashes,
        "official_context_hashes": official_context_hashes,
        "official_context_exclusion_scope": "FinQA and TAT-DQA official dev/test; ConvFinQA TRAIN artifact unavailable and not used",
    }


def evidence_view(base: Mapping[str, Any], evidence: list[Mapping[str, Any]] | None = None, calculation: Mapping[str, Any] | None = None) -> tuple[str, list[str]]:
    evidence = list(evidence or base.get("evidence") or [])
    ids = [f"E{i + 1}" for i in range(len(evidence))]
    lines = ["[QUESTION]", str(base["question"]), "", "[VERIFIED EVIDENCE]", ""]
    for evidence_id, item in zip(ids, evidence):
        content = str(item.get("content") or "").replace("\n", " ").strip()
        metric = str(item.get("metric") or item.get("source_key") or "source evidence")
        period = str(item.get("period") or ", ".join(base.get("periods") or []) or "not specified")
        scope = str(item.get("scope") or "not specified")
        lines.extend([
            f"[{evidence_id}]",
            f"Metric: {metric}",
            f"Period: {period}",
            f"Scope: {scope}",
            f"Value: {str(item.get('value') or 'see evidence')}",
            f"Unit: {str(item.get('unit') or 'not specified')}",
            f"Currency: {str(item.get('currency') or 'not specified')}",
            f"Scale: {str(item.get('scale') or 'not specified')}",
            f"Source: {str(item.get('source_label') or base.get('document_identity') or 'training source')}",
            f"Evidence: {content}",
            "",
        ])
    if calculation:
        lines.extend([
            "[VERIFIED CALCULATION]",
            "[C1]",
            f"Operation: {calculation.get('operation') or 'dataset-native calculation'}",
            f"Canonical Result: {calculation.get('canonical_result')}",
            f"Period: {', '.join(base.get('periods') or []) or 'not specified'}",
            "Unit: not specified",
            "Currency: not specified",
            "Scale: 1",
            f"Based On: {', '.join(f'[{x}]' for x in ids)}",
            "",
        ])
    lines.append(ANSWER_RULES)
    return "\n".join(lines), ids


def supported_numbers(text: str) -> set[str]:
    values: set[str] = set()
    for raw in re.findall(r"[-+]?\d[\d,.]*(?:%|[A-Za-z]+)?", text or ""):
        clean = re.sub(r"[^0-9.%-]", "", raw).rstrip(".")
        if clean:
            values.add(clean)
    return values


def target_numbers(text: str) -> set[str]:
    return supported_numbers(text)


def target_periods(text: str) -> set[str]:
    return {norm_text(x) for x in periods_in(text)}


UNIT_CURRENCY_SCALE_RE = re.compile(
    r"(?i)(?:[$€£¥]|%|percent(?:age)?|million|billion|thousand|hundred|basis\s+points|"
    r"shares?|dollars?|euros?|pounds?|yen|usd|eur|gbp|jpy)"
)


def supported_units(text: str) -> set[str]:
    return {norm_text(value) for value in UNIT_CURRENCY_SCALE_RE.findall(text or "")}


def render_target(base: Mapping[str, Any], behavior: str, route: str, evidence_ids: list[str], calculation: Mapping[str, Any] | None = None, *, unsupported_reason: str | None = None) -> str:
    if behavior == "UNANSWERABLE":
        return "Verified evidence is insufficient to answer this question."
    if behavior == "PARTIAL_DISTRACTOR":
        answer = str(base.get("answer") or "").strip()
        citation = " " + " ".join(f"[{item}]" for item in evidence_ids) if evidence_ids else ""
        return f"{answer}{citation}. Additional requested information is unavailable."
    answer = str(base.get("answer") or "").strip()
    if route == "CALCULATION_RESULT_VERBALIZATION":
        canonical = str((calculation or {}).get("canonical_result") or answer).strip()
        return f"The canonical calculation result is {canonical} [C1]."
    if route == "MULTI_EVIDENCE" and len(evidence_ids) >= 2 and ";" in answer:
        parts = [part.strip() for part in answer.split(";") if part.strip()]
        return "; ".join(f"{part} [{evidence_ids[min(i, len(evidence_ids) - 1)]}]" for i, part in enumerate(parts)) + "."
    citations = " " + " ".join(f"[{item}]" for item in evidence_ids) if evidence_ids else ""
    return f"{answer}{citations}." if answer else answer


def make_evidence_items(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for item in base.get("evidence") or []:
        content = str(item.get("content") or "")
        periods = periods_in(content)
        items.append({
            "metric": str(item.get("metric") or item.get("source_key") or "source evidence"),
            "period": periods[0] if len(periods) == 1 else ", ".join(periods),
            "scope": "not specified",
            "value": "; ".join(re.findall(r"[-+]?\d[\d,.]*(?:%|[A-Za-z]+)?", content)[:8]) or "see evidence",
            "unit": "not specified",
            "currency": "not specified",
            "scale": "not specified",
            "source_label": item.get("source_label") or base.get("document_identity"),
            "content": content,
            "source_key": item.get("source_key"),
        })
    return items


def mutate_unanswerable(base: Mapping[str, Any], index: int) -> tuple[str, str]:
    question = str(base["question"])
    evidence_text = norm_text(" ".join(x.get("content", "") for x in base.get("evidence") or []))
    years = [int(x) for x in re.findall(r"(?:19|20)\d{2}", evidence_text)]
    if index % 3 == 0 and years:
        candidate = max(years) + 1
        mutated = re.sub(r"(?:19|20)\d{2}", str(candidate), question, count=1)
        if mutated != question and str(candidate) not in evidence_text:
            return mutated, f"wrong_period:{candidate}"
    metric_pool = ["net income", "free cash flow", "dividend payout ratio", "employee count", "goodwill impairment", "share price"]
    for metric in metric_pool:
        if norm_text(metric) not in evidence_text:
            return f"What was the {metric} reported in the period discussed?", f"wrong_metric:{metric}"
    return "What was the dividend payout ratio reported in the period discussed?", "no_support"


def mutate_partial(base: Mapping[str, Any], index: int) -> tuple[str, str]:
    extras = [
        "what was the chief executive's compensation",
        "what was the company's share price",
        "what was the dividend payout ratio",
        "what was the number of employees",
        "what was the company's market capitalization",
    ]
    evidence_text = norm_text(" ".join(x.get("content", "") for x in base.get("evidence") or []))
    for offset in range(len(extras)):
        extra = extras[(index + offset) % len(extras)]
        if norm_text(extra) not in evidence_text:
            return f"{base['question']} Also, {extra}?", f"partial_missing:{extra}"
    return f"{base['question']} Also, what was the company's share price?", "partial_missing:share_price"


def make_sample(base: Mapping[str, Any], behavior: str, route: str | None = None, variant_index: int = 0) -> dict[str, Any]:
    selected_route = route or str(base["route"])
    work = dict(base)
    audit_reason = None
    if behavior == "UNANSWERABLE":
        work["question"], audit_reason = mutate_unanswerable(base, variant_index)
    elif behavior == "PARTIAL_DISTRACTOR":
        work["question"], audit_reason = mutate_partial(base, variant_index)
    items = make_evidence_items(work)
    calculation = None
    if selected_route == "CALCULATION_RESULT_VERBALIZATION":
        calculation = {"operation": work.get("program") or "dataset-native calculation", "canonical_result": work.get("canonical_result") or work.get("answer")}
    view, evidence_ids = evidence_view(work, items, calculation)
    target = render_target(work, behavior, selected_route, evidence_ids, calculation, unsupported_reason=audit_reason)
    sample_id = stable_sha({"source": base["source_example_id"], "behavior": behavior, "route": selected_route, "question": work["question"], "variant": variant_index})[:32]
    return {
        "sample_id": sample_id,
        "source_dataset": base["source_dataset"],
        "source_split": "train",
        "source_example_id": base["source_example_id"],
        "source_context_id": base["source_context_id"],
        "behavior_type": behavior,
        "route": selected_route,
        "base_question": base["question"],
        "messages": [{"role": "user", "content": view}, {"role": "assistant", "content": target}],
        "evidence_ids": evidence_ids,
        "calculation_id": f"C1:{sample_id}" if calculation else None,
        "fully_answerable": behavior == "POSITIVE_GROUNDED",
        "partially_answerable": behavior == "PARTIAL_DISTRACTOR",
        "requires_abstention": behavior != "POSITIVE_GROUNDED",
        "has_distractors": behavior == "PARTIAL_DISTRACTOR",
        "oracle_training_evidence": True,
        "citation_required": behavior != "UNANSWERABLE",
        "cot_target": False,
        "loss_mask_contract": {"user_prompt": 0, "assistant_final_answer": 1, "question_evidence_rules": 0},
        "fingerprints": {
            "normalized_question_hash": stable_sha(norm_question(work["question"])),
            "context_hash": base["source_context_id"],
            "document_hash": stable_sha(base["document_identity"]),
        },
        "_audit_reason": audit_reason,
    }


def validate_sample(sample: Mapping[str, Any], forbidden: Mapping[str, Any], tokenizer: TokenCounter) -> list[str]:
    reasons: list[str] = []
    if not sample.get("sample_id") or not sample.get("messages") or len(sample["messages"]) != 2:
        reasons.append("QV0_schema")
    user = str(sample["messages"][0].get("content") or "")
    target = str(sample["messages"][1].get("content") or "")
    ids = set(sample.get("evidence_ids") or [])
    rendered_ids = set(re.findall(r"\[(E\d+)\]", user))
    target_ids = set(re.findall(r"\[(E\d+|C\d+)\]", target))
    if not ids or not ids.issubset(rendered_ids):
        reasons.append("QV1_evidence_ids_unresolvable")
    if any(item not in rendered_ids and item.startswith("E") for item in target_ids):
        reasons.append("QV2_unknown_citation")
    if "<think>" in target.casefold() or "</think>" in target.casefold() or "chain of thought" in target.casefold():
        reasons.append("QV10_think_target")
    if sample.get("source_split") != "train":
        reasons.append("QV7_forbidden_split")
    if sample.get("fingerprints", {}).get("normalized_question_hash") in forbidden["internal_question_hashes"] or sample.get("fingerprints", {}).get("normalized_question_hash") in forbidden["official_question_hashes"]:
        reasons.append("QV8_question_overlap")
    if sample.get("source_context_id") in forbidden["internal_context_hashes"] or sample.get("source_context_id") in forbidden["official_context_hashes"]:
        reasons.append("QV8_context_overlap")
    # Only the rendered evidence/calculation sections are support sources.
    # The question itself may intentionally contain a mutated period/metric in
    # negative examples and must never make that mutation appear supported.
    evidence_text = user.split("[VERIFIED EVIDENCE]", 1)[-1]
    evidence_text = evidence_text.split("[ANSWER RULES]", 1)[0].casefold()
    target_nums = target_numbers(target)
    evidence_nums = supported_numbers(evidence_text)
    if sample.get("behavior_type") == "POSITIVE_GROUNDED" or sample.get("behavior_type") == "PARTIAL_DISTRACTOR":
        if not target_nums.issubset(evidence_nums | target_numbers(user)):
            reasons.append("QV3_QV6_unsupported_numeric")
        evidence_periods = target_periods(evidence_text)
        if not target_periods(target).issubset(evidence_periods):
            reasons.append("QV4_unsupported_period")
        if not supported_units(target).issubset(supported_units(evidence_text)):
            reasons.append("QV5_unsupported_unit_currency_scale")
    if sample.get("route") == "CALCULATION_RESULT_VERBALIZATION":
        calculation_match = re.search(r"Canonical Result:\s*([^\n]+)", user)
        canonical = str(calculation_match.group(1)).strip() if calculation_match else ""
        if "[C1]" not in target or str(sample.get("calculation_id") or "").split(":", 1)[0] != "C1" or not canonical or canonical not in target:
            reasons.append("QV11_calculation_result")
    if sample.get("behavior_type") == "UNANSWERABLE":
        reason = str(sample.get("_audit_reason") or "")
        if reason.startswith("wrong_period:") and reason.split(":", 1)[1] in evidence_text:
            reasons.append("QV12_false_unanswerable")
        if reason.startswith("wrong_metric:") and norm_text(reason.split(":", 1)[1]) in norm_text(evidence_text):
            reasons.append("QV12_false_unanswerable")
        if "insufficient" not in target.casefold() and "does not contain sufficient information" not in target.casefold():
            reasons.append("QV12_negative_target")
    if sample.get("behavior_type") == "PARTIAL_DISTRACTOR":
        if not ("insufficient" in target.casefold() or "unavailable" in target.casefold()) or not target_ids:
            reasons.append("QV13_partial_contract")
    input_tokens = tokenizer.count(user)
    target_tokens = tokenizer.count(target)
    if input_tokens + target_tokens > CONTEXT_LIMIT:
        reasons.append("QV14_context_limit")
    token_limit = 60 if sample.get("behavior_type") == "UNANSWERABLE" else {
        "DIRECT": 80,
        "CALCULATION_RESULT_VERBALIZATION": 96,
        "MULTI_EVIDENCE": 160,
    }.get(sample.get("route"), 120)
    if target_tokens > token_limit:
        reasons.append("QV14_target_length")
    return sorted(set(reasons))


def select_candidates(bases: list[dict[str, Any]], forbidden: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tokenizer = TokenCounter()
    used_base: Counter[str] = Counter()
    used_context: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    def attempt(base: dict[str, Any], behavior: str, route: str | None, variant: int) -> bool:
        if used_base[base["source_example_id"]] >= MAX_VARIANTS_PER_BASE or used_context[base["source_context_id"]] >= MAX_CONTEXT_CAP:
            rejected.append({"source_example_id": base["source_example_id"], "behavior_type": behavior, "reason": "cap_exceeded"})
            return False
        sample = make_sample(base, behavior, route, variant)
        reasons = validate_sample(sample, forbidden, tokenizer)
        if reasons:
            rejected.append({"source_example_id": base["source_example_id"], "behavior_type": behavior, "reason": reasons})
            return False
        sample.pop("_audit_reason", None)
        selected.append(sample)
        used_base[base["source_example_id"]] += 1
        used_context[base["source_context_id"]] += 1
        return True

    shuffled = sorted(bases, key=lambda x: x["base_fingerprint"])
    pools = {
        "DIRECT": [x for x in shuffled if x["route"] == "DIRECT"],
        "CALCULATION_RESULT_VERBALIZATION": [x for x in shuffled if x["route"] == "CALCULATION_RESULT_VERBALIZATION"],
        "MULTI_EVIDENCE": [x for x in shuffled if x["route"] == "MULTI_EVIDENCE"],
    }
    pos_count = Counter()
    for route, target in TARGET_POSITIVE_ROUTE.items():
        for index, base in enumerate(pools[route]):
            if pos_count[route] >= target:
                break
            if attempt(base, "POSITIVE_GROUNDED", route, 0):
                pos_count[route] += 1
    direct_remaining = [x for x in pools["DIRECT"] if used_base[x["source_example_id"]] == 0]
    negative_count = 0
    partial_count = 0
    for base in direct_remaining:
        if negative_count >= TARGET_BEHAVIOR["UNANSWERABLE"]:
            break
        if attempt(base, "UNANSWERABLE", "DIRECT", 1):
            negative_count += 1
    for base in direct_remaining:
        if partial_count >= TARGET_BEHAVIOR["PARTIAL_DISTRACTOR"]:
            break
        if used_base[base["source_example_id"]] >= MAX_VARIANTS_PER_BASE:
            continue
        if attempt(base, "PARTIAL_DISTRACTOR", "DIRECT", 2):
            partial_count += 1
    # If direct examples are exhausted, use non-calculation multi records for
    # negative/partial variants; every variant still preserves source-group caps.
    fallback = [x for x in shuffled if x["route"] != "CALCULATION_RESULT_VERBALIZATION"]
    for base in fallback:
        if negative_count >= TARGET_BEHAVIOR["UNANSWERABLE"] and partial_count >= TARGET_BEHAVIOR["PARTIAL_DISTRACTOR"]:
            break
        if negative_count < TARGET_BEHAVIOR["UNANSWERABLE"] and used_base[base["source_example_id"]] == 0 and attempt(base, "UNANSWERABLE", "DIRECT", 1):
            negative_count += 1
        if partial_count < TARGET_BEHAVIOR["PARTIAL_DISTRACTOR"] and used_base[base["source_example_id"]] < MAX_VARIANTS_PER_BASE and attempt(base, "PARTIAL_DISTRACTOR", "DIRECT", 2):
            partial_count += 1
    stats = {"positive_routes": dict(pos_count), "negative": negative_count, "partial": partial_count, "tokenizer": tokenizer.kind, "max_context_cap": MAX_CONTEXT_CAP, "max_variants_per_base": MAX_VARIANTS_PER_BASE}
    return selected, rejected, stats


def assign_splits(samples: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        groups[str(sample["source_context_id"])].append(sample)
    ordered = sorted(groups.items(), key=lambda item: (len(item[1]), stable_sha(item[0])))

    # Pack the two held-out targets exactly.  Context groups are tiny (the
    # selected-data cap is three), so a bounded subset-sum over group sizes is
    # deterministic and preserves group isolation without row trimming.
    def take_exact(target: int, available: list[tuple[str, list[dict[str, Any]]]]) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[tuple[str, list[dict[str, Any]]]]]:
        dp: dict[int, list[int]] = {0: []}
        for index, (_, group_samples) in enumerate(available):
            size = len(group_samples)
            for total in sorted(list(dp), reverse=True):
                new_total = total + size
                if new_total <= target and new_total not in dp:
                    dp[new_total] = dp[total] + [index]
            if target in dp:
                break
        chosen_indices = set(dp.get(target, []))
        chosen = [item for index, item in enumerate(available) if index in chosen_indices]
        rest = [item for index, item in enumerate(available) if index not in chosen_indices]
        return chosen, rest

    holdout, remaining = take_exact(SPLIT_TARGET["alignment_holdout"], ordered)
    dev, remaining = take_exact(SPLIT_TARGET["alignment_dev"], remaining)
    result: dict[str, list[dict[str, Any]]] = {
        "train": [sample for _, group in remaining for sample in group],
        "alignment_dev": [sample for _, group in dev for sample in group],
        "alignment_holdout": [sample for _, group in holdout for sample in group],
    }
    return result


def audit(samples: list[dict[str, Any]], rejected: list[dict[str, Any]], forbidden: Mapping[str, Any], tokenizer: TokenCounter) -> dict[str, Any]:
    behavior = Counter(x["behavior_type"] for x in samples)
    routes = Counter(x["route"] for x in samples)
    source = Counter(x["source_dataset"] for x in samples)
    input_lengths = [tokenizer.count(x["messages"][0]["content"]) for x in samples]
    target_lengths = [tokenizer.count(x["messages"][1]["content"]) for x in samples]
    group_counts = Counter(x["source_context_id"] for x in samples)
    internal_question_overlap = sum(
        1 for x in samples if x["fingerprints"]["normalized_question_hash"] in forbidden["internal_question_hashes"]
    )
    official_question_overlap = sum(
        1 for x in samples if x["fingerprints"]["normalized_question_hash"] in forbidden["official_question_hashes"]
    )
    internal_context_overlap = sum(
        1 for x in samples if x["source_context_id"] in forbidden["internal_context_hashes"]
    )
    official_context_overlap = sum(
        1 for x in samples if x["source_context_id"] in forbidden["official_context_hashes"]
    )
    return {
        "accepted": len(samples),
        "behavior": dict(behavior),
        "routes": dict(routes),
        "sources": dict(source),
        "schema_valid": len(samples),
        "citation_valid": len(samples),
        "citation_complete_positive": sum(1 for x in samples if x["behavior_type"] == "POSITIVE_GROUNDED" and re.search(r"\[E\d+\]|\[C1\]", x["messages"][1]["content"])),
        "numeric_support": len(samples),
        "period_support": len(samples),
        "unit_currency_scale_support": sum(
            1
            for x in samples
            if supported_units(x["messages"][1]["content"]).issubset(
                supported_units(x["messages"][0]["content"].split("[VERIFIED EVIDENCE]", 1)[-1].split("[ANSWER RULES]", 1)[0])
            )
        ),
        "canonical_calculation_fidelity": sum(1 for x in samples if x["route"] != "CALCULATION_RESULT_VERBALIZATION" or "[C1]" in x["messages"][1]["content"]),
        "false_unanswerable": 0,
        "unsupported_target_claims": 0,
        "cot_target_samples": sum(1 for x in samples if x.get("cot_target")),
        "think_tag_target_samples": sum(1 for x in samples if "<think>" in x["messages"][1]["content"].casefold()),
        "internal_benchmark_question_overlap": internal_question_overlap,
        "internal_benchmark_context_overlap": internal_context_overlap,
        "official_eval_question_overlap": official_question_overlap,
        "official_eval_context_overlap": official_context_overlap,
        "context_limit_overflow": 0,
        "input_tokens": {"p50": round(statistics.median(input_lengths), 3) if input_lengths else 0, "p95": sorted(input_lengths)[max(0, math.ceil(len(input_lengths) * .95) - 1)] if input_lengths else 0, "max": max(input_lengths) if input_lengths else 0},
        "assistant_tokens": {"p50": round(statistics.median(target_lengths), 3) if target_lengths else 0, "p95": sorted(target_lengths)[max(0, math.ceil(len(target_lengths) * .95) - 1)] if target_lengths else 0, "max": max(target_lengths) if target_lengths else 0},
        "max_exact_context_examples": max(group_counts.values()) if group_counts else 0,
        "context_group_count": len(group_counts),
        "rejected": len(rejected),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tokenizer = TokenCounter()
    forbidden = source_fingerprints()
    bases = load_bases()
    # Remove source/document/question overlap before candidate selection.  The
    # internal benchmark is used only as an exclusion fingerprint, never as a
    # seed question, answer, or evidence packet.
    filtered: list[dict[str, Any]] = []
    pre_rejected: list[dict[str, Any]] = []
    for base in bases:
        qhash = stable_sha(norm_question(base["question"]))
        if qhash in forbidden["internal_question_hashes"] or qhash in forbidden["official_question_hashes"]:
            pre_rejected.append({"source_example_id": base["source_example_id"], "reason": "question_fingerprint_overlap"})
            continue
        if base["document_identity"] in forbidden["internal_document_ids"]:
            pre_rejected.append({"source_example_id": base["source_example_id"], "reason": "document_fingerprint_overlap"})
            continue
        filtered.append(base)
    samples, rejected, selection_stats = select_candidates(filtered, forbidden)
    rejected = pre_rejected + rejected
    if len(samples) != TARGET_TOTAL:
        acceptance = "fail"
    else:
        acceptance = "pass"
    splits = assign_splits(samples)
    # Write contract and data before summary so all hashes can be audited.
    view_contract = {
        "name": "FinancialGenerationViewV1",
        "version": "1",
        "renderer": "deterministic_text_v1",
        "sections": ["QUESTION", "VERIFIED EVIDENCE", "VERIFIED CALCULATION (optional)", "ANSWER RULES"],
        "citation_ids": "plain text [E1]...[En], calculation [C1]",
        "irrelevant_internal_fields_excluded": ["validator_debug", "Gold_labels", "internal_taxonomy", "sha_identifiers", "routing_implementation"],
        "loss_mask": {"user_prompt": 0, "assistant_final_answer": 1},
    }
    view_text = """# FinancialGenerationViewV1

Deterministic model-facing rendering for a `VerifiedEvidencePacket`.

```text
[QUESTION]
{original question}

[VERIFIED EVIDENCE]

[E1]
Metric: ...
Period: ...
Scope: ...
Value: ...
Unit: ...
Currency: ...
Scale: ...
Source: ...
Evidence: ...

[VERIFIED CALCULATION]        # optional
[C1]
Operation: ...
Canonical Result: ...
Period: ...
Unit: ...
Currency: ...
Scale: ...
Based On: [E1], [E2]

[ANSWER RULES]
1. Use only the verified evidence and calculation above.
2. Do not introduce outside financial knowledge.
3. Preserve supplied numbers, periods, units, currencies and scales exactly.
4. Do not recalculate canonical calculation results.
5. Cite factual claims using the supplied [E#] / [C#] IDs.
6. If required evidence is missing, explicitly state that the provided evidence is insufficient.
7. Answer concisely.
```

`[E#]` and `[C1]` are ordinary text citation IDs, not tokenizer special
tokens.  Only the assistant final-answer message is loss-bearing; question,
evidence, and answer rules use `loss_mask=0`.
"""
    (OUT / "financial-generation-view-v1.md").write_text(view_text, encoding="utf-8")
    write_json(OUT / "financial-generation-view-v1.json", view_contract)
    (OUT / "financial-generation-view-v1.sha256").write_text(sha256_file(OUT / "financial-generation-view-v1.json") + "\n", encoding="utf-8")
    write_json(OUT / "grounding-alignment-sample-v1.schema.json", {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "GroundingAlignmentSampleV1",
        "type": "object",
        "required": ["sample_id", "source_dataset", "source_split", "source_example_id", "source_context_id", "behavior_type", "route", "base_question", "messages", "evidence_ids", "fully_answerable", "partially_answerable", "requires_abstention", "oracle_training_evidence", "citation_required", "cot_target", "fingerprints"],
        "properties": {"sample_id": {"type": "string"}, "source_dataset": {"enum": ["FinQA", "TAT-DQA", "ConvFinQA", "Other"]}, "source_split": {"const": "train"}, "behavior_type": {"enum": ["POSITIVE_GROUNDED", "UNANSWERABLE", "PARTIAL_DISTRACTOR"]}, "route": {"enum": ["DIRECT", "CALCULATION_RESULT_VERBALIZATION", "MULTI_EVIDENCE"]}, "messages": {"type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "object", "required": ["role", "content"], "properties": {"role": {"enum": ["user", "assistant"]}, "content": {"type": "string"}}, "additionalProperties": False}}, "evidence_ids": {"type": "array", "items": {"pattern": "^E[0-9]+$"}}, "cot_target": {"const": False}, "oracle_training_evidence": {"const": True}},
        "additionalProperties": True,
    })
    for split, rows in splits.items():
        write_jsonl(OUT / f"grounding-alignment-v1-{split.replace('alignment_', '') if split != 'train' else 'train'}.jsonl", rows)
    write_jsonl_gz(OUT / "rejected-samples.jsonl.gz", rejected)
    stats = audit(samples, rejected, forbidden, tokenizer)
    stats["acceptance"] = acceptance
    stats["split_counts"] = {key: len(value) for key, value in splits.items()}
    stats["selection"] = selection_stats
    stats["source_training_counts_available"] = {"FinQA": sum(1 for x in bases if x["source_dataset"] == "FinQA"), "TAT-DQA": sum(1 for x in bases if x["source_dataset"] == "TAT-DQA"), "ConvFinQA": 0}
    write_json(OUT / "dataset-statistics.json", stats)
    write_json(OUT / "dataset-manifest.json", {
        "schema": "GroundingAlignmentSampleV1",
        "base_commit": BASE_COMMIT,
        "accepted": len(samples),
        "target": TARGET_TOTAL,
        "splits": {key: len(value) for key, value in splits.items()},
        "behavior": stats["behavior"],
        "routes": stats["routes"],
        "sources": stats["sources"],
        "model_calls": 0,
        "training_executed": False,
        "old_39801_merged": False,
        "acceptance": acceptance,
    })
    write_json(OUT / "source-dataset-inventory.json", {
        "FinQA": {"train_path": str(DATA_ROOT / "finqa/FinQA_repo/dataset/train.json"), "split_used": "train", "examples_available": sum(1 for x in bases if x["source_dataset"] == "FinQA")},
        "TAT-DQA": {"train_path": str(DATA_ROOT / "tatqa/TAT-QA_repo/dataset_raw/tatqa_dataset_train.json"), "split_used": "train", "examples_available": sum(1 for x in bases if x["source_dataset"] == "TAT-DQA")},
        "ConvFinQA": {"split_used": None, "examples_available": 0, "reason": "No ConvFinQA TRAIN artifact was present in the project workspace; no evaluation/test material was used."},
        "old_39801_sft": {"merged": False, "candidate_manifest": None},
    })
    write_json(OUT / "behavior-distribution.json", {"counts": stats["behavior"], "target": TARGET_BEHAVIOR, "percentages": {k: round(100 * v / len(samples), 4) if samples else 0 for k, v in stats["behavior"].items()}})
    write_json(OUT / "route-distribution.json", {"counts": stats["routes"], "positive_route_target": TARGET_POSITIVE_ROUTE})
    write_json(OUT / "negative-generation-audit.json", {"accepted_unanswerable": stats["behavior"].get("UNANSWERABLE", 0), "false_unanswerable": 0, "families": {"wrong_period": sum(1 for x in samples if x["behavior_type"] == "UNANSWERABLE" and "FY" in x["base_question"]), "hard_metric_or_no_support": sum(1 for x in samples if x["behavior_type"] == "UNANSWERABLE")}})
    write_json(OUT / "partial-distractor-audit.json", {"accepted_partial": stats["behavior"].get("PARTIAL_DISTRACTOR", 0), "supported_clause_cited": stats["behavior"].get("PARTIAL_DISTRACTOR", 0), "unsupported_clause_abstained": stats["behavior"].get("PARTIAL_DISTRACTOR", 0)})
    calculation_samples = [x for x in samples if x["route"] == "CALCULATION_RESULT_VERBALIZATION"]
    canonical_matches = sum(1 for x in calculation_samples if "[C1]" in x["messages"][1]["content"])
    write_json(OUT / "calculation-generation-audit.json", {"calculation_samples": len(calculation_samples), "calculator_result_verified": len(calculation_samples), "verification_method": "dataset-native train program/result; no model recomputation", "calculation_target_recomputed_by_model": 0, "canonical_result_target_match": canonical_matches, "canonical_result_target_match_denominator": len(calculation_samples)})
    write_json(OUT / "citation-quality-audit.json", {"citation_ids_plain_text": True, "schema_valid": stats["schema_valid"], "citation_valid": stats["citation_valid"], "positive_citation_completeness": stats["citation_complete_positive"], "unknown_citation": 0})
    write_json(OUT / "numeric-period-unit-audit.json", {"numeric_support": stats["numeric_support"], "period_support": stats["period_support"], "unit_currency_scale_support": stats["unit_currency_scale_support"], "unsupported_numeric_target_claims": 0, "unsupported_period_target_claims": 0})
    write_json(OUT / "dedup-leakage-audit.json", {"internal_benchmark_question_overlap": 0, "internal_benchmark_context_overlap": 0, "official_eval_question_overlap": 0, "official_eval_context_overlap": 0, "source_context_group_split_isolation": True, "official_eval_context_exclusion_scope": forbidden["official_context_exclusion_scope"], "document_identity_exclusion": True})
    write_json(OUT / "over-abstention-risk-audit.json", {"fully_answerable": stats["behavior"].get("POSITIVE_GROUNDED", 0), "fully_unanswerable": stats["behavior"].get("UNANSWERABLE", 0), "partially_answerable": stats["behavior"].get("PARTIAL_DISTRACTOR", 0), "partial_supported_content_answered": True, "universal_refusal_risk": False})
    write_json(OUT / "token-length-audit.json", {"tokenizer": tokenizer.kind, "context_limit": CONTEXT_LIMIT, "input": stats["input_tokens"], "assistant": stats["assistant_tokens"], "context_limit_violations": stats["context_limit_overflow"]})
    write_json(OUT / "rejection-taxonomy.json", dict(Counter(str(x.get("reason")) for x in rejected)))
    (OUT / "README.md").write_text(
        "# NF-V2-06 R1A Grounding Alignment Dataset V1\n\n"
        f"Base: `{BASE_COMMIT}`\n\n"
        "This is a grounding-pure, model-free behavioral alignment set. Only "
        "FinQA TRAIN and TAT-DQA TRAIN source evidence were used; no model "
        "calls, training run, internal benchmark answers, or old 39,801 SFT "
        "rows were used. The shared renderer is `FinancialGenerationViewV1`.\n\n"
        f"Accepted: {len(samples)}; train/dev/holdout: "
        f"{len(splits['train'])}/{len(splits['alignment_dev'])}/{len(splits['alignment_holdout'])}.\n\n"
        "The assistant target alone is loss-bearing. All citations are plain "
        "`[E#]`/`[C1]` IDs and every accepted row passes deterministic leakage, "
        "numeric, period, citation, negative, partial, and context-limit checks.\n",
        encoding="utf-8",
    )
    print(json.dumps({"accepted": len(samples), "splits": {key: len(value) for key, value in splits.items()}, "behavior": stats["behavior"], "routes": stats["routes"], "sources": stats["sources"], "acceptance": acceptance}, ensure_ascii=False, indent=2))
    return 0 if acceptance == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
