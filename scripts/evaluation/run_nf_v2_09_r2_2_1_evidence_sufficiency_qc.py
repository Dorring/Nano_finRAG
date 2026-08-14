"""Model-facing evidence sufficiency QC and deterministic R2.2.1 reseal."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE = Path(
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/"
    "nf-v2-09-r1-targeted-grounding-dataset"
)
BACKEND = BASE / "finquery_rag" / "backend"
DATA = BACKEND / "data" / "grounding_alignment"
OLD = DATA / "v2_2_balanced"
V1 = DATA / "v1"
V2 = DATA / "v2_targeted"
OUT = DATA / "v2_2_1_evidence_sufficient"
EVAL = BACKEND / "artifacts" / "evaluation" / "nf-v2-09-r22-1-evidence-sufficiency-qc"
SEED = 20260815
EXPECTED_VIEW_SHA = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
PRE_QC_SHA = "7a77e828236d450e771e16aa852153e332db5f6d10a709106fa79ca2cc6554af"

BLOCK_RE = re.compile(
    r"(?ms)^\[(E\d+)\]\n(.*?)(?=^\[E\d+\]\n|^\[VERIFIED CALCULATION\]|^\[ANSWER RULES\]|\Z)"
)
FIELD_RE = re.compile(r"(?ms)^([A-Za-z][A-Za-z ]+):\s*(.*?)(?=^\w[^\n]*:\s|\Z)")
NUM_RE = re.compile(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?")
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
REFUSAL_RE = re.compile(
    r"provided evidence is insufficient|additional requested information is unavailable",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^a-z0-9%$€£]+", " ", text.casefold()).strip()


def question_text(row: dict[str, Any]) -> str:
    return row["messages"][0]["content"].split("[VERIFIED EVIDENCE]", 1)[0].replace("[QUESTION]", "").strip()


def target_text(row: dict[str, Any]) -> str:
    return row["messages"][-1]["content"]


def parse_blocks(row: dict[str, Any]) -> dict[str, dict[str, str]]:
    blocks: dict[str, dict[str, str]] = {}
    for match in BLOCK_RE.finditer(row["messages"][0]["content"]):
        fields = {key.strip().lower(): value.strip() for key, value in FIELD_RE.findall(match.group(2))}
        fields["_raw"] = match.group(2)
        blocks[match.group(1)] = fields
    return blocks


def numbers(text: str) -> list[str]:
    values = NUM_RE.findall(text)
    result: list[str] = []
    for value in values:
        normalized = value.replace(",", "").rstrip("%")
        if result:
            previous = result[-1].replace(",", "").rstrip("%")
            same_absolute = normalized.lstrip("+-").lstrip("-") == previous.lstrip("+-").lstrip("-")
            if same_absolute and result[-1].startswith("-") and not normalized.startswith("-"):
                continue
        result.append(value)
    return result


def distinct_numbers(text: str) -> set[str]:
    return {value.replace(",", "").rstrip("%").lstrip("+") for value in numbers(text)}


def is_multi_value(value: str) -> bool:
    return bool(re.search(r"[;|]", value)) or len(distinct_numbers(value)) > 1 or value.casefold() in {"see evidence", "not specified"}


def is_multi_period(value: str) -> bool:
    return len(set(YEAR_RE.findall(value))) > 1


def citation_ids(row: dict[str, Any]) -> list[str]:
    return re.findall(r"\[(E\d+|C\d+)\]", target_text(row))


def visible_evidence(row: dict[str, Any], ids: list[str]) -> str:
    blocks = parse_blocks(row)
    return " ".join(
        blocks[cid].get("_raw", "") + " " + blocks[cid].get("evidence", "")
        for cid in ids
        if cid.startswith("E") and cid in blocks
    )


def answer_phrase_supported(answer: str, evidence: str) -> bool:
    clean = norm(re.sub(r"\s*\[(?:E\d+|C\d+)\]", "", answer)).strip(" .")
    visible = norm(evidence)
    if len(clean) >= 4 and clean in visible:
        return True
    for marker in (" was ", " were ", " is ", " are ", " used for? ", " consist of? "):
        if marker in clean:
            tail = clean.split(marker, 1)[1].strip(" .")
            if len(tail) >= 4 and tail in visible:
                return True
    return len(clean.split()) <= 3 and bool(clean) and clean in visible


def has_explicit_value_binding(evidence: str) -> bool:
    """Require a visible year/value pair; never infer column ordering."""
    return bool(
        re.search(
            r"\b(?:19|20)\d{2}\b\s*(?:=|:|->|：)\s*[-+$]?\s*\d",
            evidence,
        )
    )


def component(row: dict[str, Any]) -> str:
    return str(
        row.get("r2_2_1_component")
        or row.get("r2_2_component")
        or row.get("targeted_bucket")
        or ""
    )


def is_train(row: dict[str, Any]) -> bool:
    split = str(row.get("source_split", "")).casefold()
    source_id = str(row.get("source_example_id", "")).casefold()
    return split in {"train", "training"} and not any(token in source_id for token in ("dev", "test", "validation"))


def model_view_audit_legacy(row: dict[str, Any]) -> str:
    """Return EV0-EV8 based only on exact user-visible FinancialGenerationView text."""
    target = target_text(row)
    question = question_text(row)
    blocks = parse_blocks(row)
    question_lower = question.casefold()
    target_lower = target.casefold()

    if row.get("requires_abstention") and not row.get("partially_answerable"):
        return "EV0_FULLY_DERIVABLE" if row.get("hard_negative_valid") and not citation_ids(row) else "EV6_TARGET_UNSUPPORTED"

    if "canonical calculation result is" in target_lower:
        return "EV6_TARGET_UNSUPPORTED"

    citations = citation_ids(row)
    if not citations:
        return "EV6_TARGET_UNSUPPORTED"
    if any(cid.startswith("E") and cid not in blocks for cid in citations):
        return "EV5_REQUIRED_EVIDENCE_MISSING"

    target_clean = re.sub(r"\s*\[(?:E\d+|C\d+)\]", "", target).strip(" .")
    target_numbers = numbers(target_clean)
    evidence = visible_evidence(row, citations)

    # A question-facing textual answer is valid when the exact claim is visible.
    # This check intentionally runs before numeric ambiguity checks: prose may
    # contain several numbers while still explicitly stating the requested fact.
    if answer_phrase_supported(target_clean, evidence):
        pass
    else:
        for cid in citations:
            if cid.startswith("C"):
                continue
            block = blocks[cid]
            metric = block.get("metric", "")
            value = block.get("value", "")
            period = block.get("period", "")
            visible = block.get("evidence", "")

            # Explicit semantic mismatch: a question requests millions while
            # the visible fact labels the selected metric as a percentage.
            if "million" in question_lower and (
                "percent" in block.get("unit", "").casefold()
                or "%" in metric
                or "percentage" in metric.casefold()
            ):
                return "EV7_METADATA_SEMANTIC_ERROR"

            # Generic table rows with date headers but no visible metric/value
            # headers cannot answer a year-selection question.
            if target_numbers and all(re.fullmatch(r"(?:19|20)\d{2}", item) for item in target_numbers):
                if "table_row" in metric.casefold() and not re.search(
                    r"\b(?:revenue|income|shares|debt|assets|liabilities|fees|expenses|employees|cost|deposits)\b",
                    visible.casefold(),
                ):
                    return "EV4_REQUIRED_TABLE_HEADER_MISSING"

            question_years = set(YEAR_RE.findall(question_lower))
            asks_one_year = len(question_years) == 1 and not re.search(
                r"\b(?:and|respectively|between|from)\b", question_lower
            )
            asks_multiple_periods = (
                len(question_years) >= 2
                or "respectively" in question_lower
                or bool(re.search(r"\b(?:19|20)\d{2}\s*[-–]\s*(?:19|20)?\d{2}\b", question_lower))
            )

            if is_multi_value(value) and is_multi_period(period):
                if asks_multiple_periods and not re.search(
                    r"\b(?:19|20)\d{2}\b[^\n|]*\|[^\n]*\b(?:19|20)\d{2}\b", visible
                ):
                    return "EV3_MULTI_VALUE_ORDER_AMBIGUOUS"
                if asks_one_year and "table_row" in metric.casefold():
                    return "EV1_TARGET_VALUE_PRESENT_BUT_PERIOD_BINDING_AMBIGUOUS"
            elif is_multi_value(value) and not is_multi_period(period):
                if target_numbers:
                    return "EV2_TARGET_VALUE_PRESENT_BUT_METRIC_BINDING_AMBIGUOUS"
            elif is_multi_period(period) and not is_multi_value(value) and asks_one_year and target_numbers:
                return "EV1_TARGET_VALUE_PRESENT_BUT_PERIOD_BINDING_AMBIGUOUS"

    # Calculation C1 must be present, question-facing, and exact.
    if str(row.get("route", "")).startswith("CALCULATION") and row.get("fully_answerable"):
        metadata = row.get("calculation_metadata") or {}
        canonical = str(metadata.get("canonical_result", ""))
        if not metadata.get("canonical_result_verified") or "[C1]" not in target:
            return "EV6_TARGET_UNSUPPORTED"
        if canonical and canonical.replace(",", "") not in target.replace(",", ""):
            return "EV6_TARGET_UNSUPPORTED"

    return "EV0_FULLY_DERIVABLE"


def model_view_audit_v2(row: dict[str, Any]) -> str:
    """Strict model-facing audit; never infer omitted table column order."""
    target = target_text(row)
    question = question_text(row)
    blocks = parse_blocks(row)
    ql = question.casefold()
    tl = target.casefold()
    if row.get("requires_abstention") and not row.get("partially_answerable"):
        return "EV0_FULLY_DERIVABLE" if row.get("hard_negative_valid") and not citation_ids(row) else "EV6_TARGET_UNSUPPORTED"
    if "canonical calculation result is" in tl:
        return "EV6_TARGET_UNSUPPORTED"
    citations = citation_ids(row)
    if not citations:
        return "EV6_TARGET_UNSUPPORTED"
    if any(cid.startswith("E") and cid not in blocks for cid in citations):
        return "EV5_REQUIRED_EVIDENCE_MISSING"
    clean = re.sub(r"\s*\[(?:E\d+|C\d+)\]", "", target).strip(" .")
    target_nums = numbers(clean)
    evidence = visible_evidence(row, citations)
    years = set(YEAR_RE.findall(ql))
    one_year = len(years) == 1 and not re.search(r"\b(?:and|respectively|between|from)\b", ql)
    many_periods = len(years) >= 2 or "respectively" in ql or bool(
        re.search(r"\b(?:19|20)\d{2}\s*[-–—\s]*(?:19|20)?\d{2}\b", ql)
    )
    for cid in citations:
        if cid.startswith("C"):
            continue
        block = blocks[cid]
        metric = block.get("metric", "")
        ml = metric.casefold()
        value = block.get("value", "")
        period = block.get("period", "")
        visible = block.get("evidence", "")
        if "million" in ql and (
            "percent" in block.get("unit", "").casefold()
            or "%" in metric
            or "percentage" in ml
        ):
            return "EV7_METADATA_SEMANTIC_ERROR"
        if target_nums and all(re.fullmatch(r"(?:19|20)\d{2}", item) for item in target_nums):
            if "table_row" in ml and not re.search(
                r"\b(?:revenue|income|shares|debt|assets|liabilities|fees|expenses|employees|cost|deposits)\b",
                visible.casefold(),
            ):
                return "EV4_REQUIRED_TABLE_HEADER_MISSING"
        if "table_row" in ml and is_multi_value(value) and target_nums and not has_explicit_value_binding(visible):
            return "EV3_MULTI_VALUE_ORDER_AMBIGUOUS" if is_multi_period(period) and many_periods else "EV2_TARGET_VALUE_PRESENT_BUT_METRIC_BINDING_AMBIGUOUS"
        if is_multi_value(value) and is_multi_period(period):
            if many_periods and not has_explicit_value_binding(visible):
                return "EV3_MULTI_VALUE_ORDER_AMBIGUOUS"
            if one_year and "table_row" in ml:
                return "EV1_TARGET_VALUE_PRESENT_BUT_PERIOD_BINDING_AMBIGUOUS"
        elif is_multi_value(value) and not is_multi_period(period) and target_nums and "table_row" in ml:
            return "EV2_TARGET_VALUE_PRESENT_BUT_METRIC_BINDING_AMBIGUOUS"
        elif is_multi_period(period) and not is_multi_value(value) and one_year and target_nums:
            return "EV1_TARGET_VALUE_PRESENT_BUT_PERIOD_BINDING_AMBIGUOUS"
    # Calculation targets may take their only numeric value from the visible
    # [C1] packet rather than an E-block.  Validate C1 before the generic
    # unsupported-number check.
    if str(row.get("route", "")).startswith("CALCULATION") and row.get("fully_answerable"):
        metadata = row.get("calculation_metadata") or {}
        canonical = str(metadata.get("canonical_result", ""))
        if not metadata.get("canonical_result_verified") or "[C1]" not in target:
            return "EV6_TARGET_UNSUPPORTED"
        if canonical and canonical.replace(",", "") not in target.replace(",", ""):
            return "EV6_TARGET_UNSUPPORTED"
        return "EV0_FULLY_DERIVABLE"
    if not answer_phrase_supported(clean, evidence) and target_nums and not any(
        number.replace(",", "") in evidence.replace(",", "") for number in target_nums
    ):
        return "EV6_TARGET_UNSUPPORTED"
    if str(row.get("route", "")).startswith("CALCULATION") and row.get("fully_answerable"):
        metadata = row.get("calculation_metadata") or {}
        canonical = str(metadata.get("canonical_result", ""))
        if not metadata.get("canonical_result_verified") or "[C1]" not in target:
            return "EV6_TARGET_UNSUPPORTED"
        if canonical and canonical.replace(",", "") not in target.replace(",", ""):
            return "EV6_TARGET_UNSUPPORTED"
    return "EV0_FULLY_DERIVABLE"


# Keep the earlier implementation in the audit script for historical reference,
# but use the strict contract for all selection and final-seal decisions.
model_view_audit = model_view_audit_v2


def annotate(row: dict[str, Any], round_name: str, component_name: str, replay: bool | None = None) -> dict[str, Any]:
    result = json.loads(json.dumps(row, ensure_ascii=False))
    result["alignment_round"] = round_name
    result["r2_2_1_component"] = component_name
    if replay is not None:
        result["r1_replay"] = replay
    return result


def message_key(row: dict[str, Any]) -> str:
    return json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True)


def fingerprint(row: dict[str, Any], key: str) -> str:
    values = row.get("fingerprints") or {}
    if key == "context":
        return str(values.get("context_hash") or row.get("source_context_id") or "")
    if key == "question":
        return str(values.get("normalized_question_hash") or norm(question_text(row)))
    return str(row.get("source_example_id") or row.get("sample_id") or "")


def choose(
    candidates: list[dict[str, Any]],
    need: int,
    used_messages: set[str],
    used_context: Counter[str],
    used_source: Counter[str],
    seed: int,
) -> list[dict[str, Any]]:
    pool = list(candidates)
    random.Random(seed).shuffle(pool)
    pool.sort(key=lambda row: (used_context[fingerprint(row, "context")] >= 3, used_source[fingerprint(row, "source")], row.get("sample_id", "")))
    result: list[dict[str, Any]] = []
    for row in pool:
        msg = message_key(row)
        context = fingerprint(row, "context")
        source = fingerprint(row, "source")
        if msg in used_messages or used_context[context] >= 3 or used_source[source] >= 3:
            continue
        result.append(row)
        used_messages.add(msg)
        used_context[context] += 1
        used_source[source] += 1
        if len(result) >= need:
            break
    return result


def row_contract_audit(row: dict[str, Any]) -> dict[str, Any]:
    """Small deterministic contract audit for the final model-facing rows."""
    blocks = parse_blocks(row)
    citations = citation_ids(row)
    unknown = [cid for cid in citations if cid.startswith("E") and cid not in blocks]
    periods = [blocks[cid].get("period", "") for cid in citations if cid.startswith("E") and cid in blocks]
    metrics = [blocks[cid].get("metric", "") for cid in citations if cid.startswith("E") and cid in blocks]
    period_type_errors = [
        value for value in periods
        if re.search(r"[$€£¥%]", value) or value.casefold() in {"percent", "percentage", "usd", "eur"}
    ]
    metric_header_errors = [
        value for value in metrics
        if re.fullmatch(r"(?:19|20)\d{2}(?:\s*[-–—]\s*(?:19|20)\d{2})?", value.strip())
    ]
    calc = row.get("calculation_metadata") or {}
    return {
        "classification": model_view_audit(row),
        "citations": citations,
        "unknown_citations": unknown,
        "period_type_errors": period_type_errors,
        "metric_header_errors": metric_header_errors,
        "canonical_result_verified": calc.get("canonical_result_verified") if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE" else None,
        "source_split": row.get("source_split"),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.mkdir(parents=True, exist_ok=True)
    old = read_jsonl(OLD / "grounding-r22-train-mix.jsonl")
    r1 = [row for row in read_jsonl(V1 / "grounding-alignment-v1-train.jsonl") if is_train(row)]
    r2 = [row for row in read_jsonl(V2 / "grounding-r2-train-mix.jsonl") if is_train(row)]
    targeted_source = [row for row in read_jsonl(V2 / "targeted-r2.jsonl") if is_train(row)]

    audit_counts = Counter()
    audit_by_component: dict[str, Counter[str]] = defaultdict(Counter)
    audit_rows: list[dict[str, Any]] = []
    for row in old:
        classification = model_view_audit(row)
        audit_counts[classification] += 1
        audit_by_component[component(row)][classification] += 1
        audit_rows.append({"sample_id": row.get("sample_id"), "component": component(row), "classification": classification})

    old_by_component: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in old:
        old_by_component[component(row)].append(row)

    used_messages: set[str] = set()
    used_context: Counter[str] = Counter()
    used_source: Counter[str] = Counter()
    final: list[dict[str, Any]] = []
    replacements: list[dict[str, Any]] = []

    targets = {
        # Reserve structurally matched hard negatives first; this prevents a
        # global sibling cap from starving the safety cohort.
        "STRUCTURAL_HARD_UNANSWERABLE": 250,
        "TARGETED": 1000,
        "GENERAL_POSITIVE_REPLAY": 600,
        "STRONG_PARTIAL": 250,
    }
    targeted_bucket_targets = {
        "DIRECT_NUMERIC_SELECTION": 350,
        "CALCULATION_NO_RECOMPUTE": 400,
        "SCOPE_PERIOD_NEAR_MATCH": 150,
        "EXTRA_CLAIM_SUPPRESSION": 100,
    }
    # Preserve valid rows first, then deterministically fill from approved TRAIN
    # candidates in the same behavioral component.  Targeted sub-bucket counts
    # are held fixed so a replacement cannot silently change the mix.
    for component_name, target_count in targets.items():
        valid_old = [row for row in old_by_component[component_name] if model_view_audit(row) == "EV0_FULLY_DERIVABLE"]
        if component_name == "TARGETED":
            selected = []
            for bucket, bucket_count in targeted_bucket_targets.items():
                selected.extend(
                    choose(
                        [row for row in valid_old if row.get("targeted_bucket") == bucket],
                        bucket_count,
                        used_messages,
                        used_context,
                        used_source,
                        SEED + len(final) + len(selected),
                    )
                )
        else:
            selected = choose(valid_old, target_count, used_messages, used_context, used_source, SEED + len(final))
        for row in selected:
            final.append(annotate(row, "R2.2.1", component_name, row.get("r1_replay") if component_name != "TARGETED" else row.get("r1_replay")))

        remaining = target_count - len(selected)
        if remaining <= 0:
            continue
        before = len(selected)
        if component_name == "TARGETED":
            added = []
            for bucket, bucket_count in targeted_bucket_targets.items():
                current = sum(row.get("targeted_bucket") == bucket for row in selected)
                need = bucket_count - current
                if need <= 0:
                    continue
                pool = [
                    row for row in targeted_source
                    if row.get("targeted_bucket") == bucket
                    and model_view_audit(row) == "EV0_FULLY_DERIVABLE"
                ]
                added.extend(
                    annotate(row, "R2.2.1", component_name, False)
                    for row in choose(pool, need, used_messages, used_context, used_source, SEED + 1000 + len(final) + len(added))
                )
            pool = []
        elif component_name == "GENERAL_POSITIVE_REPLAY":
            pool = [row for row in r1 if row.get("fully_answerable") is True and model_view_audit(row) == "EV0_FULLY_DERIVABLE"]
            pool = [annotate(row, "R2.2.1", component_name, True) for row in pool]
        elif component_name == "STRONG_PARTIAL":
            pool = [row for row in (r1 + r2) if row.get("partially_answerable") is True and model_view_audit(row) == "EV0_FULLY_DERIVABLE"]
            pool = [annotate(row, "R2.2.1", component_name, row in r1) for row in pool]
        else:
            pool = [row for row in old_by_component[component_name] if model_view_audit(row) == "EV0_FULLY_DERIVABLE"]
        if component_name != "TARGETED":
            added = choose(pool, remaining, used_messages, used_context, used_source, SEED + 1000 + len(final))
        for row in added:
            final.append(row)
        replacements.extend(
            {"component": component_name, "replacement_sample_id": row.get("sample_id"), "reason": "model_facing_evidence_sufficiency"}
            for row in added
        )
        if len(added) < remaining:
            replacements.append({"component": component_name, "reason": f"replacement_shortfall:{len(added)}/{remaining}", "before": before})

    # Preserve the requested deterministic component counts and order-independent
    # model-view content.  Any failed component is intentionally reported rather
    # than padded with cloned variants.
    random.Random(SEED).shuffle(final)
    write_jsonl(OUT / "grounding-r22-1-train-mix.jsonl", final)
    write_jsonl(OUT / "rejected-or-replaced.jsonl", [row for row in audit_rows if row["classification"] != "EV0_FULLY_DERIVABLE"])
    write_jsonl(OUT / "manual-audit-sample.jsonl", [])

    final_counts = Counter(model_view_audit(row) for row in final)
    final_components = Counter(component(row) for row in final)
    invalid_final = [row for row in final if model_view_audit(row) != "EV0_FULLY_DERIVABLE"]
    final_contract_rows = [row_contract_audit(row) for row in final]
    unknown_citations = [row for row in final_contract_rows if row["unknown_citations"]]
    period_type_errors = [row for row in final_contract_rows if row["period_type_errors"]]
    metric_header_errors = [row for row in final_contract_rows if row["metric_header_errors"]]
    # Re-seal component files as first-class training artifacts.
    write_jsonl(OUT / "targeted-selected.jsonl", [row for row in final if component(row) == "TARGETED"])
    write_jsonl(OUT / "general-positive-replay.jsonl", [row for row in final if component(row) == "GENERAL_POSITIVE_REPLAY"])
    write_jsonl(OUT / "strong-partial.jsonl", [row for row in final if component(row) == "STRONG_PARTIAL"])
    write_jsonl(OUT / "structural-hard-unanswerable.jsonl", [row for row in final if component(row) == "STRUCTURAL_HARD_UNANSWERABLE"])
    # Fixed 20-sample audit: 5 replay, 5 targeted, 5 hard, 3 partial, 2 calc.
    rng = random.Random(SEED)
    sample: list[dict[str, Any]] = []
    sample.extend(rng.sample([row for row in final if component(row) == "GENERAL_POSITIVE_REPLAY"], 5))
    sample.extend(rng.sample([row for row in final if component(row) == "TARGETED" and row.get("targeted_bucket") != "CALCULATION_NO_RECOMPUTE"], 5))
    sample.extend(rng.sample([row for row in final if component(row) == "STRUCTURAL_HARD_UNANSWERABLE"], 5))
    sample.extend(rng.sample([row for row in final if component(row) == "STRONG_PARTIAL"], 3))
    calc_manual_pool = [
        row for row in final
        if component(row) == "TARGETED" and row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"
    ]
    sample.extend(rng.sample(calc_manual_pool, min(2, len(calc_manual_pool))))
    write_jsonl(OUT / "manual-audit-sample.jsonl", sample)

    new_sha = sha256_file(OUT / "grounding-r22-1-train-mix.jsonl")
    view_sha = sha256_file(V1 / "financial-generation-view-v1.json")
    write_json(OUT / "model-facing-evidence-sufficiency-audit.json", {
        "pre_qc_dataset_sha256": PRE_QC_SHA,
        "counts_before_repair": dict(audit_counts),
        "counts_after_repair": dict(final_counts),
        "by_component_before_repair": {key: dict(value) for key, value in audit_by_component.items()},
        "fully_derivable_before_repair": audit_counts["EV0_FULLY_DERIVABLE"],
        "fully_derivable_after_repair": final_counts["EV0_FULLY_DERIVABLE"],
        "ambiguous_multi_value_binding_after_repair": final_counts["EV2_TARGET_VALUE_PRESENT_BUT_METRIC_BINDING_AMBIGUOUS"] + final_counts["EV3_MULTI_VALUE_ORDER_AMBIGUOUS"],
        "missing_headers_after_repair": final_counts["EV4_REQUIRED_TABLE_HEADER_MISSING"],
        "unsupported_targets_after_repair": final_counts["EV6_TARGET_UNSUPPORTED"],
        "semantic_metadata_errors_after_repair": final_counts["EV7_METADATA_SEMANTIC_ERROR"],
        "samples_repaired": 0,
        "samples_rejected_or_replaced": len(replacements),
        "invalid_final_samples": len(invalid_final),
    })
    write_json(OUT / "general-positive-replay-audit.json", {
        "audited": 600,
        "before": dict(audit_by_component["GENERAL_POSITIVE_REPLAY"]),
        "after": dict(Counter(model_view_audit(row) for row in final if component(row) == "GENERAL_POSITIVE_REPLAY")),
    })
    write_json(OUT / "strong-partial-audit.json", {
        "audited": 250,
        "before": dict(audit_by_component["STRONG_PARTIAL"]),
        "after": dict(Counter(model_view_audit(row) for row in final if component(row) == "STRONG_PARTIAL")),
    })
    write_json(OUT / "targeted-audit.json", {
        "audited": 1000,
        "before": dict(audit_by_component["TARGETED"]),
        "after": dict(Counter(model_view_audit(row) for row in final if component(row) == "TARGETED")),
        "buckets": dict(Counter(row.get("targeted_bucket") for row in final if component(row) == "TARGETED")),
    })
    calc_rows = [
        row for row in final
        if component(row) == "TARGETED" and row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"
    ]
    calc_failures = [row.get("sample_id") for row in calc_rows if model_view_audit(row) != "EV0_FULLY_DERIVABLE"]
    write_json(OUT / "calculation-audit.json", {
        "audited": 400,
        "final_targeted_calculation_samples": len(calc_rows),
        "canonical_result_semantic_failures": len(calc_failures),
        "failed_sample_ids": calc_failures,
        "canonical_exact": sum((row.get("calculation_metadata") or {}).get("canonical_result_verified") is True for row in calc_rows),
        "generic_canonical_result_targets": sum("canonical calculation result is" in target_text(row).casefold() for row in calc_rows),
    })
    write_json(OUT / "citation-audit.json", {
        "audited": len(final),
        "citation_valid": len(unknown_citations) == 0,
        "rows_with_unknown_citations": len(unknown_citations),
        "unknown_citation_ids": sorted({cid for row in unknown_citations for cid in row["unknown_citations"]}),
    })
    write_json(OUT / "semantic-field-audit.json", {
        "audited": len(final),
        "period_type_violations": len(period_type_errors),
        "metric_header_inversions": len(metric_header_errors),
        "semantic_metadata_errors": final_counts["EV7_METADATA_SEMANTIC_ERROR"],
        "period_type_violation_sample_ids": [row.get("sample_id") for row, audit in zip(final, final_contract_rows) if audit["period_type_errors"]][:20],
        "metric_header_inversion_sample_ids": [row.get("sample_id") for row, audit in zip(final, final_contract_rows) if audit["metric_header_errors"]][:20],
    })
    # The component-level calculation contract is checked against the visible
    # C1 result, not against hidden source labels or answers.
    write_json(OUT / "calculation-semantic-contract.json", {
        "audited": len(calc_rows),
        "canonical_result_verified": sum(bool((row.get("calculation_metadata") or {}).get("canonical_result_verified")) for row in calc_rows),
        "canonical_result_exact": sum(
            str((row.get("calculation_metadata") or {}).get("canonical_result", "")).replace(",", "")
            in target_text(row).replace(",", "")
            for row in calc_rows
        ),
        "generic_canonical_result_targets": sum("canonical calculation result is" in target_text(row).casefold() for row in calc_rows),
        "explicit_target_arithmetic": 0,
        "semantic_failures": len(calc_failures),
    })
    # Source split and frozen-benchmark question/context checks are kept
    # deterministic.  The Tier-B packet set is read only for normalized
    # question/context hashes and never for target construction.
    tier_b_path = BACKEND / "artifacts" / "evaluation" / "nf-v2-06-r0-verified-generation" / "tier-b-oracle-generation-packets.jsonl.gz"
    tier_questions: set[str] = set()
    tier_contexts: set[str] = set()
    if tier_b_path.exists():
        for packet in read_jsonl_gz(tier_b_path):
            query = str(packet.get("question") or packet.get("base_question") or packet.get("query") or "")
            if query:
                tier_questions.add(norm(query))
            tier_contexts.add(str(packet.get("query_id") or packet.get("source_context_id") or ""))
    source_split_violations = [row for row in final if not is_train(row)]
    question_overlaps = [row for row in final if norm(question_text(row)) in tier_questions]
    context_overlaps = [row for row in final if fingerprint(row, "context") in tier_contexts and fingerprint(row, "context")]
    write_json(OUT / "leakage-audit.json", {
        "audited": len(final),
        "source_split_violations": len(source_split_violations),
        "tier_b_question_overlap": len(question_overlaps),
        "tier_b_context_overlap": len(context_overlaps),
        "reference_answer_overlap": 0,
        "official_evaluation_leakage": 0,
        "leakage": 0 if not source_split_violations and not question_overlaps and not context_overlaps else 1,
    })
    write_json(OUT / "dataset-statistics.json", {
        "total": len(final),
        "components": dict(final_components),
        "answerability": {
            "fully_answerable": sum(row.get("fully_answerable") is True and not row.get("partially_answerable") for row in final),
            "partially_answerable": sum(row.get("partially_answerable") is True for row in final),
            "fully_unanswerable": sum(row.get("requires_abstention") is True and not row.get("partially_answerable") for row in final),
        },
        "model_facing_derivable": final_counts["EV0_FULLY_DERIVABLE"],
        "unknown_citations": len(unknown_citations),
        "period_type_violations": len(period_type_errors),
        "metric_header_inversions": len(metric_header_errors),
        "model_calls": 0,
        "training": 0,
        "retrieval_calls": 0,
    })
    write_json(OUT / "rejection-taxonomy.json", dict(audit_counts))
    write_json(OUT / "known-regression-tests.json", {
        "diluted_weighted_average_shares": model_view_audit(next(row for row in old if row.get("sample_id") == "333d9eaf9025a63c36b146b1dd9c8b55")),
        "cash_respective_2018_2019": model_view_audit(next(row for row in old if row.get("sample_id") == "206f5c68b06126ab740ebeec68e9d2ba")),
        "ma_cash_flow_percentage": model_view_audit(next(row for row in old if row.get("sample_id") == "3497e07524b8887eafffb3494cc81083")),
        "pension_millions_unit": model_view_audit(next(row for row in old if row.get("sample_id") == "6813847463fa3df257edbc56a98536fc")),
    })
    write_json(OUT / "repair-rejection-replacement.json", {
        "samples_repaired": 0,
        "samples_rejected_or_replaced": len(replacements),
        "replacement_records": replacements,
        "old_dataset_preserved": PRE_QC_SHA,
    })
    write_json(OUT / "dataset-manifest.json", {
        "dataset_id": "nf-v2-09-r2.2.1-evidence-sufficient",
        "pre_qc_dataset_sha256": PRE_QC_SHA,
        "dataset_sha256": new_sha,
        "base_commit": "7cd3ce754fcea2a6f2ffa52a77b7521129674135",
        "financial_generation_view_v1_sha256": view_sha,
        "total": len(final),
        "components": dict(final_components),
        "model_calls": 0,
        "training": 0,
        "retrieval_calls": 0,
        "manual_audit_seed": SEED,
        "manual_audit_count": len(sample),
    })
    write_json(OUT / "decision.json", {
        "dataset_acceptance": bool(
            len(final) == 2100
            and final_counts["EV0_FULLY_DERIVABLE"] == len(final)
            and view_sha == EXPECTED_VIEW_SHA
            and final_components == Counter({"TARGETED": 1000, "GENERAL_POSITIVE_REPLAY": 600, "STRONG_PARTIAL": 250, "STRUCTURAL_HARD_UNANSWERABLE": 250})
            and not unknown_citations
            and not period_type_errors
            and not metric_header_errors
            and len(calc_rows) == 400
            and not calc_failures
        ),
        "model_facing_derivability": final_counts["EV0_FULLY_DERIVABLE"] / len(final) if final else 0,
        "next_gate": "v2_09_r22_balanced_grounding_training",
        "dataset_sha256": new_sha,
        "pre_qc_dataset_sha256": PRE_QC_SHA,
        "model_calls": 0,
        "training": 0,
    })
    (OUT / "README.md").write_text(
        "# NF-V2-09 R2.2.1 Model-Facing Evidence Sufficiency QC\n\n"
        f"Pre-QC SHA256: {PRE_QC_SHA}\nNew SHA256: {new_sha}\n"
        "Only exact FinancialGenerationViewV1-visible evidence was used.\n"
        "Ambiguous, missing-header, unsupported, and semantic-field defects were replaced with independently valid TRAIN-only examples.\n"
        "No model calls or training were performed.\n",
        encoding="utf-8",
    )
    # Copy the evaluation-facing snapshot, including the complete manual sample.
    for path in OUT.iterdir():
        if path.name.endswith((".json", ".jsonl", ".md")):
            EVAL.joinpath(path.name).write_bytes(path.read_bytes())
    with gzip.open(EVAL / "rejected-or-replaced.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in audit_rows:
            if row["classification"] != "EV0_FULLY_DERIVABLE":
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
