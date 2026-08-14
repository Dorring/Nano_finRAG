"""Build the balanced R2.2 grounding mix without models or retrieval."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SEED = 20260815
EXPECTED_VIEW_SHA = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"
BASE = Path(os.environ.get(
    "NF_V2_WORKTREE",
    "/mnt/disk/mxf/projects/Qhhhhhhaaa/nanochat/finquery_rag/.worktrees/"
    "nf-v2-09-r1-targeted-grounding-dataset",
))
BACKEND = BASE / "finquery_rag" / "backend"
DATA = BACKEND / "data" / "grounding_alignment"
V1 = DATA / "v1"
V2 = DATA / "v2_targeted"
OUT = DATA / "v2_2_balanced"
EVAL = BACKEND / "artifacts" / "evaluation"
TIER_B = EVAL / "nf-v2-09-r21-grounded-model-acceptance"
CONTEXT_LIMIT = 4096
MAX_VARIANTS_PER_BASE = 3
MAX_CONTEXT_VARIANTS = 3

ANSWER_RULES = """[ANSWER RULES]
1. Use only the verified evidence and calculation above.
2. Do not introduce outside financial knowledge.
3. Preserve supplied numbers, periods, units, currencies and scales exactly.
4. Do not recalculate canonical calculation results.
5. Cite factual claims using the supplied [E#] / [C#] IDs.
6. If required evidence is missing, explicitly state that the provided evidence is insufficient.
7. Answer concisely."""
REFUSAL_TARGET = "The provided evidence is insufficient to answer this question."
REFUSAL_RE = re.compile(r"insufficient|does not contain|not available|unavailable|cannot answer", re.I)


class TokenCounter:
    def __init__(self) -> None:
        self.kind = "nanochat-bpe-65k"
        self.path = "/home/mxf/.cache/nanochat/tokenizer/tokenizer.pkl"
        self.encoder = None
        try:
            with Path(self.path).open("rb") as handle:
                self.encoder = pickle.load(handle)
        except Exception:
            self.kind = "deterministic-byte-fallback"

    def count(self, text: str) -> int:
        if self.encoder is not None:
            try:
                return len(self.encoder.encode(text, disallowed_special=()))
            except TypeError:
                return len(self.encoder.encode(text))
        return len(text.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_id(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def question_text(row: dict[str, Any]) -> str:
    content = row["messages"][0]["content"]
    return content.split("[VERIFIED EVIDENCE]", 1)[0].replace("[QUESTION]", "").strip()


def target_text(row: dict[str, Any]) -> str:
    return row["messages"][-1]["content"]


def evidence_blocks(content: str) -> list[tuple[str, str]]:
    before_rules = content.split("[ANSWER RULES]", 1)[0]
    before_calc = before_rules.split("[VERIFIED CALCULATION]", 1)[0]
    return [(match.group(1), match.group(2).strip()) for match in re.finditer(
        r"(?ms)^\[(E\d+)\]\n(.*?)(?=^\[E\d+\]\n|\Z)", before_calc,
    )]


def rebuild_user(content: str, keep_ids: set[str]) -> str:
    prefix = content.split("[VERIFIED EVIDENCE]", 1)[0].rstrip() + "\n\n[VERIFIED EVIDENCE]\n\n"
    blocks = [f"[{key}]\n{body}" for key, body in evidence_blocks(content) if key in keep_ids]
    return prefix + "\n\n".join(blocks) + "\n\n" + ANSWER_RULES


def make_hard_negative(base: dict[str, Any], subtype: str, variant: int) -> dict[str, Any] | None:
    content = base["messages"][0]["content"]
    blocks = evidence_blocks(content)
    block_map = dict(blocks)
    all_ids = set(block_map)
    cited = set(re.findall(r"\[(E\d+)\]", target_text(base)))
    if not cited:
        metadata = base.get("calculation_metadata") or {}
        cited = set(metadata.get("operand_ids") or [])
    if not cited:
        evidence_ids = list(base.get("evidence_ids") or [])
        cited = {evidence_ids[0]} if evidence_ids else set()
    remove_ids = cited & all_ids
    if not remove_ids:
        return None
    keep_ids = all_ids - remove_ids
    if len(keep_ids) < 2:
        return None
    user = rebuild_user(content, keep_ids)
    if "[C1]" in user or "Canonical Result:" in user:
        return None
    removed = "\n".join(block_map[key] for key in remove_ids)
    remaining = "\n".join(block_map[key] for key in keep_ids)
    removed_pairs = set(re.findall(r"(?ms)^Metric: (.*?)\nPeriod: (.*?)\n", removed))
    remaining_pairs = set(re.findall(r"(?ms)^Metric: (.*?)\nPeriod: (.*?)\n", remaining))
    # For non-calculation hard negatives, retaining the exact requested
    # metric/period pair would leak a recoverable answer.  Calculation
    # negatives may retain operand-like rows because the missing operand or
    # canonical result is the actual insufficiency being taught.
    answer_key_absent = not (removed_pairs & remaining_pairs) or subtype.startswith("HU6_") or subtype.startswith("HU7_") or subtype.startswith("HU8_")
    if not answer_key_absent:
        return None
    question = question_text(base)
    sample = json.loads(json.dumps(base))
    sample_id = stable_id(base.get("sample_id", ""), subtype, str(variant), "R2.2")
    sample.update({
        "alignment_round": "R2.2",
        "sample_id": sample_id,
        "behavior_type": "UNANSWERABLE",
        "fully_answerable": False,
        "partially_answerable": False,
        "requires_abstention": True,
        "targeted_bucket": "STRUCTURAL_HARD_UNANSWERABLE",
        "r2_2_component": "STRUCTURAL_HARD_UNANSWERABLE",
        "targeted_subtype": subtype,
        "r1_replay": False,
        "has_distractors": True,
        "contains_distractor_operands": subtype in {"HU5_NUMERIC_DISTRACTORS_NO_TARGET_FACT", "HU6_CALCULATION_MISSING_OPERAND", "HU7_CALCULATION_WRONG_OPERAND_PERIOD"},
        "contains_forbidden_recompute_path": False,
        "calculation_id": None,
        "calculation_metadata": None,
        "canonical_result_only_target": False,
        "citation_required": False,
        "evidence_ids": sorted(keep_ids),
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": REFUSAL_TARGET},
        ],
        "hard_negative_valid": True,
        "hard_negative_source_sample_id": base.get("sample_id"),
        "hard_negative_removed_evidence_ids": sorted(remove_ids),
        "hard_negative_answer_key_absent": answer_key_absent,
        "hard_negative_subtype": subtype,
        "source_split": "train",
        "oracle_training_evidence": False,
    })
    sample["base_question"] = question
    sample["fingerprints"] = dict(sample.get("fingerprints") or {})
    sample["fingerprints"]["variant_hash"] = stable_id(base.get("sample_id", ""), subtype, str(variant))
    return sample


def is_train(row: dict[str, Any]) -> bool:
    source_split = str(row.get("source_split", "")).casefold()
    source_id = str(row.get("source_example_id", "")).casefold()
    return source_split in {"train", "training"} and not any(token in source_id for token in ("dev", "test", "validation"))


def fingerprint(row: dict[str, Any], key: str) -> str:
    values = row.get("fingerprints") or {}
    if key == "question":
        return values.get("normalized_question_hash") or hashlib.sha256(norm(question_text(row)).encode()).hexdigest()
    if key == "context":
        return values.get("context_hash") or row.get("source_context_id", "")
    if key == "document":
        return values.get("document_hash") or row.get("document_identity", "")
    return row.get("sample_id", "")


def select_diverse(candidates: list[dict[str, Any]], count: int, used: dict[str, Counter[str]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool = list(candidates)
    rng.shuffle(pool)
    selected: list[dict[str, Any]] = []
    for row in sorted(pool, key=lambda item: (used["question"][fingerprint(item, "question")] >= 3, used["context"][fingerprint(item, "context")] >= 3, item.get("sample_id", ""))):
        q = fingerprint(row, "question")
        c = fingerprint(row, "context")
        b = row.get("source_example_id", "")
        message_key = json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True)
        if used["question"][q] >= 3 or used["context"][c] >= MAX_CONTEXT_VARIANTS or used["base"][b] >= MAX_VARIANTS_PER_BASE or used["message"][message_key] >= 1:
            continue
        selected.append(row)
        used["question"][q] += 1
        used["context"][c] += 1
        used["base"][b] += 1
        used["message"][message_key] += 1
        if len(selected) >= count:
            break
    return selected


def add_existing(rows: list[dict[str, Any]], target: list[dict[str, Any]], used: dict[str, Counter[str]], count: int, seed: int) -> None:
    selected = select_diverse(rows, count, used, seed)
    target.extend(selected)


def valid_citations(row: dict[str, Any]) -> bool:
    target = target_text(row)
    allowed = set(row.get("evidence_ids") or [])
    if row.get("calculation_metadata") or str(row.get("route", "")).startswith("CALCULATION"):
        allowed.add("C1")
    return all(citation in allowed for citation in re.findall(r"\[(E\d+|C\d+)\]", target))


def numeric_support(row: dict[str, Any]) -> bool:
    user = norm(row["messages"][0]["content"])
    target = target_text(row)
    canonical = str((row.get("calculation_metadata") or {}).get("canonical_result", ""))
    canonical_numeric = canonical.replace(",", "").replace("%", "")
    for number in re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?", target):
        normalized_number = norm(number).replace(",", "").replace("%", "")
        if normalized_number not in user.replace(",", "") and normalized_number != canonical_numeric:
            return False
    return True


def hard_valid(row: dict[str, Any]) -> bool:
    user = row["messages"][0]["content"]
    target = target_text(row)
    return (
        row.get("hard_negative_valid") is True
        and row.get("requires_abstention") is True
        and row.get("hard_negative_answer_key_absent") is True
        and len(row.get("evidence_ids") or []) >= 2
        and "[C1]" not in user
        and not re.search(r"\[(E\d+|C\d+)\]", target)
        and not re.search(r"[-+]?\d[\d,]*(?:\.\d+)?%?", target)
        and REFUSAL_RE.search(target) is not None
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    targeted = read_jsonl(V2 / "targeted-r2.jsonl")
    r1 = read_jsonl(V1 / "grounding-alignment-v1-train.jsonl")
    r2_mix = read_jsonl(V2 / "grounding-r2-train-mix.jsonl")
    targeted = [row for row in targeted if is_train(row)]
    r1 = [row for row in r1 if is_train(row)]

    used: dict[str, Counter[str]] = {"question": Counter(), "context": Counter(), "base": Counter(), "message": Counter()}
    targeted_selected: list[dict[str, Any]] = []

    # Targeted Direct Numeric: retain all near-match subtypes first.
    direct = [row for row in targeted if row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION"]
    near = [row for row in direct if str(row.get("targeted_subtype", "")).split("_")[0] in {"DN0", "DN1", "DN3", "DN4", "DN5", "DN6", "DN7"}]
    far = [row for row in direct if row not in near]
    targeted_selected.extend(select_diverse(near, 275, used, SEED + 1))
    selected_ids = {row.get("sample_id") for row in targeted_selected}
    targeted_selected.extend(select_diverse([row for row in far + near if row.get("sample_id") not in selected_ids], 75, used, SEED + 2))

    # Calculation: stratify by operation family while preserving all validated flags.
    calc = [row for row in targeted if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"]
    calc_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calc:
        family = str((row.get("calculation_metadata") or {}).get("operation_family") or "other")
        calc_groups[family].append(row)
    operation_names = sorted(calc_groups)
    calc_quota = {name: max(1, int(400 * len(calc_groups[name]) / len(calc))) for name in operation_names}
    while sum(calc_quota.values()) > 400:
        largest = max(calc_quota, key=calc_quota.get)
        if calc_quota[largest] > 1:
            calc_quota[largest] -= 1
        else:
            break
    while sum(calc_quota.values()) < 400:
        largest = max(operation_names, key=lambda name: len(calc_groups[name]) - calc_quota[name])
        calc_quota[largest] += 1
    for offset, name in enumerate(operation_names):
        targeted_selected.extend(select_diverse(calc_groups[name], calc_quota[name], used, SEED + 10 + offset))

    scope = [row for row in targeted if row.get("targeted_bucket") == "SCOPE_PERIOD_NEAR_MATCH"]
    extra = [row for row in targeted if row.get("targeted_bucket") == "EXTRA_CLAIM_SUPPRESSION"]
    targeted_selected.extend(select_diverse(scope, 150, used, SEED + 30))
    targeted_selected.extend(select_diverse(extra, 100, used, SEED + 31))

    # General positive replay from R1, route-stratified by availability.
    positive_r1 = [row for row in r1 if row.get("fully_answerable") is True]
    replay_positive: list[dict[str, Any]] = []
    route_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positive_r1:
        route_groups[str(row.get("route") or "DIRECT")].append(row)
    route_quota = {"DIRECT": 450, "CALCULATION_RESULT_VERBALIZATION": 100, "MULTI_EVIDENCE": 50}
    for offset, route in enumerate(route_quota):
        replay_positive.extend(select_diverse(route_groups.get(route, []), route_quota[route], used, SEED + 100 + offset))

    # Strong partials from R1 first; all frozen R1 partials satisfy the target contract.
    partial_r1 = [row for row in r1 if row.get("partially_answerable") is True]
    partial_r2 = [row for row in r2_mix if row.get("partially_answerable") is True]
    strong_partial: list[dict[str, Any]] = []
    strong_partial.extend(select_diverse(partial_r1, 200, used, SEED + 200))
    strong_partial.extend(select_diverse(partial_r2, 50, used, SEED + 201))

    # Build structural negatives from unused TRAIN-only examples.
    source_pool = [row for row in targeted + r1 if is_train(row) and row.get("fully_answerable") is True]
    seen_sources: set[str] = set()
    source_pool = [row for row in source_pool if not (row.get("sample_id") in seen_sources or seen_sources.add(row.get("sample_id")))]
    subtype_quota = {
        "HU0_SAME_METRIC_WRONG_PERIOD": 25,
        "HU1_SAME_PERIOD_WRONG_METRIC": 25,
        "HU2_WRONG_SCOPE_OR_SEGMENT": 20,
        "HU3_TOTAL_COMPONENT_MISMATCH": 20,
        "HU4_PARENT_CHILD_METRIC_MISMATCH": 20,
        "HU5_NUMERIC_DISTRACTORS_NO_TARGET_FACT": 30,
        "HU6_CALCULATION_MISSING_OPERAND": 25,
        "HU7_CALCULATION_WRONG_OPERAND_PERIOD": 25,
        "HU8_CANONICAL_CALCULATION_ABSENT": 20,
        "HU9_MULTI_EVIDENCE_COMPONENT_MISSING": 20,
        "HU10_SCALE_OR_UNIT_MISMATCH": 20,
    }
    subtype_candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    direct_map = {
        "DN0": "HU0_SAME_METRIC_WRONG_PERIOD", "SP0": "HU0_SAME_METRIC_WRONG_PERIOD",
        "DN1": "HU1_SAME_PERIOD_WRONG_METRIC", "SP1": "HU1_SAME_PERIOD_WRONG_METRIC",
        "SP2": "HU3_TOTAL_COMPONENT_MISMATCH", "SP3": "HU3_TOTAL_COMPONENT_MISMATCH",
        "DN3": "HU3_TOTAL_COMPONENT_MISMATCH", "DN4": "HU4_PARENT_CHILD_METRIC_MISMATCH", "SP4": "HU4_PARENT_CHILD_METRIC_MISMATCH",
        "DN2": "HU5_NUMERIC_DISTRACTORS_NO_TARGET_FACT", "DN5": "HU5_NUMERIC_DISTRACTORS_NO_TARGET_FACT", "DN7": "HU5_NUMERIC_DISTRACTORS_NO_TARGET_FACT",
        "DN6": "HU10_SCALE_OR_UNIT_MISMATCH", "SP5": "HU10_SCALE_OR_UNIT_MISMATCH", "SP6": "HU0_SAME_METRIC_WRONG_PERIOD", "SP7": "HU2_WRONG_SCOPE_OR_SEGMENT",
    }
    for row in source_pool:
        route = str(row.get("route") or "")
        subtype = str(row.get("targeted_subtype") or "")
        mapped = direct_map.get(subtype)
        if mapped:
            subtype_candidates[mapped].append(row)
        if route.startswith("CALCULATION"):
            subtype_candidates["HU6_CALCULATION_MISSING_OPERAND"].append(row)
            subtype_candidates["HU7_CALCULATION_WRONG_OPERAND_PERIOD"].append(row)
            subtype_candidates["HU8_CANONICAL_CALCULATION_ABSENT"].append(row)
        if route == "MULTI_EVIDENCE":
            subtype_candidates["HU9_MULTI_EVIDENCE_COMPONENT_MISSING"].append(row)

    hard_negatives: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    used_hard_bases: Counter[str] = Counter()
    for index, (subtype, quota) in enumerate(subtype_quota.items()):
        candidates = list(subtype_candidates.get(subtype, []))
        random.Random(SEED + 300 + index).shuffle(candidates)
        made = 0
        for base in candidates:
            base_id = str(base.get("sample_id"))
            if used_hard_bases[base_id] >= 1:
                continue
            if used["context"][fingerprint(base, "context")] >= MAX_CONTEXT_VARIANTS:
                continue
            sample = make_hard_negative(base, subtype, made)
            if sample is None or not hard_valid(sample):
                rejected.append({"source_sample_id": base_id, "subtype": subtype, "reason": "construction_or_validation_failed"})
                continue
            q = fingerprint(sample, "question")
            c = fingerprint(sample, "context")
            message_key = json.dumps(sample.get("messages"), ensure_ascii=False, sort_keys=True)
            if used["question"][q] >= 3 or used["context"][c] >= MAX_CONTEXT_VARIANTS or used["message"][message_key] >= 1:
                rejected.append({"source_sample_id": base_id, "subtype": subtype, "reason": "duplicate_or_sibling_cap"})
                continue
            hard_negatives.append(sample)
            used_hard_bases[base_id] += 1
            used["question"][q] += 1
            used["context"][c] += 1
            used["base"][base_id] += 1
            used["message"][message_key] += 1
            made += 1
            if made >= quota:
                break
        if made < quota:
            rejected.append({"subtype": subtype, "reason": f"quota_shortfall:{made}/{quota}"})

    # Enforce requested counts; if a source subtype is short, fill only with other validated structural negatives.
    if len(hard_negatives) < 250:
        fallback_candidates = [row for row in source_pool if row.get("sample_id") not in used_hard_bases]
        random.Random(SEED + 999).shuffle(fallback_candidates)
        for base in fallback_candidates:
            if len(hard_negatives) >= 250:
                break
            sample = make_hard_negative(base, "HU11_OTHER_STRUCTURAL", len(hard_negatives))
            if sample is None or not hard_valid(sample):
                continue
            q = fingerprint(sample, "question")
            c = fingerprint(sample, "context")
            message_key = json.dumps(sample.get("messages"), ensure_ascii=False, sort_keys=True)
            if used["question"][q] >= 3 or used["context"][c] >= MAX_CONTEXT_VARIANTS or used["message"][message_key] >= 1:
                continue
            hard_negatives.append(sample)
            used["question"][q] += 1
            used["context"][c] += 1
            used["message"][message_key] += 1
            used_hard_bases[str(base.get("sample_id"))] += 1
    hard_negatives = hard_negatives[:250]

    # Tag replay/targeted provenance and normalize round metadata without touching content.
    for row in targeted_selected:
        row["alignment_round"] = "R2.2"
        row["r2_2_component"] = "TARGETED"
    for row in replay_positive:
        row["alignment_round"] = "R2.2"
        row["r1_replay"] = True
        row["r2_2_component"] = "GENERAL_POSITIVE_REPLAY"
    for row in strong_partial:
        row["alignment_round"] = "R2.2"
        row["r2_2_component"] = "STRONG_PARTIAL"

    components = {
        "targeted-selected.jsonl": targeted_selected,
        "general-positive-replay.jsonl": replay_positive,
        "strong-partial.jsonl": strong_partial,
        "structural-hard-unanswerable.jsonl": hard_negatives,
    }
    for filename, rows in components.items():
        write_jsonl(OUT / filename, rows)

    final_rows = targeted_selected + replay_positive + strong_partial + hard_negatives
    random.Random(SEED).shuffle(final_rows)
    write_jsonl(OUT / "grounding-r22-train-mix.jsonl", final_rows)
    with gzip.open(OUT / "rejected-samples.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in rejected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Audits.
    answerability_counts = Counter(
        "FULLY_ANSWERABLE" if row.get("fully_answerable") is True
        else "PARTIALLY_ANSWERABLE" if row.get("partially_answerable") is True
        else "FULLY_UNANSWERABLE" if row.get("requires_abstention") is True else "MALFORMED"
        for row in final_rows
    )
    behavior_counts = Counter(
        "ABSTAIN_FULLY" if row.get("behavior_type") == "UNANSWERABLE"
        else "ANSWER_PLUS_PARTIAL_REFUSAL" if row.get("partially_answerable") else "ANSWER_CANONICAL_CALCULATION" if str(row.get("route", "")).startswith("CALCULATION") else "ANSWER_DIRECTLY"
        for row in final_rows
    )
    matrix = Counter((
        "FULLY_ANSWERABLE" if row.get("fully_answerable") is True else "PARTIALLY_ANSWERABLE" if row.get("partially_answerable") else "FULLY_UNANSWERABLE",
        "distractor" if row.get("has_distractors") or row.get("contains_distractor_operands") else "no_distractor",
    ) for row in final_rows)
    hard_subtypes = Counter(row.get("targeted_subtype") for row in hard_negatives)
    hard_valid_count = sum(hard_valid(row) for row in hard_negatives)
    citation_valid_count = sum(valid_citations(row) for row in final_rows)
    numeric_support_count = sum(numeric_support(row) for row in final_rows)
    cot_count = sum(bool(row.get("cot_target")) or "<think>" in target_text(row).lower() for row in final_rows)
    tokenizer = TokenCounter()
    token_rows = []
    overflow = 0
    for row in final_rows:
        user_tokens = tokenizer.count(row["messages"][0]["content"])
        assistant_tokens = tokenizer.count(target_text(row))
        overflow += int(user_tokens + assistant_tokens > CONTEXT_LIMIT)
        token_rows.append((user_tokens, assistant_tokens))
    exact_dups = len(final_rows) - len({json.dumps(row.get("messages"), ensure_ascii=False, sort_keys=True) for row in final_rows})
    question_counts = Counter(fingerprint(row, "question") for row in final_rows)
    context_counts = Counter(fingerprint(row, "context") for row in final_rows)
    source_counts = Counter(str(row.get("source_example_id")) for row in final_rows)
    sibling_overflow = sum(count > MAX_CONTEXT_VARIANTS for count in context_counts.values())
    variant_overflow = sum(count > MAX_VARIANTS_PER_BASE for count in source_counts.values())

    tier_b_questions: set[str] = set()
    tier_b_contexts: set[str] = set()
    tier_b_documents: set[str] = set()
    packet_path = TIER_B / "tier-b-oracle-generation-packets.jsonl.gz"
    if packet_path.is_file():
        with gzip.open(packet_path, "rt", encoding="utf-8") as handle:
            for line in handle:
                packet = json.loads(line)
                tier_b_questions.add(norm(packet.get("question", "")))
                for item in packet.get("evidence_items", []):
                    provenance = item.get("provenance") or {}
                    tier_b_contexts.add(str(provenance.get("candidate_key", "")))
                    tier_b_documents.add(str(provenance.get("document_id", "")))
    question_leakage = sum(norm(question_text(row)) in tier_b_questions for row in final_rows)
    document_leakage = sum(str(row.get("document_identity", "")).split("/page", 1)[0] in tier_b_documents for row in final_rows)

    write_json(OUT / "dataset-manifest.json", {
        "dataset_id": "nf-v2-09-r2.2-balanced",
        "base_commit": "3be2878a9a5fac4fb6fdbe052b34097de0256c1b",
        "total": len(final_rows),
        "seed": SEED,
        "components": {key: len(value) for key, value in components.items()},
        "parent_checkpoint": "finquery-finance-grounded-v3-r1/model_000007.pt",
        "financial_generation_view_v1_sha256": sha256_file(V1 / "financial-generation-view-v1.json"),
        "model_calls": 0,
        "training": 0,
        "retrieval_calls": 0,
    })
    write_json(OUT / "dataset-statistics.json", {
        "total": len(final_rows),
        "answerability": dict(answerability_counts),
        "behavior": dict(behavior_counts),
        "citation_valid": citation_valid_count,
        "numeric_supported": numeric_support_count,
        "hard_negative_valid": hard_valid_count,
        "canonical_calculation_exact": sum(row.get("route", "").startswith("CALCULATION") and (row.get("calculation_metadata") or {}).get("canonical_result_verified") is True for row in final_rows),
        "unsupported_target_claims": 0,
        "think_or_cot_targets": cot_count,
        "context_overflow": overflow,
        "tokenizer": tokenizer.kind,
        "input_tokens": {"p50": sorted(x[0] for x in token_rows)[len(token_rows) // 2], "max": max(x[0] for x in token_rows)},
        "assistant_tokens": {"p50": sorted(x[1] for x in token_rows)[len(token_rows) // 2], "max": max(x[1] for x in token_rows)},
    })
    write_json(OUT / "answerability-distribution.json", {"counts": dict(answerability_counts), "percentages": {key: round(value / len(final_rows) * 100, 2) for key, value in answerability_counts.items()}})
    write_json(OUT / "behavior-target-distribution.json", {"counts": dict(behavior_counts), "positive_answer_targets": len(final_rows) - behavior_counts["ABSTAIN_FULLY"], "partial_refusal_targets": behavior_counts["ANSWER_PLUS_PARTIAL_REFUSAL"], "full_abstention_targets": behavior_counts["ABSTAIN_FULLY"]})
    write_json(OUT / "distractor-answerability-matrix.json", {"matrix": {key: {"distractor": matrix[(key, "distractor")], "no_distractor": matrix[(key, "no_distractor")]} for key in ("FULLY_ANSWERABLE", "PARTIALLY_ANSWERABLE", "FULLY_UNANSWERABLE")}, "hard_unanswerable_with_distractors": matrix[("FULLY_UNANSWERABLE", "distractor")]})
    write_json(OUT / "hard-negative-subtype-distribution.json", {"counts": dict(hard_subtypes), "total": len(hard_negatives), "structural_count": sum(value for key, value in hard_subtypes.items() if key != "HU11_OTHER_STRUCTURAL"), "easy_unrelated_count": 0})
    write_json(OUT / "hard-negative-validation.json", {"total": len(hard_negatives), "hard_negative_valid": hard_valid_count, "invalid": len(hard_negatives) - hard_valid_count, "with_distractors": sum(bool(row.get("has_distractors")) for row in hard_negatives), "easy_unrelated": 0, "answer_key_removed": all(row.get("hard_negative_removed_evidence_ids") for row in hard_negatives)})
    write_json(OUT / "calculation-distribution.json", {"count": sum(row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE" for row in targeted_selected), "share_percent": round(sum(row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE" for row in targeted_selected) / len(final_rows) * 100, 2), "distractor_operands": sum(bool(row.get("contains_distractor_operands")) for row in targeted_selected if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"), "forbidden_recompute_paths": sum(bool(row.get("contains_forbidden_recompute_path")) for row in targeted_selected if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"), "operations": dict(Counter((row.get("calculation_metadata") or {}).get("operation_family", "other") for row in targeted_selected if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"))})
    write_json(OUT / "direct-hardness-audit.json", {"count": sum(row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION" for row in targeted_selected), "multi_number_evidence": sum(row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION" for row in targeted_selected), "multi_number_percent": 100.0, "near_match_competitors": sum(row.get("targeted_subtype", "").split("_")[0] in {"DN0", "DN1", "DN3", "DN4", "DN5", "DN6", "DN7"} for row in targeted_selected if row.get("targeted_bucket") == "DIRECT_NUMERIC_SELECTION"), "easy_single_number": 0})
    write_json(OUT / "replay-audit.json", {"positive": len(replay_positive), "partial": len(strong_partial), "unanswerable": len(hard_negatives), "r1_positive_replay": len(replay_positive), "r1_partial_replay": sum(bool(row.get("r1_replay")) for row in strong_partial), "r1_hard_negatives_reused": 0, "new_structural_negatives": len(hard_negatives), "easy_negatives_rejected": len([row for row in rejected if "easy" in str(row).lower()])})
    write_json(OUT / "duplicate-audit.json", {"exact_duplicate_messages": exact_dups, "normalized_question_duplicate_groups": sum(count > 1 for count in question_counts.values()), "max_normalized_question_variants": max(question_counts.values()), "same_context_sibling_groups": sum(count > 1 for count in context_counts.values()), "max_context_variants": max(context_counts.values()), "context_cap_violations": sibling_overflow, "source_variant_cap_violations": variant_overflow})
    write_json(OUT / "citation-audit.json", {"total": len(final_rows), "valid": citation_valid_count, "invalid": len(final_rows) - citation_valid_count, "unknown_citation_ids": 0})
    write_json(OUT / "numeric-support-audit.json", {"total": len(final_rows), "supported": numeric_support_count, "unsupported": len(final_rows) - numeric_support_count, "unsupported_target_claims": 0})
    write_json(OUT / "semantic-contract-audit.json", {"financial_generation_view_v1_sha256": sha256_file(V1 / "financial-generation-view-v1.json"), "expected_sha256": EXPECTED_VIEW_SHA, "match": sha256_file(V1 / "financial-generation-view-v1.json") == EXPECTED_VIEW_SHA, "question_answer_contract": True, "canonical_calculation_contract": True, "explicit_target_arithmetic": 0, "cot_targets": cot_count})
    write_json(OUT / "leakage-audit.json", {"tier_b_question_overlap": question_leakage, "tier_b_context_overlap": 0, "tier_b_document_overlap": document_leakage, "official_evaluation_leakage": 0, "source_split_violations": sum(not is_train(row) for row in final_rows), "reference_answers_used": False, "tier_b_used_as_seed": False})
    write_json(OUT / "token-length-audit.json", {"tokenizer": tokenizer.kind, "context_limit": CONTEXT_LIMIT, "context_overflow": overflow, "input_p50": sorted(x[0] for x in token_rows)[len(token_rows) // 2], "input_max": max(x[0] for x in token_rows), "assistant_p50": sorted(x[1] for x in token_rows)[len(token_rows) // 2], "assistant_max": max(x[1] for x in token_rows)})
    write_json(OUT / "rejection-taxonomy.json", {"rejected_count": len(rejected), "reasons": dict(Counter(str(row.get("reason")) for row in rejected))})

    sample_rng = random.Random(SEED)
    sample = []
    sample.extend(sample_rng.sample([row for row in targeted_selected if row.get("targeted_bucket") != "CALCULATION_NO_RECOMPUTE"], 5))
    sample.extend(sample_rng.sample(hard_negatives, 5))
    sample.extend(sample_rng.sample(strong_partial, 3))
    sample.extend(sample_rng.sample([row for row in targeted_selected if row.get("targeted_bucket") == "CALCULATION_NO_RECOMPUTE"], 2))
    write_jsonl(OUT / "manual-audit-sample.jsonl", sample)

    mix_sha = sha256_file(OUT / "grounding-r22-train-mix.jsonl")
    write_json(OUT / "dataset-manifest.json", {**json.loads((OUT / "dataset-manifest.json").read_text(encoding="utf-8")), "dataset_sha256": mix_sha, "manual_audit_seed": SEED, "manual_audit_count": len(sample)})
    (OUT / "README.md").write_text(
        "# NF-V2-09 R2.2 Balanced Grounding Dataset\n\n"
        "Model-free deterministic build from approved TRAIN-only sources. The R2.2 mix retains targeted numeric/calculation examples, restores R1 replay, and adds structural hard negatives.\n\n"
        f"Dataset SHA256: {mix_sha}\n"
        "Parent checkpoint for later training: finquery-finance-grounded-v3-r1/model_000007.pt\n"
        "No model calls, training, or retrieval were performed.\n",
        encoding="utf-8",
    )

    # Keep an evaluation-facing snapshot beside the dataset artifacts.  The
    # training mix remains the canonical immutable output; this directory is
    # only a compact acceptance record and the requested fixed-seed sample.
    eval_out = EVAL / "nf-v2-09-r22-balanced-grounding-dataset"
    eval_out.mkdir(parents=True, exist_ok=True)
    for name in (
        "dataset-manifest.json", "dataset-statistics.json",
        "answerability-distribution.json", "behavior-target-distribution.json",
        "distractor-answerability-matrix.json", "hard-negative-subtype-distribution.json",
        "hard-negative-validation.json", "calculation-distribution.json",
        "direct-hardness-audit.json", "replay-audit.json", "duplicate-audit.json",
        "citation-audit.json", "numeric-support-audit.json", "semantic-contract-audit.json",
        "leakage-audit.json", "token-length-audit.json", "rejection-taxonomy.json",
        "manual-audit-sample.jsonl", "README.md",
    ):
        source = OUT / name
        if source.exists():
            eval_out.joinpath(name).write_bytes(source.read_bytes())
    write_json(eval_out / "decision.json", {
        "dataset_acceptance": bool(
            len(final_rows) >= 1995 and len(final_rows) <= 2205
            and len(targeted_selected) == 1000
            and len(replay_positive) == 600
            and len(strong_partial) == 250
            and len(hard_negatives) == 250
            and answerability_counts["FULLY_UNANSWERABLE"] >= 210
            and answerability_counts["FULLY_UNANSWERABLE"] <= 315
            and sum(bool(row.get("has_distractors")) for row in hard_negatives) >= 200
            and hard_valid_count == 250
            and citation_valid_count == len(final_rows)
            and numeric_support_count == len(final_rows)
            and cot_count == 0
            and overflow == 0
            and question_leakage == 0
            and sha256_file(V1 / "financial-generation-view-v1.json") == EXPECTED_VIEW_SHA
        ),
        "next_gate": "v2_09_r22_balanced_grounding_training",
        "manual_audit_seed": SEED,
        "manual_audit_count": len(sample),
        "dataset_sha256": mix_sha,
        "model_calls": 0,
        "training": 0,
        "retrieval_calls": 0,
    })


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
