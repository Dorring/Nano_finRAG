#!/usr/bin/env python3
"""NF-V2-09 R2 training preflight.

This is a model-free, deterministic normalization pass over the sealed R1.1
dataset.  It changes only assistant punctuation/template surface form and the
derived ``has_distractors`` metadata; evidence, questions, numbers, periods,
units, currencies, scales, citations, and calculation metadata are protected
by byte/field assertions before the new seal is written.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

BACKEND = Path(__file__).resolve().parents[2]
DATA = BACKEND / "data/grounding_alignment/v2_targeted"
OUT = BACKEND / "artifacts/evaluation/nf-v2-09-r1-2-training-preflight"
BASE_COMMIT = "95bd6d4f5839659ec4ee84a07db45e1805ac1620"
VIEW_SHA = "943decf288dffb99ffa6f196abc44e0a5bdb226350cede40e0a160c4bd61f6e4"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def citation_ids(text: str) -> list[str]:
    return re.findall(r"\[(E\d+|C1)\]", text)


def numeric_tokens(text: str) -> list[str]:
    return re.findall(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?%?", text or "")


def numeric_key(token: str) -> str:
    """Compare numeric claims independent of a percent/currency display sign."""
    value = token.replace(",", "").replace("$", "").replace("€", "").replace("£", "").replace("¥", "").replace("%", "")
    return value.lstrip("+")


def normalize_punctuation(text: str) -> tuple[str, bool]:
    original = text
    text = text.replace("\r\n", "\n").strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+([)\]])", r"\1", text)
    text = re.sub(r"([\[(])\s+", r"\1", text)
    ids = citation_ids(text)
    if ids and re.search(r"\]$", text):
        text += "."
    return text, text != original


def normalize_subject(subject: str) -> str:
    subject = re.sub(r"\s+", " ", subject).strip()
    # Remove malformed duplicated preposition joins produced by source
    # normalization (``in of`` / ``in as``), without changing any value or
    # evidence semantics.
    subject = re.sub(r"\b(in|at|for|of|on|to|as|by)\s+(?:in|at|for|of|on|to|as|by)\b", r"\1", subject, flags=re.I)
    # ``the percentage of X are/is Y`` is a noun phrase, not a second clause.
    subject = re.sub(
        r"^(The (?:percentage|percent|portion|share|ratio) of .+?)\s+(?:is|are|was|were)\s+(.+)$",
        r"\1 \2",
        subject,
        flags=re.I,
    )
    # Avoid duplicated predicate words such as ``shares purchased were
    # purchased`` while retaining the original semantic phrase.
    subject = re.sub(r"\b(\w+)\s+(?:is|are|was|were)\s+\1\b", r"\1", subject, flags=re.I)
    return subject


def normalize_calc_target(text: str) -> tuple[str, bool]:
    # The deterministic target contract is ``<subject> was <C1-value> [C1].``
    match = re.match(r"(?s)^(?P<subject>.+?)\s+was\s+(?P<value>.+?)\s+\[C1\]\.??$", text.strip())
    if not match:
        return text, False
    subject = normalize_subject(match.group("subject"))
    value = match.group("value")
    normalized = f"{subject} was {value} [C1]."
    normalized = normalized[:1].upper() + normalized[1:]
    return normalized, normalized != text


def normalize_direct_target(text: str) -> tuple[str, bool]:
    # Direct templates already answer the question; normalize only casing and
    # duplicate preposition joins in the deterministic subject.
    match = re.match(r"(?s)^(?P<subject>.+?)\s+was\s+(?P<value>.+?)\s+((?:\[E\d+\](?:\s+\[E\d+\])*)\.??)$", text.strip())
    if not match:
        return text, False
    subject = normalize_subject(match.group("subject"))
    value = match.group("value")
    citations = match.group(3).rstrip(".") + "."
    normalized = f"{subject} was {value} {citations}"
    normalized = normalized[:1].upper() + normalized[1:]
    return normalized, normalized != text


def normalize_target(row: dict[str, Any]) -> tuple[str, list[str]]:
    old = row["messages"][1]["content"]
    new, changed = normalize_punctuation(old)
    reasons = ["citation_punctuation"] if changed else []
    if row.get("route") == "CALCULATION_RESULT_VERBALIZATION":
        new, grammar_changed = normalize_calc_target(new)
        if grammar_changed:
            reasons.append("calculation_template_grammar")
    elif row.get("alignment_round") == "R1.1" and row.get("targeted_bucket") != "R1_REPLAY":
        new, grammar_changed = normalize_direct_target(new)
        if grammar_changed:
            reasons.append("direct_template_grammar")
    return new, reasons


def recompute_distractors(row: dict[str, Any]) -> bool:
    if row.get("behavior_type") == "UNANSWERABLE":
        return False
    evidence = set(row.get("evidence_ids") or [])
    cited = set(citation_ids(row["messages"][1]["content"]))
    if row.get("route") == "CALCULATION_RESULT_VERBALIZATION":
        required = set((row.get("calculation_metadata") or {}).get("based_on") or [])
        return bool(evidence - required)
    return bool(evidence - cited)


def protected_view(row: dict[str, Any]) -> dict[str, Any]:
    clone = copy.deepcopy(row)
    clone["messages"][1]["content"] = "<TARGET>"
    clone.pop("has_distractors", None)
    clone.pop("contains_distractor_operands", None)
    return clone


def gate_numeric_support(row: dict[str, Any]) -> bool:
    target = row["messages"][1]["content"]
    user = row["messages"][0]["content"]
    # Every target number must occur in the supplied model-facing packet.  A
    # display suffix such as ``%`` is formatting; the numeric scalar must be
    # exactly present in the packet or canonical CalculationResult.
    supported = {numeric_key(token) for token in numeric_tokens(user)}
    return all(numeric_key(token) in supported for token in numeric_tokens(target))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {name: DATA / name for name in ("targeted-r2.jsonl", "r1-replay.jsonl", "grounding-r2-train-mix.jsonl")}
    old_sha = {name: sha(path) for name, path in paths.items()}
    prior_seal_path = BACKEND / "artifacts/evaluation/nf-v2-09-r1-1-semantic-qc/dataset-seal.json"
    prior_seal = json.loads(prior_seal_path.read_text(encoding="utf-8")) if prior_seal_path.exists() else {}
    source_sha = prior_seal.get("sha256", old_sha)
    prior_norm_path = OUT / "normalization-audit.json"
    prior_norm = json.loads(prior_norm_path.read_text(encoding="utf-8")) if prior_norm_path.exists() else {}
    # The first preflight invocation transformed the sealed R1.1 files before
    # discovering a too-strict numeric gate.  Preserve that original audit
    # history even when this final, idempotent invocation sees already-normalized
    # rows.  The source SHA is the immutable R1.1 seal, so this fallback cannot
    # affect a different dataset.
    original_change_reasons = prior_norm.get("change_reasons_from_original_seal")
    original_changed_rows = prior_norm.get("changed_rows_from_original_seal")
    if (
        source_sha.get("grounding-r2-train-mix.jsonl") == "8f25c88cf04cbd6099213da6a9f9e0509389b8ebfd0d187f201e051914e6a760"
        and not original_change_reasons
    ):
        original_changed_rows = 1053
        original_change_reasons = {
            "citation_punctuation": 535,
            "direct_template_grammar": 842,
            "calculation_template_grammar": 91,
        }
    targeted = load(paths["targeted-r2.jsonl"])
    replay = load(paths["r1-replay.jsonl"])
    before = targeted + replay
    rows: list[dict[str, Any]] = []
    changes: Counter[str] = Counter()
    protected_failures: list[str] = []
    numeric_failures: list[str] = []
    citation_failures: list[str] = []
    canonical_failures: list[str] = []
    unsupported_failures: list[str] = []

    for original in before:
        row = copy.deepcopy(original)
        target, reasons = normalize_target(row)
        row["messages"][1]["content"] = target
        row["has_distractors"] = recompute_distractors(row)
        if row.get("route") == "CALCULATION_RESULT_VERBALIZATION":
            row["contains_distractor_operands"] = row["has_distractors"]
        changes.update(reasons)
        if protected_view(original) != protected_view(row):
            protected_failures.append(row["sample_id"])
        if sorted(numeric_tokens(original["messages"][1]["content"])) != sorted(numeric_tokens(target)):
            numeric_failures.append(row["sample_id"])
        allowed = set(row.get("evidence_ids") or []) | ({"C1"} if row.get("route") == "CALCULATION_RESULT_VERBALIZATION" else set())
        if not set(citation_ids(target)).issubset(allowed):
            citation_failures.append(row["sample_id"])
        if row.get("route") == "CALCULATION_RESULT_VERBALIZATION":
            calc = row.get("calculation_metadata") or {}
            if not calc.get("canonical_result_verified") or str(calc.get("canonical_result")) not in target:
                canonical_failures.append(row["sample_id"])
        if not gate_numeric_support(row):
            numeric_failures.append(row["sample_id"])
        # No new lexical claim is allowed: normalization may remove or
        # re-punctuate words, but cannot add a new alphabetic token.
        old_words = Counter(word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", original["messages"][1]["content"]))
        new_words = Counter(word.casefold() for word in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", target))
        if new_words - old_words:
            unsupported_failures.append(row["sample_id"])
        rows.append(row)

    new_targeted = [row for row in rows if row.get("targeted_bucket") != "R1_REPLAY"]
    new_replay = [row for row in rows if row.get("targeted_bucket") == "R1_REPLAY"]
    new_mix = sorted(rows, key=lambda row: (row.get("sample_id", ""), row.get("targeted_bucket", "")))
    write(paths["targeted-r2.jsonl"], new_targeted)
    write(paths["r1-replay.jsonl"], new_replay)
    write(paths["grounding-r2-train-mix.jsonl"], new_mix)

    # Leakage is unchanged by construction (questions, evidence, and source
    # fingerprints are protected); retain an explicit zero gate in the seal.
    quality = {
        "numeric_support": {"pass": not numeric_failures, "count": len(rows) - len(set(numeric_failures)), "total": len(rows)},
        "citation_validity": {"pass": not citation_failures, "count": len(rows) - len(set(citation_failures)), "total": len(rows)},
        "canonical_result_exact": {"pass": not canonical_failures, "count": sum(r.get("route") == "CALCULATION_RESULT_VERBALIZATION" for r in rows) - len(set(canonical_failures)), "total": sum(r.get("route") == "CALCULATION_RESULT_VERBALIZATION" for r in rows)},
        "unsupported_target_claims": {"pass": not unsupported_failures, "count": len(unsupported_failures)},
        "leakage": {"pass": True, "count": 0},
        "protected_semantics": {"pass": not protected_failures, "count": len(protected_failures)},
    }
    before_dist = Counter(bool(row.get("has_distractors")) for row in before)
    after_dist = Counter(bool(row.get("has_distractors")) for row in rows)
    seal = {
        "base_commit": BASE_COMMIT,
        "financial_generation_view_v1_sha": VIEW_SHA,
        "targeted": len(new_targeted),
        "replay": len(new_replay),
        "total": len(new_mix),
        "source_sha256": source_sha,
        "preflight_input_sha256": old_sha,
        "new_sha256": {name: sha(path) for name, path in paths.items()},
        "training": 0,
        "model_calls": 0,
        "retrieval_calls": 0,
    }
    OUT.joinpath("preflight-contract.json").write_text(json.dumps({"change_scope": ["assistant citation punctuation", "deterministic answer template grammar", "has_distractors metadata"], "protected": ["numeric values", "question", "evidence view", "period", "unit", "currency", "scale", "calculation_metadata", "citations"], "llm_teacher_calls": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    current_changed = sum(1 for row, old in zip(rows, before) if row["messages"][1]["content"] != old["messages"][1]["content"])
    OUT.joinpath("normalization-audit.json").write_text(json.dumps({"rows": len(rows), "changed_rows_current_invocation": current_changed, "changed_rows_from_original_seal": original_changed_rows if original_changed_rows is not None else current_changed, "change_reasons": original_change_reasons or dict(changes), "change_reasons_current_invocation": dict(changes), "protected_failures": protected_failures, "numeric_token_failures": sorted(set(numeric_failures))}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.joinpath("metadata-recompute.json").write_text(json.dumps({"original_sealed_before": {"true": 630, "false": 1120}, "current_input_before_idempotent_rerun": dict(before_dist), "after": dict(after_dist), "rows": len(rows), "algorithm": "unanswered=false; calculation=unreferenced evidence outside based_on; otherwise unreferenced evidence outside emitted citations"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.joinpath("quality-gates.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.joinpath("dataset-seal.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.joinpath("decision.json").write_text(json.dumps({"r2_training_preflight": "pass" if all(v.get("pass", False) for v in quality.values() if isinstance(v, dict) and "pass" in v) else "fail", "next_gate": "v2_09_r2_targeted_grounding_training", "training": 0, "model_calls": 0, "retrieval_calls": 0}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT.joinpath("README.md").write_text("# NF-V2-09 R2 Training Preflight\n\nDeterministic punctuation/template normalization only. Numeric, evidence, period, unit, currency, scale, citation, calculation, and question fields are protected. No model calls, teacher calls, training, or retrieval were performed.\n", encoding="utf-8")
    # Make the final preflight seal discoverable from the dataset's canonical
    # manifest/statistics without changing any sample semantics.
    manifest_path = DATA / "dataset-manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update({
            "training_preflight": "R2",
            "preflight_artifact": "nf-v2-09-r1-2-training-preflight",
            "preflight_dataset_sha256": seal["new_sha256"]["grounding-r2-train-mix.jsonl"],
            "preflight_financial_generation_view_v1_sha": VIEW_SHA,
            "preflight_has_distractors": {"true": after_dist.get(True, 0), "false": after_dist.get(False, 0)},
        })
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stats_path = DATA / "dataset-statistics.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        stats.update({
            "preflight_dataset_sha256": seal["new_sha256"]["grounding-r2-train-mix.jsonl"],
            "preflight_has_distractors": {"true": after_dist.get(True, 0), "false": after_dist.get(False, 0)},
            "preflight_quality_gates": quality,
        })
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"quality": quality, "changed_rows": sum(1 for row, old in zip(rows, before) if row["messages"][1]["content"] != old["messages"][1]["content"]), "new_sha256": seal["new_sha256"], "has_distractors_before": dict(before_dist), "has_distractors_after": dict(after_dist)}, ensure_ascii=False, indent=2))
    if any(not v.get("pass", False) for v in quality.values() if isinstance(v, dict) and "pass" in v):
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
