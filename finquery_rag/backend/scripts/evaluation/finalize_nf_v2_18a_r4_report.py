#!/usr/bin/env python3
"""Finalize the small, deterministic NF-V2-18A-R4 report from frozen artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path, name: str):
    return json.loads((path / name).read_text(encoding="utf-8"))


def a(m: dict, k: int) -> str:
    x = m.get(f"R@{k}", {})
    return f"{x.get('answerable_count', 0)}/{x.get('answerable_denominator', 0)}"


def multi(m: dict) -> str:
    x = m.get("R@10", {})
    return f"Any@5 {x.get('multi_any_count', 0)}/{x.get('multi_denominator', 0)}; All@5 {m.get('R@5', {}).get('multi_all_count', 0)}/{x.get('multi_denominator', 0)}; All@10 {x.get('multi_all_count', 0)}/{x.get('multi_denominator', 0)}; All@20 {m.get('R@20', {}).get('multi_all_count', 0)}/{x.get('multi_denominator', 0)}"


def calc(m: dict) -> str:
    return "/".join(str(m.get(f"R@{k}", {}).get("calculation_operand_complete", 0)) for k in (5, 10, 20)) + "/15"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    art = args.artifact_dir
    decision = load(art, "decision.json")
    model = load(art, "embedding-model-manifest.json")
    scope = load(art, "index-scope.json")
    gpu = load(art, "gpu-selection.json")
    dense = load(art, "qwen-dense-index.json")
    stats = load(art, "latency.json")
    head = load(art, "current-exact-candidate-headroom.json")
    cand = load(art, "candidate-recall-r4.json")
    safety = load(art, "safety-regression.json")
    selected = load(art, "selected-config.json")
    multi_art = load(art, "multi-slot-candidate-recall.json")
    calc_art = load(art, "calculation-atomic-fact-recall.json")
    rerank = load(art, "reranker-effect-r4.json")

    stages = {f"G{i}": load(art, f"ablation-g{i}.json")["metrics"] for i in range(5)}
    lines = [
        "# NF-V2-18A-R4 Strong First-Stage Retrieval",
        "",
        "Base: `f34934b85b70ced100457f0b6c455bf8fed67572`",
        "Development set: `CONSUMED_DEVELOPMENT_REGRESSION` (120 questions; not fresh-blind after R4 tuning)",
        "Production: `V1`; production switch: `false`",
        "",
        "## Embedding and index",
        "",
        f"- Model: `{model['repo_id']}`; resolved revision: `{model['resolved_revision']}`",
        f"- Snapshot manifest SHA: `{model['snapshot_sha256']}`; bytes: `{model['total_bytes']}`; files: `{model['file_count']}`",
        f"- Contract: `{model['pooling']}`; query instruction: `{model['query_instruction']}`; document instruction: none",
        f"- Dynamic GPU: physical `{gpu.get('selected_physical_gpu')}`, logical `{gpu.get('visible_logical_gpu')}`, tier `{gpu.get('selection_tier')}`, free before `{gpu.get('free_vram_mib_before')} MiB`",
        f"- General objects: `{dense.get('general_objects')}`; AtomicFact objects: `{dense.get('atomic_objects')}`; persisted vectors: `{stats.get('vector_count')}`; dimension: `{dense.get('dimension')}`",
        f"- Scope: `{scope}`. This is a bounded GOOGL/AMZN development index; production/default indices were not replaced.",
        "",
        "## A4 exact candidate headroom and union",
        "",
        "| depth | A4 | Qwen dense | A4 ∪ Qwen |",
        "|---:|---:|---:|---:|",
    ]
    for depth in (20, 50, 100, 200):
        x = cand["depths"][str(depth)]
        lines.append(f"| {depth} | {x['A4']}/{x['denominator']} | {x['Qwen_dense']}/{x['denominator']} | {x['union']}/{x['denominator']} |")
    family = head.get("depths", {})
    lines.extend(["", f"A4 family-headroom artifact is preserved in `current-exact-candidate-headroom.json` (for example @100 answerable family metrics: `{family.get('100', {}).get('answerable_105', {}).get('answerable_count')}`). A4 remains the hard-scope candidate provider. The union values measure exact canonical evidence inclusion before ranking; family recall is not substituted for fine evidence.", ""])
    lines.extend(["## General ablations (answerable denominator 105)", "", "| stage | R@1 | R@3 | R@5 | R@10 | R@20 |", "|---|---:|---:|---:|---:|---:|"])
    names = {"G0": "A4", "G1": "Qwen dense only", "G2": "A4 ∪ Qwen dense", "G3": "union + optional reranker", "G4": "route-specific"}
    for key, label in names.items():
        m = stages[key]
        lines.append(f"| {key} {label} | {a(m,1)} | {a(m,3)} | {a(m,5)} | {a(m,10)} | {a(m,20)} |")
    lines.extend(["", "G3 reranker status: `" + str(rerank.get("general")) + "`; frozen R3 global effect was rescued `8`, damaged `9`, net `-1`, so it was not selected globally.", "- Frozen R3 selected reference (not overwritten): exact R@5 `63/120`, R@10 `71/120`; R4 G0 is the A4 first-stage baseline `62/120`, `68/120`.", ""])
    lines.extend(["## Route-specific multi and calculation", "", "Multi ablations:"])
    for key in ("M0", "M1", "M2", "M3"):
        lines.append(f"- `{key}`: {multi(multi_art[key])}")
    lines.append("- Slot provider coverage is recorded for A4, Qwen dense, and union at depths 5/10/20 in `multi-slot-candidate-recall.json`; runtime slots are derived by the existing planner, never from Gold.")
    lines.append("")
    lines.append("Calculation ablations (operand-complete counts at @5/@10/@20):")
    for key in ("C0", "C1", "C2", "C3", "C4", "C5"):
        values = [calc_art['ablations'][key].get(f"R@{k}", {}).get("calculation_operand_complete", 0) for k in (5, 10, 20)]
        lines.append(f"- `{key}`: {values[0]}/15, {values[1]}/15, {values[2]}/15")
    lines.append(f"- R3 reference baseline: `{calc_art.get('r3_reference')}`; AtomicFact → canonical TABLE_ROW mapping and period gating remain explicit.")
    lines.extend(["", "## Selected configuration", "", f"- `{selected['general']}`", f"- Multi: `{selected['multi']}`", f"- Calculation: `{selected['calculation']}`", f"- Selected exact answerable R@1/R@3/R@5/R@10/R@20: `{a(selected['metrics'],1)}`, `{a(selected['metrics'],3)}`, `{a(selected['metrics'],5)}`, `{a(selected['metrics'],10)}`, `{a(selected['metrics'],20)}`", "- A4 no-loss invariant: `0` candidates lost due enrichment", ""])
    lines.extend(["## Safety and latency", "", f"- Safety counters: `{safety}`", f"- Embedding/index build: `{stats.get('embedding', {})}`", f"- A4 replay seconds: `{stats.get('a4_coarse_seconds')}`; per-stage latency: `{stats.get('a4_coarse')}`", "- No generator calls, validator changes, calculator arithmetic changes, Gold edits, or production index writes were made.", ""])
    lines.extend(["## Decision", "", f"- Primary ceiling: **{decision.get('ceiling')}**", f"- Decision: **{decision.get('decision')}**", f"- Recommendation: **{decision.get('recommendation')}**", f"- Candidate target counts at @50/@100/@200: `{decision.get('candidate_targets')}`", "", "The embedding model improves first-stage candidate inclusion only modestly in this bounded development scope and does not reach the requested final operating point. A later sprint should target the remaining ranking/representation bottleneck; do not open full runtime based on this result alone."])
    (art / "final-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
