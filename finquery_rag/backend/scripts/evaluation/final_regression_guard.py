#!/usr/bin/env python3
"""P1.9 Phase 2: the behavioural freeze guard.

Every cleanup batch in P1.9 has to prove `BEHAVIORAL_DRIFT = 0`.  That is only a
claim if something can refute it, so this pins the things that must not move and
re-derives them from the artefacts rather than trusting a recorded number.

    # once, before any cleanup
    .venv/bin/python scripts/evaluation/final_regression_guard.py --write baseline.json

    # after every batch
    .venv/bin/python scripts/evaluation/final_regression_guard.py --check baseline.json

Exit code is 0 when every pinned fact is unchanged and 1 when any is not, so it
composes into a shell `&&` chain without anything having to read the output.

What is pinned, and why each one
--------------------------------

  benchmark bytes      V2 gold `a3d17211`, eval set `227f0341`, fixture v9
                       `c20afaec` -- the oracle. If these move, every metric
                       below is measuring something else
  V2 reproducibility   rebuilding V2 from the commit's own V1 must reproduce
                       `a3d17211` byte for byte. This is what makes the final
                       benchmark re-runnable from a clean checkout
  store bytes          the fact store the runtime reads, `0382c6a1`. The
                       benchmark is only reproducible against these bytes
  fixture integrity    the C5 guard, which must pass and would have caught the
                       D1 defect
  E2E metrics          released / correct / incorrect / refusals, re-derived from
                       the sealed replay predictions
  citation metrics     both readings, re-derived from the same predictions
  test count           the number of collected tests, so a deleted test shows up
                       as a number that moved

Retrieval Recall@K is **not** re-run here: it needs the R4 index and a full pass
over 120 cases, and it is invariant under every cleanup this phase permits.  Run
`run_nf_v3_retrieval_benchmark.py` separately when a batch could plausibly touch
retrieval; no permitted cleanup does.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

BACKEND = Path("/disk/qh/nano-finrag/finquery_rag/backend")
if not BACKEND.is_dir():
    BACKEND = Path(__file__).resolve().parents[2]
for _p in (str(BACKEND), str(BACKEND / "scripts/evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

BENCH = BACKEND / "benchmarks/tv2_canonical_v1"
V2 = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-c-v2")
PREDICTIONS = Path("/disk/qh/nano-finrag/artifacts/evaluation/p1-8-d1-c3/c3-full-v9/replay-predictions.jsonl")
STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts.jsonl")
IXBRL_STORE = Path("/disk/qh/nano-finrag/data/trusted-v2/fact-store/financial-facts-ixbrl-v1.jsonl")

#: A measurement that was not taken this run.  Distinguished from `None`, which
#: means the measurement was attempted and produced nothing -- those are
#: different states and only one of them is drift.
SKIPPED = "__skipped__"

PINNED = {    "benchmark.v2.gold": "a3d17211bbb42c83c07d1077cb1ed7ef81c2af7e0d27159cac2b58c591f8f6e4",
    "benchmark.v2.eval": "227f0341d94ab8d4b9e7ee033feaa9b40e86ec32c3137a74d2931657e9281ece",
    "benchmark.v1.gold": "3d2a0c5b7839656ce1414923ab90d03b844bc5c5fc17d82dc5e68224f466eb05",
    "fixture.v9": "c20afaec24bcab1360bbcbf9e880d9c8661c24551a1bcef2c59dc0248f3147da",
    "store": "0382c6a17f7065516c0edd0d2ca10d5ea6e0e0c7bb0dff68acae38134fc4db57",
    "metrics.released": 52,
    "metrics.correct_released": 52,
    "metrics.incorrect_release": 0,
    "metrics.correct_refusal": 43,
    "metrics.answerable": 77,
    "metrics.abstention": 43,
    "metrics.citation_precision_identity": 0.960,
    "metrics.citation_recall_identity": 0.953,
}


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rebuild_v2(destination: Path) -> str | None:
    """Rebuild V2 from this commit's V1 and return the gold digest."""

    subprocess.run(
        [sys.executable, str(BACKEND / "scripts/evaluation/build_p1_8c_benchmark_v2.py"),
         "--base", str(BENCH), "--out", str(destination)],
        check=True,
        capture_output=True,
    )
    return sha256(destination / "gold-evidence-v1.jsonl")


def measure(collect_tests: bool = False) -> dict:
    """Every pinned fact, re-derived."""

    facts: dict[str, object] = {}

    facts["benchmark.v1.gold"] = sha256(BENCH / "gold-evidence-v1.jsonl")
    facts["fixture.v9"] = sha256(BENCH / "plan-fixtures-v9.jsonl")

    rebuilt = Path("/tmp/p1-9-guard-v2")
    rebuilt.mkdir(parents=True, exist_ok=True)
    facts["benchmark.v2.gold"] = rebuild_v2(rebuilt)
    facts["benchmark.v2.eval"] = sha256(rebuilt / "canonical-eval-v1.jsonl")

    facts["store"] = sha256(STORE)

    # --- the C5 fixture guard --------------------------------------------------
    guard = subprocess.run(
        [sys.executable, str(BACKEND / "scripts/evaluation/verify_fixture_integrity.py"),
         "--eval-set", str(BENCH / "canonical-eval-v1.jsonl"),
         str(BENCH / "plan-fixtures-v9.jsonl")],
        capture_output=True,
        text=True,
    )
    facts["fixture_integrity.exit"] = guard.returncode

    # --- the sealed run, re-scored --------------------------------------------
    import score_p1_8_d1_canonical as canonical

    gold = {}
    with (rebuilt / "gold-evidence-v1.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                gold[row["id"]] = row
    predictions = {}
    with PREDICTIONS.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                predictions[row["id"]] = row

    answerable = [c for c in gold if gold[c].get("expected_outcome") == "ANSWER"]
    released = [c for c in answerable if str(predictions[c].get("release_status")) == "RELEASED"]
    correct = incorrect = 0
    for case in released:
        verdict, _a, _g = canonical.classify(
            predictions[case].get("answer"),
            gold[case].get("expected_value"),
            gold[case].get("operation"),
        )
        if verdict in ("STRICT_CORRECT", "UNSCORABLE"):
            correct += 1
        elif verdict in ("COMPARATOR_MISMATCH", "GENUINELY_INCORRECT"):
            incorrect += 1
    refusals = sum(
        1 for c in gold
        if gold[c].get("expected_outcome") == "ABSTENTION"
        and str(predictions[c].get("release_status")) != "RELEASED"
    )

    facts["metrics.answerable"] = len(answerable)
    facts["metrics.abstention"] = len(gold) - len(answerable)
    facts["metrics.released"] = len(released)
    facts["metrics.correct_released"] = correct
    facts["metrics.incorrect_release"] = incorrect
    facts["metrics.correct_refusal"] = refusals

    # --- citation, by canonical identity --------------------------------------
    import score_nf_v3_final as scorer
    from src.pdf_retrieval_v4.canonical_metric_identity import canonical_metric_id
    from measure_p1_8_identity_closure import canonical_metric, fold, period_of, quantity

    aliases: dict[str, str] = {}
    records: dict[str, dict] = {}
    for path in (STORE, IXBRL_STORE):
        if not path.is_file():
            continue
        aliases.update(scorer.load_alias_map(path))
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                for key in (record.get("candidate_key"), record.get("candidate_id"), record.get("fact_id")):
                    if key:
                        records.setdefault(str(key), record)

    def identity(record: dict | None):
        if not record:
            return None
        return (
            fold(record.get("entity")),
            canonical_metric(record, canonical_metric_id),
            period_of(record.get("period")),
            quantity(record.get("value")),
        )

    precision_hit = precision_total = recall_hit = recall_total = 0
    for case in released:
        wanted = {scorer.resolve(str(f), aliases) for f in (gold[case].get("fact_ids") or [])}
        wanted_identity = {identity(records.get(w)) for w in wanted} - {None}
        cited = [scorer.resolve(str(c), aliases) for c in (predictions[case].get("citation_ids") or [])]
        precision_total += len(cited)
        recall_total += len(wanted)
        cited_identity = set()
        for value in cited:
            ident = identity(records.get(value))
            cited_identity.add(ident)
            if ident in wanted_identity:
                precision_hit += 1
        recall_hit += len(wanted_identity & (cited_identity - {None}))

    facts["metrics.citation_precision_identity"] = round(precision_hit / precision_total, 3) if precision_total else None
    facts["metrics.citation_recall_identity"] = round(recall_hit / recall_total, 3) if recall_total else None

    # --- test count ------------------------------------------------------------
    # Opt-in: collecting the suite imports every test module, which pulls torch
    # and sentence-transformers and takes minutes.  Nothing this phase permits
    # can change it, so it is off the default path rather than in the way of it.
    if collect_tests:
        try:
            collected = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q"],
                cwd=BACKEND, capture_output=True, text=True, timeout=3600,
            )
            counts = [
                int(l.split()[0])
                for l in collected.stdout.splitlines()
                if l.split() and l.split()[0].isdigit() and "collect" in l.lower()
            ]
            facts["tests.collected"] = counts[-1] if counts else None
        except Exception:  # noqa: BLE001 - the guard reports, it does not fail here
            facts["tests.collected"] = None
    else:
        facts["tests.collected"] = SKIPPED

    return facts


def report(current: dict, baseline: dict | None) -> int:
    if baseline is None:
        for key in sorted(current):
            print(f"  {key:<38} {current[key]}")
        return 0

    drift = []
    for key in sorted(set(current) | set(baseline)):
        now, then = current.get(key), baseline.get(key)
        if now == SKIPPED or then == SKIPPED:
            continue
        if now != then:
            drift.append((key, then, now))
    for key in sorted(current):
        if current.get(key) == SKIPPED:
            print(f"  --  {key:<38} skipped")
            continue
        mark = "ok " if current.get(key) == baseline.get(key) else "DRIFT"
        print(f"  {mark} {key:<38} {current.get(key)}")
    print()
    if drift:
        print("BEHAVIORAL_DRIFT > 0")
        for key, then, now in drift:
            print(f"  {key}: {then} -> {now}")
        return 1
    print("BEHAVIORAL_DRIFT = 0")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None, help="record this run as the baseline")
    parser.add_argument("--check", type=Path, default=None, help="compare against a recorded baseline")
    parser.add_argument("--expect", action="store_true", help="also assert the PINNED table")
    parser.add_argument("--tests", action="store_true", help="also collect the test count (slow)")
    args = parser.parse_args(argv)

    current = measure(collect_tests=args.tests)

    if args.expect:
        bad = {k: (PINNED[k], current.get(k)) for k in PINNED if current.get(k) != PINNED[k]}
        if bad:
            print("PINNED ASSERTIONS FAILED")
            for key, (want, got) in bad.items():
                print(f"  {key}: want {want} got {got}")
            return 1

    baseline = json.loads(args.check.read_text(encoding="utf-8")) if args.check else None
    code = report(current, baseline)

    if args.write:
        args.write.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
        print(f"\nbaseline written to {args.write}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
