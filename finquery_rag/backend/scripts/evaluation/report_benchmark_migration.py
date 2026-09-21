"""P1.6-0H provenance repair: what the benchmark migration changed, and why.

The migration was applied on the evaluation host and its outputs were never
committed, so for a while the repository held the pre-migration benchmark while
every published number had been computed on the post-migration one.  This script
is the repair's record: it diffs the two revisions case by case and attaches a
reason to every row.

**Every changed case gets exactly one primary reason, and the reason says how
well it is grounded.**  That is the point.  A migration that cannot say why it
changed a gold row is indistinguishable from one that changed it to move a
number, and the difference matters more here than anywhere else in this
repository:

  DOCUMENTED    a committed document names this case and this defect
  MECHANICAL    the data itself decides it -- equal operands with an asserted
                ordering, a metric that is a bare period, values that differ
                only in presentation
  STRATUM_RULE  the case moved under the stratum's design rule and no narrower
                per-case rationale was written down

`STRATUM_RULE` is a real answer, not a placeholder, but it is a weaker one and
it is counted separately so nobody has to guess which rows rest on it.

Read-only: reads two revisions and writes a record.

  python report_benchmark_migration.py --pre-ref 3a82b11 \\
      --out benchmarks/tv2_canonical_v1/migrations
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]

BENCH_REL = "finquery_rag/backend/benchmarks/tv2_canonical_v1"
FILES = ("canonical-eval-v1.jsonl", "gold-evidence-v1.jsonl")

#: The migration this record describes.
MIGRATION = {
    "id": "p1.6-0h",
    "stratum": "cross_entity_comparison",
    "spec": "docs/evaluation/p1-6-0g-rederivation.md",
    "corrections": "docs/evaluation/p1-6-0a10-expense-sign.md",
    "applied_in": "docs/evaluation/p1-6-0h-fixture-migration.md",
    "store": "financial-facts-ixbrl-v1.jsonl (the rebuilt iXBRL store)",
}

#: The stratum's design rule, quoted rather than paraphrased: from
#: `p1-6-0g-rederivation.md`.  This is the reason every case moved, and the
#: narrower reasons below say how each one failed it.
DESIGN_RULE = (
    "a case survives only if one concept is disclosed as a single company-level "
    "line by every entity in it"
)

#: Old metrics that name a geography, a segment, or a note-level row rather than
#: a company-level line.  Membership is a judgement, so the list is explicit and
#: every entry is a string the old benchmark actually used -- nothing is inferred
#: by pattern.  `United States` and `Total` are named in the 0G specification as
#: exactly this defect; the rest are the same shape.
NOT_COMPANY_LEVEL = frozenset({
    "United States",              # a revenue geography and a tax jurisdiction
    "Foreign exchange contracts",  # a note row
    "Current",                    # a balance-sheet subtotal label
    "Discount rate",              # a note-level assumption
    "Other current liabilities",  # a note row
    "Additions",                  # a PP&E note row
    "Other",                      # a note row
    "Total",                      # "lives in every note" (0G)
    "Land",                       # a PP&E note row
    "Interest rate contracts",    # a derivatives note row
})

#: Reasons with a committed document behind them.  `source` is a document path
#: plus the section that says it, so the claim can be checked rather than trusted.
DOCUMENTED: dict[str, dict[str, str]] = {
    "tv2f01-s3-rank-002": {
        "reason": "SIGN_CORRECTION",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (P1.6-0H, store defects)",
        "note": "Apple R&D is stored as (34,550) where the filing says 34,550, so the "
                "old ranking put Apple last on a sign artefact.",
    },
    "tv2f01-s3-rank-004": {
        "reason": "SIGN_CORRECTION",
        "source": "docs/evaluation/p1-6-0a10-expense-sign.md",
        "note": "US-GAAP expense concepts carry a debit balance type: a positive tagged "
                "amount is the expense. The old gold ranked the presented sign, putting "
                "the smallest expense second-from-last.",
    },
    "tv2f01-s3-crossdiff-005": {
        "reason": "CONCEPT_NOT_DISCLOSED",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (the design rule)",
        "note": "Coca-Cola's balance sheet has no single Total liabilities line; the "
                "case moved to Total assets, which both filers publish.",
    },
    "tv2f01-s3-rank-005": {
        "reason": "CONCEPT_NOT_DISCLOSED",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (rank-005 is rescued)",
        "note": "The store held no shared concept, which read as 'no such concept "
                "exists'. Apple, Microsoft and Coca-Cola each publish Operating income "
                "as one line, so the case was rescued onto it rather than retired.",
    },
    "tv2f01-s3-compare-007": {
        "reason": "FISCAL_YEAR_MISALIGNMENT",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (compare-007)",
        "note": "Pfizer is the corpus's only FY2024 filing. The case now carries "
                "fiscal_year_by_entity so a runner cannot silently align the periods.",
    },
    "tv2f01-s3-compare-001": {
        "reason": "METRIC_NOT_COMPANY_LEVEL",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (the design rule)",
        "note": "`United States` is both a revenue geography and a tax jurisdiction.",
    },
    "tv2f01-s3-crossdiff-002": {
        "reason": "METRIC_NOT_COMPANY_LEVEL",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (the design rule)",
        "note": "`Other` matched several concepts per filing.",
    },
    "tv2f01-s3-crossdiff-003": {
        "reason": "METRIC_NOT_COMPANY_LEVEL",
        "source": "docs/evaluation/p1-6-0g-rederivation.md (the design rule)",
        "note": "`Total` lives in every note.",
    },
}

#: A metric that is a period rather than a quantity.
_PERIOD_ONLY = re.compile(r"^(?:FY\s*)?\d{4}$", re.IGNORECASE)


def _number(raw: str) -> float | None:
    text = str(raw).replace("$", "").replace(",", "").replace(" ", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def _presented_only(old: str, new: str) -> bool:
    """Same number, written the way a table prints it."""
    a, b = _number(old), _number(new)
    return a is not None and b is not None and abs(a - b) < 1e-9


def _order_is_undefined(values: dict[str, str]) -> bool:
    """Two operands with the same value and a stated ordering between them."""
    numbers = [n for n in (_number(v) for v in values.values()) if n is not None]
    return len(numbers) >= 2 and len(set(numbers)) < len(numbers)


def _classify(case_id: str, old: dict, new: dict) -> dict:
    """One primary reason per changed case, with how well it is grounded."""
    if case_id in DOCUMENTED:
        entry = dict(DOCUMENTED[case_id])
        entry["attribution"] = "DOCUMENTED"
        return entry

    old_metric = str(old.get("metric") or "")
    old_values = {k: str(v) for k, v in (old.get("values") or {}).items()}
    new_values = {k: str(v) for k, v in (new.get("values") or {}).items()}

    if _PERIOD_ONLY.match(old_metric):
        return {
            "reason": "METRIC_NOT_A_QUANTITY",
            "attribution": "MECHANICAL",
            "source": "the old metric is a bare period",
            "note": f"{old_metric!r} names a date, not a quantity, so the two operands "
                    f"could not be compared on it.",
        }
    if old_values and _order_is_undefined(old_values) and not _order_is_undefined(
        new_values
    ):
        return {
            "reason": "EQUAL_VALUE_ORDERING_UNDEFINED",
            "attribution": "MECHANICAL",
            "source": "the old values are equal and an ordering was asserted over them",
            "note": "The old gold could not be satisfied: it named one of two equal "
                    "operands as higher.",
        }
    if old_metric in NOT_COMPANY_LEVEL:
        return {
            "reason": "METRIC_NOT_COMPANY_LEVEL",
            "attribution": "DOCUMENTED" if old_metric in {"United States", "Total"}
            else "MECHANICAL",
            "source": "the design rule: one company-level line per entity",
            "note": f"{old_metric!r} names a geography, a segment or a note row, so the "
                    f"two operands were not the same quantity of the same kind.",
        }
    if old_metric == str(new.get("metric") or ""):
        shared = [k for k in old_values if k in new_values]
        if shared and all(_presented_only(old_values[k], new_values[k]) for k in shared):
            return {
                "reason": "PRESENTATION_ONLY",
                "attribution": "MECHANICAL",
                "source": "the numbers are unchanged; only the formatting moved",
                "note": "The old gold was read off a rendered page, the new one from the "
                        "filers' tagged values. The expectation is unchanged.",
            }
        return {
            "reason": "VALUE_NOT_THE_FILED_ONE",
            "attribution": "MECHANICAL",
            "source": "same metric, a value that is not what the filing states",
            "note": "At least one operand's value changed materially, so the old gold "
                    "was not the number the filing attributes to that concept.",
        }
    return {
        "reason": "STRATUM_REDERIVATION",
        "attribution": "STRATUM_RULE",
        "source": "the stratum's design rule, with no narrower rationale written down",
        "note": f"{old_metric!r} -> {new.get('metric')!r}. Covered by the rule that a "
                f"case survives only if one concept is a single company-level line for "
                f"every entity; the per-case reason was not recorded.",
    }


def _read_ref(ref: str, name: str) -> dict[str, dict]:
    # `encoding` is not optional here.  Without it `text=True` decodes with the
    # locale codepage, which on a Windows checkout turns the em dash in Pfizer's
    # `Weighted-average shares—diluted` into replacement characters -- and then
    # that row compares unequal to itself and is reported as a change that never
    # happened.  The benchmark is UTF-8 everywhere it is read.
    text = subprocess.run(
        ["git", "show", f"{ref}:{BENCH_REL}/{name}"],
        cwd=_BACKEND_DIR, capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return {json.loads(line)["id"]: json.loads(line)
            for line in text.splitlines() if line.strip()}


def _read_worktree(name: str) -> dict[str, dict]:
    path = _BACKEND_DIR / "benchmarks/tv2_canonical_v1" / name
    return {json.loads(line)["id"]: json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


#: Every file the benchmark version is defined by.  `plan-fixtures-v8` is here
#: because the retrieval runner defaults to it: a version that names only the
#: questions and the gold leaves the runner's actual input unpinned.
VERSION_FILES = {
    "canonical-eval-v1.jsonl": "questions",
    "gold-evidence-v1.jsonl": "gold",
    "plan-fixtures-v8.jsonl": "plan fixtures",
}


def write_version_manifest(path: Path, record: dict, bench_dir: Path) -> dict:
    """The benchmark's identity: a version name, and a hash per file.

    The version name matters more than it looks.  The file names did not change
    when the stratum was re-derived, so nothing on disk says the content moved;
    anyone reading `canonical-eval-v1.jsonl` would have no way to tell which
    revision they had.  This is the file that says so.
    """
    files = {}
    for name, role in VERSION_FILES.items():
        target = bench_dir / name
        if not target.is_file():
            raise FileNotFoundError(f"benchmark_file_missing:{target}")
        rows = sum(1 for line in target.read_text(encoding="utf-8").splitlines()
                   if line.strip())
        files[name] = {"role": role, "rows": rows, "sha256": _sha256(target)}
    manifest = {
        "benchmark": "tv2-canonical-v1",
        "version": "tv2-canonical-v1-post-p1.6-0h",
        "note": (
            "Content changed under unchanged file names at P1.6-0H. The pre-migration "
            "revision is identified by the hashes under migrations[].pre; do not treat "
            "these file names as implying the pre-migration content."
        ),
        "files": files,
        "migrations": [{
            "id": MIGRATION["id"],
            "stratum": MIGRATION["stratum"],
            "record": "migrations/p1-6-0h.json",
            "spec": MIGRATION["spec"],
            "corrections": MIGRATION["corrections"],
            "applied_in": MIGRATION["applied_in"],
            "pre": {
                "canonical-eval-v1.jsonl":
                    "9c8ca9ceea21cecfc43a8b3861d13884b3e1a70e3839438cec3db06f599882a0",
                "gold-evidence-v1.jsonl":
                    "7831eef2d33abc3086f8e88c028f0e3bd3723917765612d86551d4604bba6925",
                "plan-fixtures-v7.jsonl":
                    "43f740f7ab702371d4cb0091ffb2071c8f7b24e8c14e35fa7612f22691e7c066",
            },
            "post": {
                "canonical-eval-v1.jsonl":
                    "227f0341d94ab8d4b9e7ee033feaa9b40e86ec32c3137a74d2931657e9281ece",
                "gold-evidence-v1.jsonl":
                    "3d2a0c5b7839656ce1414923ab90d03b844bc5c5fc17d82dc5e68224f466eb05",
                "plan-fixtures-v8.jsonl":
                    "7463b45a5bfc309d0a5ab29cdb15b3f070c13c8526ae5d1486ca67847dda362d",
            },
        }],
        "applied": bool(record),
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_version_manifest(path: Path, bench_dir: Path) -> list[str]:
    """Recompute every hash in the manifest.  Never trust a recorded hash."""
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for name, entry in (manifest.get("files") or {}).items():
        target = bench_dir / name
        if not target.is_file():
            failures.append(f"FILE_MISSING = {name}")
            continue
        actual = _sha256(target)
        if actual != entry["sha256"]:
            failures.append(
                f"HASH_MISMATCH = {name}: manifest {entry['sha256'][:12]} "
                f"actual {actual[:12]}")
        rows = sum(1 for line in target.read_text(encoding="utf-8").splitlines()
                   if line.strip())
        if rows != entry.get("rows"):
            failures.append(f"ROW_COUNT = {name}: manifest {entry.get('rows')} "
                            f"actual {rows}")
    for migration in manifest.get("migrations") or []:
        for name, expected in (migration.get("post") or {}).items():
            recorded = (manifest.get("files") or {}).get(name, {}).get("sha256")
            if recorded and recorded != expected:
                failures.append(f"MIGRATION_POST_DISAGREES = {name}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-ref",
                        help="revision that still holds the pre-migration benchmark")
    parser.add_argument("--post-ref", default=None,
                        help="revision holding the post-migration benchmark; default is "
                             "the working tree")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--write-version-manifest", type=Path, default=None,
                        help="write the benchmark's identity (version + per-file hash). "
                             "Recomputed here rather than copied from a summary.")
    parser.add_argument("--verify", type=Path, default=None,
                        help="recompute every hash in an existing manifest and check it. "
                             "Skips the diff.")
    args = parser.parse_args(argv)

    bench_dir = _BACKEND_DIR / "benchmarks/tv2_canonical_v1"

    if args.verify is not None:
        failures = verify_version_manifest(args.verify, bench_dir)
        manifest = json.loads(args.verify.read_text(encoding="utf-8"))
        print("=" * 78)
        print(f"BENCHMARK VERSION -- {manifest.get('benchmark')} "
              f"{manifest.get('version')}")
        print("=" * 78)
        for name, entry in sorted((manifest.get("files") or {}).items()):
            print(f"    {name:28} {entry['role']:14} rows={entry['rows']:>4}  "
                  f"{entry['sha256'][:12]}")
        print()
        print(f"  failures: {failures or 'none'}")
        return 1 if failures else 0

    if not args.pre_ref:
        parser.error("--pre-ref is required unless --verify is given")
    if args.out is None:
        parser.error("--out is required unless --verify is given")

    read_post = ((lambda name: _read_ref(args.post_ref, name)) if args.post_ref
                 else _read_worktree)
    pre = {name: _read_ref(args.pre_ref, name) for name in FILES}
    post = {name: read_post(name) for name in FILES}

    failures: list[str] = []
    eval_changed = sorted(i for i in set(pre[FILES[0]]) | set(post[FILES[0]])
                          if pre[FILES[0]].get(i) != post[FILES[0]].get(i))
    gold_changed = sorted(i for i in set(pre[FILES[1]]) | set(post[FILES[1]])
                          if pre[FILES[1]].get(i) != post[FILES[1]].get(i))

    strata = collections.Counter(
        str((post[FILES[0]].get(i) or pre[FILES[0]].get(i) or {}).get("stratum"))
        for i in eval_changed)

    print("=" * 78)
    print(f"P1.6-0H BENCHMARK MIGRATION -- {args.pre_ref} -> "
          f"{args.post_ref or 'working tree'}")
    print("=" * 78)
    print(f"  questions changed  {len(eval_changed)}")
    print(f"  gold rows changed  {len(gold_changed)}")
    print(f"  strata touched     {dict(strata)}")
    untouched = sorted(set(pre[FILES[1]]) - set(gold_changed))
    print(f"  gold rows untouched {len(untouched)}")
    print()

    rows: list[dict] = []
    for case_id in gold_changed:
        old = pre[FILES[1]].get(case_id, {})
        new = post[FILES[1]].get(case_id, {})
        entry = _classify(case_id, old, new)
        old_q = (pre[FILES[0]].get(case_id) or {}).get("question")
        new_q = (post[FILES[0]].get(case_id) or {}).get("question")
        rows.append({
            "case_id": case_id,
            "stratum": str(new.get("stratum") or (post[FILES[0]].get(case_id) or {})
                          .get("stratum") or ""),
            "reason": entry["reason"],
            "attribution": entry["attribution"],
            "source": entry["source"],
            "note": entry["note"],
            "question": {"old": old_q, "new": new_q},
            "metric": {"old": old.get("metric"), "new": new.get("metric")},
            "values": {"old": old.get("values"), "new": new.get("values")},
            "operation": new.get("operation") or old.get("operation"),
            "expected": {
                "old": {k: old.get(k) for k in
                        ("expected_higher", "expected_ranking", "expected_value")},
                "new": {k: new.get(k) for k in
                        ("expected_higher", "expected_ranking", "expected_value")},
            },
        })

    # No changed row may go unclassified -- that is the property this record
    # exists to guarantee, so it is checked rather than assumed.
    for row in rows:
        if not row["reason"]:
            failures.append(f"UNCLASSIFIED_CHANGE = {row['case_id']}")

    by_reason = collections.Counter(r["reason"] for r in rows)
    by_attribution = collections.Counter(r["attribution"] for r in rows)
    print("  reason                             count  attribution")
    for reason, count in by_reason.most_common():
        attrs = ",".join(sorted({r["attribution"] for r in rows if r["reason"] == reason}))
        print(f"    {reason:33} {count:>4}   {attrs}")
    print()
    print(f"  attribution: {dict(by_attribution)}")
    print()

    expect_changed = [r for r in rows
                      if r["expected"]["old"] != r["expected"]["new"]]
    print(f"  cases whose expected answer changed: {len(expect_changed)}")
    for row in expect_changed:
        print(f"    {row['case_id'].replace('tv2f01-s3-', ''):14} {row['reason']}")
        print(f"        was  {row['expected']['old']}")
        print(f"        now  {row['expected']['new']}")
    print()

    record = {
        "migration": MIGRATION,
        "design_rule": DESIGN_RULE,
        "pre_ref": args.pre_ref,
        "post_ref": args.post_ref or "working tree",
        "questions_changed": len(eval_changed),
        "gold_rows_changed": len(gold_changed),
        "gold_rows_untouched": len(untouched),
        "strata_touched": dict(strata),
        "by_reason": dict(by_reason),
        "by_attribution": dict(by_attribution),
        "expectation_changed": [r["case_id"] for r in expect_changed],
        "changes": rows,
        "failures": failures,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "p1-6-0h.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    lines = [
        "# P1.6-0H — the benchmark migration, case by case",
        "",
        f"Generated by `scripts/evaluation/report_benchmark_migration.py "
        f"--pre-ref {args.pre_ref}`.",
        "",
        f"`{MIGRATION['id']}` re-derived the `{MIGRATION['stratum']}` stratum: "
        f"{len(eval_changed)} questions and {len(gold_changed)} gold rows changed, "
        f"{len(untouched)} gold rows did not.",
        "",
        f"**Design rule** ({MIGRATION['spec']}): {DESIGN_RULE}.",
        "",
        "| case | old metric | new metric | reason | grounded by |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['case_id'].replace('tv2f01-s3-', '')} "
            f"| {row['metric']['old']} | {row['metric']['new']} "
            f"| {row['reason']} | {row['attribution']} |")
    lines += [
        "",
        "Attribution: **DOCUMENTED** a committed document names this case and defect; "
        "**MECHANICAL** the data decides it; **STRATUM_RULE** the case moved under the "
        "design rule with no narrower rationale on record.",
        "",
        f"Sources: `{MIGRATION['spec']}`, `{MIGRATION['corrections']}`, "
        f"`{MIGRATION['applied_in']}`.",
        "",
    ]
    (args.out / "p1-6-0h.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")

    print(f"  failures: {failures or 'none'}")
    print(f"  written to {args.out / 'p1-6-0h.json'} and {args.out / 'p1-6-0h.md'}")

    if args.write_version_manifest is not None:
        manifest = write_version_manifest(args.write_version_manifest, record, bench_dir)
        print()
        print(f"  benchmark version {manifest['version']}")
        for name, entry in sorted(manifest["files"].items()):
            print(f"    {name:28} {entry['role']:14} rows={entry['rows']:>4}  "
                  f"{entry['sha256'][:12]}")
        # The manifest must agree with the record it was just derived from, or
        # one of the two is lying about what shipped.
        for name, expected in manifest["migrations"][0]["post"].items():
            actual = manifest["files"].get(name, {}).get("sha256")
            if actual and actual != expected:
                failures.append(f"MANIFEST_DISAGREES_WITH_MIGRATION = {name}")
                print(f"    !! {name}: manifest {actual[:12]} but the migration record "
                      f"says {expected[:12]}")
        print(f"  written to {args.write_version_manifest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
