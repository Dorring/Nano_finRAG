#!/usr/bin/env python3
"""NF-V3 H1.1 integration runner: legacy vs harness_v3 over sealed fixtures.

This is the end-to-end evidence for the H1 ablation.  It does not re-check the
harness's internal phase order -- the unit suite does that.  It answers one
question on real wiring: *given the same request and the same capabilities, do
the two runtime modes reach the same decision?*

Each fixture is executed in both modes through ``build_trusted_v2_runtime``,
the same factory the production builder calls, over the real R4 retriever, the
real Semantic Binder, the real deterministic calculator, the real generator
routing and the real release validator.  The two outcomes are compared through
``tests.harness.equivalence``, which requires every decision-bearing field to
match and strips only measurements and harness-only execution markers.

Exit status is non-zero if any fixture diverges, any fixture fails its own
expectation, or a rejected candidate is released.  A green run means the
ablation is safe to enable; it does not mean the harness is correct in general.

    python scripts/runtime/run_nf_v3_h1_harness_integration.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

BACKEND_DIR = Path(__file__).parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from tests.harness.h1_integration import run_all, sealed_digest  # noqa: E402

DEFAULT_OUTPUT = (
    BACKEND_DIR / "artifacts/runtime/nf-v3-h1-harness-core/integration-report.json"
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _render(report: dict[str, Any]) -> str:
    lines = [
        "NF-V3 H1 harness integration: legacy vs harness_v3",
        f"sealed fixture digest: {report['sealed_digest']}",
        "",
        f"{'fixture':<26} {'equiv':<6} {'legacy':<18} {'harness_v3':<18} turns",
        "-" * 78,
    ]
    for fixture in report["fixtures"]:
        lines.append(
            f"{fixture['fixture_id']:<26} "
            f"{'yes' if fixture['decision_equivalent'] else 'NO':<6} "
            f"{fixture['legacy']['status']:<18} "
            f"{fixture['harness_v3']['status']:<18} "
            f"{fixture['harness_v3']['turn_count']}"
        )
        for difference in fixture["differences"]:
            lines.append(
                f"    DIVERGED {difference['field']}: "
                f"{difference['legacy']!r} != {difference['harness_v3']!r}"
            )
        for problem in fixture["expectation_failures"]:
            lines.append(f"    FAILED   {problem}")
    summary = report["summary"]
    lines += [
        "",
        f"decision equivalence : {summary['decision_equivalence']}/{report['fixture_count']}",
        f"expectation failures : {summary['expectation_failures']}",
        f"release bypass       : {summary['release_bypass']}",
        f"false calc release   : {summary['false_calculation_release']}",
        f"infinite loop        : {summary['infinite_loop']}",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="where to write the JSON report",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="print the report without writing it to disk",
    )
    args = parser.parse_args(argv)

    report = run_all()
    payload = {**report, "sealed_digest_recomputed": sealed_digest()}
    if not args.no_write:
        _write_json(args.output, payload)
    print(_render(report))
    if not args.no_write:
        print(f"\nreport written to {args.output}")

    summary = report["summary"]
    clean = (
        summary["decision_equivalence"] == report["fixture_count"]
        and summary["expectation_failures"] == 0
        and summary["release_bypass"] == 0
        and summary["false_calculation_release"] == 0
    )
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
