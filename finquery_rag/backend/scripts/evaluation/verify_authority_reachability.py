"""P1.6-A3-W6-A: is the A3 path the only authoritative one, or does something fall back?

W4 and W5 moved authority from the legacy rules to `PeriodBindingV2` +
`EmissionAdmission`.  A migration like that fails in a specific and quiet way: the new
path is *nominally* authoritative while some fallback still calls the old selector, and
the outcome looks right because the old rule happens to agree.  Running the pipeline
cannot distinguish "the legacy rule is gone" from "the legacy rule is still there and
still agrees", so this does not rely on the outcome.

Four checks:

  1. reachability   which modules the production build actually loads, and which of them
                    reference the legacy selectors.  Read from the **loaded** module set
                    rather than from a static import graph, because the parse module is
                    loaded by `importlib` and a static walk would not see it.
  2. negative       with the legacy selectors replaced by landmines, the build must still
                    produce byte-identical records.  If anything on the path consults them,
                    this raises.
  3. positive       with `decide_emission_admission` replaced by a raiser, the build must
                    fail.  Without this, check 2 would also pass on a path that emits no
                    facts at all.
  4. the rule delta the A3 routing rule against the legacy whitelist, over the kinds the
                    legacy axis can actually produce.  Stated as a number so "one kind
                    wide" is a measurement rather than a claim.

Checks 2 and 3 are a pair on purpose.  Either alone is satisfied by a degenerate pipeline;
together they say the A3 decision is load-bearing and the legacy one is not on the path.

  python verify_authority_reachability.py --document ko_fy2025 --out <dir>
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

BUILDER = _BACKEND_DIR / "scripts/evaluation/build_store_v2.py"

#: `sys.modules` before this script imported anything of its own.  The reachability check
#: has to be measured against *this*, not against a snapshot taken once the script has
#: already imported the emitter to patch it -- the emitter is the module that defines the
#: legacy constants, so a later snapshot would hide the one module most worth looking at
#: and report a clean result that means nothing.
_PRELOADED = frozenset(sys.modules)

#: The legacy admission whitelists.  `ATOMIC_ELIGIBLE_KINDS` is the rule W4 replaced;
#: `TYPED_ELIGIBLE_KINDS` is its wider sibling.  Neither may be read on the emit path.
LEGACY_SELECTORS = ("ATOMIC_ELIGIBLE_KINDS", "TYPED_ELIGIBLE_KINDS")

#: The kinds the legacy axis classifier can actually produce, which is the domain the rule
#: comparison has to be made over.  `TemporalKind.YEAR` exists in the A3 enum but the legacy
#: axis never emits it, so counting it would report a difference that cannot occur.
LEGACY_DOMAIN = ("point", "duration", "comparison", "segment", "bucket", "category",
                 "non_temporal", "unknown")


class _Landmine:
    """A stand-in for a constant that must not be consulted.

    Raises on every access rather than returning a wrong answer, because the failure being
    hunted is silent: a fallback that reads the old whitelist and happens to agree looks
    exactly like a fallback that is gone.
    """

    def __init__(self, name: str):
        self.name = name

    def _trip(self, how: str):
        raise AssertionError(
            f"{self.name} was consulted ({how}) on the production emit path -- "
            f"the legacy admission rule is still reachable"
        )

    def __contains__(self, item):
        self._trip("membership test")

    def __iter__(self):
        self._trip("iteration")

    def __len__(self):
        self._trip("len()")

    def __bool__(self):
        self._trip("truth test")


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def render(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", default="ko_fy2025")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    builder = _module(BUILDER, "builder")
    from src.runtime.trusted_v2_canonical_fact_store import build_canonical_fact_store
    from src.pdf_retrieval_v4 import typed_evidence_emitters as emitters

    ticker, accession = builder.DOCUMENTS[args.document]

    failures: list[str] = []
    report: dict = {"phase": "P1.6-A3-W6-A", "document": args.document}

    # --- 1. what the production build actually loads -----------------------------------
    print("=== 1. modules on the production path, and what reads the legacy selectors ===")
    records, _summary = build_canonical_fact_store(
        [builder.parse_filing(ticker, accession, args.document)])
    baseline = render(records)

    repo_modules = {}
    for name, module in list(sys.modules.items()):
        if name in _PRELOADED:
            continue
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if _BACKEND_DIR in path.parents:
            repo_modules[name] = path
    print(f"    modules the build pulled in from this repository: {len(repo_modules)}")

    # A module may *define* a legacy constant -- the emitter does, deliberately, because
    # the shadow accounting measures the migration against the rule it replaced.  What no
    # module on the path may do is *read* one, so the two are counted apart.
    readers, definers = [], []
    for name, file in sorted(repo_modules.items()):
        text = file.read_text(encoding="utf-8", errors="replace")
        for lineno, line in enumerate(text.splitlines(), 1):
            for selector in LEGACY_SELECTORS:
                if not re.search(rf"\b{selector}\b", line):
                    continue
                entry = (name, file, lineno, selector, line.strip())
                (definers if re.match(rf"\s*{selector}\s*=", line) else readers).append(entry)
    print(f"    modules that define a legacy selector (allowed): {len(definers)}")
    for name, file, lineno, selector, _line in definers:
        print(f"        {file.relative_to(_BACKEND_DIR)}:{lineno}  {selector}")
    print(f"    modules on the path that READ a legacy selector: {len(readers)}")
    for name, file, lineno, selector, line in readers:
        print(f"        {file.relative_to(_BACKEND_DIR)}:{lineno}  {selector}")
        print(f"            {line[:100]}")
    if readers:
        failures.append(f"LEGACY_SELECTOR_READABLE_ON_PATH = "
                        f"{sorted({r[0] for r in readers})}")
    report["path_modules"] = sorted(repo_modules)
    report["legacy_definers"] = [{"module": n, "line": ln, "selector": s}
                                 for n, _f, ln, s, _t in definers]
    report["legacy_readers"] = [{"module": n, "line": ln, "selector": s}
                                for n, _f, ln, s, _t in readers]
    print()

    # --- 2. negative control: landmine the legacy selectors -----------------------------
    print("=== 2. with the legacy selectors landmined, the build must not touch them ===")
    originals = {name: getattr(emitters, name) for name in LEGACY_SELECTORS}
    try:
        for name in LEGACY_SELECTORS:
            setattr(emitters, name, _Landmine(name))
        mined, _ = build_canonical_fact_store(
            [builder.parse_filing(ticker, accession, args.document)])
        landmine_ok = render(mined) == baseline
    except AssertionError as exc:
        landmine_ok = False
        print(f"    LANDMINE TRIPPED: {exc}")
        failures.append(f"LEGACY_SELECTOR_CONSULTED: {exc}")
    finally:
        for name, value in originals.items():
            setattr(emitters, name, value)
    if landmine_ok:
        print(f"    build succeeded and the {len(mined)} records are byte-identical")
    report["legacy_landmine_tripped"] = not landmine_ok
    print()

    # --- 3. positive control: the A3 decision must be load-bearing ----------------------
    print("=== 3. with the A3 decision replaced by a raiser, the build must fail ===")
    original_decision = emitters.decide_emission_admission

    def _boom(*_a, **_k):
        raise AssertionError("decide_emission_admission was consulted")

    try:
        emitters.decide_emission_admission = _boom
        build_canonical_fact_store(
            [builder.parse_filing(ticker, accession, args.document)])
        print("    the build SUCCEEDED without the A3 decision -- it is not load-bearing")
        failures.append("A3_DECISION_NOT_LOAD_BEARING")
        report["a3_decision_load_bearing"] = False
    except AssertionError:
        print("    the build failed, so the A3 decision is what admits these facts")
        report["a3_decision_load_bearing"] = True
    finally:
        emitters.decide_emission_admission = original_decision
    print()

    # --- 4. the rule delta, measured ----------------------------------------------------
    print("=== 4. the A3 routing rule against the legacy whitelist ===")
    from src.pdf_retrieval_v4.period_binding import DISAGGREGATION_KINDS, TemporalKind

    legacy = set(originals["ATOMIC_ELIGIBLE_KINDS"])
    a3_ineligible = ({k.value for k in DISAGGREGATION_KINDS}
                     | {TemporalKind.NON_TEMPORAL.value})
    moved_in = sorted({k for k in LEGACY_DOMAIN
                       if k not in legacy and k not in a3_ineligible})
    # Over the whole enum, not just the legacy domain.  `YEAR` appears here and cannot
    # appear in a store, because the legacy axis never emits it -- reported so the two
    # numbers together are unambiguous rather than one of them looking like an omission.
    moved_in_enum = sorted(k.value for k in TemporalKind
                           if k.value not in legacy and k.value not in a3_ineligible)
    print(f"    the legacy axis can produce: {len(LEGACY_DOMAIN)} kinds")
    print(f"    eligible under the legacy whitelist: {sorted(legacy)}")
    print(f"    ineligible under the A3 routing rule: {sorted(a3_ineligible)}")
    print(f"    moved ineligible -> eligible, over the legacy domain: {moved_in}")
    print(f"    the same over the whole TemporalKind enum: {moved_in_enum}")
    print(f"    kinds that are in both (still eligible, unchanged): "
          f"{sorted(legacy & a3_ineligible) or 'none'}")
    if moved_in != ["unknown"]:
        failures.append(f"THE RULE MOVED MORE THAN `unknown`: {moved_in}")
    if legacy & a3_ineligible:
        failures.append(f"THE TWO RULES OVERLAP: {sorted(legacy & a3_ineligible)}")
    report["rule_delta"] = {"legacy_eligible": sorted(legacy),
                            "a3_ineligible": sorted(a3_ineligible),
                            "moved_over_legacy_domain": moved_in,
                            "moved_over_enum": moved_in_enum}
    print()

    print(f"  failures: {failures or 'none'}")
    report["failures"] = failures
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "authority-reachability.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(f"  written to {args.out / 'authority-reachability.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
