#!/usr/bin/env python3
"""Check sizes of demo assets and verify no prohibited file types.

Output: artifacts/showcase/phase8/asset-manifest.json
Exit: 0 on all pass, 1 on any failure.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = REPO_ROOT / "assets" / "demo"
OUTPUT_DIR = REPO_ROOT / "artifacts" / "showcase" / "phase8"
OUTPUT_FILE = OUTPUT_DIR / "asset-manifest.json"

MAX_FILE_SIZE_MB = 5
MAX_TOTAL_SIZE_MB = 20

PROHIBITED_EXTENSIONS = {".pth", ".bin", ".safetensors", ".pt", ".ckpt", ".log"}


def format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def main() -> int:
    results: list[dict] = []

    if not DEMO_DIR.exists() or not DEMO_DIR.is_dir():
        results.append({
            "check": "assets/demo/ directory exists",
            "pass": False,
            "detail": "assets/demo/ directory does not exist",
        })

        report = {
            "title": "Asset Size & Type Check",
            "overall_pass": False,
            "total_checks": len(results),
            "passed": 0,
            "failed": len(results),
            "directory": str(DEMO_DIR.relative_to(REPO_ROOT)),
            "total_files": 0,
            "total_size_formatted": "0 B",
            "total_size_bytes": 0,
            "checks": results,
            "files": [],
        }

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Report written to {OUTPUT_FILE}")
        print("FAIL: assets/demo/ directory does not exist")
        return 1

    # Collect all files
    all_files: list[Path] = sorted(DEMO_DIR.rglob("*"))
    files = [f for f in all_files if f.is_file()]

    file_entries: list[dict] = []
    oversized_files: list[str] = []
    prohibited_files: list[str] = []
    total_size = 0

    for fp in files:
        size = fp.stat().st_size
        total_size += size
        ext = fp.suffix.lower()
        rel_path = str(fp.relative_to(REPO_ROOT))

        entry = {
            "path": rel_path,
            "size_bytes": size,
            "size_formatted": format_size(size),
            "extension": ext,
        }

        # Check individual file size
        if size > MAX_FILE_SIZE_MB * 1024 * 1024:
            entry["oversized"] = True
            oversized_files.append(rel_path)
        else:
            entry["oversized"] = False

        # Check prohibited extension
        if ext in PROHIBITED_EXTENSIONS:
            entry["prohibited_type"] = True
            prohibited_files.append(f"{rel_path} ({ext})")
        else:
            entry["prohibited_type"] = False

        file_entries.append(entry)

    # Check 1: Each file under 5MB
    check1_pass = len(oversized_files) == 0
    results.append({
        "check": "All files under 5 MB",
        "pass": check1_pass,
        "detail": f"OK ({len(files)} files)" if check1_pass else f"Oversized: {', '.join(oversized_files)}",
    })

    # Check 2: Total directory size under 20MB
    total_size_mb = total_size / (1024 * 1024)
    check2_pass = total_size_mb <= MAX_TOTAL_SIZE_MB
    results.append({
        "check": f"Total directory size under {MAX_TOTAL_SIZE_MB} MB",
        "pass": check2_pass,
        "detail": f"Total: {total_size_mb:.2f} MB" + (" (OK)" if check2_pass else f" (exceeds {MAX_TOTAL_SIZE_MB} MB)"),
    })

    # Check 3: No model weight files
    check3_pass = len(prohibited_files) == 0
    results.append({
        "check": "No model weight files (.pth, .bin, .safetensors, .pt, .ckpt)",
        "pass": check3_pass,
        "detail": "OK" if check3_pass else f"Found: {', '.join(prohibited_files)}",
    })

    # Check 4: No .log files
    log_files = [str(f.relative_to(REPO_ROOT)) for f in files if f.suffix.lower() == ".log"]
    check4_pass = len(log_files) == 0
    results.append({
        "check": "No .log files",
        "pass": check4_pass,
        "detail": "OK" if check4_pass else f"Found: {', '.join(log_files)}",
    })

    all_pass = check1_pass and check2_pass and check3_pass and check4_pass

    report = {
        "title": "Asset Size & Type Check",
        "overall_pass": all_pass,
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "failed": sum(1 for r in results if not r["pass"]),
        "directory": str(DEMO_DIR.relative_to(REPO_ROOT)),
        "total_files": len(files),
        "total_size_formatted": format_size(total_size),
        "total_size_bytes": total_size,
        "checks": results,
        "files": file_entries,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {OUTPUT_FILE}")

    if all_pass:
        print(f"ALL {len(results)} CHECKS PASSED ({len(files)} files, {format_size(total_size)})")
        return 0
    else:
        failures = [r for r in results if not r["pass"]]
        print(f"{len(failures)}/{len(results)} CHECKS FAILED:")
        for f in failures:
            print(f"  FAIL: {f['check']} — {f['detail']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
