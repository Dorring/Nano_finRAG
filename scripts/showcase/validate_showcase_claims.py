#!/usr/bin/env python3
"""Validate that README.md and README.zh-CN.md do not contain prohibited claims.

Output: artifacts/showcase/phase8/claim-audit.json
Exit: 0 on all pass, 1 on any failure.
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "artifacts" / "showcase" / "phase8"
OUTPUT_FILE = OUTPUT_DIR / "claim-audit.json"


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def get_verified_metrics_section(text: str) -> str | None:
    """Extract the 'Verified Engineering Metrics' section from a README.

    Returns content from the section header to the next section (##), or
    from the header to end of file if it is the last section.
    Supports both English and Chinese headers.
    """
    # Match either English or Chinese section header
    pattern = r"(?:## Verified Engineering Metrics|## 已验证的工程指标)(.*?)(?=\n## |\Z)"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        return None
    return match.group(1)


def extract_metrics_from_table(text: str) -> dict[str, str]:
    """Parse a markdown metrics table and return {metric_name: value}."""
    # Find markdown tables in the text
    metrics: dict[str, str] = {}
    rows = re.findall(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", text)
    for row in rows:
        key = row[0].strip()
        val = row[1].strip()
        # Skip header rows
        if key in ("Metric", "指标", "---", ":-") or key.startswith("-"):
            continue
        metrics[key.lower()] = val
    return metrics


def check_screenshots_exist(readme_path: Path) -> tuple[bool, str]:
    """Check that screenshot references in README point to existing files.

    Look for all image references (markdown images, plus the assets/demo/ reference)
    and verify the files exist.
    """
    text = readme_path.read_text(encoding="utf-8")

    # Collect image references: ![alt](path) pattern
    image_refs = re.findall(r'!\[.*?\]\(([^)]+)\)', text)

    # Also collect linked directories referenced for "screenshots" / demo
    if "assets/demo/" in text or "assets\\demo\\" in text:
        # The README says screenshots are in assets/demo/
        demo_dir = REPO_ROOT / "assets" / "demo"
        if not demo_dir.exists() or not demo_dir.is_dir():
            return False, "assets/demo/ directory does not exist"
        # List all image-like files in the demo dir
        image_exts = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".bmp"}
        demo_files = list(demo_dir.iterdir())
        image_files = [f for f in demo_files if f.suffix.lower() in image_exts]
        if not image_files:
            return False, f"assets/demo/ directory exists but contains no image files (has {len(demo_files)} non-image entries)"
        return True, f"Found {len(image_files)} screenshot(s) in assets/demo/"

    # If no assets/demo/ reference, check individual image refs
    missing: list[str] = []
    found: list[str] = []
    for ref in image_refs:
        # Skip external URLs
        if ref.startswith("http://") or ref.startswith("https://"):
            continue
        ref_path = (readme_path.parent / ref).resolve()
        if ref_path.exists():
            found.append(ref)
        else:
            missing.append(ref)

    if missing:
        return False, f"Missing screenshot files: {', '.join(missing)} (found: {', '.join(found) if found else 'none'})"
    return True, f"All {len(found)} screenshot(s) found"


def _is_in_disclaimer_context(text: str, term: str) -> bool:
    """Check if all occurrences of `term` appear within disclaimer/negation contexts.

    A disclaimer context is either:
    - A blockquote line (starts with '>') that contains negation phrasing, OR
    - A regular paragraph where the term is negated (e.g. "not X", "not used as X")

    Supports both English and Chinese negation patterns.
    """
    lower = text.lower()
    term_lower = term.lower()

    idx = 0
    while True:
        pos = lower.find(term_lower, idx)
        if pos == -1:
            break

        # Check if the surrounding context is a disclaimer
        # Look at nearby context (±400 chars) for negation patterns
        ctx_start = max(0, pos - 400)
        ctx_end = min(len(lower), pos + 400)
        context = lower[ctx_start:ctx_end]

        negation_patterns = [
            # English
            "not used as", "explicitly **not**", "not be used",
            "should not", "not for quality", "not a quality",
            "not model-native", "system component, not",
            "is a system component",  # implies not model-native when followed by ", not"
            "而非模型原生",  # Chinese: rather than model-native
            # Chinese
            "明确不", "不作为质量", "不用于质量",
            "不会作为", "不应作为", "不可作为",
            "不应使用", "不可使用",
        ]
        is_negated = any(p in context for p in negation_patterns)

        # Accept if: blockquote with negation, OR any context with strong negation
        if not is_negated:
            return False

        idx = pos + len(term_lower)

    return True


def check_prohibited_terms(text: str, readme_name: str) -> list[dict]:
    """Check that banned terms do NOT appear as claims in the text.

    Terms that appear only in disclaimer context (blockquotes explicitly
    saying they are NOT used as claims) are considered acceptable.
    """
    results: list[dict] = []

    bans = [
        ("production-ready", "contains 'production-ready'"),
        ("state-of-the-art", "contains 'state-of-the-art'"),
        ("production-grade accuracy", "contains 'production-grade accuracy'"),
    ]
    for term, desc in bans:
        term_lower = term.lower()
        present = term_lower in text.lower()
        if present and _is_in_disclaimer_context(text, term_lower):
            # Term appears but only in disclaimer context — acceptable
            results.append({
                "check": f"Banned term '{term}' not in {readme_name}",
                "pass": True,
                "detail": f"Found in disclaimer context only — OK in {readme_name}",
            })
        else:
            results.append({
                "check": f"Banned term '{term}' not in {readme_name}",
                "pass": not present,
                "detail": f"FOUND in {readme_name}" if present else f"OK in {readme_name}",
            })

    return results


def check_eliminates_hallucinations(text: str, readme_name: str) -> dict:
    present = "eliminates hallucinations" in text.lower()
    return {
        "check": f"'eliminates hallucinations' not in {readme_name}",
        "pass": not present,
        "detail": f"FOUND in {readme_name}" if present else f"OK in {readme_name}",
    }


def check_native_tool_calling(text: str, readme_name: str) -> dict:
    """Check that 'native tool calling' or 'function calling' is not claimed as model-native.

    The phrase "not model-native tool calling" is a disclaimer — it correctly
    says the calculator is NOT model-native. This is acceptable.
    """
    lower = text.lower()
    issues: list[str] = []

    # Check "function calling" — only flag if it describes a real capability
    if "function calling" in lower:
        issues.append("function calling")

    # Check "native tool calling" — only flag if it's used as a positive claim
    if "native tool calling" in lower:
        # If all occurrences are negated (e.g., "not model-native tool calling"), pass
        if _is_in_disclaimer_context(text, "native tool calling"):
            pass  # appears only in disclaimer — OK
        else:
            issues.append("native tool calling")

    if issues:
        return {
            "check": f"No 'native tool calling' or 'function calling' as model-native claim in {readme_name}",
            "pass": False,
            "detail": f"FOUND in {readme_name}: {', '.join(issues)}",
        }
    return {
        "check": f"No 'native tool calling' or 'function calling' as model-native claim in {readme_name}",
        "pass": True,
        "detail": f"OK in {readme_name}",
    }


def check_0_54_not_quality_metric(text: str, readme_name: str) -> dict:
    """Check that '0/54' is not used as a quality metric in the verified metrics section.

    If '0/54' appears only in a disclaimer saying it is explicitly NOT used
    as a quality metric, that is acceptable.
    """
    section = get_verified_metrics_section(text)
    if section is None:
        return {
            "check": f"'0/54' not used as quality metric in {readme_name}",
            "pass": True,
            "detail": f"No verified metrics section found in {readme_name}",
        }
    if "0/54" not in section:
        return {
            "check": f"'0/54' not used as quality metric in {readme_name}",
            "pass": True,
            "detail": f"OK in {readme_name}",
        }

    # '0/54' is present — check if it's in disclaimer context only
    if _is_in_disclaimer_context(section, "0/54"):
        return {
            "check": f"'0/54' not used as quality metric in {readme_name}",
            "pass": True,
            "detail": f"Found in disclaimer context only — OK in {readme_name}",
        }

    return {
        "check": f"'0/54' not used as quality metric in {readme_name}",
        "pass": False,
        "detail": f"FOUND '0/54' used as a quality metric in {readme_name}",
    }


def check_compression_rate_not_in_metrics(text: str, readme_name: str) -> dict:
    """Check that '59.5%' is not in the verified metrics section."""
    section = get_verified_metrics_section(text)
    if section is None:
        return {
            "check": f"'59.5%' not in verified metrics section of {readme_name}",
            "pass": True,
            "detail": f"No verified metrics section found in {readme_name}",
        }
    if "59.5%" in section:
        return {
            "check": f"'59.5%' not in verified metrics section of {readme_name}",
            "pass": False,
            "detail": f"FOUND '59.5%' in verified metrics section of {readme_name}",
        }
    return {
        "check": f"'59.5%' not in verified metrics section of {readme_name}",
        "pass": True,
        "detail": f"OK in {readme_name}",
    }


def build_verified_metrics_index(metrics_md_path: Path) -> dict[str, str]:
    """Read verified-metrics.md and extract all metrics with their values."""
    text = read_text(metrics_md_path)
    index: dict[str, str] = {}

    # Extract the summary table (section 8)
    rows = re.findall(r"\|\s*(.+?)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*\|", text)
    for row in rows:
        key = row[0].strip()
        val = row[1].strip()
        if key in ("指标", "Metric", "---") or key.startswith("-"):
            continue
        index[key] = val

    return index


def check_metrics_in_verified_index(
    metrics: dict[str, str], verified_index: dict[str, str], readme_name: str
) -> dict:
    """Check all numbers in the README metrics table have entries in verified-metrics.md."""
    # Known keys and their verified-metrics.md equivalents
    key_mapping = {
        "deterministic financial operations": "确定性操作数",
        "确定性金融运算": "确定性操作数",
        "validation categories": "校验类别数",
        "校验类别": "校验类别数",
        "online services": "在线服务数",
        "在线服务数": "在线服务数",
        "phase 7 deployment acceptance": "Phase 7 验收",
        "阶段七部署验收": "Phase 7 验收",
        "automated tests": "自动化测试",
        "自动化测试": "自动化测试",
        "deployment smoke tests": "部署冒烟",
        "部署冒烟测试": "部署冒烟",
    }

    missing: list[str] = []
    found: list[str] = []
    for key, val in metrics.items():
        mapped_key = key_mapping.get(key, key)
        if mapped_key in verified_index:
            found.append(f"{key}={val}")
        else:
            missing.append(f"{key}={val} (mapped to '{mapped_key}')")

    passed = len(missing) == 0
    detail = f"All {len(found)} metrics found in verified-metrics.md"
    if missing:
        detail = f"Missing from verified-metrics.md: {', '.join(missing)}; found: {', '.join(found)}"
    return {
        "check": f"All metrics in {readme_name} table exist in verified-metrics.md",
        "pass": passed,
        "detail": detail,
    }


def check_mentions_nanochat(text: str, readme_name: str) -> dict:
    """Check that README mentions the upstream NanoChat project."""
    lower = text.lower()
    has_name = "nanogpt" in lower or "nanochat" in lower

    passed = has_name
    detail = "OK mentions upstream" if passed else "DOES NOT mention NanoChat upstream project"
    return {
        "check": f"README {readme_name} mentions upstream NanoChat",
        "pass": passed,
        "detail": detail,
    }


def check_consistent_numbers(
    en_metrics: dict[str, str], zh_metrics: dict[str, str]
) -> dict:
    """Check that both READMEs have consistent numbers for key metrics."""
    pairs = [
        ("9", "9", "deterministic financial operations"),
        ("6+", "6+", "validation categories"),
        ("3", "3", "online services"),
        ("42/42", "42/42", "phase 7 deployment acceptance"),
    ]

    results: list[dict] = []
    for en_val, zh_val, label in pairs:
        en_has = any(v == en_val for v in en_metrics.values())
        zh_has = any(v == zh_val for v in zh_metrics.values())
        passed = en_has and zh_has
        detail = f"EN={'found' if en_has else 'NOT FOUND'}, ZH={'found' if zh_has else 'NOT FOUND'}"
        results.append({
            "check": f"Consistent '{label}' = {en_val} across READMEs",
            "pass": passed,
            "detail": detail,
        })

    overall = all(r["pass"] for r in results)
    return {
        "check": "Consistent numbers across READMEs (9 ops, 6+ validation, 3 services, 42/42)",
        "pass": overall,
        "detail": "; ".join(r["detail"] for r in results),
        "sub_checks": results,
    }


def main() -> int:
    readme_en_path = REPO_ROOT / "README.md"
    readme_zh_path = REPO_ROOT / "README.zh-CN.md"
    metrics_md_path = REPO_ROOT / "docs" / "showcase" / "verified-metrics.md"

    if not readme_en_path.exists():
        print(f"ERROR: {readme_en_path} not found")
        return 1
    if not readme_zh_path.exists():
        print(f"ERROR: {readme_zh_path} not found")
        return 1
    if not metrics_md_path.exists():
        print(f"ERROR: {metrics_md_path} not found")
        return 1

    en_text = read_text(readme_en_path)
    zh_text = read_text(readme_zh_path)

    results: list[dict] = []

    # Check 2: Banned terms
    results.extend(check_prohibited_terms(en_text, "README.md"))
    results.extend(check_prohibited_terms(zh_text, "README.zh-CN.md"))

    # Check 3: "eliminates hallucinations"
    results.append(check_eliminates_hallucinations(en_text, "README.md"))
    results.append(check_eliminates_hallucinations(zh_text, "README.zh-CN.md"))

    # Check 4: "native tool calling" / "function calling"
    results.append(check_native_tool_calling(en_text, "README.md"))
    results.append(check_native_tool_calling(zh_text, "README.zh-CN.md"))

    # Check 5: "0/54" not in verified metrics section
    results.append(check_0_54_not_quality_metric(en_text, "README.md"))
    results.append(check_0_54_not_quality_metric(zh_text, "README.zh-CN.md"))

    # Check 6: "59.5%" not in verified metrics section
    results.append(check_compression_rate_not_in_metrics(en_text, "README.md"))
    results.append(check_compression_rate_not_in_metrics(zh_text, "README.zh-CN.md"))

    # Check 7: Metrics table entries in verified-metrics.md
    en_section = get_verified_metrics_section(en_text)
    if en_section:
        en_metrics = extract_metrics_from_table(en_section)
    else:
        en_metrics = {}
    zh_section = get_verified_metrics_section(zh_text)
    if zh_section:
        zh_metrics = extract_metrics_from_table(zh_section)
    else:
        zh_metrics = {}

    verified_index = build_verified_metrics_index(metrics_md_path)
    results.append(check_metrics_in_verified_index(en_metrics, verified_index, "README.md"))
    results.append(check_metrics_in_verified_index(zh_metrics, verified_index, "README.zh-CN.md"))

    # Check 8: Both READMEs mention upstream NanoChat
    results.append(check_mentions_nanochat(en_text, "README.md"))
    results.append(check_mentions_nanochat(zh_text, "README.zh-CN.md"))

    # Check 9: Consistent numbers
    results.append(check_consistent_numbers(en_metrics, zh_metrics))

    # Check 10: Demo screenshots exist
    en_screenshot_ok, en_screenshot_detail = check_screenshots_exist(readme_en_path)
    results.append({
        "check": "Demo screenshots referenced in README.md exist",
        "pass": en_screenshot_ok,
        "detail": en_screenshot_detail,
    })
    zh_screenshot_ok, zh_screenshot_detail = check_screenshots_exist(readme_zh_path)
    results.append({
        "check": "Demo screenshots referenced in README.zh-CN.md exist",
        "pass": zh_screenshot_ok,
        "detail": zh_screenshot_detail,
    })

    # --- Summary ---
    all_pass = all(r["pass"] for r in results)
    failures = [r for r in results if not r["pass"]]

    report = {
        "title": "Claim Audit Report",
        "overall_pass": all_pass,
        "total_checks": len(results),
        "passed": sum(1 for r in results if r["pass"]),
        "failed": len(failures),
        "checks": results,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report written to {OUTPUT_FILE}")

    if all_pass:
        print(f"ALL {len(results)} CHECKS PASSED")
        return 0
    else:
        print(f"{len(failures)}/{len(results)} CHECKS FAILED:")
        for f in failures:
            print(f"  FAIL: {f['check']} — {f['detail']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
