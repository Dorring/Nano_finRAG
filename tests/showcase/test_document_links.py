"""Tests for document link validity.

Verify all relative markdown links in README.md and README.zh-CN.md point
to files that exist on disk, and all docs/showcase/ files referenced from
the READMEs exist.
"""
from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
README_ZH = REPO_ROOT / "README.zh-CN.md"

# Regex for markdown links: [text](url)
LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)]+)\)")


def _extract_relative_links(md_path: Path) -> list[str]:
    """Extract all relative links from a markdown file.

    Skips http/https URLs and fragments (anchors).
    """
    if not md_path.is_file():
        return []
    text = md_path.read_text(encoding="utf-8")
    links = []
    for m in LINK_RE.finditer(text):
        url = m.group(2).strip()
        # Skip external URLs
        if url.startswith("http://") or url.startswith("https://"):
            continue
        # Skip fragments / anchors
        if url.startswith("#"):
            continue
        # Skip mailto
        if url.startswith("mailto:"):
            continue
        links.append(url)
    return links


def _resolve_link(url: str) -> Path:
    """Resolve a relative link from repo root, stripping any fragment."""
    # Remove fragment/anchor
    if "#" in url:
        url = url.split("#")[0]
    return (REPO_ROOT / url).resolve()


def test_readme_links_are_valid():
    # All relative links in README.md point to existing files
    links = _extract_relative_links(README)
    assert len(links) > 0, "README.md has no relative links to check"
    broken = []
    for url in links:
        target = _resolve_link(url)
        if not target.exists():
            broken.append(f"{url} -> {target}")
    assert not broken, \
        "Broken links in README.md:\n" + "\n".join(broken)


def test_zh_readme_links_are_valid():
    # All relative links in README.zh-CN.md point to existing files
    links = _extract_relative_links(README_ZH)
    assert len(links) > 0, "README.zh-CN.md has no relative links to check"
    broken = []
    for url in links:
        target = _resolve_link(url)
        if not target.exists():
            broken.append(f"{url} -> {target}")
    assert not broken, \
        "Broken links in README.zh-CN.md:\n" + "\n".join(broken)


def test_showcase_docs_exist():
    # All docs/showcase/ files referenced from READMEs exist
    # Collect expected files from both READMEs
    all_links = _extract_relative_links(README) + _extract_relative_links(README_ZH)
    showcase_links = set()
    for url in all_links:
        if url.startswith("docs/showcase/"):
            showcase_links.add(url)

    assert len(showcase_links) > 0, \
        "No docs/showcase/ links found in READMEs"

    missing = []
    for url in sorted(showcase_links):
        target = _resolve_link(url)
        if not target.exists():
            missing.append(f"{url} -> {target}")

    assert not missing, \
        "Missing showcase docs referenced from READMEs:\n" + "\n".join(missing)
