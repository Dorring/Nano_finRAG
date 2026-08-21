#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def stable(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    art = parser.parse_args().artifact_dir
    checks = []
    for name in ("qwen-dense-index", "atomic-fact-index"):
        value = json.loads((art / f"{name}.json").read_text(encoding="utf-8"))
        checks.append({"name": name, "ok": stable(value) == (art / f"{name}.sha256").read_text().strip(), "sha": stable(value)})
    selected = json.loads((art / "selected-config.json").read_text(encoding="utf-8"))
    config = {k: v for k, v in selected.items() if k not in {"metrics", "safety"}}
    checks.append({"name": "selected-config", "ok": stable(config) == (art / "selected-config.sha256").read_text().strip(), "sha": stable(config)})
    manifest = json.loads((art / "embedding-model-manifest.json").read_text(encoding="utf-8"))
    checks.append({"name": "embedding-manifest", "ok": manifest["snapshot_sha256"] == (art / "embedding-model-manifest.sha256").read_text().strip(), "sha": manifest["snapshot_sha256"]})
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    if not all(x["ok"] for x in checks):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
