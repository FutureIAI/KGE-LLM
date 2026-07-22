#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


BLOCKED_SUFFIXES = {
    ".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".onnx",
    ".npy", ".npz", ".parquet", ".zip", ".tar", ".tgz", ".7z", ".rar",
    ".docx", ".xlsx",
}
BLOCKED_PARTS = {
    "LLMs", "dataset", "datasets", "cache_data", "checkpoints", "logs",
    "results", "docs_remote_recon", "wandb", "internal", "private",
}
TEXT_SUFFIXES = {
    ".py", ".sh", ".md", ".txt", ".json", ".yaml", ".yml", ".toml",
    ".ini", ".cfg", ".csv", ".gitignore", ".gitattributes",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail closed on unsafe public-release content")
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-file-mb", type=float, default=5.0)
    parser.add_argument("--output", default="release_safety_report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = root / args.output
    max_bytes = int(args.max_file_mb * 1024 ** 2)
    errors = []
    files = []
    path_patterns = [
        re.compile(re.escape("/" + "mnt/data/"), re.I),
        re.compile(re.escape("/" + "Users/"), re.I),
        re.compile(re.escape("/" + "home/") + r"[^/\s]+/", re.I),
        re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ]
    secret_patterns = [
        re.compile(r"(?i)(?:api[_-]?key|access[_-]?token|password|passwd|secret)\s*[:=]\s*['\"][^'\"]+"),
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"(?i)authorization\s*:\s*bearer\s+[A-Za-z0-9._-]+"),
    ]

    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path == output:
            continue
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            errors.append({"path": str(relative), "reason": "symbolic_link"})
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        files.append({"path": str(relative), "size_bytes": size})
        if size > max_bytes:
            errors.append({"path": str(relative), "reason": "file_too_large", "size_bytes": size})
        if path.suffix.lower() in BLOCKED_SUFFIXES:
            errors.append({"path": str(relative), "reason": "blocked_file_type"})
        if any(part in BLOCKED_PARTS or part.startswith("checkpoints_") or part.startswith("cache_data_")
               for part in relative.parts):
            errors.append({"path": str(relative), "reason": "blocked_path_component"})
        is_text = path.suffix.lower() in TEXT_SUFFIXES or path.name in {
            "LICENSE", "NOTICE", "README", ".gitignore", ".gitattributes"
        }
        if is_text and path.name != Path(__file__).name:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in path_patterns:
                if pattern.search(text):
                    errors.append({"path": str(relative), "reason": "machine_specific_path_or_address"})
                    break
            for pattern in secret_patterns:
                if pattern.search(text):
                    errors.append({"path": str(relative), "reason": "possible_secret"})
                    break

    report = {
        "root": root.name,
        "file_count": len(files),
        "total_bytes": sum(item["size_bytes"] for item in files),
        "max_file_mb": args.max_file_mb,
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
