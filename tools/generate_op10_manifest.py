#!/usr/bin/env python3
"""Generate a deterministic SHA-256 manifest for an OP10 premodel directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(root: Path) -> dict:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Premodel directory does not exist: {root}")

    files = []
    total_bytes = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": sha256_file(path),
            }
        )

    return {
        "algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("premodel_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    manifest = build_manifest(args.premodel_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {manifest['file_count']} files / {manifest['total_bytes']} bytes "
        f"to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
