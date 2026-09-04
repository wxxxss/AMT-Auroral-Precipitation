import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_generate_op10_manifest_cli(tmp_path):
    bundle = tmp_path / "premodel"
    bundle.mkdir()
    (bundle / "b.txt").write_bytes(b"beta\n")
    (bundle / "a.txt").write_bytes(b"alpha\n")
    output = tmp_path / "manifest.json"

    script = Path(__file__).resolve().parents[1] / "tools" / "generate_op10_manifest.py"
    result = subprocess.run(
        [sys.executable, str(script), str(bundle), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["file_count"] == 2
    assert manifest["total_bytes"] == len(b"alpha\n") + len(b"beta\n")
    assert [item["path"] for item in manifest["files"]] == ["a.txt", "b.txt"]
    assert manifest["files"][0]["sha256"] == hashlib.sha256(b"alpha\n").hexdigest()
    assert manifest["files"][1]["sha256"] == hashlib.sha256(b"beta\n").hexdigest()
