from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXECUTABLE_DIRS = ["data", "evaluation", "method", "training", "sensitivity"]


def _python_files():
    for dirname in EXECUTABLE_DIRS:
        yield from (ROOT / dirname).rglob("*.py")


def test_public_executable_code_contains_no_working_machine_paths():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "/home/docker" in text or "/data/private/" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_public_executable_code_has_no_npu_only_imports():
    offenders = []
    for path in _python_files():
        text = path.read_text(encoding="utf-8")
        if "import torch_npu" in text or "from torch_npu" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_obsolete_first_release_evaluation_directory_is_absent():
    assert not (ROOT / "evalustion").exists()
