from pathlib import Path

from tools.prepare_op10_snapshot import patch_premodel_path


def test_patch_premodel_path_replaces_only_working_machine_path(tmp_path):
    ovation_py = tmp_path / "auroramaps" / "ovation.py"
    ovation_py.parent.mkdir(parents=True)
    original = (
        "import os\n"
        "class SeasonalFluxEstimator:\n"
        "    def __init__(self):\n"
        "        self.premodel_directory='/home/docker/data/private/AuroraData/premodel/'       #define premodel directory\n"
        "        self.other = 'unchanged'\n"
    )
    ovation_py.write_text(original, encoding="utf-8")

    changed = patch_premodel_path(ovation_py)

    assert changed is True
    patched = ovation_py.read_text(encoding="utf-8")
    assert "/home/docker/data/private/AuroraData/premodel/" not in patched
    assert "os.path.dirname(__file__)" in patched
    assert "'premodel'" in patched
    assert "self.other = 'unchanged'" in patched


def test_patch_premodel_path_is_idempotent(tmp_path):
    ovation_py = tmp_path / "auroramaps" / "ovation.py"
    ovation_py.parent.mkdir(parents=True)
    ovation_py.write_text(
        "import os\n"
        "self.premodel_directory=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'premodel')) + os.sep\n",
        encoding="utf-8",
    )

    changed = patch_premodel_path(ovation_py)

    assert changed is False
