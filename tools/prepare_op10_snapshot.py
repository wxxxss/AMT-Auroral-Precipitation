"""Prepare an extracted OP10 source snapshot for portable reproduction.

The immutable Zenodo source archive preserves the exact manuscript runtime source,
including its original working-machine ``premodel`` path. This helper modifies only
an extracted reproduction copy so that ``auroramaps/ovation.py`` resolves the
coefficient bundle from ``../premodel`` relative to the package directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path


_WORKING_PATH_LINE = (
    "self.premodel_directory='/home/docker/data/private/AuroraData/premodel/'"
)
_PORTABLE_PATH_LINE = (
    "self.premodel_directory=os.path.abspath("
    "os.path.join(os.path.dirname(__file__), '..', 'premodel')) + os.sep"
)


def patch_premodel_path(ovation_py: Path) -> bool:
    """Replace the manuscript working-machine coefficient path in one copy.

    Returns ``True`` when the exact historical path is replaced and ``False``
    when the file is already portable. Any other source state raises an error so
    that an unexpected third-party source revision is not silently modified.
    """
    ovation_py = Path(ovation_py)
    text = ovation_py.read_text(encoding="utf-8")

    if _WORKING_PATH_LINE in text:
        patched = text.replace(_WORKING_PATH_LINE, _PORTABLE_PATH_LINE, 1)
        ovation_py.write_text(patched, encoding="utf-8")
        return True

    if _PORTABLE_PATH_LINE in text:
        return False

    raise ValueError(
        "Expected OP10 premodel path was not found. Refusing to modify an "
        "unrecognized auroramaps/ovation.py snapshot."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch an extracted exact OP10 snapshot for portable use."
    )
    parser.add_argument(
        "snapshot_root",
        type=Path,
        help="Directory containing auroramaps/ovation.py and premodel/.",
    )
    args = parser.parse_args()

    root = args.snapshot_root.resolve()
    ovation_py = root / "auroramaps" / "ovation.py"
    if not ovation_py.is_file():
        raise FileNotFoundError(f"Missing {ovation_py}")
    if not (root / "premodel").is_dir():
        raise FileNotFoundError(f"Missing coefficient directory: {root / 'premodel'}")

    changed = patch_premodel_path(ovation_py)
    print("patched" if changed else "already portable")


if __name__ == "__main__":
    main()
