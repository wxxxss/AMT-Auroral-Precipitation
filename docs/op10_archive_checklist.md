# OP10 exact-archive checklist

Use this checklist on the environment that produced the revised OVATION-Prime results.

## 1. Identify the actual imported standalone source

Run from the same Python environment used for the manuscript evaluation:

```bash
python - <<'PY'
from pathlib import Path
import auroramaps
from auroramaps import ovation

print("auroramaps package:", Path(auroramaps.__file__).resolve())
print("ovation module   :", Path(ovation.__file__).resolve())
PY
```

Archive the package directory reported by this command rather than assuming a development-repository path.

## 2. Generate the exact coefficient manifest

```bash
python tools/generate_op10_manifest.py \
  /home/docker/data/private/AuroraData/premodel \
  --output op10_premodel_manifest.json
```

For the working bundle used in the revision, the expected structural inventory is 45 files: one `all_premodel_python.p` file plus 11 coefficient/probability files for each of fall, spring, summer, and winter. The SHA-256 manifest, not names or timestamps alone, is the integrity reference.

## 3. Prepare the archival payload

The final archival payload should contain:

```text
op10_exact_snapshot/
  auroramaps/                    exact standalone source directory used at runtime
  premodel/                      exact working coefficient directory
  op10_premodel_manifest.json
  LICENSE                        LGPLv3 license text
  PROVENANCE.txt                 manuscript/release provenance note
```

Create an archive without modifying the source or coefficient files:

```bash
tar -czf op10_exact_snapshot.tar.gz op10_exact_snapshot/
sha256sum op10_exact_snapshot.tar.gz > op10_exact_snapshot.tar.gz.sha256
```

## 4. GitHub versus archival release

Do not commit the 95-MB coefficient directory to GitHub. GitHub should contain the AMT-owned workflow, provenance/license material, and the generated manifest. The exact standalone source and coefficient bundle should be deposited with the immutable archival release (e.g., Zenodo) and referenced by the manuscript Open Research statement.

## 5. Final verification

After extracting the archive into a clean directory, regenerate the manifest and compare it with the archived manifest before considering the OP10 reproducibility package complete.
