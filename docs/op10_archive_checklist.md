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

Archive the package directory reported by this command rather than assuming a development-repository path. For the revised analysis, the runtime package was checked against the tracked development copy and had no local source modifications.

## 2. Generate and verify the exact coefficient manifest

Run the public manifest utility against the coefficient directory used by the runtime installation:

```bash
python tools/generate_op10_manifest.py \
  /path/to/the/runtime/premodel \
  --output op10_premodel_manifest.json
```

For the working bundle used in the revision, the verified inventory is:

- 45 files;
- 98,860,814 uncompressed bytes;
- `all_premodel_python.p`: 35,390,962 bytes;
- `all_premodel_python.p` SHA-256: `0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a`.

The complete verified manifest is committed as `third_party/auroramaps_op10/op10_premodel_manifest.json`. SHA-256 values, rather than filenames or timestamps alone, are the integrity reference.

## 3. Exact Zenodo v2 artifacts

The two exact runtime archives already prepared for the revised release are:

```text
auroramaps_op10_source_used.tar.gz
auroramaps_op10_premodel_used.tar.gz
op10_premodel_manifest.json
```

Their verified archive checksums are:

```text
fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667  auroramaps_op10_source_used.tar.gz
a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa  auroramaps_op10_premodel_used.tar.gz
```

Keep the exact source archive immutable. It intentionally preserves the historical working-machine path used by the runtime source.

## 4. Portable reproduction copy

For reproduction, extract both immutable archives into the same working directory and patch only that extracted copy:

```bash
mkdir -p op10_work
tar -xzf auroramaps_op10_source_used.tar.gz -C op10_work
tar -xzf auroramaps_op10_premodel_used.tar.gz -C op10_work
python tools/prepare_op10_snapshot.py op10_work
```

The resulting portable layout is:

```text
op10_work/
  auroramaps/
  premodel/
```

## 5. GitHub versus archival release

Do not commit the 95-MB coefficient directory to GitHub. GitHub contains the AMT-owned workflow, provenance/license material, archive metadata, and the verified manifest. Zenodo v2 should contain the exact standalone source archive and coefficient archive above, the same manifest, archive metadata/checksums, and the finalized public AMT source release/tag.

## 6. Final verification

Before publishing the immutable Zenodo version:

1. verify the two archive SHA-256 values above;
2. extract the coefficient archive into a clean directory;
3. regenerate its per-file manifest;
4. compare the regenerated manifest with `third_party/auroramaps_op10/op10_premodel_manifest.json`;
5. verify the GitHub release/tag corresponds to the reviewed public manuscript code.
