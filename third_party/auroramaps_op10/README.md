# Standalone OVATION-Prime / auroramaps snapshot

This directory documents the standalone OVATION-Prime 2010 (OP10) / `auroramaps` package used for the revised AMT manuscript comparison.

## Provenance

The working package was supplied to the AMT authors as a standalone directory rather than as a Git clone. Its Python package metadata identifies:

- package name: `auroramaps`;
- version string: `0.3`;
- author attribution: Christian Moestl / helioforecast;
- license: GNU Lesser General Public License v3.0;
- upstream project: https://github.com/helioforecast/auroramaps.

The supplied directory contained no `.git` metadata, so an exact upstream Git commit SHA cannot be reconstructed reliably and is therefore not claimed in the manuscript or this release.

The manuscript runs imported `auroramaps` from the authors' private AMT development checkout. The exact source directory used at runtime has been archived as:

```text
auroramaps_op10_source_used.tar.gz
```

with SHA-256:

```text
fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667
```

The source archive is intended for the Zenodo v2 release. GitHub retains the provenance record, license, and AMT-owned wrapper/evaluation code.

## Exact coefficient bundle used in the revised analysis

The working OP10 installation uses a `premodel` directory containing **45 files** with a total uncompressed size of **98,860,814 bytes**. The directory contains `all_premodel_python.p` plus the seasonal diffuse, monoenergetic, wave, ion, number-flux, and probability-coefficient files for fall, spring, summer, and winter.

The coefficient bundle has been archived as:

```text
auroramaps_op10_premodel_used.tar.gz
```

with SHA-256:

```text
a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa
```

The principal serialized coefficient file is:

```text
all_premodel_python.p
size:   35,390,962 bytes
sha256: 0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a
```

The coefficient files are third-party model data and are intentionally not committed to the GitHub repository. Instead, the final Zenodo archival release will contain the **exact working `premodel` directory** together with a SHA-256 manifest. Generate that manifest directly from the working directory with:

```bash
python tools/generate_op10_manifest.py \
  /path/to/premodel \
  --output op10_premodel_manifest.json
```

The manifest records every relative path, exact byte size, and SHA-256 digest. This is the byte-level reproducibility reference; a visually similar file list or an unverified current upstream download must not be substituted for the working bundle.

Machine-readable archive metadata are stored in `archive_metadata.json`.

## Release layout

The release split is:

- **GitHub:** AMT-owned source code, evaluation workflow, provenance notes, LGPLv3 license text, archive metadata, and the generated OP10 manifest;
- **Zenodo v2:** the exact standalone `auroramaps` source archive used by the authors, the exact 45-file `premodel` archive, the same manifest, and release metadata/checksums.

After extracting the archived OP10 bundle for reproduction, place or link its coefficient directory at:

```text
third_party/auroramaps_op10/premodel/
```

The GitHub `.gitignore` explicitly excludes that directory so the large third-party bundle is not accidentally committed.

## Four-hour driver

The AMT-owned comparison code uses `evaluation/ovation_driver.py` for the corrected four-hour weighted Newell coupling preprocessing. The standalone OP10 implementation and coefficient bundle remain separate third-party reproducibility artifacts.
