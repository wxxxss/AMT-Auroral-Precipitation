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

## Exact coefficient bundle used in the revised analysis

The authors' working OP10 installation uses a `premodel` directory containing **45 files** (approximately 95 MB): `all_premodel_python.p` plus the seasonal diffuse, monoenergetic, wave, ion, number-flux, and probability-coefficient files for fall, spring, summer, and winter.

The coefficient files are third-party model data and are intentionally not committed to the GitHub repository. Instead, the final archival release should contain the **exact working `premodel` directory** together with a SHA-256 manifest. Generate that manifest directly from the working directory with:

```bash
python tools/generate_op10_manifest.py \
  /home/docker/data/private/AuroraData/premodel \
  --output op10_premodel_manifest.json
```

The manifest records every relative path, exact byte size, and SHA-256 digest. This is the byte-level reproducibility reference; a visually similar file list or an unverified current upstream download must not be substituted for the working bundle.

## Release layout

The intended release split is:

- **GitHub:** AMT-owned source code, evaluation workflow, provenance notes, LGPLv3 license text, and the generated OP10 manifest;
- **Zenodo release archive:** the exact standalone `auroramaps` source used by the authors, the exact 45-file `premodel` bundle, the same manifest, and release metadata/checksums.

After extracting the archived OP10 bundle for reproduction, place or link its coefficient directory at:

```text
third_party/auroramaps_op10/premodel/
```

The GitHub `.gitignore` explicitly excludes that directory so the large third-party bundle is not accidentally committed.

## Four-hour driver

The AMT-owned comparison code uses `evaluation/ovation_driver.py` for the corrected four-hour weighted Newell coupling preprocessing. The exact standalone OP10 implementation and coefficient bundle remain a separate third-party reproducibility artifact.
