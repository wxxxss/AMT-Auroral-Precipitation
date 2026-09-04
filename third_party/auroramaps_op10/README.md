# Standalone OVATION-Prime / auroramaps snapshot

This directory is reserved for the standalone OVATION-Prime 2010 (OP10) / `auroramaps` package used for the revised AMT manuscript comparison.

## Provenance

The working package was supplied to the AMT authors as a standalone directory rather than as a Git clone. Its Python package metadata identifies:

- package name: `auroramaps`;
- version string: `0.3`;
- author attribution: Christian Moestl / helioforecast;
- license: GNU Lesser General Public License v3.0;
- upstream project: https://github.com/helioforecast/auroramaps.

The supplied directory contained no `.git` metadata, so an exact upstream Git commit SHA cannot be reconstructed reliably and is therefore not claimed in the manuscript or this release.

## Important staging note

The private development repository contains the Python source files used in the revised comparison, but the OP10 regression-coefficient bundle referenced by that standalone source (`premodel`) is stored outside the private Git repository. Consequently, the exact runnable third-party snapshot cannot be completed automatically from Git history alone.

Before this release branch is merged, the original standalone `auroramaps` source **and the exact `premodel` coefficient directory used for the revised analysis** should be copied into this directory from the authors' working package. The release must preserve the upstream attribution and LGPLv3 license. No substitute upstream commit or coefficient bundle should be presented as the exact manuscript snapshot unless it has been verified byte-for-byte against the working copy.

The AMT-owned evaluation code under `evaluation/` is already written to load the archived package from:

```text
third_party/auroramaps_op10/auroramaps/
```

and uses `evaluation/ovation_driver.py` for the corrected four-hour weighted Newell coupling preprocessing.
