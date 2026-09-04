# Third-Party Software Notices

## OVATION-Prime / auroramaps

The revised AMT manuscript compares AMT with the OVATION-Prime 2010 (OP10) model through a standalone copy of the `auroramaps` implementation that was supplied to the authors without Git metadata.

Upstream project: https://github.com/helioforecast/auroramaps

The standalone package identifies its OVATION implementation as derived from the `auroramaps`/OvationPyme lineage and states that it is distributed under the GNU Lesser General Public License v3.0 (LGPL-3.0). The upstream LGPLv3 license text is preserved under `third_party/auroramaps_op10/LICENSE`.

Because the supplied package did not contain a `.git` directory or other reliable upstream revision metadata, this repository does **not** claim an exact upstream Git commit SHA for that snapshot.

The exact source directory used by the manuscript run has been archived as `auroramaps_op10_source_used.tar.gz` with SHA-256 `fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667`.

The exact OP10 regression-coefficient bundle used in the revised analysis contains 45 files totaling 98,860,814 uncompressed bytes. It has been archived as `auroramaps_op10_premodel_used.tar.gz` with SHA-256 `a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa`. The bundled `all_premodel_python.p` file is 35,390,962 bytes with SHA-256 `0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a`.

The large third-party coefficient files are not committed to GitHub. The GitHub release preserves provenance, the upstream license, archive metadata, the SHA-256 coefficient manifest, and the AMT-owned comparison workflow. The exact source and coefficient archives are intended to be deposited with the Zenodo v2 release.

Machine-readable archive metadata are provided in `third_party/auroramaps_op10/archive_metadata.json`.

No copyright ownership of the third-party OVATION-Prime/auroramaps implementation or coefficient data is claimed by the AMT authors.
