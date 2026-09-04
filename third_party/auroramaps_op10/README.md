# Standalone OVATION-Prime / auroramaps snapshot

This directory documents the standalone OVATION-Prime 2010 (OP10) / `auroramaps` package used for the revised AMT manuscript comparison.

## Provenance

The working package was supplied to the AMT authors as a standalone directory rather than as a Git clone. Its Python package metadata identifies:

- package name: `auroramaps`;
- version string: `0.3`;
- author attribution: Christian Moestl / helioforecast;
- license: GNU Lesser General Public License v3.0;
- upstream project: https://github.com/helioforecast/auroramaps.

The supplied directory contained no `.git` metadata. An exact upstream Git commit SHA therefore cannot be reconstructed reliably and is not claimed.

The manuscript environment imported the runtime source from the authors' AMT development checkout. The tracked `auroramaps/` directory had no local Git modifications when the release snapshot was verified. The exact runtime source was archived as:

```text
auroramaps_op10_source_used.tar.gz
```

SHA-256:

```text
fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667
```

This exact standalone source archive will be distributed publicly with the Zenodo v2 release. GitHub retains the AMT-owned evaluation code, provenance record, license, archive metadata, and coefficient manifest.

## Exact coefficient bundle

The OP10 installation used a `premodel` directory containing **45 files** and **98,860,814 uncompressed bytes**. It contains `all_premodel_python.p` and the seasonal diffuse, monoenergetic, wave, ion, number-flux, and probability coefficient files.

The exact bundle is archived as:

```text
auroramaps_op10_premodel_used.tar.gz
```

SHA-256:

```text
a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa
```

The principal serialized coefficient file is:

```text
all_premodel_python.p
size:   35,390,962 bytes
sha256: 0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a
```

The complete byte-level manifest is already committed as `op10_premodel_manifest.json`. It records all 45 relative paths, exact byte sizes, and SHA-256 digests. A current upstream download must not be substituted for this verified working bundle when reproducing the manuscript baseline.

Machine-readable release metadata are stored in `archive_metadata.json`.

## Portable reproduction copy

The immutable source archive preserves the exact historical runtime source, including the working-machine `premodel` path embedded in `auroramaps/ovation.py`. Do not modify that archive.

For reproduction, extract a working copy of both Zenodo artifacts into one directory:

```bash
mkdir -p op10_work
tar -xzf auroramaps_op10_source_used.tar.gz -C op10_work
tar -xzf auroramaps_op10_premodel_used.tar.gz -C op10_work
python tools/prepare_op10_snapshot.py op10_work
```

The portable working layout is then:

```text
op10_work/
  auroramaps/
  premodel/
```

Pass this directory to the public evaluation commands using:

```text
--snapshot-root op10_work
```

`tools/prepare_op10_snapshot.py` changes only the extracted reproduction copy so that `auroramaps/ovation.py` resolves `../premodel` relative to its package directory. It refuses to patch an unrecognized source revision.

## Release split

- **GitHub:** AMT-owned model, preprocessing/training/evaluation code, third-party attribution and LGPLv3 text, archive metadata, manifest, and portable-snapshot tooling.
- **Zenodo v2:** the exact standalone `auroramaps` source archive, exact 45-file `premodel` archive, the same manifest, and the finalized GitHub release snapshot.

## Four-hour driver

The revised AMT comparisons use `evaluation/ovation_driver.py` to construct the standard four-hour weighted Newell coupling driver. The solar wind is first aggregated to hourly means, then the current hour and the preceding three hours are combined with weights `a`, `0.65`, `0.65^2`, and `0.65^3`, where `a` is the fraction of the current hour elapsed.
