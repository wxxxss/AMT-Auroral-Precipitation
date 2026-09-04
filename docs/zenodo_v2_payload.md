# Zenodo v2 payload for the revised AMT manuscript

This file records the release-specific artifacts that must accompany the finalized public AMT source release.

## Required OP10 artifacts

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `auroramaps_op10_source_used.tar.gz` | Exact standalone `auroramaps` source directory imported by the manuscript evaluation environment | `fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667` |
| `auroramaps_op10_premodel_used.tar.gz` | Exact 45-file OP10 `premodel` coefficient bundle | `a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa` |
| `op10_premodel_manifest.json` | Per-file relative path, byte size, and SHA-256 manifest for the exact coefficient bundle | committed under `third_party/auroramaps_op10/` |

The coefficient directory contains 45 files totaling 98,860,814 uncompressed bytes. `all_premodel_python.p` is 35,390,962 bytes with SHA-256:

```text
0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a
```

## Verified provenance state

The runtime import was confirmed as the tracked package under the AMT development checkout, and `git status --short -- auroramaps` plus `git diff -- auroramaps` produced no output at verification time. The package itself contains no `.git` metadata from its upstream distribution, so no upstream commit SHA is claimed.

## Public AMT source release

Zenodo v2 should also archive the finalized GitHub release/tag corresponding to the revised manuscript. Do not create the immutable Zenodo version until `release/v2-reproducibility` has passed final review and is merged/tagged.

## Release order

1. Verify `third_party/auroramaps_op10/op10_premodel_manifest.json` is present and matches the archived coefficient bundle.
2. Run the full GitHub test workflow and final public-path/provenance audit.
3. Review and merge `release/v2-reproducibility` into the public default branch.
4. Create the manuscript release/tag from the merged public source.
5. Create a new Zenodo version containing the GitHub release plus the exact OP10 source archive, coefficient archive, manifest, and archive metadata.
6. Record the newly assigned **version DOI** in the GitHub README and manuscript Open Research Statement.

Do not replace the exact standalone OP10 artifacts with a current upstream checkout merely to obtain a Git SHA; the archived runtime snapshot and checksums are the reproducibility reference for the revised comparison.
