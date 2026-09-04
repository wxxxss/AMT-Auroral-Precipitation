# Zenodo v2 payload for the revised AMT manuscript

This file records the release-specific third-party artifacts that must accompany the finalized public AMT source release.

## Required OP10 artifacts

| Artifact | Purpose | SHA-256 |
|---|---|---|
| `auroramaps_op10_source_used.tar.gz` | Exact standalone `auroramaps` source directory imported by the manuscript evaluation environment | `fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667` |
| `auroramaps_op10_premodel_used.tar.gz` | Exact 45-file OP10 `premodel` coefficient bundle | `a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa` |
| `op10_premodel_manifest.json` | Per-file relative path, byte size, and SHA-256 manifest for the coefficient bundle | generated from the working bundle; add the exact file to this repository before release |

The coefficient directory contains 45 files totaling 98,860,814 uncompressed bytes. `all_premodel_python.p` is 35,390,962 bytes and has SHA-256:

```text
0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a
```

## Public AMT source release

Zenodo v2 should also archive the finalized GitHub release/tag corresponding to the revised manuscript. Do not create the immutable Zenodo version until the `release/v2-reproducibility` branch has passed final review and has been merged/tagged.

## Release order

1. Add the complete `op10_premodel_manifest.json` to `third_party/auroramaps_op10/`.
2. Verify that the runtime `auroramaps` source has not been locally modified relative to the source snapshot being archived, or explicitly document any local-only modifications.
3. Run the full GitHub test workflow and final public-path/provenance audit.
4. Merge the release branch and create the manuscript release/tag.
5. Create a new Zenodo version containing the GitHub release plus the exact OP10 artifacts above.
6. Record the newly assigned version DOI in the GitHub README and manuscript Open Research Statement.
