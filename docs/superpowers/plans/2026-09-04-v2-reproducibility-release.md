# AMT v2 Reproducibility Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a curated, reproducible v2 code release for the AMT manuscript by migrating the final scientific implementation from the private development repository into the public repository without exposing raw data, private paths, obsolete evaluators, or unrelated development artifacts.

**Architecture:** The public repository will expose four layers: AMT model/data/training code, manuscript evaluation code, sensitivity-analysis code, and the exact standalone OVATION-Prime/auroramaps snapshot used for the revised comparison. Public entry points must use relative paths or CLI arguments, while raw data and trained artifacts remain external unless explicitly archived for the release.

**Tech Stack:** Python, PyTorch, pandas, NumPy, scikit-learn, SciPy, Matplotlib, Git/GitHub, Zenodo.

**Spec:** User-approved migration design in the 2026-09-04 ChatGPT session.

## Global Constraints

- Never publish raw OMNI, DMSP/SSJ, IMAGE, private parquet files, credentials, or user-specific filesystem content.
- Do not publish obsolete instantaneous-OVATION evaluators as manuscript reproduction scripts.
- The OVATION-Prime code must be labeled as a standalone OP10 snapshot with unrecoverable upstream Git metadata; preserve upstream attribution and LGPLv3 licensing.
- Final manuscript evaluators must use the corrected four-hour weighted OVATION driver where applicable.
- Scientific thresholds, sample definitions, and reported metrics must match the revised manuscript.
- Work only on branch `release/v2-reproducibility` until review.

---

### Task 1: Audit and release scaffold

**Files:**
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `.gitignore` additions if needed
- Modify: `README.md`

- [ ] Inventory final private-branch files used for the manuscript and classify them as publish / sanitize / exclude.
- [ ] Identify all absolute private paths and environment-specific imports in files selected for publication.
- [ ] Record the public repository release structure and provenance rules in README/third-party notices.
- [ ] Commit the release scaffold.

### Task 2: Publish core AMT implementation

**Files:**
- Create/replace: `method/model.py`
- Create: `method/loss.py`
- Create: `data/dataset_v4.py`
- Create: `training/train_amt.py`

- [ ] Migrate the final shared-backbone four-head architecture.
- [ ] Migrate only the loss actually used for the paper, not experimental loss variants.
- [ ] Migrate the 116-dimensional feature construction and skip-feature implementation.
- [ ] Replace hard-coded private paths in the training entry point with command-line parameters.
- [ ] Verify imports and configuration values against the manuscript.
- [ ] Commit the core implementation.

### Task 3: Publish final inference and OVATION comparison code

**Files:**
- Create: `evaluation/infer_v4_utils.py`
- Create: corrected point/global comparison script using four-hour OVATION driver
- Create: corrected hemispheric-power comparison script
- Create: corrected IMAGE boundary comparison script

- [ ] Migrate shared inference utilities with configurable paths/devices.
- [ ] Ensure all manuscript-facing OVATION comparison entry points use the corrected four-hour weighted driver.
- [ ] Exclude old instantaneous-OVATION manuscript evaluators from the v2 reproduction workflow.
- [ ] Commit the comparison layer.

### Task 4: Publish reviewer-added statistical evaluation packages

**Files:**
- Create: boundary-statistics evaluator/utilities and focused tests.
- Create: MLT--MLAT spatial evaluator/plotter and focused tests.
- Create: history-length sensitivity evaluator/configuration and focused tests.

- [ ] Migrate the exact-time IMAGE boundary inventory/statistics implementation, including duplicate-timestamp and backward-OMNI matching tests.
- [ ] Migrate the final 48x40 MLT--MLAT diagnostic used for Figure 9 and its resolution-sensitivity support code.
- [ ] Migrate the common-subset 60/90/120/180/240-min history sensitivity workflow.
- [ ] Commit each analysis package after import/static checks.

### Task 5: Archive the standalone OVATION-Prime snapshot

**Files:**
- Create: `third_party/auroramaps_op10/README.md`
- Create: `third_party/auroramaps_op10/LICENSE`
- Create: source files copied from the exact standalone package used in the revised analysis.
- Create: `third_party/auroramaps_op10/SHA256SUMS` or equivalent manifest when the exact byte-level snapshot is available.

- [ ] Preserve source attribution and LGPLv3 license text.
- [ ] State explicitly that the supplied package had no `.git` metadata and therefore no reliable upstream commit SHA can be claimed.
- [ ] Distinguish upstream project URL from the exact archived snapshot used in this study.
- [ ] Commit the third-party snapshot.

### Task 6: README, verification, and release handoff

**Files:**
- Modify: `README.md`

- [ ] Document repository structure, data sources, environment, model inputs, training, inference, manuscript evaluations, OVATION provenance, and citation guidance.
- [ ] Verify that no selected public file contains private absolute paths, secrets, raw-data payloads, or stale manuscript-result wording.
- [ ] Verify that final public scripts refer to the corrected four-hour OVATION workflow and current sample definitions.
- [ ] Open a PR from `release/v2-reproducibility` to `main` for user review.
- [ ] After merge, user creates Zenodo v2; update README and manuscript Open Research Statement with the resulting DOI.
