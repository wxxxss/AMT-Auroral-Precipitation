# AMT: Auroral Multi-Task Deep Learning Model

Public code and reproducibility workflow for the revised manuscript:

> **Multi-Task Deep Learning Net of Solar Wind Driven Global Auroral Particle Precipitation**

This repository is a curated research release. Its scientific defaults are aligned with the **revised manuscript**, rather than with older development scripts from the private research repository.

## What AMT predicts

AMT predicts four auroral precipitation energy-flux channels:

1. diffuse electrons;
2. monoenergetic electrons;
3. broadband electrons;
4. ions.

The production model uses a 116-dimensional solar-wind/IMF driver vector and nine spatiotemporal skip features. The shared encoder is `116 -> 1024 -> 512 -> 256`; the 256-dimensional representation is concatenated with the nine skip features and passed to four independent `265 -> 128 -> 64 -> 1` regression heads. Dropout is 0.2 and the four log-flux outputs are clipped to `[-6.5, 4.0]`.

The production model contains approximately 0.951 million trainable parameters.

## Repository layout

```text
method/
  model.py                          AMT architecture
  loss.py                           asymmetric four-head regression loss

data/
  preprocess_ssj.py                 DMSP/SSJ extraction, classification, QC and folding
  preprocess_omni_v4.py             OMNI QC, 5-min grid, interpolation, lags and matching
  dataset_v4.py                     116-D AMT features, targets, scaling and skip features

training/
  train_amt.py                      manuscript training/model-selection configuration

evaluation/
  infer_v4_utils.py                 shared AMT inference utilities
  ovation_driver.py                 four-hour weighted Newell driver
  ovation_model.py                  archived OP10 loading/interpolation helpers
  evaluate_pixelwise_ovation.py     sampled 2014 DMSP/SSJ AMT--OVATION comparison
  evaluate_global_skill.py          Table 4 aggregate skill metrics
  boundary_statistics_utils.py      IMAGE boundary utilities
  evaluate_boundary_statistics.py   978-time IMAGE boundary evaluation
  evaluate_hp_timing.py             17 March 2015 hemispheric-power timing workflow
  spatial_diagnostic_utils.py       MLT--MLAT spatial statistics
  evaluate_spatial_mlt_mlat.py      paired spatial error evaluation
  evaluate_spatial_resolution_sensitivity.py
  plot_spatial_diagnostic_polar.py  polar spatial diagnostic

sensitivity/
  history_sensitivity_config.py
  history_sensitivity_eval_utils.py
  train_history_sensitivity.py
  evaluate_history_sensitivity.py

third_party/auroramaps_op10/
  LICENSE
  README.md
  archive_metadata.json
  op10_premodel_manifest.json

tools/
  generate_op10_manifest.py
  prepare_op10_snapshot.py

tests/                              focused scientific regression tests
```

Raw observations, large derived parquet files, fitted scalers, model checkpoints and the 95 MB third-party OP10 coefficient directory are not committed to GitHub.

## Environment

Python 3.11 is used by the repository CI. A local environment can be prepared with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The public entry points use standard PyTorch device arguments. The manuscript training configuration used float32 on a single NPU; users may run the same code on an appropriate installed NPU backend, CUDA device, or CPU where practical.

## 1. DMSP/SSJ preprocessing

The manuscript uses DMSP F16/F17/F18 SSJ/5 measurements. `data/preprocess_ssj.py` implements the target preparation described in Section 2.2:

- background: peak electron differential energy flux `< 1e5 eV/(cm^2 s sr eV)`;
- monoenergetic: peak `>= 2e8`, average electron energy `>= 1000 eV`, and a two-sided spectral decrease;
- broadband: at least three channels `>= 2e8` and average electron energy `>= 100 eV`, excluding spectra already classified as monoenergetic;
- remaining valid non-background spectra with average electron energy `< 1e4 eV`: diffuse;
- spectra that satisfy none of the retained classification rules are discarded;
- electron and ion total energy fluxes are converted with `pi * 1.602e-12` to `erg cm^-2 s^-1`;
- negative, non-finite, or nonphysical total-flux records above the adopted 1000 `erg cm^-2 s^-1` QC limit are removed;
- only `|MLAT| = 50--90 deg` is retained;
- for the hemispherically normalized data set, Southern-Hemisphere MLAT is folded to `|MLAT|`, MLT is unchanged, and `src_hemi` is retained.

Example preparation of the chronological train/validation source and the held-out 2014 source:

```bash
python data/preprocess_ssj.py \
  --cdf-root /path/to/dmsp_ssj \
  --satellites f16 f17 f18 \
  --start-year 2009 \
  --end-year 2013 \
  --output data_local/ssj_2009_2013_fold.parquet

python data/preprocess_ssj.py \
  --cdf-root /path/to/dmsp_ssj \
  --satellites f16 f17 f18 \
  --start-year 2014 \
  --end-year 2014 \
  --output data_local/ssj_2014_fold.parquet
```

Southern-Hemisphere seasonal phase, dipole-tilt sign, and solar-zenith-angle proxy adjustments are applied later by `data/dataset_v4.py`. The real UTC is retained for solar-wind matching.

## 2. OMNI processing and chronological matching

The upstream driver uses seven 5-min OMNI quantities:

```text
Bx (GSE), By (GSM), Bz (GSM), Vx, Vy, Vz, P_dyn
```

The public preprocessing reproduces the manuscript rules:

- magnetic-field components outside `[-500, 500] nT` -> missing;
- velocity components outside `[-3000, 3000] km/s` -> missing;
- dynamic pressure outside `0--90 nPa` -> missing;
- regular 5-min grid;
- only complete internal gaps of at most 30 min are linearly interpolated;
- the required history variables are generated only from the present or earlier OMNI records;
- each SSJ record is matched to the most recent preceding OMNI row with a maximum 10-min tolerance.

Build the 120-min OMNI history used by the production model:

```bash
python data/preprocess_omni_v4.py build-omni \
  --cdf-root /path/to/omni_hro2_5min \
  --start-year 2008 \
  --end-year 2014 \
  --history-minutes 120 \
  --output data_local/omni_2008_2014_hist120.parquet
```

Create the chronological 2009--2012 training set and 2013 validation set using stratified downsampling that preserves the natural auroral-class proportions:

```bash
python data/preprocess_omni_v4.py train-val \
  --ssj-parquet data_local/ssj_2009_2013_fold.parquet \
  --omni-parquet data_local/omni_2008_2014_hist120.parquet \
  --train-output data_local/final_train.parquet \
  --val-output data_local/final_val.parquet
```

Build the full held-out 2014 test set without test-set subsampling:

```bash
python data/preprocess_omni_v4.py test \
  --ssj-parquet data_local/ssj_2014_fold.parquet \
  --omni-parquet data_local/omni_2008_2014_hist120.parquet \
  --output data_local/final_test_2014.parquet
```

With the exact manuscript source data and preprocessing, the reported split sizes are 18,396,472 training samples, 2,254,298 validation samples, and 44,205,248 held-out 2014 test samples.

## 3. AMT feature vector

`data/dataset_v4.py` constructs the manuscript feature representation. The current state contains the seven primitive OMNI variables and ten deterministic derived descriptors: Newell coupling, convection electric field `Ey=-Vx*Bz`, sine/cosine IMF clock angle, southward IMF, transverse magnetic-field magnitude, total magnetic-field magnitude, solar-wind speed, Akasofu epsilon, and dynamic-pressure tendency.

Newell coupling, southward IMF and Akasofu epsilon use `log(1+x)` compression. Four descriptors -- Newell coupling, `Ey`, southward IMF and sine clock angle -- are retained every 5 min from 5 through 120 min before the SSJ observation. One-hour averages of Newell coupling, `Ey` and southward IMF are also included. The resulting production driver dimension is 116.

The regression targets are the four precipitation channels in `log10(F + 1e-6)` space. The solar-wind scaler is fitted on the training set and reused unchanged for validation and test data.

## 4. Train the manuscript configuration

```bash
python training/train_amt.py \
  --train-parquet data_local/final_train.parquet \
  --val-parquet data_local/final_val.parquet \
  --output-dir outputs/amt_training \
  --device cuda
```

The defaults mirror the manuscript training table:

- optimizer: AdamW;
- initial learning rate: `3e-4`;
- weight decay: `1e-2`;
- batch size: `8192`;
- ReduceLROnPlateau: factor `0.5`, patience `8`, minimum learning rate `1e-6`;
- gradient L2 clipping: `5`;
- maximum epochs: `100`;
- early-stopping patience: `50` epochs on aggregate validation loss;
- full training snapshot every `5` epochs;
- float32 precision;
- checkpoint selection by minimum aggregate 2013 validation loss.

The asymmetric regression loss uses active threshold `tau=-5` and underprediction penalties `(5, 50, 50, 10)` for diffuse, monoenergetic, broadband and ion channels.

`outputs/amt_training/` contains the fitted scaler, minimum-validation-loss checkpoint, training history/configuration, and resumable five-epoch snapshots.

## 5. Exact OVATION-Prime / OP10 provenance

All revised comparisons use an OP10 implementation corresponding to the public `helioforecast/auroramaps` codebase with the standard four-hour weighted solar-wind coupling driver.

The working OP10 code was supplied to the authors as a **standalone package**, not obtained by cloning the upstream Git repository. The working directory contained no `.git` metadata, so this release does not invent or claim an unrecoverable upstream commit SHA.

The exact runtime artifacts already identified for the Zenodo v2 record are:

```text
auroramaps_op10_source_used.tar.gz
auroramaps_op10_premodel_used.tar.gz
op10_premodel_manifest.json
```

Checksums:

```text
fc62174f367864457e83622915476ddb5c4a26382b6eca5a9ca27cd9a00f9667  auroramaps_op10_source_used.tar.gz
a68f40945b52b0e77f1e18db37aea598d5ee226f30b7ce66a1e27cf35b1e57fa  auroramaps_op10_premodel_used.tar.gz
```

The exact `premodel` bundle contains 45 files totaling 98,860,814 uncompressed bytes. `all_premodel_python.p` is 35,390,962 bytes with SHA-256:

```text
0a4e913e6bb375a0a49babbc2d322f114f7d045174aebd401a8cc3ec0e01cc7a
```

The complete per-file manifest is committed as `third_party/auroramaps_op10/op10_premodel_manifest.json`.

The immutable source archive intentionally preserves the historical working-machine coefficient path. For a portable reproduction copy, extract both archives into one directory and patch **only the extracted copy**:

```bash
mkdir -p op10_work
tar -xzf auroramaps_op10_source_used.tar.gz -C op10_work
tar -xzf auroramaps_op10_premodel_used.tar.gz -C op10_work
python tools/prepare_op10_snapshot.py op10_work
```

The resulting layout is:

```text
op10_work/
  auroramaps/
  premodel/
```

`evaluation/ovation_driver.py` implements the four-hour weighting used by the manuscript: hourly solar-wind means are converted to Newell coupling values and the current hour plus the preceding three hours are combined with weights `a`, `0.65`, `0.65^2`, `0.65^3`, where `a` is the fraction of the current hour elapsed.

See `third_party/auroramaps_op10/README.md` and `THIRD_PARTY_NOTICES.md` for attribution and license information.

## 6. Pixel-wise AMT--OVATION comparison

Section 4.2.1 uses the full held-out 2014 DMSP/SSJ test set as the source population. The comparison groups are:

- Storm: `Bz < -10 nT` or `P_dyn > 5 nPa`;
- Quiet: `Bz >= -2 nT` and `P_dyn <= 3 nPa`;
- All: no activity filtering.

Up to 30,000 independent UTCs are sampled for Storm and Quiet and up to 60,000 for All; every SSJ record at a selected UTC is retained.

```bash
python evaluation/evaluate_pixelwise_ovation.py \
  --test-data data_local/final_test_2014.parquet \
  --ovation-omni data_local/omni_for_ovation.parquet \
  --model-path outputs/amt_training/aurora_v4_best.pth \
  --scaler-path outputs/amt_training/sw_scaler_v4.pkl \
  --snapshot-root op10_work \
  --output-dir outputs/pixelwise_ovation \
  --seed 42 \
  --device cuda
```

The evaluator reports Pearson correlation, log-space RMSE and prediction efficiency referenced to the observation mean. Its `all_paired.parquet` output is also the source for the manuscript aggregate skill summary.

## 7. Global skill summary

Table 4 uses the same sampled 2014 **All** evaluation set as Section 4.2.1. The aggregate metrics retain paired rows for which both model predictions exceed `1e-4 erg cm^-2 s^-1`. Pearson `R`, KGE and NMedAE are computed in log10 total-flux space; ROC AUC, CSI and accuracy use the physical activity threshold `E_tot >= 0.5 erg cm^-2 s^-1`.

```bash
python evaluation/evaluate_global_skill.py \
  --paired-data outputs/pixelwise_ovation/all_paired.parquet \
  --output-dir outputs/global_skill
```

This command writes the filtered paired sample and the six AMT/OVATION-Prime metrics used for the aggregate comparison.

## 8. IMAGE boundary evaluation

```bash
python evaluation/evaluate_boundary_statistics.py \
  --ealb-txt /path/to/EALB_wic_v2.txt \
  --palb-txt /path/to/PALB_wic_v2.txt \
  --omni-parquet /path/to/omni_with_120min_history.parquet \
  --model-path outputs/amt_training/aurora_v4_best.pth \
  --scaler-path outputs/amt_training/sw_scaler_v4.pkl \
  --snapshot-root op10_work \
  --thin-minutes 60 \
  --device cuda
```

The manuscript protocol requires paired EALB/PALB coverage in at least 18 of 24 MLT sectors, exact IMAGE timestamps, backward-only OMNI matching within 10 min, complete 120-min AMT and four-hour OVATION histories, and one-hour chronological thinning. The resulting evaluation sample contains 978 times: 371 Quiet, 473 Moderate and 134 Strong. Boundary thresholds of 0.25, 0.50 and 1.00 `erg cm^-2 s^-1` are evaluated.

## 9. Hemispheric-power model-response timing

The event-scale comparison uses 17 March 2015, 04:00--18:00 UT, at 5-min cadence. AMT and OVATION-Prime are evaluated on the same polar grid and integrated with the same spherical area elements.

```bash
python evaluation/evaluate_hp_timing.py \
  --omni-history /path/to/omni_history_including_2015.parquet \
  --model-path outputs/amt_training/aurora_v4_best.pth \
  --scaler-path outputs/amt_training/sw_scaler_v4.pkl \
  --snapshot-root op10_work \
  --output-dir outputs/hp_timing \
  --device cuda
```

This diagnostic is a comparison of model-response timing under the two input formulations. It is not presented as an independent observational validation of hemispheric-power accuracy.

## 10. MLT--MLAT spatial diagnostic

The spatial diagnostic deliberately reuses the exact same unique-UTC **All** sampling function and seed as the Section 4.2.1 pixel-wise evaluator, so it follows the manuscript requirement that the aggregate and spatial analyses use the same sampled evaluation set.

```bash
python evaluation/evaluate_spatial_mlt_mlat.py \
  --test-data data_local/final_test_2014.parquet \
  --ovation-omni /path/to/omni_for_ovation.parquet \
  --model-path outputs/amt_training/aurora_v4_best.pth \
  --scaler-path outputs/amt_training/sw_scaler_v4.pkl \
  --snapshot-root op10_work \
  --mlt-bin-hours 0.5 \
  --mlat-bin-deg 1.0 \
  --min-count 20 \
  --seed 42 \
  --device cuda
```

Render the 48 x 40 polar diagnostic with:

```bash
python evaluation/plot_spatial_diagnostic_polar.py \
  --predictions outputs/spatial_diagnostic/spatial_predictions.parquet \
  --mlt-bins 48 \
  --mlat-bins 40 \
  --min-count 20
```

The manuscript analysis contains 1,352 valid spatial bins; AMT has lower local median absolute log-flux error in 89.0% of them, with median `MedAE_OV - MedAE_AMT = 0.466 dex`.

## 11. Solar-wind history sensitivity

The controlled experiment uses 60, 90, 120, 180 and 240 min histories. All five models use the same row population valid for the full 240-min history. The corresponding driver dimensions are 68, 92, 116, 164 and 212.

```bash
python sensitivity/train_history_sensitivity.py \
  --history-minutes 120 \
  --train-parquet /path/to/common_240min_train.parquet \
  --val-parquet /path/to/common_240min_val.parquet \
  --output-root outputs/history_sensitivity \
  --seed 42 \
  --device cuda
```

After all five histories have been trained:

```bash
python sensitivity/evaluate_history_sensitivity.py \
  --test-parquet /path/to/common_240min_2014_test.parquet \
  --run-root outputs/history_sensitivity \
  --output-dir outputs/history_sensitivity_evaluation \
  --seed 42 \
  --device cuda
```

The common manuscript subset contains 12,814,552 training samples, 1,525,176 validation samples and 28,982,325 held-out 2014 test samples.

## Tests

```bash
python -m pytest -q
```

CI runs the focused regression suite on every branch update. The tests cover AMT architecture/loss, manuscript training defaults, SSJ classification/folding, OMNI QC/interpolation/backward matching, shared evaluation sampling, history-dependent feature dimensions, four-hour OVATION weighting, IMAGE matching/boundary statistics, aggregate global skill, MLT--MLAT spatial diagnostics, HP integration, and OP10 archive/manifest utilities.

## Reproducibility policy

This repository is **not a mirror of the private development repository**. Historical exploratory loss functions, obsolete radar diagnostics, instantaneous-OVATION comparison scripts, private executable filesystem paths, raw observations and intermediate artifacts are intentionally excluded. When a historical development implementation differs from the revised manuscript, the public release follows the revised manuscript.

## Citation and Zenodo

A Zenodo **v2** record will be created after this reproducibility branch is reviewed and merged. The new version DOI will then be added here and to the manuscript Open Research Statement.

Until that DOI is assigned, cite the manuscript and this GitHub repository. The exact standalone OP10 source archive and exact coefficient bundle described above are part of the planned Zenodo v2 payload.
