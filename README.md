# AMT: Auroral Multi-Task Deep Learning Model

This repository contains the public source code and manuscript evaluation workflow for the **Auroral Multi-Task (AMT)** model described in:

> **Multi-Task Deep Learning Net of Solar Wind Driven Global Auroral Particle Precipitation**

This v2 reproducibility release corresponds to the revised manuscript and includes the corrected OVATION-Prime comparison, multi-time IMAGE boundary statistics, MLT--MLAT spatial diagnostics, and solar-wind history-length sensitivity analysis.

## Model overview

AMT predicts four auroral precipitation energy-flux channels:

1. diffuse electrons,
2. monoenergetic electrons,
3. broadband electrons, and
4. ions.

The production model uses a shared solar-wind encoder followed by four channel-specific regression heads. The solar-wind driver contains **116 features** built from current OMNI variables, physically motivated derived quantities, 5-min lagged descriptors over the preceding **120 min**, and one-hour aggregate descriptors. Nine spatial/temporal geometry features bypass the shared encoder and enter each regression head directly.

The manuscript model configuration is:

- solar-wind input dimension: 116;
- skip-feature dimension: 9;
- shared encoder: 1024 -> 512 -> 256;
- regression-head hidden widths: 128 -> 64 -> 1;
- dropout: 0.2;
- optimizer: AdamW;
- initial learning rate: `3e-4`;
- weight decay: `1e-2`;
- asymmetric underprediction penalties: `(5, 50, 50, 10)` for diffuse, monoenergetic, broadband, and ion channels.

## Repository structure

```text
method/
  model.py                         AMT architecture
  loss.py                          final asymmetric multi-task loss

data/
  dataset_v4.py                    116-D feature construction and dataset

training/
  train_amt.py                     configurable manuscript-model training entry point

evaluation/
  infer_v4_utils.py                AMT inference utilities
  ovation_driver.py                corrected four-hour weighted Newell driver
  ovation_model.py                 OP10 loading/interpolation helpers
  boundary_statistics_utils.py     IMAGE boundary statistics utilities
  evaluate_boundary_statistics.py  multi-time IMAGE boundary evaluation
  spatial_diagnostic_utils.py      MLT--MLAT binning and spatial metrics
  evaluate_spatial_mlt_mlat.py     paired AMT--OVATION spatial evaluation
  plot_spatial_diagnostic_polar.py final polar Figure-9 diagnostic
  evaluate_spatial_resolution_sensitivity.py

sensitivity/
  history_sensitivity_config.py
  history_sensitivity_eval_utils.py
  train_history_sensitivity.py
  evaluate_history_sensitivity.py

third_party/
  auroramaps_op10/                 standalone OP10/auroramaps snapshot used in revision

tests/                             focused scientific/implementation regression tests
```

The older `evalustion/` directory belongs to the first public release and is not the authoritative workflow for the revised manuscript. The v2 workflow is under `evaluation/`.

## Data

Raw observational data are **not redistributed** through GitHub. The workflow expects locally prepared data derived from the official sources used in the manuscript:

- NASA OMNI 5-min solar-wind and IMF data;
- Defense Meteorological Satellite Program (DMSP) Special Sensor J (SSJ) particle precipitation observations;
- IMAGE/WIC EALB and PALB auroral-boundary products for the boundary evaluation.

Large parquet files, trained checkpoints, scalers, and generated results are intentionally ignored by Git. The final Zenodo release can be used to archive release-specific artifacts that are appropriate for redistribution.

### Required columns for AMT

The AMT dataset expects current OMNI variables

```text
Bx, By, Bz, Vx, Vy, Vz, P_dyn
```

plus the corresponding 5-min lag columns required by the selected history horizon, together with

```text
utc, mlat, mlt, aurora_type, ele_energy_flux, ion_energy_flux
```

and, when available, `src_hemi` for the hemispheric-folding convention described in the manuscript.

## Environment

A minimal Python environment can be created from:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The public code is device-configurable. CPU and CUDA-compatible PyTorch are supported by the released entry points. The original large-scale training and revision experiments were performed on an NPU environment; users of such hardware can adapt the `--device` argument to their installed PyTorch backend.

## Train the manuscript AMT configuration

```bash
python training/train_amt.py \
  --train-parquet /path/to/final_train.parquet \
  --val-parquet /path/to/final_val.parquet \
  --output-dir outputs/amt_training \
  --device cuda
```

The output directory contains the fitted solar-wind scaler, the minimum-validation-loss checkpoint, the training configuration, and the training history.

## Corrected OVATION-Prime baseline

The revised manuscript uses the **OVATION-Prime 2010 (OP10)** implementation corresponding to the public `helioforecast/auroramaps` codebase. The authors received the working copy as a standalone package without `.git` metadata, so this release does **not** claim an unrecoverable upstream commit SHA.

The manuscript comparison uses the standard four-hour weighted Newell coupling driver implemented in `evaluation/ovation_driver.py`:

1. solar wind is aggregated to hourly means;
2. the Newell coupling function is computed for each hourly state;
3. the current hour and the preceding three hours are combined;
4. historical weights are `0.65`, `0.65^2`, and `0.65^3`, while the current-hour weight is the fraction of the current hour that has elapsed.

For folded Southern-Hemisphere SSJ samples, the real UTC is retained for the four-hour solar-wind history and only the OP10 seasonal phase is shifted by 182 days.

The standalone third-party source used for the revised comparison is staged under `third_party/auroramaps_op10/`. See `THIRD_PARTY_NOTICES.md` for provenance and licensing details.

## IMAGE boundary statistics

The revised boundary analysis is run with:

```bash
python evaluation/evaluate_boundary_statistics.py \
  --ealb-txt /path/to/EALB_wic_v2.txt \
  --palb-txt /path/to/PALB_wic_v2.txt \
  --omni-parquet /path/to/omni_with_120min_history.parquet \
  --inventory-only
```

The formal manuscript run then uses the trained AMT model and scaler:

```bash
python evaluation/evaluate_boundary_statistics.py \
  --ealb-txt /path/to/EALB_wic_v2.txt \
  --palb-txt /path/to/PALB_wic_v2.txt \
  --omni-parquet /path/to/omni_with_120min_history.parquet \
  --model-path /path/to/aurora_v4_best.pth \
  --scaler-path /path/to/sw_scaler_v4.pkl \
  --thin-minutes 60 \
  --device cuda
```

The manuscript protocol uses:

- paired IMAGE EALB/PALB coverage >= 18 of 24 MLT sectors;
- exact IMAGE timestamps;
- backward-only OMNI matching with a maximum 10-min offset;
- a complete 120-min AMT history;
- one-hour chronological thinning;
- activity groups Quiet, Moderate, and Strong;
- flux thresholds 0.25, 0.50, and 1.00 erg cm^-2 s^-1;
- paired AMT/OVATION scoring on the same valid IMAGE MLT sectors.

The final corrected inventory contains 12,443 eligible IMAGE times, from which the one-hour thinning retains 978 evaluation times (371 Quiet, 473 Moderate, and 134 Strong).

## MLT--MLAT spatial diagnostic

Generate the paired predictions and spatial metrics with:

```bash
python evaluation/evaluate_spatial_mlt_mlat.py \
  --test-data /path/to/held_out_2014_ssj.parquet \
  --ovation-omni /path/to/omni_2014_2015.parquet \
  --model-path /path/to/aurora_v4_best.pth \
  --scaler-path /path/to/sw_scaler_v4.pkl \
  --mlt-bin-hours 0.5 \
  --mlat-bin-deg 1.0 \
  --min-count 20 \
  --seed 42 \
  --device cuda
```

Then render the polar diagnostic used in the revised manuscript:

```bash
python evaluation/plot_spatial_diagnostic_polar.py \
  --predictions outputs/spatial_diagnostic/spatial_predictions.parquet \
  --mlt-bins 48 \
  --mlat-bins 40 \
  --min-count 20
```

For the final 48 x 40 grid with at least 20 paired samples per displayed bin, the manuscript analysis contains 1,352 valid bins. AMT has the lower local median absolute log-flux error in approximately 89.0% of these bins, with a median `MedAE_OV - MedAE_AMT` of 0.466 dex.

The grid-resolution post-processing check can be reproduced with:

```bash
python evaluation/evaluate_spatial_resolution_sensitivity.py \
  --predictions outputs/spatial_diagnostic/spatial_predictions.parquet
```

## Solar-wind history-length sensitivity

The controlled sensitivity experiment compares 60, 90, 120, 180, and 240 min of recent solar-wind history. All five models must use the **same row population valid for the complete 240-min history**, so only the exposed lag horizon changes.

Expected solar-wind input dimensions are:

| History | Input dimension |
|---:|---:|
| 60 min | 68 |
| 90 min | 92 |
| 120 min | 116 |
| 180 min | 164 |
| 240 min | 212 |

Train one configuration, for example:

```bash
python sensitivity/train_history_sensitivity.py \
  --history-minutes 120 \
  --train-parquet /path/to/common_240min_train.parquet \
  --val-parquet /path/to/common_240min_val.parquet \
  --output-root outputs/history_sensitivity \
  --seed 42 \
  --device cuda
```

After all five runs are complete:

```bash
python sensitivity/evaluate_history_sensitivity.py \
  --test-parquet /path/to/common_240min_2014_test.parquet \
  --run-root outputs/history_sensitivity \
  --output-dir outputs/history_sensitivity_evaluation \
  --seed 42 \
  --device cuda
```

The common manuscript subset contains 12,814,552 training samples, 1,525,176 validation samples, and 28,982,325 held-out 2014 test samples.

## Tests

```bash
python -m pytest -q
```

The focused tests cover the AMT architecture/loss, history-dependent input dimensions, exact backward IMAGE--OMNI matching, duplicate IMAGE timestamps, common-bin boundary scoring, four-hour OVATION driver construction, and MLT--MLAT spatial metrics.

## Reproducibility and provenance

This repository is a **curated manuscript release**, not a mirror of the private development repository. Obsolete experimental loss functions, instantaneous-OVATION evaluators, private filesystem paths, raw data, and intermediate development artifacts are intentionally excluded.

Third-party OVATION-Prime attribution and license information are recorded in `THIRD_PARTY_NOTICES.md`.

## Citation and Zenodo

A Zenodo v2 record will be created from the finalized public release after review of this branch. The v2 DOI will be added here and to the manuscript Open Research Statement after Zenodo assigns it.

If you use AMT before the v2 DOI is assigned, please cite the manuscript and this GitHub repository.
