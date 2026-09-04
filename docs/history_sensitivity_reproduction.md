# Reproducing the solar-wind history-length sensitivity experiment

The revised manuscript compares 60, 90, 120, 180, and 240 min histories on **exactly the same train/validation/test rows**. The common population is defined by requiring a complete 240-min OMNI history first; each model then exposes only the lag columns corresponding to its assigned history length.

## 1. Build the common 240-min OMNI table

Use the same raw OMNI source and quality-control rules as the production 120-min pipeline, changing only the history horizon:

```bash
python data/preprocess_omni_v4.py build-omni \
  --cdf-root /path/to/omni_hro2_5min \
  --start-year 2008 \
  --end-year 2014 \
  --history-minutes 240 \
  --output data_local/omni_2008_2014_hist240_common.parquet
```

## 2. Build common train/validation rows

```bash
python data/preprocess_omni_v4.py train-val \
  --ssj-parquet data_local/ssj_2009_2013_fold.parquet \
  --omni-parquet data_local/omni_2008_2014_hist240_common.parquet \
  --train-output data_local/common_240min_train.parquet \
  --val-output data_local/common_240min_val.parquet
```

## 3. Build the common held-out 2014 rows

```bash
python data/preprocess_omni_v4.py test \
  --ssj-parquet data_local/ssj_2014_fold.parquet \
  --omni-parquet data_local/omni_2008_2014_hist240_common.parquet \
  --output data_local/common_240min_2014_test.parquet
```

For the exact manuscript inputs this produces 12,814,552 training samples, 1,525,176 validation samples, and 28,982,325 held-out 2014 test samples.

## 4. Train all five configurations

Run `sensitivity/train_history_sensitivity.py` for each history in `60 90 120 180 240`, always passing the same two common parquet files. The script changes the exposed lag horizon and corresponding input dimension while retaining the manuscript optimization defaults and checkpoint-selection criterion.

Example:

```bash
python sensitivity/train_history_sensitivity.py \
  --history-minutes 120 \
  --train-parquet data_local/common_240min_train.parquet \
  --val-parquet data_local/common_240min_val.parquet \
  --output-root outputs/history_sensitivity \
  --seed 42 \
  --device cuda
```

Expected driver dimensions are 68, 92, 116, 164, and 212 for 60, 90, 120, 180, and 240 min, respectively.

## 5. Evaluate the common 2014 subset

```bash
python sensitivity/evaluate_history_sensitivity.py \
  --test-parquet data_local/common_240min_2014_test.parquet \
  --run-root outputs/history_sensitivity \
  --output-dir outputs/history_sensitivity_evaluation \
  --seed 42 \
  --device cuda
```

The evaluator reports the all-sample metrics and the manuscript robustness check that excludes predictions below `1e-4 erg cm^-2 s^-1`.
