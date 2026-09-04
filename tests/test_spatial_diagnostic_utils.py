import numpy as np
import pandas as pd
import pytest

from evaluation.spatial_diagnostic_utils import compute_spatial_metrics, summarize_spatial_metrics


def test_spatial_delta_medae_positive_when_amt_is_better():
    df = pd.DataFrame(
        {
            "mlt": [0.1, 0.2, 0.3, 0.4],
            "mlat": [60.1, 60.2, 60.3, 60.4],
            "true_flux": [1.0, 2.0, 4.0, 8.0],
            "pred_mlp_total": [1.0, 2.0, 4.0, 8.0],
            "pred_ovation": [0.1, 0.2, 0.4, 0.8],
        }
    )
    metrics = compute_spatial_metrics(
        df,
        mlt_bin_hours=0.5,
        mlat_bin_deg=1.0,
        mlat_min=50.0,
        mlat_max=90.0,
        min_count=4,
    )
    valid = metrics[metrics["valid"]]
    assert len(valid) == 1
    assert valid.iloc[0]["delta_medae"] > 0
    summary = summarize_spatial_metrics(metrics)
    assert summary["n_valid_bins"] == 1
    assert summary["amt_better_bin_fraction"] == pytest.approx(1.0)


def test_near_zero_finite_predictions_are_retained_with_log_floor():
    df = pd.DataFrame(
        {
            "mlt": [1.0, 1.1],
            "mlat": [70.0, 70.1],
            "true_flux": [0.0, 1e-8],
            "pred_mlp_total": [0.0, 0.0],
            "pred_ovation": [0.0, 1e-9],
        }
    )
    metrics = compute_spatial_metrics(df, min_count=2)
    valid = metrics[metrics["valid"]]
    assert len(valid) == 1
    assert np.isfinite(valid.iloc[0]["amt_medae"])
    assert np.isfinite(valid.iloc[0]["ov_medae"])
