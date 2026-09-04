import numpy as np
import pandas as pd

from evaluation.evaluate_global_skill import (
    binary_skill_metrics,
    continuous_skill_metrics,
    filter_global_skill_sample,
)


def test_global_skill_filter_requires_both_model_predictions_above_threshold():
    paired = pd.DataFrame(
        {
            "obs_total": [1.0, 2.0, 3.0, 4.0],
            "pred_amt_total": [1.0, 1e-5, 2.0, np.nan],
            "pred_ovation": [1.0, 2.0, 1e-5, 2.0],
        }
    )
    out = filter_global_skill_sample(paired, prediction_min=1e-4)
    assert len(out) == 1
    assert out.iloc[0]["obs_total"] == 1.0


def test_continuous_metrics_are_perfect_for_identical_nonconstant_fluxes():
    flux = np.array([0.1, 0.3, 1.0, 3.0], dtype=float)
    metrics = continuous_skill_metrics(flux, flux, epsilon=1e-6)
    assert np.isclose(metrics["R"], 1.0)
    assert np.isclose(metrics["KGE"], 1.0)
    assert np.isclose(metrics["NMedAE"], 0.0)


def test_binary_metrics_match_manuscript_threshold_definition():
    observed = np.array([0.1, 1.0, 2.0, 0.2], dtype=float)
    predicted = np.array([0.1, 0.9, 0.8, 0.2], dtype=float)
    metrics = binary_skill_metrics(observed, predicted, activity_threshold=0.5)
    assert np.isclose(metrics["ROC_AUC"], 1.0)
    assert np.isclose(metrics["CSI"], 1.0)
    assert np.isclose(metrics["Accuracy"], 1.0)
