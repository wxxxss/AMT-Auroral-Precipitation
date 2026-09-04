import numpy as np
import pandas as pd

from evaluation.evaluate_pixelwise_ovation import (
    activity_mask,
    regression_metrics_log10,
    sample_group_by_unique_utc,
)


def test_activity_groups_match_manuscript_definitions():
    df = pd.DataFrame(
        {
            "Bz": [-11.0, -1.0, -5.0, -1.0],
            "P_dyn": [2.0, 2.0, 6.0, 4.0],
        }
    )
    assert activity_mask(df, "Storm").tolist() == [True, False, True, False]
    assert activity_mask(df, "Quiet").tolist() == [False, True, False, False]
    assert activity_mask(df, "All").tolist() == [True, True, True, True]


def test_sampling_selects_unique_utcs_but_retains_all_records_at_each_time():
    df = pd.DataFrame(
        {
            "utc": pd.to_datetime(
                [
                    "2014-01-01 00:00",
                    "2014-01-01 00:00",
                    "2014-01-01 00:05",
                    "2014-01-01 00:05",
                    "2014-01-01 00:10",
                ]
            ),
            "Bz": [-1.0] * 5,
            "P_dyn": [2.0] * 5,
        }
    )

    sampled = sample_group_by_unique_utc(df, "All", max_unique_times=2, seed=42)

    assert sampled["utc"].nunique() == 2
    counts = sampled.groupby("utc").size()
    # A selected timestamp must bring every SSJ record sharing that timestamp.
    for utc in counts.index:
        assert counts.loc[utc] == (df["utc"] == utc).sum()


def test_log_metrics_use_observation_mean_for_prediction_efficiency():
    observed = np.array([1.0, 10.0, 100.0])
    predicted = np.array([1.0, 10.0, 10.0])

    metrics = regression_metrics_log10(observed, predicted)

    obs_log = np.log10(observed)
    pred_log = np.log10(predicted)
    expected_rmse = np.sqrt(np.mean((pred_log - obs_log) ** 2))
    expected_pe = 1.0 - np.mean((pred_log - obs_log) ** 2) / np.mean(
        (obs_log - obs_log.mean()) ** 2
    )
    assert np.isclose(metrics["RMSE"], expected_rmse)
    assert np.isclose(metrics["PE"], expected_pe)
