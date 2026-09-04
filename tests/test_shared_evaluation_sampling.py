import numpy as np
import pandas as pd

from evaluation.evaluate_pixelwise_ovation import sample_group_by_unique_utc
from evaluation.evaluate_spatial_mlt_mlat import sample_by_utc


def test_spatial_sampling_matches_pixelwise_all_sampling_exactly():
    times = pd.date_range("2014-01-01", periods=20, freq="5min")
    df = pd.DataFrame(
        {
            "utc": np.repeat(times, 2),
            "Bz": 0.0,
            "P_dyn": 2.0,
            "ele_energy_flux": 1.0,
            "ion_energy_flux": 0.1,
        }
    )

    pixelwise = sample_group_by_unique_utc(
        df,
        "All",
        max_unique_times=7,
        seed=42,
    )
    spatial = sample_by_utc(df, 7, seed=42)

    assert set(pixelwise["utc"].unique()) == set(spatial["utc"].unique())
