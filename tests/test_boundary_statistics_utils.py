import numpy as np
import pandas as pd
import pytest

from evaluation.boundary_statistics_utils import (
    build_image_boundary_table,
    find_backward_omni_row,
    paired_boundary_metrics,
    thin_by_time,
)


def test_backward_omni_match_never_uses_future_row():
    omni = pd.DataFrame(
        {
            "utc": pd.to_datetime(["2001-01-01 03:30:00", "2001-01-01 03:35:00"]),
            "Bz": [-5.0, -6.0],
        }
    )
    target = pd.Timestamp("2001-01-01 03:34:40")
    row = find_backward_omni_row(omni, target, tolerance=pd.Timedelta("10min"))
    assert row["utc"] == pd.Timestamp("2001-01-01 03:30:00")
    assert row["_dt"] == pd.Timedelta(minutes=4, seconds=40)


def _boundary_frame(values):
    data = {"Year": [2001, 2001], "SOY": [0, 0]}
    for i in range(24):
        data[f"MLT_{i}"] = [values[i], values[i]]
    return pd.DataFrame(data)


def test_identical_image_duplicates_are_removed():
    values = np.arange(24, dtype=float) + 60.0
    table = build_image_boundary_table(_boundary_frame(values), _boundary_frame(values + 5.0))
    assert len(table) == 1
    assert table.loc[0, "paired_valid_count"] == 24


def test_conflicting_image_duplicates_fail():
    values = np.arange(24, dtype=float) + 60.0
    ealb = _boundary_frame(values)
    ealb.loc[1, "MLT_3"] += 1.0
    with pytest.raises(ValueError, match="conflicting duplicate"):
        build_image_boundary_table(ealb, _boundary_frame(values + 5.0))


def test_paired_metrics_use_common_valid_bins():
    gt = [60.0, 61.0, 62.0, 63.0]
    amt = [60.0, np.nan, 64.0, 63.0]
    ov = [62.0, 61.0, np.nan, 64.0]
    result = paired_boundary_metrics(gt, amt, ov)
    assert result["n_common_valid"] == 2
    assert result["amt_mae"] == pytest.approx(0.0)
    assert result["ov_mae"] == pytest.approx(1.5)


def test_time_thinning_is_global_chronological_greedy():
    df = pd.DataFrame(
        {"utc": pd.to_datetime(["2001-01-01 00:00", "2001-01-01 00:30", "2001-01-01 01:00", "2001-01-01 02:01"])}
    )
    thinned = thin_by_time(df, pd.Timedelta(minutes=60))
    assert thinned["utc"].tolist() == [
        pd.Timestamp("2001-01-01 00:00"),
        pd.Timestamp("2001-01-01 01:00"),
        pd.Timestamp("2001-01-01 02:01"),
    ]
