import numpy as np
import pandas as pd
import pytest

from evaluation.ovation_driver import build_ovation_weighted_ec, calculate_newell_coupling


def test_constant_hourly_solar_wind_preserves_constant_coupling():
    times = pd.date_range("2014-01-01 00:00", periods=5 * 12, freq="5min")
    df = pd.DataFrame(
        {
            "utc": times,
            "Bx": 1.0,
            "By": 2.0,
            "Bz": -5.0,
            "Vx": -400.0,
            "Vy": 0.0,
            "Vz": 0.0,
        }
    )
    target = pd.Timestamp("2014-01-01 04:30")
    result = build_ovation_weighted_ec(df, [target])[0]
    expected = calculate_newell_coupling(2.0, -5.0, 400.0)
    assert result == pytest.approx(expected)


def test_four_hour_driver_returns_nan_without_complete_history():
    times = pd.date_range("2014-01-01 00:00", periods=24, freq="5min")
    df = pd.DataFrame(
        {
            "utc": times,
            "Bx": 1.0,
            "By": 2.0,
            "Bz": -5.0,
            "Vx": -400.0,
            "Vy": 0.0,
            "Vz": 0.0,
        }
    )
    value = build_ovation_weighted_ec(df, [pd.Timestamp("2014-01-01 01:30")])[0]
    assert np.isnan(value)
